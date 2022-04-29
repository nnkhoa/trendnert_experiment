import pandas as pd
import json
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
from gensim import models

from tqdm import tqdm


def get_most_frequent_bigram_title(text, top_n=100):
    cv = CountVectorizer(ngram_range=(2, 3), min_df=10, stop_words='english', lowercase=True)
    freq_matrix = cv.fit_transform(text)

    freq = dict(zip(cv.get_feature_names(), freq_matrix.toarray().sum(axis=0)))

    sorted_freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))

    ngram_list = [ngram for ngram in list(sorted_freq.keys())]

    ngram_list_length = len(ngram_list)
    i = 0
    while i < ngram_list_length:
        ngram = ngram_list[i]
        tmp_list = ngram_list.copy()
        tmp_list.remove(ngram)

        if any(ngram in entry for entry in tmp_list):
            ngram_list.remove(ngram)
            ngram_list_length = len(ngram_list)
        else:
            i += 1
    return ngram_list[:top_n]


def map_kws_to_topics(kw_list, label_list, threshold=0.6):
    st_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')

    topic_kw_map = []
    kw_embeddings = st_model.encode(kw_list, convert_to_tensor=True)

    for label in label_list:
        map_dict = {}
        label_embeddings = st_model.encode(label.lower(), convert_to_tensor=True)
        cosine_sims = util.cos_sim(label_embeddings, kw_embeddings)

        score_dict = {}
        for i, value in enumerate(kw_list):
            score_dict[value] = cosine_sims[0][i].item()

        score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))

        relevant_kw_list = []

        for key, value in score_dict.items():
            if value > threshold:
                relevant_kw_list.append(key)

        map_dict['label'] = label
        map_dict['keywords'] = relevant_kw_list

        topic_kw_map.append(map_dict)

    return topic_kw_map


def get_ranked_matrix(matrix_similarity, sorted_matrix):
    # Getting the pair of keyword ranking based on similarity score
    # This is redundant to do since the core concept is literally sorting
    # But, gotta follow the paper's explanation orz
    rank = 0
    ranked_matrix = np.zeros(shape=(matrix_similarity.shape[0], matrix_similarity.shape[0]))

    for i in range(matrix_similarity.shape[0]):
        for j in range(matrix_similarity.shape[0]):
            temp = sorted_matrix[i][j]
            for k in range(matrix_similarity.shape[0]):
                for p in range(matrix_similarity.shape[0]):
                    if (matrix_similarity[k][p] != 1.0) and (temp != 1.0):
                        if matrix_similarity[k][p] == temp:
                            ranked_matrix[k][p] = rank + 1
                            rank += 1
                            continue
    return np.triu(ranked_matrix, 1)


def leap2trend_ranking(kw_list, year):
    model_filename = WORD2VEC_PATH + f'year_{year}.model'
    vector_filename = f'vector_{year}.kv'
    if not os.path.exists(WORD2VEC_PATH + vector_filename):
        print('Creating Keyed Vectors file')
        print('loading model...')
        model = models.Word2Vec.load(model_filename)
        model.wv.save(WORD2VEC_PATH + vector_filename)

    word_vector = models.KeyedVectors.load(WORD2VEC_PATH + vector_filename)

    vector_list = []
    kw_presented = unique_kw.copy()
    for kw in kw_list:
        try:
            kw_vector = word_vector[kw]
            vector_list.append(kw_vector)
        except KeyError:
            print(f'{kw} did not appear in this period. Skipping')
            kw_presented.remove(kw)
            continue

    if len(vector_list):
        # reduced_similarity_matrix = np.zeros((len(kw_list), len(kw_list)))
        # kw_presented = kw_list.copy()
        similarity_matrix = cosine_similarity(vector_list)
        reduced_similarity_matrix = np.round(np.triu(similarity_matrix, 1), 3)

    # sort similarity score descending
        sorted_vector = np.flip(np.sort(np.concatenate(reduced_similarity_matrix, axis=0)))
        sorted_matrix = np.reshape(sorted_vector, (reduced_similarity_matrix.shape[0], reduced_similarity_matrix.shape[1]))

        # redundant but needed OMFG
        ranked_matrix = get_ranked_matrix(reduced_similarity_matrix, sorted_matrix)

        pair_keyword_rankings = []

        for i in range(ranked_matrix.shape[0]):
            for j in range(i, ranked_matrix.shape[0]):
                if kw_presented[i] != kw_presented[j]:
                    pair_keyword_rankings.append(('-'.join([kw_presented[i], kw_presented[j]]),
                                                  ranked_matrix[i][j],
                                                  reduced_similarity_matrix[i][j]))
        # the higher the similarity score the better the rank, rank 1 being the best
        pair_keyword_rankings.sort(key=lambda value: value[1])

        idx = [entry[0] for entry in pair_keyword_rankings]
        similarity_rank = [entry[1] for entry in pair_keyword_rankings]

        return pd.Series(similarity_rank, index=idx)

    return pd.Series([])


if __name__ == '__main__':
    WORD2VEC_PATH = '/Users/khoanguyen/Workspace/dataset/trendnert/w2v_model/'
    data_path = '/Users/khoanguyen/Workspace/dataset/trendnert/trendnert_partial.gz'

    data = pd.read_json(data_path, lines=True, compression='gzip')
    title_list = data['title'].tolist()

    label_list = data['label'].unique().tolist()
    label_list.remove(None)

    if not os.path.exists('kw_label_map.json'):
        print('Creating keyword-label map!')
        top_ngram = get_most_frequent_bigram_title(title_list, top_n=500)

        topic_kw_map = map_kws_to_topics(top_ngram, label_list, threshold=0.55)

        with open('kw_label_map.json', 'w+') as f:
            json.dump(topic_kw_map, f, indent=4)

    with open('kw_label_map.json', 'r+') as f:
        topic_kw_map = json.load(f)

    kw_mapped = []

    for entry in topic_kw_map:
        kw_mapped.extend(entry['keywords'])

    unique_kw = list(set(kw_mapped))

    unique_kw = [kw.replace(' ', '_') for kw in unique_kw]

    kw_pair_ranking_df = pd.DataFrame(columns=range(2000, 2016))

    for year in tqdm(range(2000, 2016)):
        ranking = leap2trend_ranking(unique_kw, year)

        kw_pair_ranking_df[year] = ranking
    # kw_pair_ranking_df.fillna(0, inplace=True)
    kw_pair_ranking_df.T.to_csv('leap2trend.csv')
