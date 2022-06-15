import pandas as pd
import json
import os
import numpy as np
import pickle

from scipy.stats import linregress

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
from gensim import models

from tqdm import tqdm


def get_most_frequent_bigram_title(text, top_n=100):
    # bigram_model = models.Phrases(text, min_count=10, progress_per=10000)
    #
    # model = models.Word2Vec(vector_size=200, min_count=10, window=6, sg=0, workers=3)
    #
    # model.build_vocab(bigram_model[text])
    #
    # model.train(bigram_model[text],
    #             total_examples=model.corpus_count,
    #             epochs=model.epochs,
    #             report_delay=1)
    #
    # vocab_count = {}
    # for ngram in model.wv.key_to_index:
    #     vocab_count[ngram] = model.wv.key_to_index[ngram].count
    #
    # sorted_freq = dict(sorted(vocab_count.items(), key=lambda item: item[1], reverse=True))

    '''
    NOTE: most frequent bigram/trigram by sklearn CountVectorizer
    '''
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

        if not relevant_kw_list or len(relevant_kw_list) < 5:
            relevant_kw_list = [key for key in list(score_dict.keys())[:5]]

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


def find_label(kw, kw_label_map):
    labels = []
    for mapping in kw_label_map:
        if kw.replace('_', ' ') in mapping['keywords']:
            labels.append(mapping['label'])

    return labels


def leap2trend_ranking(kw_list, filename, word2vec_path):
    model_filename = word2vec_path + f'{filename}.model'
    vector_filename = f'vector_{filename}.kv'
    if not os.path.exists(word2vec_path + vector_filename):
        print('Creating Keyed Vectors file')
        print('loading model...')
        model = models.Word2Vec.load(model_filename)
        model.wv.save(word2vec_path + vector_filename)

    word_vector = models.KeyedVectors.load(word2vec_path + vector_filename)

    vector_list = []
    kw_presented = kw_list.copy()
    for kw in kw_list:
        if kw in word_vector:
            kw_vector = word_vector[kw]
            vector_list.append(kw_vector)
        else:
            print(f'{kw} could not be found in model. Removing...')
            kw_presented.remove(kw)
            continue
            # print(f'{kw} could not be found in model. Attempting to calculate keyword vector by averaging components')
            # kw_component = kw.split('_')
            # if (kw_component[0] in word_vector) and (kw_component[1] in word_vector):
            #     vector_list.append(np.mean([word_vector[kw_component[0]], word_vector[kw_component[1]]], axis=0))
            # else:
            #     print(f'{kw}\'s components did not appear in this time period. Removing...')
            #     kw_presented.remove(kw)
            #     continue
    print(f'Keyword found: {len(kw_presented)}/{len(kw_list)}')
    if len(vector_list):
        # reduced_similarity_matrix = np.zeros((len(kw_list), len(kw_list)))
        # kw_presented = kw_list.copy()
        similarity_matrix = cosine_similarity(vector_list)
        reduced_similarity_matrix = np.round(np.triu(similarity_matrix, 1), 3)

        # sort similarity score descending
        sorted_vector = np.flip(np.sort(np.concatenate(reduced_similarity_matrix, axis=0)))
        sorted_matrix = np.reshape(sorted_vector,
                                   (reduced_similarity_matrix.shape[0], reduced_similarity_matrix.shape[1]))

        # redundant but needed OMFG
        ranked_matrix = get_ranked_matrix(reduced_similarity_matrix, sorted_matrix)

        pair_keyword_rankings = []

        for i in range(ranked_matrix.shape[0]):
            for j in range(i, ranked_matrix.shape[0]):
                if kw_presented[i] != kw_presented[j]:
                    pair_keyword_rankings.append(('-'.join([kw_presented[i].replace('_', ' '), kw_presented[j].replace('_', ' ')]),
                                                  ranked_matrix[i][j],
                                                  reduced_similarity_matrix[i][j]))

        # the higher the similarity score the better the rank, rank 1 being the best
        pair_keyword_rankings.sort(key=lambda value: value[1])

        pair_keyword_rankings_ordered = []
        for pair in pair_keyword_rankings:
            if pair[1] != 0:
                pair = (pair[0], pair_keyword_rankings.index(pair) + 1, pair[2])
            pair_keyword_rankings_ordered.append(pair)

        idx = [entry[0] for entry in pair_keyword_rankings_ordered]
        similarity_rank = [entry[1] for entry in pair_keyword_rankings_ordered]

        return pd.Series(similarity_rank, index=idx)

    return pd.Series([])


def linregress_rank_evolution(kw_ranking_df):
    results_dict = []
    for kw_pair in kw_ranking_df.columns.tolist():
        linear_regression = linregress(kw_ranking_df.index.tolist(),
                                       kw_ranking_df[kw_pair].fillna(method='ffill').tolist())

        lin_regress_result_dict = {'slope': linear_regression[0], 'correlation_coefficient': linear_regression[2]}
        results_dict.append(lin_regress_result_dict)

    return results_dict


def apply_to_edf():
    monthly_file = ['2019-07-01', '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01',
                    '2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
                    '2020-05-01', '2020-06-01', '2020-07-01']

    kw_list = ['microsoft team', 'artificial intelligence',
               'digital transformation', 'microsoft azure',
               'cloud computing', 'cloud solution', 'health care',
               'storage server', 'data center', 'remote work']

    kw_list = [kw.replace(' ', '_') for kw in kw_list]
    word2vec_path = '/Users/khoanguyen/Workspace/dataset/edf_msft/processed_data/w2v_model/'


    ranking_list = []

    for data_file in tqdm(monthly_file):
        print(data_file)
        ranking = leap2trend_ranking(kw_list, data_file, word2vec_path)

        ranking_list.append(ranking)

    kw_pair_ranking_df = pd.concat(ranking_list, axis=1)
    kw_pair_ranking_df.columns = monthly_file
    kw_pair_ranking_df.T.to_csv('edf_original_leap2trend.csv')


if __name__ == '__main__':
    # '''
    WORD2VEC_PATH = '/Users/khoanguyen/Workspace/dataset/trendnert/w2v_model/'
    data_path = '/Users/khoanguyen/Workspace/dataset/trendnert/trendnert_partial.gz'

    data = pd.read_json(data_path, lines=True, compression='gzip')
    title_list = data['title'].tolist()

    label_list = data['label'].unique().tolist()
    label_list.remove(None)

    if not os.path.exists('kw_label_map_gensim.json'):
        print('Creating keyword-label map!')
        top_ngram = get_most_frequent_bigram_title(title_list, top_n=500)

        topic_kw_map = map_kws_to_topics(top_ngram, label_list, threshold=0.55)

        with open('kw_label_map_gensim.json', 'w+') as f:
            json.dump(topic_kw_map, f, indent=4)

    with open('kw_label_map_gensim.json', 'r+') as f:
        topic_kw_map = json.load(f)

    kw_mapped = []

    for entry in topic_kw_map:
        kw_mapped.extend(entry['keywords'])

    unique_kw = list(set(kw_mapped))

    unique_kw = [kw.replace(' ', '_') for kw in unique_kw]

    kw_pair_ranking_df = pd.DataFrame()
    # columns=range(2000, 2016)
    for year in tqdm(range(2000, 2016)):
        filename = 'year_' + str(year)
        ranking = leap2trend_ranking(unique_kw, filename, WORD2VEC_PATH)

        # kw_pair_ranking_df[year] = ranking
        kw_pair_ranking_df = pd.concat([kw_pair_ranking_df, ranking.rename(str(year))], axis=1)
    # kw_pair_ranking_df.fillna(0, inplace=True)
    kw_pair_ranking_df.T.to_csv('leap2trend.csv')
    # '''
    # apply_to_edf()
