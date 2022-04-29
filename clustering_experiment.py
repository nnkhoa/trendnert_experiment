import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.data import Sentence
from tqdm import tqdm
import os
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import json


def split_into_chunks(list_file, num_chunks):
    return [list_file[i:i + num_chunks] for i in range(0, len(list_file), num_chunks)]


def document_representation(text_list):
    document_model = SentenceTransformerDocumentEmbeddings('sentence-transformers/stsb-roberta-base-v2')

    doc_emb_list = []

    for text in tqdm(text_list):
        doc_sent = Sentence(text)
        document_model.embed(doc_sent)
        doc_vector = doc_sent.embedding.detach().numpy()
        doc_emb_list.append(doc_vector)

    # root_path, file_name = os.path.split(output_path)
    # if not os.path.exists(root_path):
    #     os.mkdir(root_path)
    doc_emb_array = np.asarray(doc_emb_list)
    # np.savetxt(output_path, doc_emb_array)
    return doc_emb_array


def convert_corpus_to_vector(text_df, output_path, workers=1):
    title_list = text_df['title'].tolist()
    abstract_list = text_df['paperAbstract'].tolist()

    text_list = [title + '. ' + abstract
                 for title, abstract in zip(title_list, abstract_list)]

    if workers > 1:
        doc_per_chunk = len(text_df) // workers
        text_list_chunking = split_into_chunks(text_list, doc_per_chunk)
        with Pool(workers) as p:
            queue = p.map(document_representation, text_list_chunking)
        # document_representation(text_list, trendnert_text_trf_path)
        doc_emb_arr = np.vstack(queue)
    else:
        doc_emb_arr = document_representation(text_list)

    root_path, file_name = os.path.split(output_path)
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    np.savetxt(output_path, doc_emb_arr)


def phrase_representation(phrase_list, output_path):
    sentence_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')

    phrase_emb_list = []

    phrase_embeddings = sentence_model.encode(phrase_list)
    # phrase_emb_array = np.asarray(phrase_embeddings)

    root_path, file_name = os.path.split(output_path)
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    np.savetxt(output_path, phrase_embeddings)


def identify_cluster(text_df, phrase_emb_arr, doc_emb_arr, output_path):
    phrase_list = text_df['label'].unique().tolist()
    cluster_id_list = text_df['cid'].dropna().unique().tolist()

    results = []

    for cluster_id in tqdm(cluster_id_list):
        cluster_doc = text_df.loc[text_df['cid'] == cluster_id]
        cluster_doc_emb_arr = doc_emb_arr[cluster_doc.index.tolist()]
        cluster_centroid = cluster_doc_emb_arr.mean(axis=0)

        similarity_calc = cosine_similarity(phrase_emb_arr, [cluster_centroid])
        phrase_similarity = pd.DataFrame(similarity_calc,
                                         index=phrase_list,
                                         columns=['similarity'])

        phrase_similarity.sort_values(by='similarity', ascending=False, inplace=True)
        phrase_ranking = phrase_similarity.index.tolist()

        cluster_label = cluster_doc['label'].value_counts().index.to_list()

        result_dict = {"cluster_id": cluster_id,
                       "cluster_label": cluster_label,
                       "prediction": phrase_ranking}
        results.append(result_dict)

    if not os.path.exists(output_path):
        root_path, _ = os.path.split(output_path)
        if not os.path.exists(root_path):
            os.mkdir(root_path)

    with open(output_path, 'w+') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    trendnert_text_trf_path = '/Users/khoanguyen/Workspace/dataset/trendnert/embeddings/trendnert_partial.txt'
    trendnert_label_emb_path = '/Users/khoanguyen/Workspace/dataset/trendnert/embeddings/label_emb.txt'
    trendnert_data_path = '/Users/khoanguyen/Workspace/dataset/trendnert/trendnert_partial.gz'
    cluster_label_prediction = '/Users/khoanguyen/Workspace/dataset/trendnert/output/cluster_label_prediction.json'
    core_count = 4

    print('Loading Text data')
    trendnert_data = pd.read_json(trendnert_data_path, compression='gzip', lines=True)
    label_list = trendnert_data['label'].unique().tolist()

    if not os.path.exists(trendnert_label_emb_path):
        print("Generating labels trf representation")
        phrase_representation(label_list, trendnert_label_emb_path)

    if not os.path.exists(trendnert_text_trf_path):
        print("Generating doc trf representation")
        convert_corpus_to_vector(trendnert_text_trf_path, trendnert_text_trf_path, workers=4)

    print('Loading vector representation')
    trendnert_text_emb = np.genfromtxt(trendnert_text_trf_path)
    trendnert_label_emb = np.genfromtxt(trendnert_label_emb_path)

    print("Identifying clusters")
    identify_cluster(trendnert_data, trendnert_label_emb, trendnert_text_emb, cluster_label_prediction)