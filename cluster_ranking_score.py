import json
import numpy as np
import logging
from sklearn.metrics import classification_report


def get_rank(gold_list, list_predictions, max_k=3):
    list_predictions = list_predictions[:max_k]
    rank = []
    for gold_idx, gold in enumerate(gold_list):
        try:
            rank.append(abs(list_predictions.index(gold) - gold_idx) + 1)
        except ValueError:
            rank.append(max_k + 1)
    return rank


def compute_average_rank(Y_test, predictions):
    ranks = []
    for idx, y in enumerate(Y_test):
        logging.debug(f"y={Y_test[idx]}, predictions={predictions[idx]}")
        ranks.extend(get_rank(Y_test[idx], predictions[idx], max_k=len(Y_test[idx]) + 3))
    return np.mean(ranks)


def compute_precision(Y_test, predictions):
    one_label_groundtruth = []
    one_label_prediction = []
    for idx, y_test in enumerate(Y_test):
        if len(y_test) == 1:
            one_label_groundtruth.append(y_test[0])
            one_label_prediction.append(predictions[idx][0])

    print(classification_report(one_label_groundtruth, one_label_prediction))


if __name__ == '__main__':
    prediction_file = '/Users/khoanguyen/Workspace/dataset/trendnert/output/cluster_label_prediction.json'

    results = json.load(open(prediction_file, 'r'))

    groundtruth = [entry['cluster_label'] for entry in results]
    prediction = [entry['prediction'] for entry in results]

    print("Average Rank:", compute_average_rank(groundtruth, prediction))

    compute_precision(groundtruth, prediction)

    '''
    REMARK: As the data is incomplete (chances are each cluster can contain more topics than shown in 15% partial
    data that we have), this experiment needs to put on hold
    '''

