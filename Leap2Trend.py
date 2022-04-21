from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def get_most_frequent_bigram_title(text, top_n=100):
    cv = CountVectorizer(ngram_range=(2, 2), min_df=10, stop_words='english', lowercase=True)
    freq_matrix = cv.fit_transform(text)

    freq = dict(zip(cv.get_feature_names(), freq_matrix.toarray().sum(axis=0)))

    sorted_freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))

    return [bigram for bigram in list(sorted_freq.keys())[:top_n]]


if __name__ == '__main__':
    data_path = '/Users/khoanguyen/Workspace/dataset/trendnert/trendnert_partial.gz'

    data = pd.read_json(data_path, lines=True, compression='gzip')
    title_list = data['title'].tolist()

    top_bigram = get_most_frequent_bigram_title(title_list, top_n=500)

    for bigram in top_bigram:
        print(bigram)
