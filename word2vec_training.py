from gensim import models
import glob
import os
from tqdm import tqdm


def w2v_training(train_data_path, model_output, skip_gram=True):
    sentences = models.word2vec.LineSentence(train_data_path)
    bigram_model = models.Phrases(sentences, min_count=10, progress_per=10000)

    if skip_gram:
        model = models.Word2Vec(vector_size=200, min_count=10, window=6, sg=1, workers=3)
    else:
        model = models.Word2Vec(vector_size=200, min_count=10, window=6, sg=0, workers=3)

    model.build_vocab(bigram_model[sentences])

    model.train(bigram_model[sentences],
                total_examples=model.corpus_count,
                epochs=model.epochs,
                report_delay=1)

    model.save(model_output)


if __name__ == '__main__':
    text_path = '/Users/khoanguyen/Workspace/dataset/trendnert/paperAbstractTitle/'
    file_pattern = 'year_'
    model_path = '/Users/khoanguyen/Workspace/dataset/trendnert/w2v_model/'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    search_pattern = text_path + file_pattern + '*'

    text_path_list = glob.glob(search_pattern)

    for text_path in tqdm(text_path_list):
        _, text_filename = os.path.split(text_path)

        model_output = model_path + text_filename + '.model'

        w2v_training(text_path, model_output)

