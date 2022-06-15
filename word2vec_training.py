from gensim import models
import glob
import os
from tqdm import tqdm
from optparse import OptionParser


def w2v_training(train_data_path, model_output, skip_gram=True):
    sentences = models.word2vec.LineSentence(train_data_path)
    bigram_model = models.Phrases(sentences, min_count=10, threshold=5, progress_per=10000)

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


def get_args():
    parser = OptionParser()

    parser.add_option("--data-folder", dest="data_folder",
                      help="Path the folder containing the data")
    parser.add_option("--file-prefix", dest="file_prefix",
                      help="File prefix for glob to grab what is needed")
    parser.add_option("--model-path", dest="model_path", default=None,
                      help="Path for outputting model")
    (options, args) = parser.parse_args()

    return options


if __name__ == '__main__':
    cmd_argument = get_args()
    text_path = cmd_argument.data_folder
    file_pattern = cmd_argument.file_prefix
    model_path = cmd_argument.model_path

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    search_pattern = text_path + file_pattern + '*'

    text_path_list = glob.glob(search_pattern)

    for text_path in tqdm(text_path_list):
        _, text_filename = os.path.split(text_path)

        model_output = model_path + text_filename + '.model'

        w2v_training(text_path, model_output)

