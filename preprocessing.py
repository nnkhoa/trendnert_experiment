import itertools
import pandas as pd
from tqdm import tqdm
import gensim
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
import nltk.data

tqdm.pandas()


def text_preprocess(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    text = re.sub(r'\d+', '', text)
    text = text.casefold()
    text = remove_stopwords(text)

    sentences = tokenizer.tokenize(text)

    return sentences


if __name__ == '__main__':
    data_path = '/Users/khoanguyen/Workspace/dataset/trendnert/trendnert_partial.gz'
    save_text_path = '/Users/khoanguyen/Workspace/dataset/trendnert/paperAbstractTitle/'
    data = pd.read_json(data_path, lines=True, compression='gzip')

    year_list = data['year'].unique().tolist()

    for year in tqdm(year_list):
        data_by_year = data.loc[data['year'] == year]

        doc_sentences = data_by_year['paperAbstract'].apply(lambda x: text_preprocess(x)).tolist()
        paper_title_list = data_by_year['title'].apply(lambda x: text_preprocess(x)).tolist()

        for i, doc in enumerate(doc_sentences):
            try:
                doc.insert(0, paper_title_list[i][0])
            except IndexError:
                pass

        output_name = 'year_' + str(year)
        with open(save_text_path + output_name, 'w+') as f:
            for doc in doc_sentences:
                for sentence in doc:
                    f.write(f'{sentence}\n')
