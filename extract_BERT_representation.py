import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import re
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm
import pickle


def text_preprocess(text):
    text = re.sub(r'\d+', '', text)
    text = text.casefold()
    text = remove_stopwords(text)

    return text


def encode_text(text, model, tokenizer):
    text = '[CLS] ' + text + ' [SEP]'

    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    chunk = (len(tokenized_text) // 512) + 1

    token_vecs_sum = []

    for i in range(0, chunk):
        tokenized_chunk = tokenized_text[i:i+512]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_chunk)

        segments_id = [1] * len(tokenized_chunk)

        tokens_tensor = torch.tensor([indexed_tokens])

        segments_tensor = torch.tensor([segments_id])

    # segments_id = [1] * len(tokenized_text)
    #
    # tokens_tensor = torch.tensor([indexed_tokens])
    #
    # segments_tensor = torch.tensor([segments_id])

    # print(tokens_tensor.size())
    # print(segments_tensor.size())

        with torch.no_grad():
            output = model(tokens_tensor, segments_tensor)
            # get hidden state from all layers
            hidden_states = output[2]

        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # `token_embeddings` is a [190 x 13 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

    # print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    return tokenized_text, token_vecs_sum


def encode_text_new(text_list, model, tokenizer):
    tokenized_text = tokenizer(text_list, padding=True, truncation=True, return_tensor='pt')

    ids = tokenized_text['input_ids']
    mask = tokenized_text['attention_mask']

    output = model(ids, mask)
    hidden_states = output[2]

    token_embeddings = torch.stack(hidden_states)
    token_embeddings = token_embeddings.permute(1, 2, 0, 3)


def get_term_representation(encoded_text, tokenized_text, term, tokenizer):
    tokenized_term = tokenizer.tokenize(term)

    term_reps = []

    for i in range(len(tokenized_text)):
        if tokenized_text[i:i+len(tokenized_term)] == tokenized_term:
            term_token_tensors = torch.stack(encoded_text[i:i+len(tokenized_term)])
            term_vector = torch.mean(term_token_tensors, dim=0)

            term_reps.append(term_vector)

    # print('Term appearances: %d' % len(term_reps))

    if len(term_reps) == 0:
        vector_size = encoded_text[0].shape[0]
        empty_vector = torch.tensor([0] * vector_size, dtype=torch.float)
        term_reps.append(empty_vector)

    return term_reps


if __name__ == '__main__':
    data_path = '/Users/khoanguyen/Workspace/dataset/trendnert/trendnert_partial.gz'

    bert_model_name = 'allenai/scibert_scivocab_uncased'
    # bert_model_name = 'bert-base-uncased'

    data = pd.read_json(data_path, lines=True, compression='gzip')

    data_not_null = data.loc[data['label'].notnull()]
    data_not_null['processed_text'] = data_not_null['paperAbstract'].apply(lambda x: text_preprocess(x))
    year_list = sorted(data_not_null['year'].unique())

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)

    # tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    # model = AutoModel.from_pretrained(bert_model_name, output_hidden_states=True)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    kw_list = ['neural network', 'face detection', 'transfer learning',
               'artificial intelligence', 'object recognition', 'machine learning',
               'energy consumption', 'genetic algorithms', 'virtual machines',
               'information extraction']

    term_vector_df = pd.DataFrame(columns=kw_list, index=year_list)

    for year in tqdm(year_list[10:30]):
        print(f'Current Year: {year}')
        data_at_year = data_not_null.loc[data_not_null['year'] == year]

        text_list = data_at_year['processed_text'].tolist()

        corpus_tokenized = []
        corpus_token_vecs = []

        for text in tqdm(text_list):
            tokenized_text, token_vecs = encode_text(text, model=model, tokenizer=tokenizer)
            corpus_tokenized.append(tokenized_text)
            corpus_token_vecs.append(token_vecs)

        for term in tqdm(kw_list):
            print(f'\n Term: {term}')
            term_representations = []
            for i in range(len(text_list)):
                term_representations.extend(get_term_representation(encoded_text=corpus_token_vecs[i],
                                                                    tokenized_text=corpus_tokenized[i],
                                                                    term=term, tokenizer=tokenizer))

            empty_tensor = torch.tensor([0] * corpus_token_vecs[0][0].shape[0], dtype=torch.float)

            term_representations_non_zero = []
            for representation in term_representations:
                if torch.equal(representation, empty_tensor):
                    continue
                term_representations_non_zero.append(representation)

            if len(term_representations_non_zero) == 0:
                print('Term did not appear in this period')
                term_final_vector = empty_tensor
            else:
                print('Total Appearances: %d' % len(term_representations_non_zero))
                term_tensors = torch.stack(term_representations_non_zero)
                term_final_vector = torch.mean(term_tensors, dim=0)

            term_vector_df[term] = term_vector_df[term].astype(object)
            term_vector_df.at[year, term] = term_final_vector

    with open('term_representation_scibert_1985_2005.pickle', 'wb+') as f:
        pickle.dump(term_vector_df, f)
