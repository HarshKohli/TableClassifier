# Author: Harsh Kohli
# Date Created: 24-04-2021

import json
import pickle
import random
from utils.embedding_utils import compute_embeddings, index_embeddings, get_hardest_negatives
from nltk import word_tokenize


def create_batches(data, tables, batch_size):
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in
               range((len(data) + batch_size - 1) // batch_size)]
    questions, headers, table_words = [], [], []
    for batch in batches:
        question, header, words = [], [], []
        for datum in batch:
            question.append(datum['question_tokens'])
            table_info = tables[datum['table_id']]
            a, b = [], []
            for col in table_info:
                a.append(col['header'])
                b.append(col['words'])
            header.append(a)
            words.append(b)
        questions.append(question)
        headers.append(header)
        table_words.append(words)
    return questions, headers, table_words


def load_preprocessed_data(config):
    serialized_data_file = open(config['preprocessed_data_path'], 'rb')
    data = pickle.load(serialized_data_file)
    train_data, dev_data, test_data = data['train_data'], data['dev_data'], data['test_data']
    train_tables, dev_tables, test_tables = data['train_tables'], data['dev_tables'], data['test_tables']
    return train_data, dev_data, test_data, train_tables, dev_tables, test_tables


def read_data(data_file, tables_file, real_proxy_token):
    sample_file = open(data_file, "r", encoding='utf8')
    table_file = open(tables_file, "r", encoding='utf8')
    samples_data, tables_data, all_questions, all_tables = [], {}, [], set()
    for line in sample_file.readlines():
        sample = json.loads(line)
        question, table_id = sample['question'], sample['table_id']
        question_tokens = cleanly_tokenize(question)
        all_questions.append(question_tokens)
        samples_data.append({'table_id': table_id, 'question_tokens': question_tokens, 'label': 1})
    for line in table_file.readlines():
        table = json.loads(line)
        table_id, header, types, rows = table['id'], table['header'], table['types'], table['rows']
        all_tables.add(table_id)
        col_words = {}
        for row in rows:
            for index, col in enumerate(row):
                if types[index] == 'real':
                    if index not in col_words:
                        col_words[index] = real_proxy_token
                    continue
                one_cell = cleanly_tokenize(col)
                if index not in col_words:
                    col_words[index] = one_cell
                else:
                    col_words[index] = col_words[index] + ' ' + one_cell
        one_table_data = []
        for col_no, one_col_words in col_words.items():
            header_tokens = cleanly_tokenize(header[col_no])
            one_table_data.append({'header': header_tokens, 'words': one_col_words})
        tables_data[table_id] = one_table_data
    return samples_data, tables_data, all_questions, all_tables


def process_test_data(data_file, tables_file, real_proxy_token):
    samples_data, tables_data, _, all_tables = read_data(data_file, tables_file, real_proxy_token)
    negatives = []
    for sample in samples_data:
        table_id = sample['table_id']
        for neg_id in all_tables:
            if neg_id != table_id:
                negatives.append({'table_id': neg_id, 'question_tokens': sample['question_tokens'], 'label': 0})
    samples_data.extend(negatives)
    return samples_data, tables_data


def process_train_data(config, nnlm_embedder):
    train_index = config['train_index']
    samples_data, tables_data, all_questions, _ = read_data(config['train_data'], config['train_tables'],
                                                            config['real_proxy_token'])
    embeddings = compute_embeddings(all_questions, nnlm_embedder, config['batch_size'])
    index, id2_embed, id2_question = 0, {}, {}
    for embedding, sample in zip(embeddings, samples_data):
        id2_embed[index] = embedding.numpy()
        index = index + 1
    index_embeddings(id2_embed, train_index)
    hardest_negatives, random_negatives, dim = [], [], id2_embed[0].size
    if config['include_hardest_negatives']:
        print('Computing Hardest Negatives...')
        hardest_negatives = get_hardest_negatives(samples_data, train_index, dim)
    if config['include_random_negatives']:
        print('Computing Random Negatives...')
        for sample in samples_data:
            random_index = random.randint(0, len(samples_data) - 1)
            random_negative = samples_data[random_index]
            if random_negative['table_id'] != sample['table_id']:
                random_negatives.append(
                    {'table_id': random_negative['table_id'], 'question_tokens': sample['question_tokens'], 'label': 0})
    samples_data.extend(hardest_negatives)
    samples_data.extend(random_negatives)
    random.shuffle(samples_data)
    return samples_data, tables_data


def cleanly_tokenize(text):
    tokens = word_tokenize(
        text.lower().replace("-", " - ").replace('–', ' – ').replace("''", '" ').replace("``", '" ').replace("/",
                                                                                                             " / "))
    return " ".join(tokens)
