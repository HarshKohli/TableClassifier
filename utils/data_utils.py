# Author: Harsh Kohli
# Date Created: 24-04-2021

import json
import pickle
import random
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
    samples_data, tables_data = [], {}
    for line in sample_file.readlines():
        sample = json.loads(line)
        question, table_id = sample['question'], sample['table_id']
        question_tokens = cleanly_tokenize(question)
        samples_data.append({'table_id': table_id, 'question_tokens': question_tokens})
    for line in table_file.readlines():
        table = json.loads(line)
        table_id, header, types, rows = table['id'], table['header'], table['types'], table['rows']
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
    random.shuffle(samples_data)
    return samples_data, tables_data


def cleanly_tokenize(text):
    tokens = word_tokenize(
        text.lower().replace("-", " - ").replace('–', ' – ').replace("''", '" ').replace("``", '" ').replace("/",
                                                                                                             " / "))
    return " ".join(tokens)
