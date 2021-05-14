# Author: Harsh Kohli
# Date Created: 24-04-2021

import json
import pickle
import random
from utils.embedding_utils import compute_embeddings, index_embeddings, get_hardest_negatives
from nltk import word_tokenize


def create_train_batches(data, tables, config):
    batch_size, pad_token = config['batch_size'], config['pad_token']
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in
               range((len(data) + batch_size - 1) // batch_size)]
    questions, headers, table_words, labels, all_num_cols, masks = [], [], [], [], [], []
    for index, batch in enumerate(batches):
        question, header, words, label, num_cols = [], [], [], [], []
        max_cols = 0
        for datum in batch:
            question.append(datum['question_tokens'])
            table_info = tables[datum['table_id']]
            a, b = [], []
            num_col = len(table_info)
            num_cols.append(num_col)
            if num_col > max_cols:
                max_cols = num_col
            for col in table_info:
                a.append(col['header'])
                b.append(col['words'])
            header.append(a)
            words.append(b)
            label.append(datum['label'])
        mask = [[0.0 for _ in range(max_cols)] for _ in range(len(header))]
        for index2, one_mask in enumerate(mask):
            for index3 in range(num_cols[index2]):
                one_mask[index3] = 1.0
        header = [x + [pad_token] * (max_cols - len(x)) for x in header]
        words = [x + [pad_token] * (max_cols - len(x)) for x in words]
        masks.append(mask)
        labels.append(label)
        questions.append(question)
        all_num_cols.append(num_cols)
        headers.append(header)
        table_words.append(words)
    return questions, headers, table_words, labels, all_num_cols, masks


def create_tables_batches(tables, config):
    batch_size, pad_token = config['batch_size'], config['pad_token']
    tables_list = [(x, tables[x]) for x in tables]
    batches = [tables_list[i * batch_size:(i + 1) * batch_size] for i in
               range((len(tables_list) + batch_size - 1) // batch_size)]
    headers, table_words, all_num_cols, masks, all_table_ids = [], [], [], [], []
    for index, batch in enumerate(batches):
        header, words, label, num_cols, table_ids = [], [], [], [], []
        max_cols = 0
        for table_id, table_info in batch:
            a, b = [], []
            num_col = len(table_info)
            num_cols.append(num_col)
            if num_col > max_cols:
                max_cols = num_col
            for col in table_info:
                a.append(col['header'])
                b.append(col['words'])
            header.append(a)
            words.append(b)
            table_ids.append(table_id)
        mask = [[0.0 for _ in range(max_cols)] for _ in range(len(header))]
        for index2, one_mask in enumerate(mask):
            for index3 in range(num_cols[index2]):
                one_mask[index3] = 1.0
        header = [x + [pad_token] * (max_cols - len(x)) for x in header]
        words = [x + [pad_token] * (max_cols - len(x)) for x in words]
        all_table_ids.append(table_ids)
        masks.append(mask)
        all_num_cols.append(num_cols)
        headers.append(header)
        table_words.append(words)
    return headers, table_words, all_num_cols, masks, all_table_ids


def create_samples_batches(samples, batch_size):
    batches = [samples[i * batch_size:(i + 1) * batch_size] for i in
               range((len(samples) + batch_size - 1) // batch_size)]
    return batches


def load_preprocessed_data(config):
    serialized_data_file = open(config['preprocessed_data_path'], 'rb')
    data = pickle.load(serialized_data_file)
    return data


def read_data(data_file, tables_file, real_proxy_token):
    sample_file = open(data_file, "r", encoding='utf8')
    table_file = open(tables_file, "r", encoding='utf8')
    samples_data, tables_data, all_questions, all_tables = [], {}, [], set()
    for line in sample_file.readlines():
        sample = json.loads(line)
        question, table_id = sample['question'], sample['table_id']
        question_tokens = cleanly_tokenize(question)
        all_questions.append(question_tokens)
        samples_data.append({'table_id': table_id, 'question_tokens': question_tokens, 'label': 1.0})
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
    sample_file.close()
    table_file.close()
    return samples_data, tables_data, all_questions


def isolate_in_domain_test(samples_data, all_questions):
    train_samples, train_questions, test_samples = [], [], []
    table_counts = {}

    for sample, question in zip(samples_data, all_questions):
        table_id = sample['table_id']
        if table_id not in table_counts:
            table_counts[table_id] = 1
        elif table_counts[table_id] == 5:
            test_samples.append(sample)
        else:
            train_samples.append(sample)
            train_questions.append(question)
            table_counts[table_id] = table_counts[table_id] + 1

    return train_samples, train_questions, test_samples


def process_train_data(config, nnlm_embedder, data_file, tables_file):
    train_index = config['train_index']
    samples_data, tables_data, all_questions = read_data(data_file, tables_file, config['real_proxy_token'])
    if config['use_in_domain_test']:
        samples_data, all_questions, in_domain_test = isolate_in_domain_test(samples_data, all_questions)
    else:
        in_domain_test = None
    embeddings = compute_embeddings(all_questions, nnlm_embedder, config['batch_size'])
    index, id2_embed, id2_question = 0, {}, {}
    for embedding, sample in zip(embeddings, samples_data):
        id2_embed[index] = embedding.numpy()
        index = index + 1
    index_embeddings(id2_embed, train_index, config['dim'])
    hardest_negatives, random_negatives = [], []
    if config['include_hardest_negatives']:
        print('Computing Hardest Negatives...')
        hardest_negatives = get_hardest_negatives(samples_data, train_index, config['dim'])
    if config['include_random_negatives']:
        print('Computing Random Negatives...')
        for sample in samples_data:
            random_index = random.randint(0, len(samples_data) - 1)
            random_negative = samples_data[random_index]
            if random_negative['table_id'] != sample['table_id']:
                random_negatives.append(
                    {'table_id': random_negative['table_id'], 'question_tokens': sample['question_tokens'],
                     'label': 0.0})
    samples_data.extend(hardest_negatives)
    samples_data.extend(random_negatives)
    random.shuffle(samples_data)
    return samples_data, tables_data, in_domain_test


def cleanly_tokenize(text):
    text = text.lower()
    tokens = word_tokenize(
        text.lower().replace("-", " - ").replace('–', ' – ').replace("''", '" ').replace("``", '" ').replace("/",
                                                                                                             " / "))
    return " ".join(tokens)
