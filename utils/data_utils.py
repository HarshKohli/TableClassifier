# Author: Harsh Kohli
# Date Created: 24-04-2021

import json
from nltk import word_tokenize


def read_data(data_file, tables_file):
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
                        col_words[index] = '<unk>'
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
    return samples_data, tables_data


def cleanly_tokenize(text):
    tokens = word_tokenize(text.lower().replace("-", " - ").replace('–', ' – ').replace("''", '" ').replace("``", '" '))
    return " ".join(tokens)
