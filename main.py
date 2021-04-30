# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
from utils.data_utils import load_preprocessed_data, create_batches

config = yaml.safe_load(open('config.yml', 'r'))

batch_size = config['batch_size']
train_data, dev_data, test_data, train_tables, dev_tables, test_tables = load_preprocessed_data(config)
train_questions, train_headers, train_table_words = create_batches(train_data, train_tables, batch_size)
dev_questions, dev_headers, dev_table_words = create_batches(train_data, train_tables, batch_size)
test_questions, test_headers, test_table_words = create_batches(train_data, train_tables, batch_size)
