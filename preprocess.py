# Author: Harsh Kohli
# Date Created: 24-04-2021

import yaml
import pickle
import tensorflow_hub as hub
from utils.data_utils import create_tables_batches, process_train_data, create_train_batches, read_data, \
    create_samples_batches

config = yaml.safe_load(open('config.yml', 'r'))
nnlm_embedder = hub.load(config['tf_hub_model'])
batch_size = config['batch_size']

print('Processing Train Data...')
train_data, train_tables, in_domain_test = process_train_data(config, nnlm_embedder, config['train_data'],
                                                              config['train_tables'])
train_batches = create_train_batches(train_data, train_tables, config)

train_samples_batches = create_samples_batches(train_data, batch_size)
train_tables_batches = create_tables_batches(train_tables, config)

if config['use_in_domain_test']:
    in_domain_test_batches = create_samples_batches(in_domain_test, batch_size)

print('Processing Dev Data...')
dev_data, dev_tables, _ = read_data(config['dev_data'], config['dev_tables'], config['real_proxy_token'])
dev_samples_batches = create_samples_batches(dev_data, batch_size)
dev_tables_batches = create_tables_batches(dev_tables, config)

print('Processing Test Data...')
test_data, test_tables, _ = read_data(config['test_data'], config['test_tables'], config['real_proxy_token'])
test_samples_batches = create_samples_batches(test_data, batch_size)
test_tables_batches = create_tables_batches(test_tables, config)

all_data = {
    'train_batches': train_batches,
    'dev_samples_batches': dev_samples_batches,
    'dev_tables_batches': dev_tables_batches,
    'test_samples_batches': test_samples_batches,
    'test_tables_batches': test_tables_batches,
    'train_tables_batches': train_tables_batches,
    'train_samples_batches': train_samples_batches
}

if config['use_in_domain_test']:
    all_data['train_tables_batches'] = train_tables_batches
    all_data['in_domain_test_batches'] = in_domain_test_batches

pickle_file = open(config['preprocessed_data_path'], 'wb')
pickle.dump(all_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
pickle_file.close()
