# Author: Harsh Kohli
# Date Created: 24-04-2021

import yaml
import pickle
import tensorflow_hub as hub
from utils.data_utils import process_test_data, process_train_data

config = yaml.safe_load(open('config.yml', 'r'))
nnlm_embedder = hub.load(config['tf_hub_model'])

real_proxy_token = config['real_proxy_token']
print('Processing Train Data...')
train_data, train_tables = process_train_data(config, nnlm_embedder)

print('Processing Dev and Test Data...')
dev_data, dev_tables = process_test_data(config['dev_data'], config['dev_tables'], real_proxy_token)
test_data, test_tables = process_test_data(config['test_data'], config['test_tables'], real_proxy_token)

all_data = {'train_data': train_data, 'dev_data': dev_data, 'test_data': test_data,
            'train_tables': train_tables, 'dev_tables': dev_tables, 'test_tables': test_tables}
pickle_file = open(config['preprocessed_data_path'], 'wb')
pickle.dump(all_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
pickle_file.close()
