# Author: Harsh Kohli
# Date Created: 24-04-2021

import yaml
import pickle
from utils.data_utils import read_data

config = yaml.safe_load(open('config.yml', 'r'))

train_data, train_tables = read_data(config['train_data'], config['train_tables'])
dev_data, dev_tables = read_data(config['train_data'], config['train_tables'])
test_data, test_tables = read_data(config['train_data'], config['train_tables'])

all_data = {'train_data': train_data, 'dev_data': dev_data, 'test_data': test_data,
            'train_tables': train_tables, 'dev_tables': dev_tables, 'test_tables': test_tables}
pickle_file = open(config['preprocessed_data_path'], 'wb')
pickle.dump(all_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
pickle_file.close()
