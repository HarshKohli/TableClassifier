# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
import pickle

config = yaml.safe_load(open('config.yml', 'r'))
serialized_data_file = open(config['preprocessed_data_path'], 'rb')
data = pickle.load(serialized_data_file)
