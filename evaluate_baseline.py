# Author: Harsh Kohli
# Date Created: 11-08-2021

import yaml
import os
from utils.training_utils import baseline_metrics_logger
from utils.data_utils import read_data, get_ranks_baseline

config = yaml.safe_load(open('config.yml', 'r'))

print('Computing Dev Basline...')
data, tables, _ = read_data(config['dev_data'], config['dev_tables'], config['real_proxy_token'])
dev_ranks = get_ranks_baseline(data, tables)

print('Computing Test Basline...')
data, tables, _ = read_data(config['test_data'], config['test_tables'], config['real_proxy_token'])
test_ranks = get_ranks_baseline(data, tables)

log_path = os.path.join(config['log_dir'], config['baseline_name'])
find_p = config['find_p']
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file = open(os.path.join(log_path, config['baseline_name'] + '.tsv'), 'w', encoding='utf8')
log_file.write('Eval Set' + '\t' + 'MRR')
for p in find_p:
    log_file.write('\tP@' + str(p))
log_file.write('\n')
baseline_metrics_logger(dev_ranks, test_ranks, log_file, find_p)
log_file.close()
