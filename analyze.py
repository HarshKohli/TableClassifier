# Author: Harsh Kohli
# Date created: 3/31/2021

import json

f = open('dataset/train.jsonl', 'r', encoding='utf8')
print(len(f.readlines()))
# for line in f.readlines():
#     one_dict = json.loads(line)
#     print('here')
#
# train_tables = json.load(f)
# print('Done')
