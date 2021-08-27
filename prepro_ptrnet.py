# Author: Harsh Kohli
# Date Created: 12-08-2021

import os
import yaml
import pickle
import tensorflow as tf
from utils.embedding_utils import index_embeddings, get_top_k_tables
from utils.training_utils import test_query_encoder, test_table_encoder
from utils.data_utils import load_preprocessed_data, create_ptrnet_dataset

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

model_name = config['best_model']
dim, k, batch_size = config['dim'], config['top_k'], config['ptrnet_batch_size']

data = load_preprocessed_data(config['preprocessed_data_path'])
train_headers, train_table_words, train_all_num_cols, train_masks, train_table_ids = data['train_tables_batches']
dev_headers, dev_table_words, dev_all_num_cols, dev_masks, dev_table_ids = data['dev_tables_batches']
test_headers, test_table_words, test_all_num_cols, test_masks, test_table_ids = data['test_tables_batches']

model = tf.keras.models.load_model(os.path.join(config['model_dir'], config['best_model']))


@tf.function
def query_embedding_step(question):
    return model.get_query_embedding(question)


@tf.function
def table_embedding_step(header, table_word, all_num_col, mask):
    return model.get_table_embedding(header, table_word, all_num_col, mask)


dev_index = os.path.join(config['index_dir'], config['dev_tables_index'])

print('Computing Dev Query Embeddings...')
dev_sample_info_dict = test_query_encoder(data['dev_samples_batches'], query_embedding_step)
print('Computing Dev Table Embeddings...')
dev_index_to_vec, dev_id_to_index = test_table_encoder(dev_headers, dev_table_words, dev_all_num_cols, dev_masks,
                                                       dev_table_ids, table_embedding_step)

index_embeddings(dev_index_to_vec, dev_index, dim)
print('Indexing Dev Table Embeddings...')
get_top_k_tables(dev_sample_info_dict, dev_id_to_index, dev_index, dim, k)

test_index = os.path.join(config['index_dir'], config['test_tables_index'])

print('Computing Test Query Embeddings...')
test_sample_info_dict = test_query_encoder(data['test_samples_batches'], query_embedding_step)
print('Computing Test Table Embeddings...')
test_index_to_vec, test_id_to_index = test_table_encoder(test_headers, test_table_words, test_all_num_cols, test_masks,
                                                         test_table_ids, table_embedding_step)

index_embeddings(test_index_to_vec, test_index, dim)
print('Indexing Test Table Embeddings...')
get_top_k_tables(test_sample_info_dict, test_id_to_index, test_index, dim, k)

train_index = os.path.join(config['index_dir'], config['train_tables_index'])

print('Computing Train Query Embeddings...')
train_sample_info_dict = test_query_encoder(data['train_samples_batches'], query_embedding_step)
print('Computing Train Table Embeddings...')
train_index_to_vec, train_id_to_index = test_table_encoder(train_headers, train_table_words, train_all_num_cols,
                                                           train_masks, train_table_ids, table_embedding_step)

index_embeddings(train_index_to_vec, train_index, dim)
print('Indexing Train Table Embeddings...')
get_top_k_tables(train_sample_info_dict, train_id_to_index, train_index, dim, k)

dev_batches = create_ptrnet_dataset(dev_sample_info_dict, batch_size)
test_batches = create_ptrnet_dataset(test_sample_info_dict, batch_size)
train_batches = create_ptrnet_dataset(train_sample_info_dict, batch_size)

all_data = {
    'train_batches': train_batches,
    'dev_batches': dev_batches,
    'test_batches': test_batches,
    'train_index_to_vec': train_index_to_vec,
    'dev_index_to_vec': dev_index_to_vec,
    'test_index_to_vec': test_index_to_vec
}

pickle_file = open(config['preprocessed_ptrnet_data'], 'wb')
pickle.dump(all_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
pickle_file.close()
