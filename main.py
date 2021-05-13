# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
import tensorflow as tf
import os
from models import QuerySchemaEncoder
from utils.training_utils import train_epoch, test_query_encoder, test_table_encoder
from utils.embedding_utils import index_embeddings
from utils.data_utils import load_preprocessed_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

batch_size = config['batch_size']
model_name = config['model_name']
data = load_preprocessed_data(config)

questions, headers, table_words, labels, all_num_cols, masks = data['train_batches']
train_headers, train_table_words, train_all_num_cols, train_masks, train_table_ids = data['train_tables_batches']
dev_headers, dev_table_words, dev_all_num_cols, dev_masks, dev_table_ids = data['dev_tables_batches']
test_headers, test_table_words, test_all_num_cols, test_masks, test_table_ids = data['test_tables_batches']

model = QuerySchemaEncoder(config)
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
save_path = os.path.join(config['model_dir'], model_name)


@tf.function
def train_step(question, header, table_word, label, all_num_col, mask):
    with tf.GradientTape() as tape:
        loss = model([question, header, table_word, label, all_num_col, mask])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        (grad, var)
        for (grad, var) in zip(gradients, model.trainable_variables)
        if grad is not None
    )
    return loss


@tf.function
def query_embedding_step(question):
    return model.get_query_embedding(question)


@tf.function
def table_embedding_step(headers, table_words, all_num_cols, masks):
    return model.get_table_embedding(headers, table_words, all_num_cols, masks)


train_iterations = len(questions)
print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    train_epoch(questions, headers, table_words, labels, all_num_cols, masks, train_iterations, train_step)
    print('Completed Epoch. Saving Latest Model...')
    tf.saved_model.save(model, os.path.join(save_path, str(epoch_num)))

    print('Computing In-Domain Query Embeddings...')
    indomain_sample_info_dict = test_query_encoder(data['in_domain_test_batches'], query_embedding_step)
    print('Computing Train Table Embeddings...')
    train_index_to_vec, train_id_to_index = test_table_encoder(train_headers, train_table_words, train_all_num_cols,
                                                               train_masks, train_table_ids, table_embedding_step)
    print('Indexing Train Table Embeddings...')
    index_embeddings(train_index_to_vec,
                     os.path.join(config['index_dir'], model_name, str(epoch_num), config['train_tables_index']),
                     config['dim'])

    print('Computing Dev Query Embeddings...')
    dev_sample_info_dict = test_query_encoder(data['dev_samples_batches'], query_embedding_step)
    print('Computing Dev Table Embeddings...')
    dev_index_to_vec, dev_id_to_index = test_table_encoder(dev_headers, dev_table_words, dev_all_num_cols, dev_masks,
                                                           dev_table_ids, table_embedding_step)
    print('Indexing Dev Table Embeddings...')
    index_embeddings(dev_index_to_vec,
                     os.path.join(config['index_dir'], model_name, str(epoch_num), config['dev_tables_index']),
                     config['dim'])

    print('Computing Test Query Embeddings...')
    test_sample_info_dict = test_query_encoder(data['test_samples_batches'], query_embedding_step)
    print('Computing Test Table Embeddings...')
    test_index_to_vec, test_id_to_index = test_table_encoder(test_headers, test_table_words, test_all_num_cols,
                                                             test_masks, test_table_ids, table_embedding_step)
    print('Indexing Test Table Embeddings...')
    index_embeddings(test_index_to_vec,
                     os.path.join(config['index_dir'], model_name, str(epoch_num), config['test_tables_index']),
                     config['dim'])

    break
