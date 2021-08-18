# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
import tensorflow as tf
import os
from models import QuerySchemaEncoder
from utils.training_utils import train_epoch, test_query_encoder, test_table_encoder, metrics_logger
from utils.embedding_utils import index_embeddings
from utils.data_utils import load_preprocessed_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

batch_size = config['batch_size']
model_name = config['model_name']
data = load_preprocessed_data(config)

questions, headers, table_words, labels, all_num_cols, masks = data['train_batches']
if config['use_in_domain_test']:
    train_headers, train_table_words, train_all_num_cols, train_masks, train_table_ids = data['train_tables_batches']
dev_headers, dev_table_words, dev_all_num_cols, dev_masks, dev_table_ids = data['dev_tables_batches']
test_headers, test_table_words, test_all_num_cols, test_masks, test_table_ids = data['test_tables_batches']

model = QuerySchemaEncoder(config)
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
save_path = os.path.join(config['model_dir'], model_name)
log_path = os.path.join(config['log_dir'], model_name)
dim = config['dim']


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
def table_embedding_step(header, table_word, all_num_col, mask):
    return model.get_table_embedding(header, table_word, all_num_col, mask)


train_iterations = len(questions)
print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    train_epoch(questions, headers, table_words, labels, all_num_cols, masks, train_iterations, train_step)
    print('Completed Epoch. Saving Latest Model...')
    tf.keras.models.save_model(model, os.path.join(save_path, str(epoch_num)))

    index_dir = os.path.join(config['index_dir'], model_name, str(epoch_num))
    train_index = os.path.join(index_dir, config['train_tables_index'])
    dev_index = os.path.join(index_dir, config['dev_tables_index'])
    test_index = os.path.join(index_dir, config['test_tables_index'])

    if config['use_in_domain_test']:
        print('Computing In-Domain Query Embeddings...')
        indomain_sample_info_dict = test_query_encoder(data['in_domain_test_batches'], query_embedding_step)
        print('Computing Train Table Embeddings...')
        train_index_to_vec, train_id_to_index = test_table_encoder(train_headers, train_table_words, train_all_num_cols,
                                                                   train_masks, train_table_ids, table_embedding_step)
        print('Indexing Train Table Embeddings...')
        index_embeddings(train_index_to_vec, train_index, dim)

    print('Computing Dev Query Embeddings...')
    dev_sample_info_dict = test_query_encoder(data['dev_samples_batches'], query_embedding_step)
    print('Computing Dev Table Embeddings...')
    dev_index_to_vec, dev_id_to_index = test_table_encoder(dev_headers, dev_table_words, dev_all_num_cols, dev_masks,
                                                           dev_table_ids, table_embedding_step)
    print('Indexing Dev Table Embeddings...')
    index_embeddings(dev_index_to_vec, dev_index, dim)

    print('Computing Test Query Embeddings...')
    test_sample_info_dict = test_query_encoder(data['test_samples_batches'], query_embedding_step)
    print('Computing Test Table Embeddings...')
    test_index_to_vec, test_id_to_index = test_table_encoder(test_headers, test_table_words, test_all_num_cols,
                                                             test_masks, test_table_ids, table_embedding_step)
    print('Indexing Test Table Embeddings...')
    index_embeddings(test_index_to_vec, test_index, dim)

    print('Logging Metrics...')
    p_req = config['find_p']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = open(os.path.join(log_path, str(epoch_num) + '.tsv'), 'w', encoding='utf8')
    log_file.write('Eval Set' + '\t' + 'MRR')
    for p in p_req:
        log_file.write('\tP@' + str(p))
    log_file.write('\n')
    if config['use_in_domain_test']:
        metrics_logger(indomain_sample_info_dict, train_id_to_index, train_index, dim, 'in_domain_test', p_req,
                       log_file)
    metrics_logger(dev_sample_info_dict, dev_id_to_index, dev_index, dim, 'dev', p_req, log_file)
    metrics_logger(test_sample_info_dict, test_id_to_index, test_index, dim, 'test', p_req, log_file)
    log_file.close()
