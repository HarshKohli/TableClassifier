# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
import tensorflow as tf
import os
import numpy as np
from models import QuerySchemaEncoder
from utils.data_utils import load_preprocessed_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

batch_size = config['batch_size']
data = load_preprocessed_data(config)

questions, headers, table_words, labels, all_num_cols, masks = data['train_batches']
train_headers, train_table_words, train_all_num_cols, train_masks = data['train_tables_batches']
dev_headers, dev_table_words, dev_all_num_cols, dev_masks = data['dev_tables_batches']
test_headers, test_table_words, test_all_num_cols, test_masks = data['test_tables_batches']

model = QuerySchemaEncoder(config)
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
save_path = os.path.join(config['model_dir'], config['model_name'])


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


train_iterations = len(questions)
print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    for index, (question, header, table_word, label, all_num_col, mask) in enumerate(
            zip(questions, headers, table_words, labels, all_num_cols, masks)):
        a, b, c, d, e, f = np.array(question), np.array(header), np.array(table_word), np.array(label), np.array(
            all_num_col), np.array(mask)
        loss = train_step(a, b, c, d, e, f)
        if index % 100 == 0:
            print('Done ' + str(index) + ' train iterations out of ' + str(train_iterations) + ' Loss is ' + str(
                float(loss.numpy())))
    print('Completed Epoch. Saving Latest Model...')
    tf.saved_model.save(model, save_path, str(epoch_num))
