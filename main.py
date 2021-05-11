# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
import tensorflow as tf
import os
import numpy as np
from models import QuerySchemaEncoder
from tensorflow.keras.losses import CategoricalCrossentropy
from utils.data_utils import load_preprocessed_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.run_functions_eagerly(True)
config = yaml.safe_load(open('config.yml', 'r'))

batch_size = config['batch_size']
data = load_preprocessed_data(config)
print('here')
# train_questions, train_headers, train_table_words, train_labels, train_num_cols, train_masks = create_batches(
#     train_data, train_tables, config)
# dev_questions, dev_headers, dev_table_words, dev_labels, dev_num_cols, dev_masks = create_batches(train_data,
#                                                                                                   train_tables, config)
# test_questions, test_headers, test_table_words, test_labels, test_num_cols, test_masks = create_batches(train_data,
#                                                                                                         train_tables,
#                                                                                                         config)
# train_iterations, dev_iterations, test_iterations = len(train_questions), len(dev_questions), len(test_questions)
#
# model = QuerySchemaEncoder(config)
# loss_obj = CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
# save_path = os.path.join(config['model_dir'], config['model_name'])
#
#
# @tf.function
# def train_step(train_question, train_header, train_table_word, labels):
#     with tf.GradientTape() as tape:
#         logits = model([train_question, train_header, train_table_word], training=True)
#         loss = loss_obj(y_true=labels, y_pred=logits)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss
#
#
# print('Starting Training...')
# for epoch_num in range(config['num_epochs']):
#     print('Starting Epoch: ' + str(epoch_num))
#     for index, (train_question, train_header, train_table_word, train_label) in enumerate(
#             zip(train_questions, train_headers, train_table_words, train_labels)):
#         loss = train_step(train_question, train_header, train_table_word, train_label)
#         if index % 5 == 0:
#             print('Done ' + str(index) + ' train iterations out of ' + str(train_iterations) + ' Loss is ' + str(
#                 float(loss.numpy())))
#         break
#     print('Completed Epoch. Saving Latest Model...')
#     tf.saved_model.save(model, save_path + str(epoch_num))
