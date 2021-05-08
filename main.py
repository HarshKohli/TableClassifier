# Author: Harsh Kohli
# Date Created: 27-04-2021

import yaml
import tensorflow as tf
from models import QuerySchemaEncoder
from tensorflow.keras.losses import CategoricalCrossentropy
from utils.data_utils import load_preprocessed_data, create_batches

tf.config.experimental_run_functions_eagerly(True)
config = yaml.safe_load(open('config.yml', 'r'))

batch_size = config['batch_size']
train_data, dev_data, test_data, train_tables, dev_tables, test_tables = load_preprocessed_data(config)
train_questions, train_headers, train_table_words, train_labels = create_batches(train_data, train_tables, batch_size)
dev_questions, dev_headers, dev_table_words, dev_labels = create_batches(train_data, train_tables, batch_size)
test_questions, test_headers, test_table_words, test_labels = create_batches(train_data, train_tables, batch_size)

model = QuerySchemaEncoder(config)
loss_obj = CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)


@tf.function
def train_step(train_question, train_header, train_table_word, labels):
    with tf.GradientTape() as tape:
        logits = model([train_question, train_header, train_table_word], training=True)
        loss = loss_obj(y_true=labels, y_pred=logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


for epoch_num in range(config['num_epochs']):
    for train_question, train_header, train_table_word, train_label in zip(train_questions, train_headers,
                                                                           train_table_words, train_labels):
        loss = train_step(train_question, train_header, train_table_word, train_label)
        print(float(loss.numpy()))
