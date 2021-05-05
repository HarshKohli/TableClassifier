# Author: Harsh Kohli
# Date Created: 02-05-2021

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_hub as hub


class QuerySchemaEncoder(Model):
    def __init__(self, config):
        super(QuerySchemaEncoder, self).__init__()
        self.nnlm_embedder = hub.load(config['tf_hub_model'])

        self.table_encoder_dense = tf.keras.models.Sequential()
        self.table_encoder_dense.add(tf.keras.Input(shape=(256,)))
        self.table_encoder_dense.add(tf.keras.layers.Dense(192, activation='relu'))
        self.table_encoder_dense.add(tf.keras.layers.Dense(128, activation='relu'))

        self.query_encoder_dense = tf.keras.models.Sequential()
        self.query_encoder_dense.add(tf.keras.Input(shape=(128,)))
        self.query_encoder_dense.add(tf.keras.layers.Dense(128, activation='relu'))

        self.output_layers = tf.keras.models.Sequential()
        self.output_layers.add(tf.keras.Input(shape=(256,)))
        self.output_layers.add(tf.keras.layers.Dense(128, activation='relu'))
        self.output_layers.add(tf.keras.layers.Dense(64, activation='relu'))
        self.output_layers.add(tf.keras.layers.Dense(32, activation='relu'))
        self.output_layers.add(tf.keras.layers.Dense(2, activation='softmax'))

    def call(self, features, training):
        questions, headers, table_words = features

        table_encodings = []
        for train_header, train_table_word in zip(headers, table_words):
            header_embedding = self.nnlm_embedder(train_header)
            word_embedding = self.nnlm_embedder(train_table_word)
            dense_output_1 = self.table_encoder_dense(tf.concat((header_embedding, word_embedding), axis=1))
            dense_output_1 = tf.math.reduce_mean(dense_output_1, axis=0)
            table_encodings.append(dense_output_1)
        table_encodings = tf.convert_to_tensor(table_encodings)

        query_embeddings = self.nnlm_embedder(questions)
        query_encodings = self.query_encoder_dense(query_embeddings)

        output_prob = self.output_layers(tf.concat((table_encodings, query_encodings), axis=1))
        return output_prob
