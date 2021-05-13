# Author: Harsh Kohli
# Date Created: 02-05-2021

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_hub as hub


class QuerySchemaEncoder(Model):
    def __init__(self, config):
        super(QuerySchemaEncoder, self).__init__()
        self.nnlm_embedder = hub.load(config['tf_hub_model'])
        self.margin = config['contrastive_loss_margin']

        self.table_encoder_dense = tf.keras.models.Sequential()
        self.table_encoder_dense.add(tf.keras.layers.InputLayer(input_shape=(256,)))
        self.table_encoder_dense.add(tf.keras.layers.Dense(192, activation='relu'))
        self.table_encoder_dense.add(tf.keras.layers.Dense(128, activation='relu'))

        self.query_encoder_dense = tf.keras.models.Sequential()
        self.query_encoder_dense.add(tf.keras.layers.InputLayer(input_shape=(128,)))
        self.query_encoder_dense.add(tf.keras.layers.Dense(128, activation='relu'))

    def call(self, features):
        questions, headers, table_words, labels, all_num_cols, masks = features
        table_encodings = self.get_table_embedding(headers, table_words, all_num_cols, masks)
        question_encodings = self.get_query_embedding(questions)
        d = tf.reduce_sum(tf.square(question_encodings - table_encodings), 1)
        d_sqrt = tf.sqrt(d)
        loss = labels * tf.square(tf.maximum(0., self.margin - d_sqrt)) + (
                tf.ones(shape=[tf.shape(labels)[0]]) - labels) * d
        loss = 0.5 * tf.reduce_mean(loss)
        return loss

    def get_table_embedding(self, headers, table_words, all_num_cols, masks):
        header_embeddings = self.nnlm_embedder(tf.reshape(headers, [-1]))
        table_word_embeddings = self.nnlm_embedder(tf.reshape(table_words, [-1]))
        table_encodings = self.table_encoder_dense(tf.concat((header_embeddings, table_word_embeddings), axis=1))
        table_encodings = tf.reshape(table_encodings, [tf.shape(table_words)[0], tf.shape(table_words)[1], -1])
        expanded_masks = tf.expand_dims(masks, -1)
        masks_broadcasted = tf.broadcast_to(expanded_masks,
                                            shape=[tf.shape(table_encodings)[0], tf.shape(table_encodings)[1],
                                                   tf.shape(table_encodings)[2]])
        table_encodings = tf.math.multiply(table_encodings, masks_broadcasted)
        table_encodings = tf.math.reduce_sum(table_encodings, axis=1)
        num_cols_normalized = tf.broadcast_to(tf.expand_dims(tf.dtypes.cast(all_num_cols, tf.float32), -1),
                                              shape=[tf.shape(table_encodings)[0], tf.shape(table_encodings)[1]])
        table_encodings = tf.math.divide(table_encodings, num_cols_normalized)
        return table_encodings

    def get_query_embedding(self, questions):
        question_embeddings = self.nnlm_embedder(questions)
        return self.query_encoder_dense(question_embeddings)
        #return question_embeddings
