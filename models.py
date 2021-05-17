# Author: Harsh Kohli
# Date Created: 02-05-2021

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental import preprocessing


class QuerySchemaEncoder(Model):
    def __init__(self, config):
        super(QuerySchemaEncoder, self).__init__()
        self.nnlm_embedder = hub.load(config['tf_hub_model'])
        self.margin = config['contrastive_loss_margin']
        dim = config['dim']

        self.use_char_embed = config['use_char_embedding']
        if self.use_char_embed:
            self.char_vocab = config['character_vocab']
            self.char_embedding_dim = config['char_embedding_dim']
            self.ids_from_chars = preprocessing.StringLookup(vocabulary=self.char_vocab, mask_token=None)

            self.char_encoder = tf.keras.models.Sequential()
            self.char_encoder.add(tf.keras.layers.Embedding(len(self.char_vocab), self.char_embedding_dim))
            self.char_encoder.add(tf.keras.layers.LSTM(self.char_embedding_dim, activation='relu'))

            self.char_word_combiner = tf.keras.models.Sequential()
            self.char_word_combiner.add(tf.keras.layers.InputLayer(input_shape=(dim + self.char_embedding_dim,)))
            self.char_word_combiner.add(tf.keras.layers.Dense(dim, activation='relu'))

        self.table_encoder_dense = tf.keras.models.Sequential()
        self.table_encoder_dense.add(tf.keras.layers.InputLayer(input_shape=(2 * dim,)))
        self.table_encoder_dense.add(tf.keras.layers.Dense(1.5 * dim, activation='relu'))
        self.table_encoder_dense.add(tf.keras.layers.Dense(dim, activation='relu'))

        self.use_lstm_query_encoder = config['use_lstm_query_encoder']

        if self.use_lstm_query_encoder:
            self.query_encoder = tf.keras.models.Sequential()
            self.query_encoder.add(tf.keras.layers.LSTM(dim, activation='relu'))

        else:
            self.query_encoder = tf.keras.models.Sequential()
            self.query_encoder.add(tf.keras.layers.InputLayer(input_shape=(dim,)))
            self.query_encoder.add(tf.keras.layers.Dense(dim, activation='relu'))
            self.query_encoder.add(tf.keras.layers.Dense(dim, activation='relu'))

    def call(self, features, **kwargs):
        questions, headers, table_words, labels, all_num_cols, masks = features
        table_encodings = self.get_table_embedding(headers, table_words, all_num_cols, masks)
        question_encodings = self.get_query_embedding(questions)
        d = tf.reduce_sum(tf.square(question_encodings - table_encodings), 1)
        d_sqrt = tf.sqrt(d)
        loss = labels * d + (tf.ones(shape=[tf.shape(labels)[0]]) - labels) * tf.square(
            tf.maximum(0., self.margin - d_sqrt))
        loss = 0.5 * tf.reduce_mean(loss)
        return loss

    def get_table_embedding(self, headers, table_words, all_num_cols, masks):
        header_embeddings = self.nnlm_embedder(tf.reshape(headers, [-1]))
        table_word_embeddings = self.nnlm_embedder(tf.reshape(table_words, [-1]))

        if self.use_char_embed:
            header_char_embeddings = self.get_char_embeddings(tf.reshape(headers, [-1]))
            table_char_embeddings = self.get_char_embeddings(tf.reshape(table_words, [-1]))
            header_embeddings = self.char_word_combiner(tf.concat((header_embeddings, header_char_embeddings), axis=1))
            table_word_embeddings = self.char_word_combiner(tf.concat((table_word_embeddings, table_char_embeddings),
                                                                      axis=1))

        table_encodings = self.table_encoder_dense(tf.concat((header_embeddings, table_word_embeddings), axis=1))
        table_encodings = tf.reshape(table_encodings, [tf.shape(table_words)[0], tf.shape(table_words)[1], -1])
        expanded_masks = tf.dtypes.cast(tf.expand_dims(masks, -1), tf.float32)
        masks_broadcasted = tf.broadcast_to(expanded_masks,
                                            shape=[tf.shape(table_encodings)[0], tf.shape(table_encodings)[1],
                                                   tf.shape(table_encodings)[2]])
        table_encodings = tf.math.multiply(table_encodings, masks_broadcasted)
        table_encodings = tf.math.reduce_sum(table_encodings, axis=1)
        num_cols_normalized = tf.broadcast_to(tf.expand_dims(tf.dtypes.cast(all_num_cols, tf.float32), -1),
                                              shape=[tf.shape(table_encodings)[0], tf.shape(table_encodings)[1]])
        table_encodings = tf.math.divide(table_encodings, num_cols_normalized)
        table_encodings = tf.math.l2_normalize(table_encodings, axis=1)
        return table_encodings

    def get_query_embedding(self, questions):
        if self.use_lstm_query_encoder:
            question_embeddings = self.get_lstm_word_embeddings(questions)
        else:
            question_embeddings = self.nnlm_embedder(questions)
            question_embeddings = self.query_encoder(question_embeddings)

        if self.use_char_embed:
            question_char_embeddings = self.get_char_embeddings(tf.reshape(questions, [-1]))
            question_embeddings = self.char_word_combiner(tf.concat((question_embeddings, question_char_embeddings),
                                                                    axis=1))

        question_embeddings = tf.math.l2_normalize(question_embeddings, axis=1)
        return question_embeddings

    def get_lstm_word_embeddings(self, text):
        tokenized = tf.strings.split(text)
        tokenized_lengths = tf.reshape(tf.map_fn(tf.shape, tokenized, fn_output_signature=tf.int32), [-1])
        word_encoded = self.nnlm_embedder(tokenized.flat_values)
        reshaped_word_embeddings = tf.RaggedTensor.from_row_lengths(word_encoded, tokenized_lengths)
        return self.query_encoder(reshaped_word_embeddings)

    def get_char_embeddings(self, text):
        tokenized = tf.strings.split(text)
        tokenized_lengths = tf.reshape(tf.map_fn(tf.shape, tokenized, fn_output_signature=tf.int32), [-1])
        chars = tf.strings.unicode_split(tokenized.flat_values, input_encoding='UTF-8')
        ids = self.ids_from_chars(chars)
        char_encoded = self.char_encoder(ids)
        reshaped_char_embeds = tf.RaggedTensor.from_row_lengths(char_encoded, tokenized_lengths)
        return tf.math.reduce_mean(reshaped_char_embeds, axis=1)
