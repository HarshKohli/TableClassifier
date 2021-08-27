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

    @tf.function
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

    @tf.function
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


class PointingNetworkDecoder(Model):
    def __init__(self, config):
        super(PointingNetworkDecoder, self).__init__()
        dim = config['dim']
        initializer = tf.keras.initializers.GlorotUniform()
        self.num_tables = config['top_k']
        self.decoding_iterations = config['seq_decoding_iterations']

        self.table_encoder_dense = tf.keras.models.Sequential()
        self.table_encoder_dense.add(tf.keras.layers.InputLayer(input_shape=(2 * dim,)))
        self.table_encoder_dense.add(tf.keras.layers.Dense(1.5 * dim, activation='relu', kernel_initializer=initializer))
        self.table_encoder_dense.add(tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer))

        self.decoding_mlp = tf.keras.models.Sequential()
        self.decoding_mlp.add(tf.keras.layers.InputLayer(input_shape=(4 * dim,)))
        self.decoding_mlp.add(tf.keras.layers.Dense(2 * dim, activation='relu', kernel_initializer=initializer))
        self.decoding_mlp.add(tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer))
        self.decoding_mlp.add(tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer))

        self.decoding_lstm_cell = tf.keras.layers.LSTMCell(2 * dim, activation='relu', kernel_initializer=initializer)

    def call(self, features, **kwargs):
        query, tables, labels = features
        query_expanded = tf.broadcast_to(tf.expand_dims(query, axis=1), shape=tf.shape(tables))
        query_table_combined = tf.concat((tables, query_expanded), axis=2)
        query_table_combined = tf.reshape(query_table_combined, shape=[-1, tf.shape(query_table_combined)[2]])
        query_aware_table_rep = self.table_encoder_dense(query_table_combined)
        query_aware_table_rep = tf.reshape(query_aware_table_rep, shape=[tf.shape(tables)[0], tf.shape(tables)[1], -1])
        logits, output_pointers = self.iterative_pointing_decoder(query_aware_table_rep)

        if not kwargs['training']:
            return logits[-1]

        ce_loss = tf.zeros(shape=[tf.shape(labels)[0]])
        for logit in logits:
            ce_loss = ce_loss + tf.nn.softmax_cross_entropy_with_logits(labels, logit)
        ce_loss = tf.reduce_mean(ce_loss)
        return ce_loss

    def decoding_step(self, query_aware_suggestions, input_shape, output, pointer_encodings):
        suggestions_flattened = tf.reshape(query_aware_suggestions, shape=[-1, input_shape[2]])
        output_broadcasted = tf.tile(output, (input_shape[1], 1))
        pointer_encodings_broadcasted = tf.tile(pointer_encodings, (input_shape[1], 1))
        suggestion_scores = self.decoding_mlp(
            tf.concat((suggestions_flattened, output_broadcasted, pointer_encodings_broadcasted), axis=1))
        suggestion_scores = tf.reshape(suggestion_scores, shape=[input_shape[0], -1])
        return tf.nn.softmax(suggestion_scores)

    def iterative_pointing_decoder(self, query_aware_table_reps):
        input_shape = tf.shape(query_aware_table_reps)
        batch_size = input_shape[0]
        output_pointers = tf.broadcast_to(tf.cast(self.num_tables / 2, tf.int32), shape=[batch_size])
        state = self.decoding_lstm_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        not_settled = tf.tile([True], (batch_size,))
        logits = []
        for iteration in range(self.decoding_iterations):
            pointer_encodings = tf.gather_nd(query_aware_table_reps,
                                             tf.stack([tf.range(batch_size), output_pointers], axis=1))
            output, state = self.decoding_lstm_cell(pointer_encodings, state)
            if iteration == 0:
                suggestion_scores = self.decoding_step(query_aware_table_reps, input_shape, output, pointer_encodings)
            else:
                prev_scores = logits[iteration - 1]
                suggestion_scores = tf.cond(pred=tf.reduce_any(input_tensor=not_settled),
                                            true_fn=lambda: self.decoding_step(query_aware_table_reps, input_shape,
                                                                               output, pointer_encodings),
                                            false_fn=lambda: prev_scores)
            new_pointers = tf.cast(tf.math.argmax(suggestion_scores, axis=1), dtype=tf.int32)
            if iteration == 0:
                not_settled = tf.tile([True], (batch_size,))
            else:
                not_settled = tf.math.not_equal(output_pointers, new_pointers)
            output_pointers = new_pointers
            logits.append(suggestion_scores)

        return logits, output_pointers
