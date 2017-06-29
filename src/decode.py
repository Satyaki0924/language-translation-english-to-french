import tensorflow as tf


class Decode(object):
    @staticmethod
    def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                             output_fn, keep_prob):
        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)
        logits = tf.nn.dropout(output_fn(train_pred), keep_prob)
        return logits

    @staticmethod
    def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                             maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length,
            vocab_size)
        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn,
                                                                        scope=decoding_scope)
        inference_logits = tf.nn.dropout(inference_logits, keep_prob)
        return inference_logits

    @staticmethod
    def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                       num_layers, target_vocab_to_int, keep_prob):
        cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

        with tf.variable_scope("decoding") as decoding_scope:
            output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)

            train_logits = Decode().decoding_layer_train(encoder_state, cell, dec_embed_input, sequence_length,
                                                         decoding_scope, output_fn, keep_prob)
        with tf.variable_scope("decoding", reuse=True) as decoding_scope:
            start_of_sequence_id = target_vocab_to_int['<GO>']
            end_of_sequence_id = target_vocab_to_int['<EOS>']
            inference_logits = Decode().decoding_layer_infer(encoder_state, cell, dec_embeddings,
                                                             start_of_sequence_id, end_of_sequence_id,
                                                             sequence_length - 1,
                                                             vocab_size,
                                                             decoding_scope, output_fn, keep_prob)
        return train_logits, inference_logits
