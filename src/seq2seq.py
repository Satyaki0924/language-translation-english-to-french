import tensorflow as tf

from src.decode import Decode
from src.encode import Encode
from src.process_inputs import Inputs


class Seq2seq(object):
    @staticmethod
    def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size,
                      target_vocab_size,
                      enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
        enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
        encoder_state = Encode().encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
        target_data = Inputs().process_decoding_input(target_data, target_vocab_to_int, batch_size)
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
        target_embed = tf.nn.embedding_lookup(dec_embeddings, target_data)
        return Decode().decoding_layer(target_embed, dec_embeddings,
                                       encoder_state, target_vocab_size,
                                       sequence_length, rnn_size, num_layers,
                                       target_vocab_to_int, keep_prob)
