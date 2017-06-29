import tensorflow as tf


class Encode(object):

    @staticmethod
    def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
        cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        cell_ = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        _cell_ = tf.contrib.rnn.DropoutWrapper(cell_, keep_prob)
        _, enc_state = tf.nn.dynamic_rnn(_cell_, rnn_inputs, dtype=tf.float32)
        return enc_state
