import tensorflow as tf


class Inputs(object):
    def __init__(self):
        self.inputs, self.targets, self.learning_rate, self.keep_probability = None, None, None, None

    def model_inputs(self):
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='input')
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='target')
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='lr')
        self.keep_probability = tf.placeholder(tf.float32, shape=None, name='keep_prob')

    @staticmethod
    def process_decoding_input(target_data, target_vocab_to_int, batch_size):
        l_word = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        return tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), l_word], 1)

    def get(self):
        self.model_inputs()
        return self.inputs, self.targets, self.learning_rate, self.keep_probability
