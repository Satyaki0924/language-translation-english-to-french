import os
import pickle

import tensorflow as tf

from src.preprocess import PreProcess
from src.process_inputs import Inputs
from src.seq2seq import Seq2seq
from src.set_params import Params
from src.train import Train


class Main(object):
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))

    def load_process(self):
        with open(self.path + '/saves/pre_process.p', mode='rb') as in_file:
            return pickle.load(in_file)

    def main(self):
        train_graph = tf.Graph()
        save_path = self.path + '/checkpoints/dev'
        source_path = self.path + '/data/small_vocab_en'
        target_path = self.path + '/data/small_vocab_fr'
        PreProcess(source_path, target_path).process_and_save_data()
        _, batch_size, rnn_size, num_layers, encoding_embedding_size, decoding_embedding_size, _, _ = \
            Params().get()
        (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = \
            self.load_process()
        max_source_sentence_length = max([len(sentence) for sentence in source_int_text])
        with train_graph.as_default():
            input_data, targets, lr, keep_prob = Inputs().get()
            sequence_length = tf.placeholder_with_default(
                max_source_sentence_length, None, name='sequence_length')
            input_shape = tf.shape(input_data)
            train_logits, inference_logits = Seq2seq().seq2seq_model(
                tf.reverse(input_data, [-1]), targets, keep_prob, batch_size,
                sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
                encoding_embedding_size, decoding_embedding_size,
                rnn_size, num_layers, target_vocab_to_int)
            tf.identity(inference_logits, 'logits')
            with tf.name_scope("optimization"):
                cost = tf.contrib.seq2seq.sequence_loss(train_logits, targets,
                                                        tf.ones([input_shape[0], sequence_length]))
                optimizer = tf.train.AdamOptimizer(lr)
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                                    for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)
        Train(source_int_text, target_int_text, train_graph, train_op, cost,
              input_data, targets, lr, sequence_length, keep_prob, inference_logits, save_path).train()
