import os
import pickle

import numpy as np


class Functions(object):
    @staticmethod
    def pad_sentence_batch(sentence_batch):
        codes = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [codes['<PAD>']] * (max_sentence - len(sentence))
                for sentence in sentence_batch]

    @staticmethod
    def load_params():
        with open(os.path.dirname(os.path.realpath(
                os.path.dirname(os.path.realpath(__file__)))) + '/saves/params.p',
                  mode='rb') as in_file:
            return pickle.load(in_file)

    @staticmethod
    def save_params(params):
        with open(os.path.dirname(os.path.realpath(
                os.path.dirname(os.path.realpath(__file__)))) + '/saves/params.p', 'wb') as out_file:
            pickle.dump(params, out_file)

    @staticmethod
    def batch_data(source, target, batch_size):
        for batch_i in range(0, len(source) // batch_size):
            start_i = batch_i * batch_size
            source_batch = source[start_i:start_i + batch_size]
            target_batch = target[start_i:start_i + batch_size]
            yield np.array(Functions().pad_sentence_batch(source_batch)), \
                  np.array(Functions().pad_sentence_batch(target_batch))

    @staticmethod
    def save_points(points, name):
        with open(os.path.dirname(os.path.realpath(
                os.path.dirname(os.path.realpath(__file__)))) + '/points/' + name + '.txt',
                  mode='a') as in_file:
            in_file.write(str(points) + '\n')

    @staticmethod
    def read_points(name):
        with open(os.path.dirname(os.path.realpath(
                os.path.dirname(os.path.realpath(__file__)))) + '/points/' + name + '.txt',
                  mode='r') as out_file:
            return out_file.read()
