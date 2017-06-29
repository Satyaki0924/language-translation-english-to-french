import numpy as np
import tensorflow as tf

from src.functions import Functions
from src.main import Main


class Translate(object):
    def __init__(self):
        self.source_vocab_to_int, self.target_vocab_to_int, \
        self.source_int_to_vocab, self.target_int_to_vocab, self.load_path = None, None, None, None, None

    def loader(self):
        _, (self.source_vocab_to_int, self.target_vocab_to_int), \
        (self.source_int_to_vocab, self.target_int_to_vocab) = Main().load_process()
        self.load_path = Functions().load_params()

    @staticmethod
    def sentence_to_seq(sentence, vocab_to_int):
        sentence = sentence.lower().split(" ")
        ids = [vocab_to_int[s] if s in vocab_to_int else vocab_to_int['<UNK>'] for s in sentence]
        return ids

    def translate(self):
        self.loader()
        while True:
            try:
                translate_sentence = str(input('>> Enter sentence in English\n'))
                break
            except:
                pass
        translate_sentence = self.sentence_to_seq(translate_sentence, self.source_vocab_to_int)

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(self.load_path + '.meta')
            loader.restore(sess, self.load_path)
            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('logits:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
            translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]
        print('Input')
        print('  Word Ids:      {}'.format([i for i in translate_sentence]))
        print('  English Words: {}'.format([self.source_int_to_vocab[i] for i in translate_sentence]))
        print('\nPrediction')
        print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
        print('  French Words: {}'.format([self.target_int_to_vocab[i]
                                           for i in np.argmax(translate_logits, 1)]))
