import copy
import os
import pickle


class PreProcess(object):
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}

    @staticmethod
    def load_data(path):
        input_file = os.path.join(path)
        with open(input_file, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def convert_to_ids(source, vocab_to_int, add_eos=False):
        source = source.split(" ")
        source = [s for s in source if s]
        encoded = [vocab_to_int[s] for s in source]
        if add_eos:
            encoded += [vocab_to_int['<EOS>']]
        return encoded

    def text_to_ids(self, source_text, target_text, source_vocab_to_int, target_vocab_to_int):
        source_list = [self.convert_to_ids(s, source_vocab_to_int, False) for s in source_text.split("\n")]
        target_list = [self.convert_to_ids(s, target_vocab_to_int, True) for s in target_text.split("\n")]
        return source_list, target_list

    def create_lookup_tables(self, text):
        vocab = set(text.split())
        vocab_to_int = copy.copy(self.CODES)
        for v_i, v in enumerate(vocab, len(self.CODES)):
            vocab_to_int[v] = v_i
        int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
        return vocab_to_int, int_to_vocab

    def process_and_save_data(self):
        source_text = self.load_data(self.source_path)
        target_text = self.load_data(self.target_path)
        source_text = source_text.lower()
        target_text = target_text.lower()
        source_vocab_to_int, source_int_to_vocab = self.create_lookup_tables(source_text)
        target_vocab_to_int, target_int_to_vocab = self.create_lookup_tables(target_text)
        source_text, target_text = self.text_to_ids(source_text, target_text,
                                                    source_vocab_to_int, target_vocab_to_int)
        with open(os.path.dirname(os.path.realpath(self.path)) + '/saves/pre_process.p', 'wb') as out_file:
            pickle.dump((
                (source_text, target_text),
                (source_vocab_to_int, target_vocab_to_int),
                (source_int_to_vocab, target_int_to_vocab)), out_file)
