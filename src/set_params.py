class Params(object):
    def __init__(self, epochs=5, batch_size=256,
                 rnn_size=512, num_layers=2, encoding_embedding_size=256,
                 decoding_embedding_size=256, learning_rate=0.001, keep_probability=0.5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.encoding_embedding_size = encoding_embedding_size
        self.decoding_embedding_size = decoding_embedding_size
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability

    def get(self):
        return self.epochs, self.batch_size, self.rnn_size, self.num_layers, \
               self.encoding_embedding_size, self.decoding_embedding_size, \
               self.learning_rate, self.keep_probability
