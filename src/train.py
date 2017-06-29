import sys
import time

import numpy as np
import tensorflow as tf

from src.functions import Functions
from src.set_params import Params


class Train(object):
    def __init__(self, source_int_text, target_int_text, train_graph, train_op, cost,
                 input_data, targets, lr, sequence_length, keep_prob, inference_logits, save_path):
        self.source_int_text = source_int_text
        self.target_int_text = target_int_text
        self.train_graph = train_graph
        self.train_op = train_op
        self.cost = cost
        self.input_data = input_data
        self.targets = targets
        self.lr = lr
        self.sequence_length = sequence_length
        self.keep_prob = keep_prob
        self.inference_logits = inference_logits
        self.save_path = save_path

    @staticmethod
    def get_accuracy(target, logits):
        max_seq = max(target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(target, [(0, 0), (0, max_seq - target.shape[1])], 'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(logits, [(0, 0), (0, max_seq - logits.shape[1]), (0, 0)], 'constant')
        return np.mean(np.equal(target, np.argmax(logits, 2)))

    def train(self):
        epochs, batch_size, rnn_size, _, _, _, learning_rate, keep_probability = Params().get()
        train_source = self.source_int_text[batch_size:]
        train_target = self.target_int_text[batch_size:]
        valid_source = Functions().pad_sentence_batch(self.source_int_text[:batch_size])
        valid_target = Functions().pad_sentence_batch(self.target_int_text[:batch_size])
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(epochs):
                for batch_i, (source_batch, target_batch) \
                        in enumerate(Functions().batch_data(train_source, train_target, batch_size)):
                    start_time = time.time()
                    _, loss = sess.run([self.train_op, self.cost],
                                       {self.input_data: source_batch, self.targets: target_batch,
                                        self.lr: learning_rate, self.sequence_length: target_batch.shape[1],
                                        self.keep_prob: keep_probability})
                    batch_train_logits = sess.run(self.inference_logits,
                                                  {self.input_data: source_batch, self.keep_prob: 1.0})
                    batch_valid_logits = sess.run(self.inference_logits,
                                                  {self.input_data: valid_source, self.keep_prob: 1.0})
                    train_acc = self.get_accuracy(target_batch, batch_train_logits)
                    valid_acc = self.get_accuracy(np.array(valid_target), batch_valid_logits)
                    end_time = time.time()
                    time_ = end_time - start_time
                    sys.stdout.write(
                        '\r epoch: {:>3} batch: {:>4}/{} - train_acc: {:>6.3f}, validation_acc: {:>6.3f}, Loss: {:>6.3f} time: {:>6.3f}s'.format(
                            epoch_i, batch_i, len(self.source_int_text) // batch_size, train_acc, valid_acc,
                            loss, time_))
                    Functions.save_points(loss, 'loss')
                    Functions.save_points(train_acc, 'train_acc')
                    Functions.save_points(valid_acc, 'valid_acc')
                    Functions.save_points(time_, 'time')
            saver = tf.train.Saver()
            saver.save(sess, self.save_path)
            sys.stdout.write('\r Training finished, model saved....')
            Functions().save_params(self.save_path)
