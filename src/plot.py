import os

import matplotlib.pyplot as plt
from matplotlib import style

from src.functions import Functions

style.use('ggplot')


class Plot(object):
    @staticmethod
    def plot():
        plt.figure(figsize=(20, 10))
        plt.subplot(221)
        loss = list(map(float, [i for i in Functions.read_points('loss').split('\n') if i]))
        all = plt.plot(loss, c='r')
        plt.legend(all, 'Loss')
        plt.title("Training Loss")
        plt.subplot(222)
        train_acc = list(map(float, [i for i in Functions.read_points('train_acc').split('\n') if i]))
        all1 = plt.plot(train_acc, c='g')
        plt.legend(all1, 'Training Accuracy')
        plt.title("Training Accuracy")
        plt.subplot(223)
        valid_acc = list(map(float, [i for i in Functions.read_points('valid_acc').split('\n') if i]))
        all2 = plt.plot(valid_acc, c='purple')
        plt.legend(all2, 'Validation accuracy')
        plt.title("Validation accuracy")
        plt.subplot(224)
        time = list(map(float, [i for i in Functions.read_points('time').split('\n') if i]))
        all3 = plt.plot(time, c='b')
        plt.legend(all3, 'time')
        plt.title("Time")
        plt.savefig(os.path.dirname(os.path.realpath(
            os.path.dirname(os.path.realpath(__file__)))) + '/graphs/plot.png')
        print('*** Image of plot saved successfully ***')



