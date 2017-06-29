from distutils.version import LooseVersion

import tensorflow as tf

from src.main import Main
from src.plot import Plot
from src.translate import Translate


class Controller(object):
    @staticmethod
    def main(choice):
        try:
            assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion(
                '1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}' \
                .format(tf.__version__)
            print('TensorFlow Version: {}'.format(tf.__version__))
            print('*****Author: Satyaki Sanyal*****')
            print('***This project must only be used for educational purpose***')
            if choice == 1:
                if not tf.test.gpu_device_name():
                    print('*** ERROR: No GPU found. Please use a GPU to train your neural network. ***')
                else:
                    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
                    Main().main()

            elif choice == 2:
                Translate().translate()

            elif choice == 3:
                Plot().plot()

            else:
                print('*** Error: Wrong choice ***')
        except Exception as exc:
            print('*** Error: ' + str(exc) + ' ***')
