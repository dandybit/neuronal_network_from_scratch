import sys
import numpy as np
from numpy import random
from network import *
from utils import *
matplotlib.use('TkAgg')

def main() -> None:
    # the parse_dataset flats the dataset
    train_X, train_y, test_X, test_y = parse_dataset()


    dataset_train = Dataset(train_X, train_y, batch_size=32)
    dataset_test = Dataset(test_X, test_y, batch_size=32)

    # first layer -> input layer.
    # n_neurons, activation_function
    layer_test = [DenseLayer(n_neurons=train_X.shape[1], activation_func=Plain),
                  DenseLayer(n_neurons=256, activation_func=ReLU),
                  DenseLayer(n_neurons=256, activation_func=ReLU),
                  DenseLayer(n_neurons=10, activation_func=Softmax)]

    network = Network(layer_test, optimizer=SGD(learning_rate=0.0001), loss_func=CategoricalCrossEntropy)

    results, error = network.train(dataset_train, epochs=32)

    network.test(dataset_test)


    """
    # 2D for CNN layers
    train_X = np.reshape(train_X, (train_X.shape[0], 28, 28))
    test_X = np.reshape(test_X, (test_X.shape[0], 28, 28))

    dataset_train = Dataset(train_X, train_y, batch_size=32)
    dataset_test = Dataset(test_X, test_y, batch_size=32)

    print(dataset_train.get_dataset().shape)
    print(dataset_test.get_dataset().shape)

    layer_test = [[train_X.shape[1], Plain, DenseLayer], [256, ReLU, DenseLayer], [256, ReLU, DenseLayer], [10, Softmax, DenseLayer]]
    """

if __name__ == "__main__":
    main()
