import sys
import numpy as np
from numpy import random
from network import *
from utils import *
matplotlib.use('TkAgg')

def main() -> None:
    train_X, train_y, test_X, test_y = parse_dataset()

    dataset_train = Dataset(train_X, train_y, batch_size=32)
    dataset_test = Dataset(test_X, test_y, batch_size=32)

    layer_test = [[train_X.shape[1], Plain], [256, ReLU], [256, ReLU], [10, Softmax]]

    network = Network(layer_test, optimizer=SGD(learning_rate=0.0001), loss_func=CategoricalCrossEntropy)

    results, error = network.train(dataset_train, epochs=32)

    network.test(dataset_test)

if __name__ == "__main__":
    main()
