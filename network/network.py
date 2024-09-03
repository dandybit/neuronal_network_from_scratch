import sys
import time

import numpy as np
from .dataset import *
from .activations import *
from .optimizers import *
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Network:
    def __init__(self, layers_init: list, optimizer=SGD(), loss_func=CategoricalCrossEntropy) -> None:
        self.layers_init = layers_init
        self.loss_func = loss_func()
        self.optimizer = optimizer
        self.input_layer = Layer(self.layers_init[0][0], Flat, 1)
        self.layers = []

        last_layer = self.input_layer
        for layer in self.layers_init[1:]:
            current_layer = Layer(layer[0], layer[1], last_layer.get_shape()[0])
            self.layers.append(Layer(layer[0], layer[1], last_layer.get_shape()[0]))
            last_layer = current_layer

    def train(self, inputs: Dataset, epochs=1) -> list:

        # batch level forward, backward
        for epoch in range(epochs):
            results = []
            errors = []
            for x, y in inputs:
                output_f = self._forward(x)
                results.append(output_f)
                errors.append(self.loss_func(y, output_f))
                self._backward(output_f, y)
                print(f"\rCURRENT EPOCH: {epoch} - ERROR: {np.mean(np.array(errors))}", end="", flush=True)
            print(f"\rCURRENT EPOCH: {epoch} - ERROR_BATCH_MEAN: {np.mean(np.array(errors))}\n", flush=True)

        return [np.array(results), np.array(errors)]

    def _forward(self, x: np.ndarray) -> np.ndarray:
        last_output = self.layers[0](x)
        for layer in self.layers[1:]:
            last_output = layer(last_output)

        return last_output

    def _backward(self, y_p: np.ndarray, y_t: np.ndarray) -> None:
        last_output = self.optimizer.first_layer(y_p, y_t, self.layers[-1])
        last_layer = self.layers[-1]

        for layer in self.layers[-2::-1]:
            last_output = self.optimizer(last_output, layer, last_layer.get_weights_bias()[0])
            last_layer = layer

    def test(self, inputs: Dataset) -> None:
        results = []
        errors = []
        for x, y in inputs:
            output_f = self._forward(x)
            results.append(output_f)
            errors.append(self.loss_func(y, output_f))
            self._backward(output_f, y)
        print(f" ERROR MEAN TEST SET: {np.mean(np.array(errors))}", flush=True)
        print("DISPLAY CONFUSION MATRIX")
        # Display confusion matrix
        ConfusionMatrixDisplay.from_predictions(np.reshape(np.argmax(inputs.get_labels(), axis=-1), (-1,)),
                                                np.reshape(np.argmax(np.array(results), axis=-1), (-1,)),
                                                normalize="true", values_format=".0%")
        plt.show()


class Layer:
    def __init__(self, n_neurons: int, activation_func, n_neurons_last_layer: int) -> None:
        self.n_neurons = n_neurons
        self.activation_func = activation_func()
        self.derivative_act_func = derivative(self.activation_func)
        self.n_neurons_last_layer = n_neurons_last_layer
        self.inner_weights = np.random.normal(0, 0.01, size=(self.n_neurons, self.n_neurons_last_layer))
        self.inner_bias = np.zeros(self.n_neurons)
        self.last_input = []

    def __call__(self, output_last_layer: np.ndarray) -> np.ndarray:
        self.last_input = output_last_layer
        self.output_x = np.matmul(output_last_layer, self.inner_weights.T) + self.inner_bias
        return self.activation_func(self.output_x)

    def get_shape(self) -> tuple:
        return self.inner_weights.shape

    def get_weights_bias(self) -> tuple:
        return self.inner_weights, self.inner_bias

    def set_weights_bias(self, new_weights_bias: list) -> None:
        self.inner_weights = new_weights_bias[0]
        self.inner_bias = new_weights_bias[1]

    def get_derivative(self) -> object:
        return self.derivative_act_func

    def get_last_input(self) -> np.ndarray:
        return self.last_input

    def get_output_no_act(self) -> np.ndarray:
        return self.output_x

