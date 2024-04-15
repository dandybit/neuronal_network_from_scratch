import sys
import time

from abc import ABC, abstractmethod
import numpy as np
from .dataset import *
from .activations import *
from .optimizers import *
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Network:
    def __init__(self, layers_init: list, optimizer=SGD(), loss_func=CategoricalCrossEntropy) -> None:
        self.layers = layers_init
        self.loss_func = loss_func()
        self.optimizer = optimizer
        self.input_layer = self.layers[0]
        self.input_layer.build_weights(1, init_layer=True)

        last_layer = self.input_layer
        for layer in self.layers[1:]:
            layer.build_weights(last_layer.n_neurons)
            last_layer = layer

        self.layers = self.layers[1:]

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
            if layer.gradient:
                last_output = self.optimizer(last_output, layer, last_layer.get_weights_bias()[0])
                last_layer = layer
            else:
                layer.set_weights_bias(last_layer.get_weights_bias())
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
    def __init__(self, n_neurons, activation_func) -> None:
        self.n_neurons = n_neurons
        self.activation_func = activation_func()
        self.derivative_act_func = derivative(self.activation_func)
        self.inner_bias = np.zeros(self.n_neurons)
        self.last_input = []
        self.gradient = True

    def __call__(self, output_last_layer: np.ndarray) -> np.ndarray:
        self.last_input = output_last_layer
        self.output_x = (np.dot(output_last_layer, self.inner_weights.T) + self.inner_bias).copy()
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


class DenseLayer(Layer):
    def __init__(self, n_neurons: int, activation_func) -> None:
        super().__init__(n_neurons, activation_func)

    def build_weights(self, n_neurons_last_layer: int, initializer=None, init_layer=False) -> None:
        if not init_layer:
            self.n_neurons_last_layer = n_neurons_last_layer
            self.inner_weights = np.random.normal(0, 0.01, size=(self.n_neurons, self.n_neurons_last_layer))
        else:
            self.inner_weights = np.ones((1, self.n_neurons))
            self.inner_bias = np.zeros(self.n_neurons)


class CNNLayer(Layer):
    def __init__(self, filters: int, kernel: tuple, activation_func, stride=1, padding=1) -> None:
        super().__init__(filters, activation_func)
        self.kernel = kernel
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.filters = filters


    def __call__(self, output_last_layer: np.ndarray) -> np.ndarray:
        # binary images
        if len(output_last_layer.shape) == 3:
            output_last_layer = output_last_layer[:, :, :, np.newaxis]

        self.last_input = output_last_layer

        output_k = torch_conv_op(output_last_layer, self.inner_weights, self.stride, self.padding)

        # add bias
        output_k += self.inner_bias

        self.output_x = output_k.copy()

        # activation function
        output_k = self.derivative_act_func(output_k)

        # return output_k
        return output_k

    # method use to build weights in DenseLayer, tracking kernel and n_strides
    def build_weights(self, n, init_layer=False):
        if not init_layer:
            if len(n) == 2:
                self.inner_weights = np.random.normal(0, 0.01, size=(1, self.kernel[0], self.kernel[1], self.filters))
                self.inner_bias = np.zeros(self.filters)
            else:
                self.inner_weights = np.random.normal(0, 0.01, size=(n[-1], self.kernel[0], self.kernel[1], self.filters))
                self.inner_bias = np.zeros(self.filters)

            self.n_neurons = (int(((n[0] + 2 * self.padding[0] - self.kernel[0]) / self.stride[0]) + 1),
                              int(((n[1] + 2 * self.padding[1] - self.kernel[1]) / self.stride[1]) + 1),
                              self.filters)

        else:
            self.inner_weights = np.random.normal(0, 0.01, size=(n, self.kernel[0], self.kernel[1], self.filters))
            self.inner_bias = np.zeros(self.filters)
            # passing image size via first conv2d
            self.n_neurons = self.kernel


class FlattenLayer(Layer):
    def __init__(self):
        self.gradient = False

    def __call__(self, x):
        self.original_size = x
        output_x = np.reshape(x, (x.shape[0], -1))
        self.n_neurons = output_x.shape[1]
        return output_x

    def build_weights(self, n, init_layer=False):
        self.n_neurons = n[0] * n[1] * n[2]

    def backward_process(self, x):
        return np.reshape(x, self.original_size.shape)

    def get_weights_bias(self) -> tuple:
        # infer n_filters tuple([-1]) and dim image from input shape
        return self.inner_weights, self.inner_bias
