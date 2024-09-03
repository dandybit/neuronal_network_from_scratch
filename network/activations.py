import numpy as np
import math
from functools import singledispatch


class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, 0, x)


class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + math.exp(-x))


class Flat:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


class Softmax:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        # number stability
        z_exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=-1, keepdims=True)


"""
class SoftmaxDerivative:
    def __call__(self, x):
        # jacobian matrix
        # outputs softmax, x = vector softmax results x = softmax(x)
        # i = j => softmax(x) * (1 - softmax(x)) = softmax(x) - softmax(x)^2 diagonal
        # i != j => -softmax(x) * softmax(x') | x' => all values from vector softmax (rows)
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
"""


class CategoricalCrossEntropy:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon=1e-8) -> np.ndarray:
        loss = -np.sum(y_true * np.log(y_pred + epsilon), axis=-1)
        return np.mean(loss)


# obtain derivative function from activation functions
@singledispatch
def derivative(obj: str) -> str:
    raise ValueError


@derivative.register
def _(func: ReLU) -> object:
    class ReLUDerivative:
        def __call__(self, x: np.ndarray) -> np.ndarray:
            return np.where(x > 0, 1, 0)

    return ReLUDerivative()


@derivative.register
def _(func: Softmax) -> object:
    class CrossSoftmaxDerivative:
        def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            return y_pred - y_true

    return CrossSoftmaxDerivative()


@derivative.register
def _(func: Flat) -> object:
    class FlatDerivative:
        def __call__(self, x: np.ndarray) -> int:
            return 1

    return FlatDerivative()
