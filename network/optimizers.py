import numpy as np
from utils.nn_utils import *
import network.network as nn

class SGD:
    """
    Implementation of the Stochastic Gradient Descent.
    """
    def __init__(self, learning_rate=0.005):
        self.learning_rate = learning_rate

    def first_layer(self, y_pred: np.ndarray, y_true: np.ndarray, layer) -> np.ndarray:
        """
        Complete the chain rule derivative, obtain gradient values in the point,
        and update weights and bias for the current layer.

        Parameters:
        y_pred (np.ndarray): predictions of the last layer.
        y_true (np.ndarray): dataset labels.

        returns:
        np.ndarray: derivative for the next step of the chain
        """
        weights, bias = layer.get_weights_bias()
        x = layer.get_last_input()

        # derivative loss function and last activation function
        # init chain rule.
        dL_dW = layer.get_derivative()(y_true, y_pred)

        # obtain gradient value at the point
        gradient = np.dot(dL_dW.T, x) / x.shape[0]  # batch_size

        db = np.sum(dL_dW, axis=0) / x.shape[0]  # batch_size

        # update weights, negate gradients because we are
        # trying to minimize error.
        weights -= self.learning_rate * gradient
        bias -= self.learning_rate * db

        layer.set_weights_bias([weights, bias])

        # return derivative for the next steps of the chain rule
        return dL_dW


    def __call__(self, y: np.ndarray, layer, last_weights: np.ndarray) -> np.ndarray:
        """
        Complete the chain rule derivative, obtain gradient values in the point,
        and update weights and bias for the current layer.

        Parameters:
        y (np.ndarray): last derivative (chain rule).
        last_weights (np.ndarray): weights from the previous layer (backward pass).

        returns:
        np.ndarray: derivative for the next step of the chain
        """

        weights, bias = layer.get_weights_bias()
        x = layer.get_last_input()
        xw_b = layer.get_output_no_act()

        # act_f(lineal_f) we need to multiply last_weights with the accumulate
        # derivative, in order to complete the chain.
        # check for DENSE -> CNN CASE y.shape == 4
        if isinstance(layer, nn.CNNLayer) and len(y.shape) == 4:
            dA = torch_conv_op(y, last_weights, 1)
        else:
            dA = np.matmul(y, last_weights)

        # obtain derivative activation function
        d_act_f = layer.get_derivative()(xw_b)

        # complete chain rule
        if isinstance(layer, nn.CNNLayer):
            dL_dW = np.reshape(dA, d_act_f.shape) * d_act_f
            # obtain gradient value at the point
            print("+" * 100)
            print(x.shape)
            print(dL_dW.shape)
            print(layer.inner_weights.shape)
            print("+" * 100)
            gradient, db = op_back_conv2d(x, dL_dW, layer.inner_weights)
            gradient = gradient / x.shape[0]
            db = db / x.shape[0]
        else:
            dL_dW = dA * d_act_f
            # obtain gradient value at the point
            gradient = np.matmul(dL_dW.T, x) / x.shape[0]
            db = np.sum(dL_dW, axis=0, keepdims=True) / x.shape[0]


        # update weights, negate gradients because we are
        # trying to minimize error.
        if isinstance(layer, nn.CNNLayer):
            weights = weights - self.learning_rate * gradient
            bias = bias - self.learning_rate * db
        else:
            weights = weights - self.learning_rate * gradient
            bias = bias - self.learning_rate * db

        layer.set_weights_bias([weights, bias])

        # return derivative for the next steps of the chain rule
        return dL_dW