import numpy as np


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
        # init chain rule. dL/dZ
        dL_dA = layer.get_derivative()(y_true, y_pred)

        # obtain gradient value in the point
        gradient = np.matmul(dL_dA.T, x) / x.shape[0]  # batch_size

        db = np.sum(dL_dA, axis=0) / x.shape[0]  # batch_size

        # update weights, negate gradients because we are
        # trying to minimize error.
        weights -= self.learning_rate * gradient
        bias -= self.learning_rate * db

        layer.set_weights_bias([weights, bias])

        # return derivative for the next steps of the chain rule
        return dL_dA


    def __call__(self, dA_dZ: np.ndarray, layer, last_weights: np.ndarray) -> np.ndarray:
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
        xw_b = layer.get_output_no_act() # Z output

        # act_f(lineal_f) we need to multiply last_weights with the accumulate
        # derivative, in order to complete the chain.
        # chain completed for the last layer.
        dZ_dW = np.matmul(dA_dZ, last_weights)

        # obtain derivative activation function.
        d_act_f = layer.get_derivative()(xw_b)

        # complete chain rule.
        dA_dZ = dZ_dW * d_act_f

        # obtain gradient value in the point
        gradient = np.matmul(dA_dZ.T, x) / x.shape[0]

        db = np.sum(dA_dZ, axis=0, keepdims=True) / x.shape[0]

        # update weights, negate gradients because we are
        # trying to minimize error.
        weights = weights - self.learning_rate * gradient
        bias = bias - self.learning_rate * db

        layer.set_weights_bias([weights, bias])

        # return derivative for the next steps of the chain rule
        return dA_dZ
