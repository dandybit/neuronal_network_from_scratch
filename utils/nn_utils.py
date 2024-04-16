import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_conv_op(image: np.ndarray, kernel: np.ndarray, stride=(1, 1), padding=1) -> np.ndarray:
    image_in = torch.tensor(np.reshape(image, (image.shape[0], image.shape[-1], image.shape[1], image.shape[2])),
                            dtype=torch.float32)

    kernel_in = torch.tensor(np.reshape(kernel, (kernel.shape[0], kernel.shape[-1], kernel.shape[1], kernel.shape[2])),
                             dtype=torch.float32)

    conv_op = F.conv2d(image_in, kernel_in, stride=stride, padding=padding).numpy()

    return np.reshape(conv_op, (conv_op.shape[0], conv_op.shape[2], conv_op.shape[3], conv_op.shape[1]))


def op_conv2d(image: np.ndarray, kernel: np.ndarray, stride=(1, 1), padding=(1, 1)) -> np.ndarray:
    output_rows_dim = int(((image.shape[1] + 2 * padding[0] - kernel.shape[1]) / stride[0])) + 1
    output_col_dim = int(((image.shape[2] + 2 * padding[1] - kernel.shape[2]) / stride[1])) + 1
    output_image = np.zeros((image.shape[0], output_rows_dim, output_col_dim, kernel.shape[0]))

    # padded image
    input_padded = np.pad(image,
                          pad_width=((0, 0),
                                     (padding[0], padding[0]),
                                     (padding[1], padding[1]),
                                     (0, 0)),
                          mode='constant', constant_values=0)

    range_col = input_padded.shape[1] - kernel.shape[1]
    range_row = input_padded.shape[2] - kernel.shape[2]
    # kernel -> n_filters = image channels, i, j, depth filter
    # n_filters
    for i in range(0, kernel.shape[0]):
        for x in range(0, range_col, stride[0]):
            for y in range(0, range_row, stride[1]):
                conv_op = kernel[i:i+1, :, :, :] * input_padded[:, x:x+kernel.shape[1], y:y+kernel.shape[2], :]
                # sum values in all channels
                conv_op = np.sum(conv_op, axis=(1, 2, 3))
                conv_op = np.reshape(conv_op, (image.shape[0], 1, 1, 1))
                # re-dim to sum in output_k
                output_image[:, x:x+1, y:y+1, i:i+1] += conv_op

    return output_image


def op_back_conv2d(X: np.ndarray, dL_dW: np.ndarray, kernel: np.ndarray, stride=(1, 1), padding=None) -> tuple:
    gradient = np.zeros(kernel.shape)
    partial_w = np.zeros((dL_dW.shape[-1], dL_dW.shape[1], dL_dW.shape[2], X.shape[-1]))
    dB = np.sum(dL_dW, axis=(0, 1, 2))

    acc_sum = 0
    for i in range(0, dL_dW.shape[-1]):
        for ii in range(0, X.shape[-1]):
            acc_sum += dL_dW[:, :, :, i:i+1] * X[:, :, :, ii:ii+1]
        partial_w[i] = np.mean(acc_sum, axis=0)

    range_col = X.shape[1] - kernel.shape[1]
    range_row = X.shape[2] - kernel.shape[2]
    # kernel -> n_filters = image channels, i, j, depth filter
    # n_filters
    for i in range(0, kernel.shape[0]):
        for x in range(0, range_col, stride[0]):
            for y in range(0, range_row, stride[1]):
                gradient[i:i+1, :, :, :] -= partial_w[i:i+1, x:x + kernel.shape[1], y:y + kernel.shape[2], :]

    return gradient, dB


def dA_conv(Y: np.ndarray, last_weights: np.ndarray, stride=(1, 1), padding=None):
    # padded type
    input_padded = np.pad(Y,
                          pad_width=((0, 0),
                                     (0, Y.shape[0] % last_weights.shape[1] + 1),
                                     (0, Y.shape[1] % last_weights.shape[2] + 1),
                                     (0, 0)),
                          mode='constant', constant_values=0)

    dX = np.zeros_like(input_padded)
    for i in range(Y.shape[0]):
        for x in range(0, Y.shape[1], stride[0]):
            for y in range(0, Y.shape[2], stride[1]):
                for z in range(0, Y.shape[3]):
                    dX[i:i + 1, x:x + last_weights.shape[1], y:y + last_weights.shape[2], z:z + 1] += (
                            last_weights[:, :, :, z:z + 1] *
                            input_padded[i:i + 1, x:x + last_weights.shape[1], y:y + last_weights.shape[2], z:z + 1])

    dX = dX[:, 0:Y.shape[1], 0:Y.shape[2], :]

    return dX
