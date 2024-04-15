import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_conv_op(image: np.ndarray, kernel: np.ndarray, stride=(1, 1), padding=1) -> np.ndarray:

    image_in = torch.tensor(np.reshape(image, (image.shape[0], image.shape[-1], image.shape[1], image.shape[2])), dtype=torch.float32)
    kernel_in = torch.tensor(np.reshape(kernel, (kernel.shape[-1], kernel.shape[0], kernel.shape[1], kernel.shape[2])), dtype=torch.float32)

    print("*" * 100)
    print(image_in.shape)
    print(kernel_in.shape)
    print("*" * 100)

    conv_op = F.conv2d(image_in, kernel_in, stride=stride, padding=padding).numpy()

    return np.reshape(conv_op, (conv_op.shape[0], conv_op.shape[2], conv_op.shape[3], conv_op.shape[1]))

def op_conv2d(image: np.ndarray, kernel: np.ndarray, stride=(1, 1), padding=None) -> np.ndarray:
    output_image = np.zeros(image.shape)

    # padded type
    input_padded = np.pad(image,
                          pad_width=((0, 0),
                                     (0, kernel.shape[1] % image.shape[1]),
                                     (0, kernel.shape[2] % image.shape[2]),
                                     (0, 0)),
                          mode='constant', constant_values=0)

    for x in range(0, image.shape[1], stride[0]):
        for y in range(0, image.shape[2], stride[1]):
            for z in range(0, image.shape[3]):
                conv_op = kernel * input_padded[:, x:x + kernel.shape[1], y:y + kernel.shape[2], z:z + 1]
                conv_op = np.sum(conv_op, axis=(1, 2))
                # re-dim to sum in output_k
                conv_op = conv_op[:, np.newaxis, np.newaxis, :]
                output_image[:, x:x + 1, y:y + 1, :] += conv_op

    return output_image


def op_back_conv2d(X: np.ndarray, dL_dW: np.ndarray, kernel: np.ndarray, stride=(1, 1), padding=None) -> tuple:
    gradient = np.zeros(kernel.shape)
    dB = np.sum(dL_dW, axis=(0, 1, 2))

    # padded type
    input_padded = np.pad(X,
                          pad_width=((0, 0),
                                     (0, kernel.shape[1] % X.shape[1]),
                                     (0, kernel.shape[2] % X.shape[2]),
                                     (0, 0)),
                          mode='constant', constant_values=0)

    for i in range(X.shape[0]):
        for x in range(0, X.shape[1], stride[0]):
            for y in range(0, X.shape[2], stride[1]):
                for z in range(0, X.shape[3]):
                    conv_op = input_padded[i:i+1, x:x + kernel.shape[1], y:y + kernel.shape[2], z:z+1]
                    # reverse convolution
                    gradient += conv_op * dL_dW[i, x, y, z]

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
                    dX[i:i+1, x:x+last_weights.shape[1], y:y+last_weights.shape[2], z:z+1] += (
                            last_weights[:, :, :, z:z+1] *
                            input_padded[i:i+1, x:x+last_weights.shape[1], y:y+last_weights.shape[2], z:z+1])

    dX = dX[:, 0:Y.shape[1], 0:Y.shape[2], :]

    return dX


