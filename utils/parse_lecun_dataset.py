import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# MNIST DATASET.
# http://yann.lecun.com/exdb/mnist/
def parse_dataset() -> tuple:
    for file in glob.glob("dataset_example/train/*"):
        if "images" in file:
            with open(file, 'rb') as f:
                train_X = _parse_train_x(bytearray(f.read()))
            f.close()
        else:
            with open(file, 'rb') as f:
                train_y = _parse_train_y(bytearray(f.read()))
            f.close()

    for file in glob.glob("dataset_example/test/*"):
        if "images" in file:
            with open(file, 'rb') as f:
                test_X = _parse_train_x(bytearray(f.read()))
            f.close()
        else:
            with open(file, 'rb') as f:
                test_y = _parse_train_y(bytearray(f.read()))
            f.close()

    return train_X, train_y, test_X, test_y


def _parse_train_x(train_raw) -> np.ndarray:
    magig_number = int.from_bytes(train_raw[:4], byteorder="big")
    number_images = int.from_bytes(train_raw[4:8], byteorder="big")
    number_rows = int.from_bytes(train_raw[8:12], byteorder="big")
    number_columns = int.from_bytes(train_raw[12:16], byteorder="big")

    total_size_image = number_columns * number_rows
    byte_image = np.frombuffer(train_raw[16:], dtype=np.uint8)

    images_flat = byte_image.reshape(number_images, total_size_image)

    return images_flat


def _parse_train_y(label_raw) -> np.ndarray:
    magic_number = int.from_bytes(label_raw[:4], byteorder="big")
    number_labels = int.from_bytes(label_raw[4:8], byteorder="big")

    encoder = OneHotEncoder(sparse_output=False)
    values = [[x] for x in range(10)]
    encoder.fit(values)

    labels = np.stack([encoder.transform([[x]]) for x in label_raw[8:]])
    labels = labels.reshape(labels.shape[0], len(values), )

    return labels
