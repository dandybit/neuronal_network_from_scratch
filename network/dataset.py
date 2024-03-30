import numpy as np

class Dataset:
    def __init__(self, data, labels, batch_size=32, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        if data.shape[0] % batch_size == 0:
            self.data = np.reshape(data, (-1, batch_size, data.shape[-1]))
            self.labels = np.reshape(labels, (-1, batch_size, labels.shape[-1]))
        else:
            drop_ele = data.shape[0] % batch_size
            self.data = np.reshape(data[:-drop_ele], (-1, batch_size, data.shape[-1]))
            self.labels = np.reshape(labels[:-drop_ele], (-1, batch_size, labels.shape[-1]))
        self.batch_size = batch_size

    def get_dataset(self):
        return self.data

    def get_labels(self):
        return self.labels

    def __iter__(self):
        return DatasetIterator(self.data, self.labels)


class DatasetIterator:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.index = 0

    def __next__(self):
        try:
            n_item = (self.data[self.index], self.labels[self.index])
        except IndexError:
            raise StopIteration()

        self.index += 1
        return n_item

    def __iter__(self):
        return self
