import torch

from d2l import torch as d2l

class SyntheticRegressionData(d2l.DataModule):

    def __init__(self, w, b, std_dev, num_train, num_val, batch_size):

        # Reshaping w to have (number of features X 1)
        self._W = w.reshape(-1, 1)
        self._b = b
        self._std_dev = std_dev
        self._num_train = num_train
        self._num_val = num_val
        self._batch_size = batch_size

        total_num_examples = self._num_train + self._num_val

        # X has shape (number of examples X number of features)
        self._X = torch.randn(total_num_examples, len(self._W))

        noise = torch.randn(total_num_examples, 1) + self._std_dev
        self._y = torch.matmul(self._X, self._W) + self._b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_train(self):
        return self._num_train

    @property
    def batch_size(self):
        return self._batch_size
