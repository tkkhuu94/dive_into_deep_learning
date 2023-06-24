from d2l import torch as d2l
import torch

import sgd


class LinearRegressionScratch(d2l.Module):

    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.lr = lr
        self.sigma = sigma

        self.w = torch.normal(mean=0,
                               std=sigma,
                               size=(num_inputs, 1),
                               requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def __str__(self):
        return 'W: {}, b: {}'.format(self.w, self.b)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y)**2 / 2
        return l.mean()

    def configure_optimizers(self):
        return sgd.SGD(params=[self.w, self.b], lr=self.lr)


class LinearRegression(d2l.Module):
    def __init__(self, num_inputs, lr):
        super().__init__()
        self.lr = lr
        self.net = torch.nn.Linear(in_features=num_inputs, out_features=1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def __str__(self):
        return 'W: {}, b: {}'.format(self.w, self.b)
    
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = torch.nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    @property
    def w(self):
        return self.net.weight.data
    
    @property
    def b(self):
        return self.net.bias.data
