class SGD(object):
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr

    def step(self):
        for param in self._params:
            param -= self._lr * param.grad

    def zero_grad(self):
        for param in self._params:
            if param.grad is not None:
                param.grad.zero_()