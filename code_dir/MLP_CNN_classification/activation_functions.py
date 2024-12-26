import warnings
from abc import abstractmethod
from code_dir.MLP_CNN_classification.config import xp
# import numpy as xp


class ActivationFunction:

    def __init__(self):
        pass

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def function_derivative(self, x):
        pass



class ReLU(ActivationFunction):
    def function(self, x):
        return xp.maximum(0, x)

    def function_derivative(self, x):
        return (x > 0).astype(xp.float32)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def function(self, x):
        return xp.where(x > 0, x, self.alpha * x)

    def function_derivative(self, x):
        return xp.where(x > 0, 1, self.alpha)


class Softmax(ActivationFunction):
    def function(self, x):
        exp_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True)).astype(xp.float64)
        return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)

    def function_derivative(self, x):
        s = self.function(x)
        jacobian = xp.array([xp.diagflat(s_i.flatten()) - xp.outer(s_i, s_i.T) for s_i in s])
        return jacobian


class Tanh(ActivationFunction):
    def function(self, x):
        return xp.tanh(x)

    def function_derivative(self, x):
        return 1 - xp.tanh(x) ** 2

