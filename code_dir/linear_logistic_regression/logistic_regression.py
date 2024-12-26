import numpy as np
from code_dir.linear_logistic_regression.regression_class import Regression, MiniBatchGradientDescent
from sklearn.metrics import accuracy_score


def logistic(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(Regression, MiniBatchGradientDescent):
    def __init__(self, add_bias=True, learning_rate=0.01, max_iters=1e4, batch_size=32, epsilon=1e-4):
        Regression.__init__(self, add_bias)
        MiniBatchGradientDescent.__init__(self, learning_rate=learning_rate, max_iters=max_iters, epsilon=1e-4,
                                          batch_size=batch_size)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iters = max_iters

    def fit(self, x, y):
        self.gradient_descent(x, y, self.figure_of_merit)
        return self

    def predict(self, x):
        cost = logistic(np.dot(x, self.weight))
        return [0 if this_cost <= 0.5 else 1 for this_cost in cost]

    def gradient_fn(self, x, y):
        N, D = x.shape
        yh = logistic(np.dot(x, self.weight))
        return np.dot(x.T, yh - y) / N

    def cost_function(self, x, y):
        z = np.dot(x, self.weight)
        exp_z = np.exp(z)
        return np.mean(np.log1p(exp_z) - y * z)

    def figure_of_merit(self, x, y):
        return accuracy_score(y, self.predict(x))

