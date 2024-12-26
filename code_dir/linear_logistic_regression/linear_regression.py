import numpy as np

from code_dir.linear_logistic_regression.regression_class import Regression, MiniBatchGradientDescent


class LinearRegression(Regression, MiniBatchGradientDescent):

    def __init__(self, regression_type, add_bias=True, learning_rate=0.01, max_iters=1e4, batch_size=32, epsilon=1e-4,
                 stand=False, regularization=0, reg_type=1, momentum=0):
        Regression.__init__(self, add_bias, stand=stand, regularization=regularization, reg_type=reg_type)
        MiniBatchGradientDescent.__init__(self, learning_rate=learning_rate, max_iters=max_iters, epsilon=epsilon, batch_size=batch_size, momentum=momentum)
        self.regression_type = regression_type

    def fit(self, x, y):
        if self.regression_type == "normal":
            # self.weight, self.cost, _, _ = np.linalg.lstsq(x, y)
            XTX = np.dot(np.transpose(x), x)
            XTXinv = np.linalg.pinv(XTX + self.reg_lambda * np.eye(XTX.shape[0], XTX.shape[1]))
            XTy = np.dot(np.transpose(x), y)
            self.weight = XTXinv @ XTy
        elif self.regression_type == "sgd":
            self.sgd_fit(x, y)
        return self

    def predict(self, x):
        yh = x@self.weight
        return yh

    def cost_function(self, x, y):
        return .5 * np.mean((self.predict(x) - y)**2) + self.reg_lambda * np.sum(np.abs(self.weight[1:]) ** self.reg_type)

    def gradient_fn(self, x, y):
        yh = self.predict(x)
        N, D = x.shape
        grad = .5 * np.dot(yh - y, x) / N
        grad[1:] += self.reg_lambda * np.sign(self.weight[1:]) * np.abs(self.weight[1:]) ** (self.reg_type - 1)
        return grad

    def sgd_fit(self, x, y):
        self.gradient_descent(x, y, self.cost_function)
        return self

    def figure_of_merit(self, x, y):
        return .5 * np.mean((self.predict(x) - y)**2)
