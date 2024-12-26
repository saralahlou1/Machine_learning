import math
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from code_dir.MLP_CNN_classification.config import xp
from code_dir.MLP_CNN_classification.activation_functions import ActivationFunction


def one_hot_converter(y_one_hot):
    y_normal = xp.argmax(y_one_hot, axis=1)
    if xp is not np:
        y_normal = xp.asnumpy(y_normal)
    return y_normal


class MLP:
    layer_sizes: List[int]
    learning_rate: float
    activation_funcs: List[ActivationFunction]
    weights: List[xp.ndarray]
    biases: List[xp.ndarray]
    batch_size: int
    epochs: int
    batch_number: int
    epsilon: float
    momentum: float
    reg_type: int
    reg_lambda: float
    accuracy_per_epoch: List[float]

    def __init__(self, layer_sizes, activation_funcs, learning_rate=0.025, batch_size=-1, epochs=50,
                 momentum=0, reg_lambda=0, reg_type=2):
        self.momentum = momentum
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_funcs = activation_funcs
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.accuracy_per_epoch = []
        self.test_accuracy_per_epoch = []

    def forward(self, X):
        activations = [X]
        for dex, weight in enumerate(self.weights):
            z = xp.matmul(activations[-1], weight) + self.biases[dex]
            a = self.activation_funcs[dex].function(z)
            activations.append(a)
        return activations

    def backpropagation(self, X, y):
        activations = self.forward(X)

        weight_grads = [xp.zeros_like(w) for w in self.weights]
        bias_grads = [xp.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y
        for dex in reversed(range(len(self.weights))):

            weight_grads[dex] = xp.matmul(activations[dex].T, delta) / self.batch_size + \
                                self.reg_lambda * xp.sign(self.weights[dex]) * xp.abs(self.weights[dex]) ** (self.reg_type - 1)
            bias_grads[dex] = xp.sum(delta, axis=0, keepdims=True) / self.batch_size

            if dex > 0:
                delta = xp.matmul(delta, self.weights[dex].T) * self.activation_funcs[dex - 1].function_derivative(
                    activations[dex])

        return weight_grads, bias_grads

    def update_parameters(self, weight_grads, bias_grads, old_delta_weights):

        for dex, weight_grad in enumerate(weight_grads):
            delta_weight = (1 - self.momentum) * weight_grad + self.momentum * old_delta_weights[dex]
            self.weights[dex] -= self.learning_rate * delta_weight
            self.biases[dex] -= self.learning_rate * bias_grads[dex]
            old_delta_weights[dex] = delta_weight

        return old_delta_weights

    def _create_mini_batches(self, x, y):
        if self.batch_size == -1:
            self.batch_number = 1
            yield x, y
        else:
            indices = xp.random.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            for start in range(0, len(x), self.batch_size):
                end = start + self.batch_size
                yield x_shuffled[start:end], y_shuffled[start:end]

    def mpl_gradient_descent(self, X, y, x_test, y_test):
        self.weights = [xp.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.1 for i in
                        range(len(self.layer_sizes) - 1)]
        self.biases = [xp.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]

        t = 0
        self.batch_number = math.ceil(len(X) / self.batch_size)
        old_delta_weights = [xp.zeros_like(weight) for weight in self.weights]
        with tqdm(total=self.epochs, desc="Epochs", leave=True) as epoch_pbar:
            while t < self.epochs:
                for x_batch, y_batch in self._create_mini_batches(X, y):
                    weight_grads, bias_grads = self.backpropagation(x_batch, y_batch)
                    old_delta_weights = self.update_parameters(weight_grads, bias_grads, old_delta_weights)
                t += 1
                self.accuracy_per_epoch.append(self.mlp_figure_of_merit(X, y))
                self.test_accuracy_per_epoch.append(self.mlp_figure_of_merit(x_test, y_test))
                epoch_pbar.update(1)
            epoch_pbar.close()

    def cost_function(self, x, y):
        log_preds = xp.log(self.predict(x) + 1e-12)
        labels_one_hot = xp.eye(self.predict(x).shape[1])[y]
        return -xp.sum(labels_one_hot * log_preds) / self.predict(x).shape[0] + \
               self.reg_lambda * sum([xp.abs(weight) ** self.reg_type for weight in self.weights])


    def mlp_figure_of_merit(self, x, y):
        y_true = one_hot_converter(y)
        y_pred = one_hot_converter(self.predict(x))
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def predict(self, x):
        return self.forward(x)[-1]
