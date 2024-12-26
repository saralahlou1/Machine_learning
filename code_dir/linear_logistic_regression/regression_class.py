import math
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from code_dir.linear_logistic_regression.data_class import UCIdata
from code_dir.timer import Timer


def one_dim_handler(x):
    if x.ndim == 1:
        return x[:, None]
    else:
        return x


class Regression(ABC):

    def __init__(self, add_bias=True, stand=False, regularization=0, reg_type=2):
        self.add_bias = add_bias
        self.weight = None
        self.cost = None
        self.stand = stand
        self.reg_lambda = regularization
        self.reg_type = reg_type

    def do_add_bias(self, x):
        x = one_dim_handler(x)
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])
        return x

    def set_reg_lambda(self, reg):
        self.reg_lambda = reg

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def cost_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def figure_of_merit(self, *args, **kwargs):
        pass

    def train_and_test_cost(self, test_percentage, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage, random_state=42)
        self.fit(x_train, y_train)
        train_residual = self.figure_of_merit(x_train, y_train)
        test_residual = self.figure_of_merit(x_test, y_test)
        return train_residual, test_residual

    def correlation_fit(self, data_set: UCIdata, num_corr_features: int, test_percentages=(0.4,), high=True):
        correlation_array, feature_name = data_set.correlation_to_name()
        if high:
            left_feature_name = list(feature_name[len(feature_name) - num_corr_features:])
        else:
            left_feature_name = list(feature_name[: num_corr_features])

        if self.stand:
            x = data_set.standarize()
            x = x.to_numpy()
        else:
            x = data_set.all_data_features[left_feature_name].to_numpy()
        x = self.do_add_bias(x)
        y = data_set.all_data_targets[data_set.target_to_classify].to_numpy()

        train_residuals, test_residuals, weights = self.loop_percentage_test(test_percentages, x, y)
        return train_residuals, test_residuals, weights

    def loop_percentage_test(self, percentages, x, y):
        train_cost_list = []
        test_cost_list = []
        prediction_list = []

        # for percentage_test in tqdm(percentages, desc="Percent test loop...", leave=False):
        for percentage_test in percentages:
            train_cost, test_cost = self.train_and_test_cost(percentage_test, x, y)
            train_cost_list.append(train_cost)
            test_cost_list.append(test_cost)
            prediction_list.append(self.predict(x))

        return train_cost_list, test_cost_list, prediction_list

    def feature_number_loop(self, data_set: UCIdata, test_percentages, high=True):
        feature_numbers = [feat_num for feat_num in range(1, len(data_set.all_data_features.columns) + 1)]
        train_cost, test_cost, corr_weights = [], [], []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.correlation_fit, data_set, feat_num, test_percentages, high)
                for feat_num in feature_numbers]

            for future in tqdm(futures, total=len(feature_numbers), desc="Number of features Loop...", leave=True,
                               position=1):
                train_cost_list, test_cost_list, weights = future.result()
                train_cost.append(train_cost_list)
                test_cost.append(test_cost_list)
                corr_weights.append(weights)

        return feature_numbers, train_cost, test_cost, corr_weights

    def series_feature_number_loop(self, data_set: UCIdata, test_percentages, high=True):
        feature_numbers = [feat_num for feat_num in range(1, len(data_set.all_data_features.columns) + 1)]
        train_cost, test_cost, corr_weights = [], [], []
        for feat_num in tqdm(feature_numbers, desc="Number of features Loop...", leave=True):
            train_cost_list, test_cost_list, weights = self.correlation_fit(data_set, feat_num,
                                                                            test_percentages=test_percentages,
                                                                            high=high)

            train_cost.append(train_cost_list)
            test_cost.append(test_cost_list)
            corr_weights.append(weights)

        return feature_numbers, train_cost, test_cost, corr_weights

    def do_train_validation(self, x_test, y_test, x_train, y_train, train_index, val_index, reg_lambda_list):
        train_costs = []
        validate_costs = []
        test_costs = []
        x_act_train, x_val = x_train[train_index], x_train[val_index]
        y_act_train, y_val = y_train[train_index], y_train[val_index]
        weights = []
        for reg_lambda in reg_lambda_list:
            self.set_reg_lambda(reg_lambda)
            self.fit(x_act_train, y_act_train)
            train_costs.append(self.figure_of_merit(x_act_train, y_act_train))
            validate_costs.append(self.figure_of_merit(x_val, y_val))
            test_costs.append(self.figure_of_merit(x_test, y_test))
            weights.append(self.weight)
        return train_costs, validate_costs, test_costs, weights

    def _process_lambda(self, reg_lambda, x_act_train, y_act_train, x_val, y_val, x_test, y_test):
        """A helper function to process each regularization lambda."""
        self.set_reg_lambda(reg_lambda)
        self.fit(x_act_train, y_act_train)
        train_cost = self.figure_of_merit(x_act_train, y_act_train)
        validate_cost = self.figure_of_merit(x_val, y_val)
        test_cost = self.figure_of_merit(x_test, y_test)
        weight = self.weight
        return train_cost, validate_cost, test_cost, weight

    def par_do_train_validation(self, x_test, y_test, x_train, y_train, train_index, val_index, reg_lambda_list):
        # Preparing the data splits
        x_act_train, x_val = x_train[train_index], x_train[val_index]
        y_act_train, y_val = y_train[train_index], y_train[val_index]

        train_costs = []
        validate_costs = []
        test_costs = []
        weights = []

        # Use ProcessPoolExecutor to parallelize the loop over reg_lambda_list
        with ProcessPoolExecutor(max_workers=4) as executor:
            # Submit tasks to executor and store futures
            futures = [
                executor.submit(self._process_lambda, reg_lambda, x_act_train, y_act_train, x_val, y_val, x_test, y_test)
                for reg_lambda in reg_lambda_list
            ]

            # Process the futures as they complete and display progress with tqdm
            for future in tqdm(futures, total=len(reg_lambda_list), desc="RegLambda Loop", leave=False):
                train_cost, validate_cost, test_cost, weight = future.result()
                train_costs.append(train_cost)
                validate_costs.append(validate_cost)
                test_costs.append(test_cost)
                weights.append(weight)

        return train_costs, validate_costs, test_costs, weights

class MiniBatchGradientDescent(ABC):

    def __init__(self, learning_rate=0.001, max_iters=1e4, epsilon=1e-4, batch_size=32, momentum=0):
        self.batch_number = None
        self.weight = None
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.cost_history = []
        self.weight_history = []
        self.momentum = momentum

    def set_batch_size(self, size):
        self.batch_size = size

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    def _create_mini_batches(self, x, y):
        """Shuffle and create mini-batches from the dataset."""

        if self.batch_size == -1:
            self.batch_number = 1
            yield x, y

        else:
            indices = np.random.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            for start in range(0, len(x), self.batch_size):
                end = start + self.batch_size
                yield x_shuffled[start:end], y_shuffled[start:end]

    def gradient_descent(self, x, y, figure_of_merit):
        N, D = x.shape
        self.weight = np.zeros(D)
        self.weight_history = []
        grad = np.inf
        t = 0
        self.batch_number = math.ceil(len(x) / self.batch_size)
        old_delta_weight = 0
        with tqdm(total=self.max_iters, desc="Epochs", leave=False) as epoch_pbar:
            while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
                for x_batch, y_batch in self._create_mini_batches(x, y):
                    grad = self.gradient_fn(x_batch, y_batch)

                    delta_weight = (1 - self.momentum) * grad + self.momentum * old_delta_weight
                    old_delta_weight = delta_weight

                    self.weight = self.weight - self.learning_rate * delta_weight
                    self.cost_history.append(figure_of_merit(x, y))
                    self.weight_history.append(self.weight)
                t += 1
                epoch_pbar.update(1)
            epoch_pbar.close()

    @abstractmethod
    def gradient_fn(self, x, y):
        pass

    def series_cost_wrt_iteration(self, x, y, figure_of_merit, learning_rates):
        times = []
        timing = Timer(False)
        all_cost_history = []
        for dex, rate in tqdm(enumerate(learning_rates), total=len(learning_rates), leave=False,
                              desc="Learning rate loop..."):
            self.learning_rate = rate
            deep_x, deep_y = x.copy(), y.copy()
            timing.start()
            self.gradient_descent(deep_x, deep_y, figure_of_merit)
            timing.stop()
            times.append(timing.elapsed_time)
            timing.reset()
            all_cost_history.append(self.cost_history)
            self.cost_history = []
        return times, all_cost_history

    def cost_wrt_iteration(self, x, y, cost_function, learning_rates):
        accuracies = []
        times = []
        all_cost_history = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_gradient_descent, self, x, y, cost_function, rate)
                for rate in learning_rates
            ]
            for future in tqdm(futures, total=len(learning_rates), leave=True, desc="Learning rate loop..."):
                gradient_accuracy, elapsed_time, cost_history = future.result()
                accuracies.append(gradient_accuracy)
                times.append(elapsed_time)
                all_cost_history.append(cost_history)

        return accuracies, times, all_cost_history

    def series_batch_cost_wrt_iteration(self, x, y, figure_of_merit, learning_rates, batch_size_list):
        batch_times, batch_cost_history = [], []
        for batch in tqdm(batch_size_list, total=len(batch_size_list), leave=True, desc="Batches size loop..."):
            self.set_batch_size(batch)
            rate_times, rate_cost_history = self.series_cost_wrt_iteration(x, y, figure_of_merit, learning_rates)
            batch_times.append(rate_times)
            batch_cost_history.append(rate_cost_history)
        return batch_times, batch_cost_history

    def batch_cost_wrt_iteration(self, x, y, cost_function, learning_rates, batch_size_list):
        batch_accuracies, batch_times, batch_cost_history = [], [], []

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_batch, self, x, y, cost_function, learning_rates, batch): batch
                for batch in batch_size_list
            }
            for future in tqdm(as_completed(futures), total=len(batch_size_list), leave=True,
                               desc="Batches size loop..."):
                batch = futures[future]
                try:
                    rate_accuracies, rate_times, rate_cost_history = future.result()
                    batch_accuracies.append(rate_accuracies)
                    batch_times.append(rate_times)
                    batch_cost_history.append(rate_cost_history)
                except Exception as exc:
                    print(f"Batch {batch} generated an exception: {exc}")

        return batch_accuracies, batch_times, batch_cost_history


def run_gradient_descent(instance, x, y, cost_function, rate):
    """Helper function to be used in parallel.
    It performs gradient descent for a given learning rate."""
    if instance.batch_size == -1:
        instance.max_iters = 50
    deep_x, deep_y = x.copy(), y.copy()
    timing = Timer(False)
    instance.learning_rate = rate
    timing.start()
    instance.gradient_descent(deep_x, deep_y, cost_function)
    timing.stop()
    elapsed_time = timing.elapsed_time
    cost_history = instance.cost_history.copy()
    instance.cost_history = []
    gradient_accuracy = instance.figure_of_merit(x, y)
    return gradient_accuracy, elapsed_time, cost_history


def process_batch(instance, x, y, cost_function, learning_rates, batch):
    instance.set_batch_size(batch)
    return instance.cost_wrt_iteration(x, y, cost_function, learning_rates)
