from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from code_dir.linear_logistic_regression.regression_class import Regression


class SyntheticData(ABC):
    sampled_function: Callable[[np.ndarray], np.ndarray]
    nonlinear_basis_func: Callable[[np.ndarray], np.ndarray]
    noise: Callable[[float], np.ndarray]

    def __init__(self, sampled_function, nonlinear_basis_func, sample_interval, num_samples, noise):
        self.nonlinear_basis_func = nonlinear_basis_func
        self.sampled_function = sampled_function
        self.noise = noise
        self.sample_interval = sample_interval
        self.num_samples = num_samples

    def x_y_values(self):
        x = np.linspace(self.sample_interval[0], self.sample_interval[1], self.num_samples)
        y = self.sampled_function(x)
        return x, y


    def get_x_for_regression(self, num_basis_func, *args):
        x, y = self.x_y_values()
        basis_parameter = np.linspace(self.sample_interval[0], self.sample_interval[1], num_basis_func)

        results = [self.nonlinear_basis_func(x, mu, *args) for mu in basis_parameter]
        return np.array(results).T, y + self.noise(len(y))


    def percentage_loop(self, regression_model: Regression, num_basis_func: int, percentages):
        x, y = self.get_x_for_regression(num_basis_func, 1)
        x = regression_model.do_add_bias(x)
        train_cost_list, test_cost_list, prediction_list = regression_model.loop_percentage_test(percentages, x, y)
        # print(f"Min testing cost num_basis {num_basis_func}: {percentages[test_cost_list.index(np.min(test_cost_list))]}")
        # print(f"Min training cost num_basis {num_basis_func}: {percentages[train_cost_list.index(np.min(train_cost_list))]}")
        return train_cost_list, test_cost_list, prediction_list

    def num_basis_fun_percent_loop(self, regression_model: Regression, nums_basis_func, percentages):
        train_costs, test_costs, predictions = [], [], []
        for num_basis_func in nums_basis_func:
            train_cost_list, test_cost_list, prediction_list = self.percentage_loop(regression_model, num_basis_func, percentages)
            train_costs.append(train_cost_list)
            test_costs.append(test_cost_list)
            predictions.append(prediction_list)

        return train_costs, test_costs, predictions


    def num_basis_fun_loop(self, regression_model: Regression, nums_basis_func, percentage):
        train_costs, test_costs, xs, ys, y_hs, weights = [], [], [], [], [], []
        for num_basis_func in nums_basis_func:
            x, y = self.get_x_for_regression(num_basis_func, 1)
            x = regression_model.do_add_bias(x)
            train_residual, test_residual = regression_model.train_and_test_cost(percentage, x, y)
            train_costs.append(train_residual)
            test_costs.append(test_residual)
            y_hs.append(regression_model.predict(x))
            weights.append(regression_model.weight)
            xs.append(x)

        return train_costs, test_costs, y_hs, weights, xs


    def cross_validation(self, num_basis_func, num_partitions, test_percentage, regression_model: Regression, reg_lambda_list):
        kf = KFold(n_splits=num_partitions)
        x, y = self.get_x_for_regression(num_basis_func, 1)
        x = regression_model.do_add_bias(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage, random_state=42)
        all_train_cost, all_validate_cost, all_test_cost, weights = [], [], [], []
        for train_index, val_index in tqdm(kf.split(x_train), desc="K-Fold Validation", leave=False, total=num_partitions):
            train_costs, validate_costs, test_costs, weight = \
                regression_model.do_train_validation(x_test, y_test, x_train, y_train, train_index, val_index, reg_lambda_list)
            all_train_cost.append(train_costs)
            all_validate_cost.append(validate_costs)
            weights.append(weight)
            all_test_cost.append(test_costs)

        return all_train_cost, all_validate_cost, all_test_cost, weights

    def par_cross_validation(self, num_basis_func, num_partitions, test_percentage, regression_model: Regression, reg_lambda_list):
        kf = KFold(n_splits=num_partitions)
        x, y = self.get_x_for_regression(num_basis_func, 1)
        x = regression_model.do_add_bias(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage, random_state=42)
        all_train_cost, all_validate_cost, all_test_cost, weights = [], [], [], []
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    _process_k_fold, regression_model, x_test, y_test, x_train, y_train, train_index, val_index,
                    reg_lambda_list
                ) for train_index, val_index in kf.split(x_train)
            ]

            for future in tqdm(futures, total=num_partitions, desc="K-Fold Validation", leave=False):
                train_costs, validate_costs, test_costs, weight = future.result()
                all_train_cost.append(train_costs)
                all_validate_cost.append(validate_costs)
                all_test_cost.append(test_costs)
                weights.append(weight)

        return all_train_cost, all_validate_cost, all_test_cost, weights



def _process_k_fold(regression_model: Regression, x_test, y_test, x_train, y_train, train_index, val_index, reg_lambda_list):
    """
    Helper function to handle each fold in parallel.
    Calls the `do_train_validation` method on the model.
    """
    return regression_model.par_do_train_validation(x_test, y_test, x_train, y_train, train_index, val_index, reg_lambda_list)

