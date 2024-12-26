import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from code_dir.linear_logistic_regression.linear_regression import LinearRegression
from code_dir.linear_logistic_regression.run import grid, get_color
from code_dir.regularization_and_model_evaluation.synthetic_data import SyntheticData


def sample_function_1(truc):
    return np.sin(np.sqrt(truc)) + np.cos(truc) + np.sin(truc)


def sample_function_2(truc):
    return -4 * truc + 10


def basis_function(truc, mu, sigma):
    return np.exp(-((truc - mu) / sigma) ** 2)


def series_multiple_data_cross_valid(model, num_repetition, lambdas, num_basis_functions=20, number_data_points=50, par=False):
    sample_interval = [0, 20]
    all_train_cost, all_validate_cost, all_test_cost, all_weights = [], [], [], []
    all_predictions, all_noise = [], []
    noise = np.random.randn
    synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points, noise)
    x, y = synthetic_data.get_x_for_regression(num_basis_functions, 1)
    x = model.do_add_bias(x)
    for _ in tqdm(range(num_repetition), desc="Dataset rep...", leave=True, total=num_repetition):
        np.random.seed(random.randint(1, 1000))
        noise = np.random.randn
        synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points, noise)
        if par:
            one_data_train_cost, one_data_validate_cost, one_data_test_cost, one_data_weights = \
                synthetic_data.par_cross_validation(num_basis_functions, 10, 0.2, model, lambdas)
        else:
            one_data_train_cost, one_data_validate_cost, one_data_test_cost, one_data_weights = \
                synthetic_data.cross_validation(num_basis_functions, 10, 0.2, model, lambdas)
        cross_lambda_predictions = [[x @ lambda_weight for lambda_weight in cross_weight] for cross_weight in
                                    one_data_weights]
        all_train_cost.append(one_data_train_cost)
        all_validate_cost.append(one_data_validate_cost)
        all_test_cost.append(one_data_test_cost)
        all_weights.append(one_data_weights)
        all_predictions.append(cross_lambda_predictions)
        all_noise.append(noise(len(y)))

    return np.array(all_train_cost), np.array(all_validate_cost), np.array(all_test_cost), np.array(all_weights), \
           np.array(all_predictions), np.array(all_noise)


def _process_repetition(model, num_basis_functions, number_data_points, lambdas, sample_interval):
    """
    Helper function to process a single repetition of data generation and cross-validation.
    """
    noise = np.random.randn
    synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points, noise)
    x, y = synthetic_data.get_x_for_regression(num_basis_functions, 1)
    x = model.do_add_bias(x)

    np.random.seed(random.randint(1, 1000))
    noise = np.random.randn
    synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points, noise)

    one_data_train_cost, one_data_validate_cost, one_data_test_cost, one_data_weights = \
        synthetic_data.cross_validation(num_basis_functions, 10, 0.2, model, lambdas)

    cross_lambda_predictions = [[x @ lambda_weight for lambda_weight in cross_weight] for cross_weight in
                                one_data_weights]

    return one_data_train_cost, one_data_validate_cost, one_data_test_cost, one_data_weights, cross_lambda_predictions, noise(
        len(y))


def multiple_data_cross_valid(model, num_repetition, lambdas, num_basis_functions=20, number_data_points=50):
    sample_interval = [0, 20]
    all_train_cost, all_validate_cost, all_test_cost, all_weights = [], [], [], []
    all_predictions, all_noise = [], []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _process_repetition, model, num_basis_functions, number_data_points, lambdas, sample_interval
            ) for _ in range(num_repetition)
        ]

        for future in tqdm(futures, desc="Dataset rep...", leave=True, total=num_repetition):
            one_data_train_cost, one_data_validate_cost, one_data_test_cost, one_data_weights, cross_lambda_predictions, noise = future.result()

            all_train_cost.append(one_data_train_cost)
            all_validate_cost.append(one_data_validate_cost)
            all_test_cost.append(one_data_test_cost)
            all_weights.append(one_data_weights)
            all_predictions.append(cross_lambda_predictions)
            all_noise.append(noise)

    return np.array(all_train_cost), np.array(all_validate_cost), np.array(all_test_cost), np.array(all_weights), \
           np.array(all_predictions), np.array(all_noise)



def variance_bias_calc(synthetic_data: SyntheticData, all_predictions, all_noise):
    """
    Returns a cross x lambda array. Return[0] is an array of bias_sqyared for example for one cross_validation wrt lambda.
    """
    expected_across_data = np.mean(all_predictions, axis=0)
    expected_noise_squared = np.mean(np.mean(0.5 * all_noise ** 2, axis=0), axis=0)
    variance = np.mean(np.mean((all_predictions - expected_across_data) ** 2, axis=0), axis=2)
    bias_squared = np.mean((expected_across_data - synthetic_data.x_y_values()[1]) ** 2, axis=2)
    return bias_squared, variance, expected_noise_squared


def cross_cost_lambda_plotter(reg_lambda_list, y_label_list, errors, optimum_lam, *args, ax=None, log_scale=True):
    for dex, list_to_plot in enumerate(args):
        mean_to_plot = np.mean(list_to_plot, axis=0)
        if errors[dex] is None:
            std_to_plot = np.std(list_to_plot, axis=0)
        else:
            std_to_plot = errors[dex]
        if y_label_list[dex] == "weights":
            mean_to_plot = mean_to_plot.T
            for dex_2, weight in enumerate(mean_to_plot[1:]):
                ax[dex].plot(reg_lambda_list, weight, marker='o', markersize=2, linestyle='-', linewidth=1,
                             color=get_color(dex_2, len(mean_to_plot)))

        else:
            for dex_3, cross_correlation in enumerate(list_to_plot):
                if dex_3 == 0:
                    ax[dex].plot(reg_lambda_list, cross_correlation, marker='o', markersize=5, linestyle='-',
                                 linewidth=2,
                                 color="green", label=f"{len(list_to_plot)} cross validations")
                else:
                    ax[dex].plot(reg_lambda_list, cross_correlation, marker='o', markersize=5, linestyle='-',
                                 linewidth=2, color="green")

            ax[dex].plot(reg_lambda_list, mean_to_plot, marker='o', markersize=5, linestyle='-', linewidth=2,
                         color="red", label=f"Average across the {len(list_to_plot)} \n cross validations")
            ax[dex].errorbar(reg_lambda_list, mean_to_plot, yerr=std_to_plot, fmt='none', ecolor='red', capsize=5)
            if log_scale:
                ax[dex].set_yscale('log')

        ax[dex].axvline(x=optimum_lam, color='k', linestyle='--', linewidth=2)
        grid(ax[dex])
        ax[dex].set_xlabel("lambda")
        ax[dex].set_ylabel(y_label_list[dex])
        ax[dex].set_xscale('log')
        ax[dex].legend()


if __name__ == "__main__":
    num_basis_func = 80
    num_data_points = 35
    sam_interval = [0, 20]
    all_lambda = np.logspace(-4, 2, 35)
    num_reps = 50
    run_noise = np.random.randn
    synth_data = SyntheticData(sample_function_1, basis_function, sam_interval, num_data_points, run_noise)
    with PdfPages(f"code_dir/hw_2/analysis_files/cross_validation_to_100.pdf") as line_pdf:
        for regularization_type in [1, 2]:
            linear_epoch = 10000
            linear_model = LinearRegression("sgd", add_bias=True, learning_rate=0.01, epsilon=1e-6,
                                            max_iters=linear_epoch, batch_size=-1, stand=False,
                                            reg_type=regularization_type)
            train_cost, validate_cost, test_cost, weights, predictions, noises = \
                multiple_data_cross_valid(linear_model, num_reps, all_lambda,
                                          num_basis_functions=num_basis_func, number_data_points=num_data_points)

            bias, variances, noise_var = variance_bias_calc(synth_data, predictions, noises)
            std_train_cost_rep = np.std(train_cost, axis=0)
            std_train_cost = np.sqrt(np.sum(std_train_cost_rep ** 2, axis=0)) / len(std_train_cost_rep)
            train_cost = np.mean(train_cost, axis=0)
            std_validate_cost_rep = np.std(validate_cost, axis=0)
            std_validate_cost = np.sqrt(np.sum(std_validate_cost_rep ** 2, axis=0)) / len(std_validate_cost_rep)
            validate_cost = np.mean(validate_cost, axis=0)
            std_test_cost_rep = np.std(test_cost, axis=0)
            std_test_cost = np.sqrt(np.sum(std_test_cost_rep ** 2, axis=0)) / len(std_test_cost_rep)
            test_cost = np.mean(test_cost, axis=0)
            weights = np.mean(weights, axis=0)
            f_6, ax_6 = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))
            opt_lambda = all_lambda[np.argmin(np.mean(validate_cost, axis=0))]
            print(f"L{regularization_type} lambda: {opt_lambda}")
            if regularization_type == 2:
                log_sca = False
            else:
                log_sca = True
            cross_cost_lambda_plotter(all_lambda,
                                      ["Training cost", "validation cost", "Testing cost", "weights",
                                       r"$bias^2$", "variances", r"$bias^2 + variances$",
                                       r"$bias^2 + variances + noise\ variance$"],
                                      [std_train_cost, std_validate_cost, std_test_cost, None, None, None, None, None],
                                      opt_lambda, train_cost, validate_cost, test_cost, weights, bias, variances,
                                      bias + variances, bias + variances + noise_var, ax=ax_6.flatten(), log_scale=log_sca)
            f_6.suptitle(f"Regularization L{regularization_type}")
            f_6.tight_layout()
            f_6.savefig(f"code_dir/hw_2/analysis_files/l{regularization_type}_to_100_reg")
            line_pdf.savefig(f_6)

    linear_model = LinearRegression("normal", reg_type=2)
    train_cost, validate_cost, test_cost, weights, predictions, noises = \
        series_multiple_data_cross_valid(linear_model, num_reps, all_lambda,
                                         num_basis_functions=num_basis_func, number_data_points=num_data_points, par=False)

    with PdfPages(f"code_dir/hw_2/analysis_files/analytical_l2.pdf") as line_pdf:
        bias, variances, noise_var = variance_bias_calc(synth_data, predictions, noises)
        std_train_cost_rep = np.std(train_cost, axis=0)
        std_train_cost = np.sqrt(np.sum(std_train_cost_rep ** 2, axis=0)) / len(std_train_cost_rep)
        train_cost = np.mean(train_cost, axis=0)
        std_validate_cost_rep = np.std(validate_cost, axis=0)
        std_validate_cost = np.sqrt(np.sum(std_validate_cost_rep ** 2, axis=0)) / len(std_validate_cost_rep)
        validate_cost = np.mean(validate_cost, axis=0)
        std_test_cost_rep = np.std(test_cost, axis=0)
        std_test_cost = np.sqrt(np.sum(std_test_cost_rep ** 2, axis=0)) / len(std_test_cost_rep)
        test_cost = np.mean(test_cost, axis=0)
        weights = np.mean(weights, axis=0)
        f_6, ax_6 = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))
        opt_lambda = all_lambda[np.argmin(np.mean(validate_cost, axis=0))]
        print(f"L{2} lambda: {opt_lambda}")
        cross_cost_lambda_plotter(all_lambda,
                                  ["Training cost", "validation cost", "Testing cost", "weights",
                                   r"$bias^2$", "variances", r"$bias^2 + variances$",
                                   r"$bias^2 + variances + noise\ variance$"],
                                  [std_train_cost, std_validate_cost, std_test_cost, None, None, None, None, None],
                                  opt_lambda, train_cost, validate_cost, test_cost, weights, bias, variances,
                                  bias + variances, bias + variances + noise_var, ax=ax_6.flatten(), log_scale=True)
        f_6.suptitle(f"Regularization L{2}")
        f_6.tight_layout()
        f_6.savefig(f"code_dir/hw_2/analysis_files/l{2}_new_reg_closed_form")
        line_pdf.savefig(f_6)
