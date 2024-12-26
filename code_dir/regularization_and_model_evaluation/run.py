import itertools
import math
import random

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
from code_dir.linear_logistic_regression.linear_regression import LinearRegression
from code_dir.linear_logistic_regression.run import grid, get_color
from code_dir.regularization_and_model_evaluation.synthetic_data import SyntheticData


def sample_function_1(truc):
    return np.sin(np.sqrt(truc)) + np.cos(truc) + np.sin(truc)


def sample_function_2(truc):
    return -4 * truc + 10


def basis_function(truc, mu, sigma):
    return np.exp(-((truc - mu) / sigma) ** 2)


def plot_only_fun_noise(synth_data: SyntheticData, f=None, ax=None):
    x_values, y_values = synth_data.x_y_values()
    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    ax.plot(x_values, y_values, 'b-', label='data')
    ax.plot(x_values, y_values + 0.5 * synth_data.noise(synth_data.num_samples), 'r.', label='noise + data')
    ax.set_xlabel('x')
    ax.set_ylabel(r'y')
    grid(ax)
    ax.legend()


def synthetic_data_wrt_num_basis(synth_data: SyntheticData, ys_predicted, x_values, number_basis, all_weights, all_xs,
                                 f=None, ax=None,
                                 f2=None, ax2=None, point_line="-", ylabel="y"):
    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    for dex, y_predicted in enumerate(ys_predicted):
        ax[dex].plot(x_values, synth_data.sampled_function(x_values), 'k-', label='data', linewidth=3)
        ax[dex].plot(x_values, synth_data.sampled_function(x_values) + 0.5 * synth_data.noise(synth_data.num_samples),
                     "r.",
                     label='noise + data', markersize=5)
        ax[dex].plot(x_values, y_predicted, point_line, label=f'Nb basis function = {number_basis[dex]}',
                     color="blue")
        ax[dex].set_xlabel("x")
        ax[dex].set_ylabel(ylabel)
        grid(ax[dex])
        ax[dex].legend()
        this_xs = np.array(all_xs[dex]).T
        for dex_2, func_xs in enumerate(this_xs):
            if dex_2 == 0:
                ax2[dex].plot(x_values, all_weights[dex][dex_2] * func_xs, point_line,
                              label=f'Nb basis fn \n= {number_basis[dex]}',
                              color=get_color(dex_2, len(this_xs)))
            else:
                ax2[dex].plot(x_values, all_weights[dex][dex_2] * func_xs, point_line, color=get_color(dex_2, len(this_xs)))
            ax2[dex].set_xlabel("x")
            ax2[dex].set_ylabel(ylabel)
            grid(ax2[dex])
            ax2[dex].legend()

    f.tight_layout()
    f2.tight_layout(rect=[0, 0, 0.9, 1])
    cbar_ax = f2.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('plasma')
    cbar = f2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label='Number of Basis Functions')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', 'Total\nnum\nbasis'])


def synthetic_data_wrt_percentages_color_basis(train_cost_list, test_cost_list, number_basis, number_of_data_points, f=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    train_cost_list = number_of_data_points * np.array(train_cost_list).T
    test_cost_list = number_of_data_points * np.array(test_cost_list).T
    percentage = 1
    ax[0].plot(number_basis, train_cost_list[percentage], marker='o', markersize=5, linestyle='-', linewidth=2,
               color="red", label=f'Training cost')
    ax[0].plot(number_basis, test_cost_list[percentage], marker='o', markersize=5, linestyle='-', linewidth=2,
               color="blue", label=f'Testing cost')
    ax[0].set_xlabel("Number of basis")
    ax[0].set_ylabel("Cost (SSE)")
    ax[1].set_xlabel("Number of basis")
    ax[1].set_ylabel("Cost (SSE)")
    ax[0].axvline(x=number_basis[np.argmin(test_cost_list[percentage])], color='k', linestyle='--', linewidth=2,
                  label=f"Min testing cost at nb basis= {number_basis[np.argmin(test_cost_list[percentage])]}")
    ax[1].axvline(x=number_basis[np.argmin(test_cost_list[percentage])], color='k', linestyle='--', linewidth=2,
                  label=f"Min testing cost at nb basis= {number_basis[np.argmin(test_cost_list[percentage])]}")

    ax[1].plot(number_basis, train_cost_list[percentage], marker='o', markersize=5, linestyle='-', linewidth=2,
               color="red", label=f'Training cost')
    ax[1].plot(number_basis, test_cost_list[percentage], marker='o', markersize=5, linestyle='-', linewidth=2,
               color="blue", label=f'Testing cost')
    ax[1].set_ylim(-5, 100)
    ax[0].legend()
    grid(ax[1])
    grid(ax[0])
    ax[1].legend()
    f.tight_layout()


def repetition_plotter(x_value, yhs, one_train_costs, one_test_costs, nums_basis, data_ax, cost_ax, iteration, tot_itr):
    for dex, yh in enumerate(yhs):
        if iteration == 0:
            data_ax[dex].plot(x_value, yh, linestyle='-', linewidth=2,
                              color="green", label=f"{tot_itr} fitted models, \n nb basis={nums_basis[dex]}")
            data_ax[dex].legend()
        else:
            data_ax[dex].plot(x_value, yh, linestyle='-', linewidth=2,
                              color="green")
    if iteration == 0:
        cost_ax[0].plot(nums_basis, one_train_costs, linestyle='-', linewidth=2,
                        color="green", label=f"{tot_itr} training costs")
        cost_ax[1].plot(nums_basis, one_test_costs, linestyle='-', linewidth=2,
                        color="green", label=f"{tot_itr} testing costs")
        cost_ax[0].legend()
        cost_ax[1].legend()
    else:
        cost_ax[0].plot(nums_basis, one_train_costs, linestyle='-', linewidth=2,
                        color="green")
        cost_ax[1].plot(nums_basis, one_test_costs, linestyle='-', linewidth=2,
                        color="green")


def mean_plotter(x_value, y_values, yhs_s, all_train_costs, all_test_costs, nums_basis, data_ax, cost_ax):
    mean_yhs = np.mean(np.array(yhs_s), axis=0)
    mean_train_costs = np.mean(np.array(all_train_costs), axis=0)
    mean_test_costs = np.mean(np.array(all_test_costs), axis=0)
    for dex, mean_yh in enumerate(mean_yhs):
        data_ax[dex].plot(x_value, y_values, linestyle='-', linewidth=2,
                          color="blue", label="Ground truth")
        data_ax[dex].plot(x_value, mean_yh, linestyle='-', linewidth=1,
                          color="red", label=f"Average of {len(yhs_s)} repetitions")

        grid(data_ax[dex])
        data_ax[dex].set_xlabel("x")
        data_ax[dex].set_ylabel("y")
        data_ax[dex].legend()

    cost_ax[0].plot(nums_basis, mean_train_costs, marker='o', markersize=5, linestyle='-', linewidth=2,
                    color="red", label=f"Average of the {len(yhs_s)} training costs")
    cost_ax[1].plot(nums_basis, mean_test_costs, marker='o', markersize=5, linestyle='-', linewidth=2,
                    color="red", label=f"Average of the {len(yhs_s)} testing costs")
    grid(cost_ax[0])
    cost_ax[0].set_xlabel("Number of basis")
    cost_ax[0].set_ylabel("Training cost (MSE)")
    cost_ax[0].legend()
    cost_ax[1].legend()
    grid(cost_ax[1])
    cost_ax[1].set_xlabel("Number of basis")
    cost_ax[1].set_ylabel("Testing cost (MSE)")


def cross_cost_lambda_plotter(reg_lambda_list, y_label_list, optimum_lam, *args, ax=None):
    for dex, list_to_plot in enumerate(args):
        mean_to_plot = np.mean(np.array(list_to_plot), axis=0)
        std_to_plot = np.std(np.array(list_to_plot), axis=0)
        if y_label_list[dex] == "weights":
            mean_to_plot = mean_to_plot.T
            for dex_2, weight in enumerate(mean_to_plot[::2]):
                ax[dex].plot(reg_lambda_list, weight, marker='o', markersize=2, linestyle='-', linewidth=1,
                             color=get_color(2 * dex_2, len(mean_to_plot)))
            for dex_2, weight in enumerate(mean_to_plot[1::2]):
                ax[dex + 1].plot(reg_lambda_list, weight, marker='o', markersize=2, linestyle='-', linewidth=1,
                                 color=get_color(2 * dex_2 + 1, len(mean_to_plot)))
            grid(ax[dex])
            ax[dex].set_xlabel("lambda")
            ax[dex].set_ylabel(y_label_list[dex])
            ax[dex].set_xscale('log')
            grid(ax[dex + 1])
            ax[dex + 1].axvline(x=optimum_lam, color='k', linestyle='--', linewidth=2)
            ax[dex].axvline(x=optimum_lam, color='k', linestyle='--', linewidth=2)
            ax[dex + 1].set_xlabel("lambda")
            ax[dex + 1].set_ylabel(y_label_list[dex])
            ax[dex + 1].set_xscale('log')
        else:
            for cross_correlation in list_to_plot:
                ax[dex].plot(reg_lambda_list, cross_correlation, marker='o', markersize=5, linestyle='-', linewidth=2,
                             color="green")

            ax[dex].plot(reg_lambda_list, mean_to_plot, marker='o', markersize=5, linestyle='-', linewidth=2,
                         color="red")
            ax[dex].axvline(x=optimum_lam, color='k', linestyle='--', linewidth=2)
            ax[dex].errorbar(reg_lambda_list, mean_to_plot, yerr=std_to_plot, fmt='none', ecolor='red', capsize=5)
            grid(ax[dex])
            ax[dex].set_xlabel("lambda")
            ax[dex].set_ylabel(y_label_list[dex])
            ax[dex].set_xscale('log')


def plot_contour(f, x1bound, x2bound, resolution, ax):
    x1range = np.linspace(x1bound[0], x1bound[1], resolution)
    x2range = np.linspace(x2bound[0], x2bound[1], resolution)
    xg, yg = np.meshgrid(x1range, x2range)
    zg = np.zeros_like(xg)
    for i, j in itertools.product(range(resolution), range(resolution)):
        zg[i][j] = f([xg[i, j], yg[i, j]])
    ax.contour(xg, yg, zg, 100)


if __name__ == "__main__":
    size = (22, 10)

    number_data_points = 100
    sample_interval = [0, 20]

    ######################################## Part 1 ########################################
    percentages_to_test = [round(0.1 * i, 2) for i in range(1, 10)]
    num_basis = [i * 10 for i in range(1, 11)]

    ######## Plot the synthetic data + noise ########
    # np.random.seed(72)
    np.random.seed(random.randint(1, 1000))
    noise = np.random.randn

    linear_model = LinearRegression("normal")
    synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points, noise)

    with PdfPages(f"code_dir/regularization_and_model_evaluation/analysis_files/complete_analysis.pdf") as line_pdf:

        f_1, ax_1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
        plot_only_fun_noise(synthetic_data, f_1, ax_1)
        f_1.tight_layout()
        f_1.savefig("code_dir/regularization_and_model_evaluation/analysis_files/data_noise")
        line_pdf.savefig(f_1)

        ######## Plot fit for different number of basis functions ########
        _, _, y_hs, weights, xs = synthetic_data.num_basis_fun_loop(linear_model, num_basis, 0.001)
        x, _ = synthetic_data.x_y_values()
        f_2, ax_2 = plt.subplots(nrows=2, ncols=5, figsize=(16, 7))
        f_22, ax_22 = plt.subplots(nrows=2, ncols=5, figsize=(16, 7))
        synthetic_data_wrt_num_basis(synthetic_data, y_hs, x, num_basis, weights, xs, f=f_2, ax=ax_2.flatten(),
                                     ax2=ax_22.flatten(), f2=f_22)
        f_2.savefig("code_dir/regularization_and_model_evaluation/analysis_files/fit_wrt_nb_basis")
        f_22.savefig("code_dir/regularization_and_model_evaluation/analysis_files/basis_func_plot")
        line_pdf.savefig(f_22)
        line_pdf.savefig(f_2)

        ######## Plot train/test cost vs percentages (train_costs[i] is the ith num of basis) ########
        train_costs, test_costs, predictions = synthetic_data.num_basis_fun_percent_loop(linear_model, num_basis,
                                                                                         percentages_to_test)
        f_3, ax_3 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        synthetic_data_wrt_percentages_color_basis(train_costs, test_costs, num_basis,
                                                   number_data_points, f=f_3,
                                                   ax=ax_3.flatten())
        f_3.tight_layout()
        f_3.savefig("code_dir/regularization_and_model_evaluation/analysis_files/cost_wrt_percentage_nb_basis")
        line_pdf.savefig(f_3)

        ############################## Part 2 ##############################

        num_repetitions = 10
        number_data_points = 150
        all_yh_s, all_train, all_test = [], [], []
        f_4, ax_4 = plt.subplots(nrows=2, ncols=5, figsize=(16, 7))
        f_5, ax_5 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points, noise)
        x, y = synthetic_data.x_y_values()
        for idx, rep in enumerate(range(num_repetitions)):
            np.random.seed(random.randint(1, 1000))
            noise = np.random.randn
            synthetic_data = SyntheticData(sample_function_1, basis_function, sample_interval, number_data_points,
                                           noise)
            train_costs, test_costs, y_hs, _, _ = synthetic_data.num_basis_fun_loop(linear_model, num_basis, 0.2)
            repetition_plotter(x, y_hs, train_costs, test_costs, num_basis, ax_4.flatten(), ax_5.flatten(), idx,
                               num_repetitions)
            all_yh_s.append(y_hs)
            all_train.append(train_costs)
            all_test.append(test_costs)

        mean_plotter(x, y, all_yh_s, all_train, all_test, num_basis, ax_4.flatten(), ax_5.flatten())
        f_4.tight_layout()
        f_5.tight_layout()
        f_4.savefig("code_dir/regularization_and_model_evaluation/analysis_files/rep_fit_wrt_nb_basis")
        line_pdf.savefig(f_4)
        f_5.savefig("code_dir/regularization_and_model_evaluation/analysis_files/rep_cost_wrt_nb_basis")
        line_pdf.savefig(f_5)

        ############################## Part 4 ##############################

        num_basis_functions = 20
        np.random.seed(random.randint(1, 1000))
        noise = np.random.randn
        sample_interval = [0, 10]
        number_data_points = 50
        synthetic_data = SyntheticData(sample_function_2, basis_function, sample_interval, number_data_points, noise)
        x, y = synthetic_data.x_y_values()
        y = y + noise(len(y))
        for reg_type in [1, 2]:
            linear_epoch = 100000
            linear_model_gd = LinearRegression("sgd", add_bias=True, learning_rate=0.01, epsilon=1e-6,
                                               max_iters=linear_epoch, batch_size=-1, stand=False, reg_type=reg_type, momentum=0.8)
            if reg_type == 2:
                lambdas = [0, 0.01, 0.1, 1, 10, 100]
                momontums = [0, 0, 0, 0, 0, 0]
                f_7, ax_7 = plt.subplots(nrows=2, ncols=math.ceil(len(lambdas) / 2), figsize=(16, 7))
            else:
                lambdas = [0, 0.01, 0.1, 1, 50, 50, 100, 100]
                momontums = [0, 0, 0, 0, 0, 0.9, 0, 0.9]
                f_7, ax_7 = plt.subplots(nrows=2, ncols=math.ceil(len(lambdas) / 2), figsize=(16, 7))
            contour_bound = 20
            ax_7 = ax_7.flatten()
            for idx, (reg_lam, momentum) in enumerate(zip(lambdas, momontums)):
                linear_model_gd = LinearRegression("sgd", add_bias=True, learning_rate=0.01, epsilon=1e-6,
                                                   max_iters=linear_epoch, batch_size=-1, stand=False,
                                                   reg_type=reg_type, momentum=momentum)
                cost = lambda w: .5 * np.mean((w[0] + w[1] * x - y) ** 2)
                l2_penalty = lambda w: np.dot(w, w) / 2
                l1_penalty = lambda w: np.sum(np.abs(w))
                cost_plus_l_i = lambda w: cost(w) + reg_lam * (
                            (reg_type - 1) * l2_penalty(w) + (2 - reg_type) * l1_penalty(w))

                plot_contour(cost_plus_l_i, [-contour_bound, contour_bound], [-contour_bound, contour_bound], 100,
                             ax_7[idx])
                linear_model_gd.set_reg_lambda(reg_lam)
                x = linear_model_gd.do_add_bias(x)
                linear_model_gd.fit(x, y)
                history = np.array(linear_model_gd.weight_history)
                ax_7[idx].plot(history[:, 1], history[:, 0], marker='o', markersize=2, linestyle='-', linewidth=1,
                               color="red")
                grid(ax_7[idx])
                ax_7[idx].set_xlabel(r"$w_0$")
                ax_7[idx].set_ylabel(r"$w_1$")
                if momentum != 0:
                    ax_7[idx].set_title(f"lambda = {reg_lam} with momentum = {momentum}")
                else:
                    ax_7[idx].set_title(f"lambda = {reg_lam}")
                x, y = synthetic_data.x_y_values()
                y = y + noise(len(y))

            f_7.suptitle(f"Regularization L{reg_type}")
            f_7.tight_layout()
            f_7.savefig(f"code_dir/regularization_and_model_evaluation/analysis_files/contour_l{reg_type}")
            line_pdf.savefig(f_7)

