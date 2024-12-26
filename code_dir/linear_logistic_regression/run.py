import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from code_dir.linear_logistic_regression.data_class import Temperature, Diabetes
from code_dir.linear_logistic_regression.linear_regression import LinearRegression
from code_dir.linear_logistic_regression.logistic_regression import LogisticRegression


def convergence_time(cost_history, tolerance):
    nb_it = len(cost_history)
    min_cost = np.min(cost_history)
    convergence_point = min_cost + tolerance * min_cost
    nb_it_to_converge = -1

    for i in range(nb_it):
        if cost_history[i] <= convergence_point:
            convergence_point = cost_history[i]
            nb_it_to_converge = i
            break

    return convergence_point, nb_it_to_converge


def grid(axi):
    # Show the major grid and style it slightly.
    axi.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    axi.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    axi.minorticks_on()
    axi.set_axisbelow(True)


def get_color(index, num_colors=10, color_skeme='plasma'):
    cmap = plt.get_cmap(color_skeme)
    return cmap(index/num_colors)

def cost_wrt_training_percentage(percentages, feature_numbers, train_cost, test_cost, corr_weights, select=-1, point_line="-", f=None, ax=None, ylabel="Cost"):

    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    if select > 1:
        feature_numbers = feature_numbers[::select]
        train_cost, test_cost, corr_weights = train_cost[::select], test_cost[::select], corr_weights[::select]

    for dex, (train_cost_per_feature, test_cost_per_feature, weights) in enumerate(zip(train_cost, test_cost, corr_weights)):
        ax[0].plot(np.array(percentages), train_cost_per_feature, point_line, label=f'Nb feature= {feature_numbers[dex]}', color=get_color(dex, len(feature_numbers)))
        ax[0].set_title("Train figure of merit plot")
        ax[0].set_xlabel("Testing percentage")
        ax[0].set_ylabel(ylabel)

        ax[1].plot(percentages, test_cost_per_feature, point_line, label=f'Nb feature= {feature_numbers[dex]}', color=get_color(dex, len(feature_numbers)))
        ax[1].set_title("Testing figure of merit plot")
        ax[1].set_xlabel("Testing percentage")
        ax[1].set_ylabel(ylabel)

        weights_to_plot = [weight[1] for weight in weights]

        ax[2].plot(percentages, weights_to_plot, point_line, label=f'Nb feature= {feature_numbers[dex]}',
                   color=get_color(dex, len(feature_numbers)))
        ax[2].set_title("Weight plot")
        ax[2].set_xlabel("Testing percentage")
        ax[2].set_ylabel("Weight of the most/least correlated feature")

        grid(ax[1])
        grid(ax[2])
        grid(ax[0])



def cost_wrt_num_features(percentages, feature_numbers, train_cost, test_cost, select=-1, point_line="-", f=None, ax=None, ylabel="cost"):

    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    train_cost, test_cost = np.array(train_cost).T, np.array(test_cost).T

    if select > 1:
        feature_numbers = feature_numbers[::select]
        train_cost, test_cost = train_cost[::select], test_cost[::select]

    for dex, (train_cost_per_feature, test_cost_per_feature) in enumerate(zip(train_cost, test_cost)):

        ax[0].plot(feature_numbers, train_cost_per_feature, point_line, label=f'Nb feature= {percentages[dex]}', color=get_color(dex, len(percentages)))
        ax[0].set_title("Train figure of merit plot")
        ax[0].set_xlabel("Number of features")
        ax[0].set_ylabel(ylabel)

        ax[1].plot(feature_numbers, test_cost_per_feature, point_line, label=f'Nb feature= {percentages[dex]}', color=get_color(dex, len(percentages)))
        ax[1].set_title("Testing figure of merit plot")
        ax[1].set_xlabel("Number of features")
        ax[1].set_ylabel(ylabel)

        grid(ax[1])
        grid(ax[0])



def rate_cost_wrt_iteration(batch_accuracies, batch_times, batch_cost_history, batch_size_list, learning_rates, f=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(nrows=2, ncols=len(batch_size_list), figsize=(16, 4))

    convergence = []
    for dex, batch in enumerate(batch_size_list):
        for index, rate in enumerate(learning_rates):

            converge_it = convergence_time(batch_cost_history[dex][index], 0.01)[1]
            convergence.append(converge_it)

            total_iterations_2 = [i for i in range(len(batch_cost_history[dex][index]))]
            ax[0][dex].plot(total_iterations_2, batch_cost_history[dex][index], '-', label=f'Rate {rate}',
                            color=get_color(index, len(learning_rates)))
            ax[0][dex].set_title(f"Batch size {batch}")
            ax[0][dex].set_xlabel("Iteration")
            ax[0][dex].set_ylabel("Figure of merit")
            ax[0][dex].legend()
            grid(ax[0][dex])
        ax[1][dex].plot(learning_rates, convergence, "ro")
        # ax[1][dex].plot(learning_rates, batch_times[dex], "ro")
        ax[1][dex].set_title(f"Batch size {batch}")
        ax[1][dex].set_xlabel("Learning rate")
        ax[1][dex].set_ylabel("Convergence time")
        grid(ax[1][dex])
        ax[1][dex].legend()
        ax[1][dex].set_xscale('log')

        ax[2][dex].plot(learning_rates, batch_accuracies[dex], "ro")
        ax[2][dex].set_title(f"Batch size {batch}")
        ax[2][dex].set_xlabel("Learning rate")
        ax[2][dex].set_ylabel("Figure of merit")
        grid(ax[2][dex])
        ax[2][dex].legend()
        ax[2][dex].set_xscale('log')
        convergence = []

        f.tight_layout()


def batch_cost_wrt_iteration(batch_accuracies, batch_times, batch_cost_history, batch_size_list, learning_rates, f=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(nrows=2, ncols=len(learning_rates), figsize=(16, 4))

    batch_times = np.array(batch_times).T
    batch_accuracies = np.array(batch_accuracies).T
    convergence = []
    for index, rate in enumerate(learning_rates):
        for dex, batch in enumerate(batch_size_list):

            converge_it = convergence_time(batch_cost_history[dex][index], 0.01)[1]
            convergence.append(converge_it)

            total_iterations_2 = [i for i in range(len(batch_cost_history[dex][index]))]
            ax[0][index].plot(total_iterations_2, batch_cost_history[dex][index], '-', label=f'Batch {batch}',
                              color=get_color(dex, len(batch_size_list)))
            ax[0][index].set_title(f"Learning rates {rate}")
            ax[0][index].set_xlabel("Iteration")
            ax[0][index].set_ylabel("Figure of merit")
            ax[0][index].legend()
            grid(ax[0][index])
        ax[1][index].plot(batch_size_list, convergence, "ro")
        # ax[1][index].plot(batch_size_list, batch_times[index], "ro")
        ax[1][index].set_title(f"Learning rates {rate}")
        ax[1][index].set_xlabel("Batch size")
        ax[1][index].set_ylabel("Convergence time")
        grid(ax[1][index])
        ax[1][index].legend()
        ax[1][index].set_xscale('log')

        ax[2][index].plot(batch_size_list, batch_accuracies[index], "ro")
        ax[2][index].set_title(f"Learning rates {rate}")
        ax[2][index].set_xlabel("Batch size")
        ax[2][index].set_ylabel("Figure of merit")
        grid(ax[2][index])
        ax[2][index].legend()
        ax[2][index].set_xscale('log')
        convergence = []

        f.tight_layout()



if __name__ == "__main__":
    percentages_to_test = [round(0.1 * i, 2) for i in range(1, 10)]
    rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    batch_list = [8, 16, 32, 64, 128]
    size = (22, 10)
    highest_corr = False

    temperature = Temperature(925, 'aveOralM')
    temperature.nan_handler()
    diabetes = Diabetes(891, 'Diabetes_binary')

    logistic_epoch = 2
    logistic_model = LogisticRegression(learning_rate=0.01, epsilon=1e-6, max_iters=logistic_epoch, batch_size=-1)

    linear_epoch = 5000
    linear_model_gd = LinearRegression("sgd", add_bias=True, learning_rate=0.01, epsilon=1e-6, max_iters=linear_epoch,
                                       batch_size=8, stand=True)
    linear_model = LinearRegression("normal")



    with PdfPages(f"complete_analysis.pdf") as line_pdf:

        print("Starting linear sgd cost vs iteration with color rate\n")
        x = temperature.standarize()
        x = x.to_numpy()
        x = linear_model_gd.do_add_bias(x)
        y = temperature.all_data_targets[temperature.target_to_classify].to_numpy()
        batch_accuracies_1, batch_times_1, batch_cost_history_1 = linear_model_gd.batch_cost_wrt_iteration(x, y,
                                                                                                           linear_model_gd.cost_function,
                                                                                                           rate_list,
                                                                                                           batch_list)

        del x
        del y

        batch_accuracies_1.reverse()
        batch_times_1.reverse()
        batch_cost_history_1.reverse()
        fig, axs = plt.subplots(nrows=3, ncols=len(batch_list), figsize=size)

        rate_cost_wrt_iteration(batch_accuracies_1, batch_times_1, batch_cost_history_1, batch_list, rate_list, f=fig,
                                ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Row 1 shows the cost with respect to the number of iterations for different learning rates, \n"
                     f"while row 2 shows the time required to complete all the iterations. Each column represents a \n"
                     f"different batch size. This was done for the linear mini batch gradient descent with {linear_epoch} epochs.")
        fig.tight_layout()
        fig.savefig("mini_batch_linear_rate")
        line_pdf.savefig(fig)

        fig, axs = plt.subplots(nrows=3, ncols=len(rate_list), figsize=size)
        batch_cost_wrt_iteration(batch_accuracies_1, batch_times_1, batch_cost_history_1, batch_list, rate_list, f=fig,
                                 ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Row 1 shows the cost with respect to the number of iterations for different batch sizes, \n"
                     f"while row 2 shows the time required to complete all the iterations. Each column represents a \n"
                     f"different learning rate. This was done for the linear mini batch gradient descent with {linear_epoch} epochs.")
        fig.tight_layout()
        fig.savefig("mini_batch_linear_batch")
        line_pdf.savefig(fig)

        print("Starting analytic linear cost vs feature/percentage\n")

        feature_numbers_1, train_cost_1, test_cost_1, corr_weights_1 = \
            linear_model.feature_number_loop(temperature, percentages_to_test, high=highest_corr)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size)
        cost_wrt_training_percentage(percentages_to_test, feature_numbers_1, train_cost_1, test_cost_1, corr_weights_1,
                                     select=-1, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Cost of the training/testing data with respect to the percentage of testing data. Different \n"
                     f"colors represent different numbers of features included in the training (starting from the\n"
                     f"most correlated one alone and adding other features one by one). This was done for analytic \n"
                     f"linear regression.")
        fig.tight_layout()
        fig.savefig("linear_analytic_percentage")
        line_pdf.savefig(fig)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=size)
        cost_wrt_num_features(percentages_to_test, feature_numbers_1, train_cost_1, test_cost_1, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(
            f"Cost of the training/testing data with respect to the number of features included in the training.\n"
            f" Different colors represent different percentages of testing data. This was done for analytic \n"
            f"linear regression.")
        fig.tight_layout()
        fig.savefig("linear_analytic_feature")
        line_pdf.savefig(fig)

        print("Starting sgd linear cost vs feature/percentage\n")

        linear_epoch = 75000
        linear_batch = 128
        linear_model_gd = LinearRegression("sgd", add_bias=True, learning_rate=0.01, epsilon=1e-14,
                                           max_iters=linear_epoch,
                                           batch_size=linear_batch, stand=True)

        feature_numbers_1, train_cost_1, test_cost_1, corr_weights_1 = \
            linear_model_gd.feature_number_loop(temperature, percentages_to_test, high=highest_corr)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size)
        cost_wrt_training_percentage(percentages_to_test, feature_numbers_1, train_cost_1, test_cost_1, corr_weights_1,
                                     select=-1, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Cost of the training/testing data with respect to the percentage of testing data. Different \n"
                     f"colors represent different numbers of features included in the training (starting from the \n"
                     f"most correlated one alone and adding other features one by one). This was done for the mini batch \n"
                     f"stochastic gradient descent for linear regression with a learning rate of 0.01, mini batch size of \n"
                     f"{linear_batch}, and number of epochs {linear_epoch}.")
        fig.tight_layout()
        fig.savefig("linear_sgd_percent")
        line_pdf.savefig(fig)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=size)
        cost_wrt_num_features(percentages_to_test, feature_numbers_1, train_cost_1, test_cost_1, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(
            f"Cost of the training/testing data with respect to the number of features included in the training.\n"
            f" Different colors represent different percentages of testing data. This was done for the mini batch \n"
            f"stochastic gradient descent for linear regression with a learning rate of 0.01, mini batch size of \n"
            f"{linear_batch}, and number of epochs {linear_epoch}.")
        fig.tight_layout()
        fig.savefig("linear_sgd_feature")
        line_pdf.savefig(fig)



        print("Starting logistic sgd cost vs iteration with color rate\n")
        x = diabetes.standarize()
        x = x.to_numpy()
        x = logistic_model.do_add_bias(x)
        y = diabetes.all_data_targets[diabetes.target_to_classify].to_numpy()
        batch_accuracies_1, batch_times_1, batch_cost_history_1 = logistic_model.batch_cost_wrt_iteration(x, y,
                                                                                                          logistic_model.cost_function,
                                                                                                          rate_list, batch_list)
        del x
        del y

        batch_accuracies_1.reverse()
        batch_times_1.reverse()
        batch_cost_history_1.reverse()

        fig, axs = plt.subplots(nrows=3, ncols=len(batch_list), figsize=size)
        rate_cost_wrt_iteration(batch_accuracies_1, batch_times_1, batch_cost_history_1, batch_list, rate_list, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Row 1 shows the cost with respect to the number of iterations for different learning rates, \n"
                     f"while row 2 shows the time required to complete all the iterations. Each column represents a \n"
                     f"different batch size. This was done for the logistic mini batch gradient descent with {logistic_epoch} epochs.")
        fig.tight_layout()
        fig.savefig("logistic_rate")
        line_pdf.savefig(fig)

        fig, axs = plt.subplots(nrows=3, ncols=len(rate_list), figsize=size)
        batch_cost_wrt_iteration(batch_accuracies_1, batch_times_1, batch_cost_history_1, batch_list, rate_list, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Row 1 shows the cost with respect to the number of iterations for different batch sizes, \n"
                     f"while row 2 shows the time required to complete all the iterations. Each column represents a \n"
                     f"different learning rate. This was done for the logistic mini batch gradient descent with {logistic_epoch} epochs.")
        fig.tight_layout()
        fig.savefig("logistic_batch")
        line_pdf.savefig(fig)



        logistic_epoch = 5000
        logistic_model = LogisticRegression(learning_rate=0.01, epsilon=1e-14, max_iters=logistic_epoch, batch_size=-1)


        print("Starting full batch logistic cost vs feature/percentage\n")
        feature_numbers_1, train_cost_1, test_cost_1, corr_weights_1 = \
            logistic_model.feature_number_loop(diabetes, percentages_to_test, high=highest_corr)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=size)
        cost_wrt_training_percentage(percentages_to_test, feature_numbers_1, train_cost_1, test_cost_1, corr_weights_1,
                                     select=-1, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(f"Cost of the training/testing data with respect to the percentage of testing data. Different \n"
                     f"colors represent different numbers of features included in the training (starting from the \n"
                     f"most correlated one alone and adding other features one by one). This was done for the full batch \n"
                     f"logistic regression with a learning rate of 0.01 with {logistic_epoch} iterations.")
        fig.tight_layout()
        fig.savefig("logistic_percent")
        line_pdf.savefig(fig)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=size)
        cost_wrt_num_features(percentages_to_test, feature_numbers_1, train_cost_1, test_cost_1, f=fig, ax=axs)
        for axis in axs.flatten():
            grid(axis)
        fig.suptitle(
            f"Cost of the training/testing data with respect to the number of features included in the training.\n"
            f" Different colors represent different percentages of testing data. This was done for analytic \n"
            f"logistic regression with a learning rate of 0.01 with {logistic_epoch} iterations.")
        fig.tight_layout()
        fig.savefig("logistic_feature")
        line_pdf.savefig(fig)


