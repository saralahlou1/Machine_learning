import argparse
import gc


import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import code_dir.MLP_CNN_classification.config as config
from code_dir.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Choose between CPU (NumPy) and GPU (CuPy) processing.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU computation with CuPy.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gpu:
        import cupy as cp
        config.xp = cp
    else:
        config.xp = np

    from code_dir.MLP_CNN_classification.activation_functions import ReLU, Softmax, Tanh, LeakyReLU
    from code_dir.MLP_CNN_classification.multilayer_perceptron import MLP
    from code_dir.MLP_CNN_classification.data_class_nn import DataClassNN
    from code_dir.MLP_CNN_classification.plotting_functions import plot_hyper_parameter_choice, \
        par_hyper_parameter_choice, cnn_hyper_parameter_choice, series_cnn_hyper_parameter_choice
    from code_dir.MLP_CNN_classification.activation_functions import ReLU, Softmax, Tanh, LeakyReLU
    from code_dir.MLP_CNN_classification.plotting_functions import plot_hyper_parameter_choice, \
        par_hyper_parameter_choice, data_loader_getter
    from code_dir.MLP_CNN_classification.convolution_nn import SimpleCNN, train_model, PreTrained

    ###################################################### Dataset 1 ######################################################

    all_data = DataClassNN(size=28)

    y_train = all_data.train_labels
    x_train = all_data.train_data
    y_test = all_data.val_labels
    x_test = all_data.val_data

    input_neuron_num = x_train.shape[-1]
    output_neuron_num = y_train.shape[-1]

    choice_epoch = 100
    choice_batch = 500
    choice_rate = 0.035
    momuntum = 0.99
    choice_lambda = 0.005

    used_epoch = 50
    used_batch = 100
    used_rate = 0.005
    used_momentum = 0.75

    learning_rates = [0.05, 0.025, 0.01, 0.005, 0.001]
    batch_sizes = [256, 512, 1024, 2048]
    lambdas = [0, 0.001, 0.01, 0.05, 0.1]

    ############################## Hyper-parameter choice ##############################

    choice_num_layer = [input_neuron_num, 256, 256, output_neuron_num]
    choice_act_functions = [ReLU(), ReLU(), Softmax()]

    mlp = MLP(choice_num_layer, choice_act_functions, batch_size=choice_batch, epochs=choice_epoch)
    print("Learning rates ...")
    rate_accuracies, rate_accuracies_per_epoch, rate_times = \
        par_hyper_parameter_choice(mlp, "learning_rate", learning_rates, x_train, y_train, x_test, y_test)
    f_1, ax_1 = plt.subplots(nrows=5, ncols=3, figsize=(16, 12))
    ax_1 = ax_1.flatten()
    plot_hyper_parameter_choice("learning rate", learning_rates, rate_accuracies, rate_accuracies_per_epoch,
                                rate_times, f=f_1, ax=ax_1, row=0)
    ax_1[1].set_title(f"Learning rate dependence")



    mlp = MLP(choice_num_layer, choice_act_functions, learning_rate=choice_rate, epochs=choice_epoch)
    print("Batch sizes ...")
    batch_accuracies, batch_accuracies_per_epoch, batch_times = \
        par_hyper_parameter_choice(mlp, "batch_size", batch_sizes, x_train, y_train, x_test, y_test)
    plot_hyper_parameter_choice("batch size", batch_sizes, batch_accuracies, batch_accuracies_per_epoch,
                                batch_times, f=f_1, ax=ax_1, row=1)
    ax_1[4].set_title(f"Batch size dependence")




    for reg_dex, reg_type in enumerate([1, 2]):
        print(f"L{reg_type} regularization ...")
        mlp = MLP(choice_num_layer, choice_act_functions, batch_size=choice_batch, learning_rate=choice_rate,
                  epochs=choice_epoch, reg_type=reg_type)
        reg_accuracies, reg_accuracies_per_epoch, reg_times = \
            par_hyper_parameter_choice(mlp, "reg_lambda", lambdas, x_train, y_train, x_test, y_test)
        plot_hyper_parameter_choice("Lambda", lambdas, reg_accuracies, reg_accuracies_per_epoch,
                                    reg_times, f=f_1, ax=ax_1, row=reg_dex + 2)
        ax_1[3*reg_dex + 7].set_title(f"L{reg_type} regularization")



    print(f"Momuntum effect ...")
    mlp = MLP(choice_num_layer, choice_act_functions, batch_size=choice_batch, learning_rate=choice_rate,
              epochs=choice_epoch, reg_type=1, momentum=momuntum)
    momun_reg_accuracies, momun_reg_accuracies_per_epoch, momun_reg_times = \
        par_hyper_parameter_choice(mlp, "reg_lambda", lambdas, x_train, y_train, x_test, y_test)
    plot_hyper_parameter_choice("Lambda", lambdas, momun_reg_accuracies,
                                momun_reg_accuracies_per_epoch,
                                momun_reg_times, f=f_1, ax=ax_1, row=4)
    ax_1[13].set_title(f"L{1} regularization, momentum {momuntum}")

    f_1.tight_layout()

    with PdfPages(f"code_dir/hw_3/analysis_files/parameter_choice.pdf") as line_pdf:
        line_pdf.savefig(f_1)

    del batch_accuracies, batch_accuracies_per_epoch, batch_times, momun_reg_accuracies, \
        momun_reg_accuracies_per_epoch, momun_reg_times, reg_accuracies, reg_accuracies_per_epoch, reg_times, \
        rate_accuracies, rate_accuracies_per_epoch, rate_times

    ############################## Part 1 (depth) ##############################

    all_data = DataClassNN(size=28)

    y_train = all_data.train_labels
    x_train = all_data.train_data
    y_test = all_data.test_labels
    x_test = all_data.test_data

    input_neuron_num = x_train.shape[-1]
    output_neuron_num = y_train.shape[-1]

    layer_size_lists = [[input_neuron_num, output_neuron_num], [input_neuron_num, 256, output_neuron_num],
                        [input_neuron_num, 256, 256, output_neuron_num],
                        [input_neuron_num, 256, 256, 256, output_neuron_num],
                        [input_neuron_num, 256, 256, 256, 256, output_neuron_num]]

    activation_function_lists = [[Softmax()], [ReLU(), Softmax()], [ReLU(), ReLU(), Softmax()],
                                 [ReLU(), ReLU(), ReLU(), Softmax()],
                                 [ReLU(), ReLU(), ReLU(), ReLU(), Softmax()]]

    parameters_list = [(ai, bi) for ai, bi in zip(layer_size_lists, activation_function_lists)]

    print(f"ReLU depth ...")

    mlp = MLP(choice_num_layer, choice_act_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate)
    depth_accuracies, depth_accuracies_per_epoch, depth_times = \
        par_hyper_parameter_choice(mlp, ["layer_sizes", "activation_funcs"],
                                   parameters_list, x_train, y_train, x_test, y_test)

    width_to_append = (depth_accuracies[2], depth_accuracies_per_epoch[2], depth_times[2])



    f_2, ax_2 = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    ax_2 = ax_2.flatten()
    plot_hyper_parameter_choice("Network Depth", np.arange(0, len(depth_accuracies)),
                                depth_accuracies, depth_accuracies_per_epoch, depth_times,
                                f=f_2, ax=ax_2, row=0, scale="linear", validation=False)
    ax_2[1].set_title(f"Depth dependence")



    del depth_accuracies, depth_accuracies_per_epoch, depth_times

    ############################## Part 1.1 (width) ##############################
    #
    def put_in(tests, trains, times):
        tests, trains, times = list(tests), list(trains), list(times)
        tests.insert(0, width_to_append[0]), trains.insert(0, width_to_append[1]), times.insert(0, width_to_append[2])
        return np.array(tests), np.array(trains), np.array(times)



    print(f"ReLU width...")

    layer_sizes = [[input_neuron_num, 512, 512, output_neuron_num],
                   [input_neuron_num, 1024, 1024, output_neuron_num],
                   [input_neuron_num, 2048, 2048, output_neuron_num]]

    activation_functions = [ReLU(), ReLU(), Softmax()]

    mlp = MLP(choice_num_layer, activation_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate)

    width_accuracies, width_accuracies_per_epoch, width_times = \
        par_hyper_parameter_choice(mlp, "layer_sizes", layer_sizes, x_train, y_train, x_test, y_test)

    width_accuracies, width_accuracies_per_epoch, width_times = \
        put_in(width_accuracies, width_accuracies_per_epoch, width_times)


    plot_hyper_parameter_choice("Network width", [256, 512, 1024, 2048],
                                width_accuracies, width_accuracies_per_epoch, width_times,
                                f=f_2, ax=ax_2, row=1, scale="log", validation=False)
    ax_2[4].set_title(f"Width dependence")

    del width_accuracies, width_accuracies_per_epoch, width_times


    ############################## Part 2 ##############################

    print(f"Different activations ...")

    diff_activation_functions = [[LeakyReLU(), LeakyReLU(), Softmax()], [Tanh(), Tanh(), Softmax()]]


    mlp = MLP(choice_num_layer, choice_act_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate)

    activation_accuracies, activation_accuracies_per_epoch, activation_times = \
        par_hyper_parameter_choice(mlp, "activation_funcs", diff_activation_functions, x_train, y_train, x_test, y_test)

    activation_accuracies, activation_accuracies_per_epoch, activation_times = \
        put_in(activation_accuracies, activation_accuracies_per_epoch, activation_times)


    plot_hyper_parameter_choice("Activation", ["ReLU", "Leaky ReLU", "tanh"],
                                activation_accuracies, activation_accuracies_per_epoch, activation_times,
                                f=f_2, ax=ax_2, row=2, scale="log", validation=False)

    ax_2[7].set_title(f"Activation type dependence")

    f_2.tight_layout()
    with PdfPages(f"code_dir/hw_3/analysis_files/parts_1_2.pdf") as line_pdf:
        line_pdf.savefig(f_2)

    ############################## Part 3 ##############################

    layer_sizes = [input_neuron_num, 256, 256, output_neuron_num]

    activation_functions = [ReLU(), ReLU(), Softmax()]


    print(f"Regularization ...")
    f_3, ax_3 = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    ax_3 = ax_3.flatten()
    mlp = MLP(layer_sizes, activation_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate, reg_lambda=choice_lambda)
    lambda_accuracies, lambda_accuracies_per_epoch, lambda_times = \
        par_hyper_parameter_choice(mlp, "reg_type", [1, 2], x_train, y_train, x_test, y_test)

    lambda_accuracies, lambda_accuracies_per_epoch, lambda_times = \
        put_in(lambda_accuracies, lambda_accuracies_per_epoch, lambda_times)
    plot_hyper_parameter_choice("Regularization type", ["No reg", "L1 reg", "L2 reg"],
                                lambda_accuracies, lambda_accuracies_per_epoch, lambda_times,
                                f=f_3, ax=ax_3, row=0, scale="linear", validation=False)

    ax_3[1].set_title(f"Regularization type dependence")


    optimum_mlp_accuracies, optimum_mlp_accuracies_per_epoch, optimum_mlp_times = \
        lambda_accuracies[2], lambda_accuracies_per_epoch[2], lambda_times[2]

    ############################## Part 4 ##############################


    all_data = DataClassNN(size=28, normalize=False)

    y_train = all_data.train_labels
    x_train = all_data.train_data
    y_test = all_data.test_labels
    x_test = all_data.test_data

    print(f"Normalization ...")

    mlp = MLP(layer_sizes, activation_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate)

    norm_accuracies, norm_accuracies_per_epoch, norm_times = \
        par_hyper_parameter_choice(mlp, "reg_type", [2], x_train, y_train, x_test, y_test)

    ############################## Part 5 ##############################

    all_data = DataClassNN(size=128)

    y_train = all_data.train_labels
    x_train = all_data.train_data
    y_test = all_data.test_labels
    x_test = all_data.test_data

    input_neuron_num = x_train.shape[-1]
    output_neuron_num = y_train.shape[-1]

    layer_sizes = [input_neuron_num, 256, 256, output_neuron_num]

    print(f"128 pix ...")

    mlp = MLP(layer_sizes, activation_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate)

    large_pix_accuracies, large_pix_accuracies_per_epoch, large_pix_times = \
        par_hyper_parameter_choice(mlp, "reg_type", [2], x_train, y_train, x_test, y_test)

    all_data = DataClassNN(size=128, normalize=False)

    y_train = all_data.train_labels
    x_train = all_data.train_data
    y_test = all_data.test_labels
    x_test = all_data.test_data

    input_neuron_num = x_train.shape[-1]
    output_neuron_num = y_train.shape[-1]

    layer_sizes = [input_neuron_num, 256, 256, output_neuron_num]

    print(f"128 pix unnormalized ...")

    mlp = MLP(layer_sizes, activation_functions, batch_size=choice_batch, epochs=choice_epoch,
              learning_rate=choice_rate)

    un_large_pix_accuracies, un_large_pix_accuracies_per_epoch, un_large_pix_times = \
        par_hyper_parameter_choice(mlp, "reg_type", [2], x_train, y_train, x_test, y_test)


    plot_accuracies, plot_accuracies_per_epoch, plot_times = \
        [norm_accuracies[0], large_pix_accuracies[0], un_large_pix_accuracies[0]], \
        [norm_accuracies_per_epoch[0], large_pix_accuracies_per_epoch[0], un_large_pix_accuracies_per_epoch[0]], \
                  [norm_times[0], large_pix_times[0], un_large_pix_times[0]]

    plot_accuracies, plot_accuracies_per_epoch, plot_times = \
        put_in(plot_accuracies, plot_accuracies_per_epoch, plot_times)

    plot_hyper_parameter_choice("", ["Nomalized \n28 pix", "Un-nomalized \n28 pix", "Nomalized \n128 pix", "Un-Nomalized \n128 pix"],
                                plot_accuracies, plot_accuracies_per_epoch, plot_times,
                                f=f_3, ax=ax_3, row=1, scale="linear", validation=False)

    ax_3[4].set_title(f"Normalization and data size dependence")


    f_3.tight_layout()
    with PdfPages(f"code_dir/hw_3/analysis_files/parts_3_4_5.pdf") as line_pdf:
        line_pdf.savefig(f_3)


    del y_train, x_train, x_test, y_test, all_data, plot_accuracies, plot_accuracies_per_epoch, plot_times,\
        norm_accuracies, norm_accuracies_per_epoch, norm_times
    gc.collect()

    ############################## CNN ##############################

    print("CNN")

    timing = Timer(False)
    criterion = nn.CrossEntropyLoss()

    print(f"Epochs: {used_epoch}, batches: {used_batch}, rate: {used_rate}, momentum: {used_momentum}")

    small_training, small_testing = data_loader_getter(28, batch_size=used_batch)
    small_model = SimpleCNN(28, epochs=used_epoch)
    optimizer = optim.SGD(small_model.parameters(), lr=used_rate, momentum=used_momentum)
    timing.start()
    train_model(small_model, small_model, criterion, optimizer, small_training, small_testing)
    timing.stop()

    small_cnn_accuracies, small_cnn_accuracies_per_epoch, small_cnn_times = \
        small_model.test_accuracy_per_epoch, small_model.accuracy_per_epoch, timing.elapsed_time

    del small_training, small_testing, small_model
    gc.collect()

    large_training, large_testing = data_loader_getter(128, batch_size=used_batch)
    large_model = SimpleCNN(128, epochs=used_epoch)
    optimizer = optim.SGD(large_model.parameters(), lr=used_rate, momentum=used_momentum)
    timing.start()
    train_model(large_model, large_model, criterion, optimizer, large_training, large_testing)
    timing.stop()

    large_cnn_accuracies, large_cnn_accuracies_per_epoch, large_cnn_times = \
        large_model.test_accuracy_per_epoch, large_model.accuracy_per_epoch, timing.elapsed_time

    del large_training, large_testing, large_model
    gc.collect()

    pre_trained_training, pre_trained_testing = data_loader_getter(28, batch_size=used_batch, three_channel=True)
    fc_neurons = [512, 512, 512, 11]
    activations = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
    pre_trained = PreTrained(fc_neurons, activations, epochs=used_epoch)
    resnet_optimizer = optim.SGD(pre_trained.resnet.fc.parameters(), lr=used_rate, momentum=used_momentum)
    timing.start()
    train_model(pre_trained, pre_trained.resnet, criterion, resnet_optimizer, pre_trained_training, pre_trained_testing)
    timing.stop()

    pre_small_cnn_accuracies, pre_small_cnn_accuracies_per_epoch, pre_small_cnn_times = \
        pre_trained.test_accuracy_per_epoch, pre_trained.accuracy_per_epoch, timing.elapsed_time

    del pre_trained_training, pre_trained_testing, pre_trained
    gc.collect()

    large_pre_trained_training, large_pre_trained_testing = data_loader_getter(128, batch_size=used_batch,
                                                                               three_channel=True)
    fc_neurons = [512, 512, 512, 11]
    activations = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
    pre_trained = PreTrained(fc_neurons, activations, epochs=used_epoch)
    resnet_optimizer = optim.Adam(pre_trained.resnet.fc.parameters(), lr=used_rate)
    timing.start()
    train_model(pre_trained, pre_trained.resnet, criterion, resnet_optimizer,
                large_pre_trained_training, large_pre_trained_testing)
    timing.stop()

    pre_large_cnn_accuracies, pre_large_cnn_accuracies_per_epoch, pre_large_cnn_times = \
        pre_trained.test_accuracy_per_epoch, pre_trained.accuracy_per_epoch, timing.elapsed_time

    del large_pre_trained_training, large_pre_trained_testing, pre_trained
    gc.collect()

    all_times = np.array(
        [optimum_mlp_times, small_cnn_times, large_cnn_times, pre_small_cnn_times, pre_large_cnn_times])
    all_test_accuracies = np.array([optimum_mlp_accuracies[:used_epoch], small_cnn_accuracies, large_cnn_accuracies,
                                    pre_small_cnn_accuracies, pre_large_cnn_accuracies])
    all_train_accuracies = np.array([optimum_mlp_accuracies_per_epoch[:used_epoch], small_cnn_accuracies_per_epoch,
                                     large_cnn_accuracies_per_epoch,
                                     pre_small_cnn_accuracies_per_epoch, pre_large_cnn_accuracies_per_epoch])

    f_4, ax_4 = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    ax_4 = ax_4.flatten()

    plot_hyper_parameter_choice("", ["optimum MLP", "28 pix CNN", "128 pix CNN", "Pre-trained \n28 pix CNN",
                                     "Pre-trained \n128 pix CNN"],
                                all_test_accuracies, all_train_accuracies, all_times,
                                f=f_4, ax=ax_4, row=0, scale="linear", validation=False)



    ############################## CNN hyper-parameter choice ##############################

    print("CNN parameters")

    activations_width = [nn.ReLU(), nn.ReLU()]
    exp_fc_neurons_width = [[256, 256, 11], [1024, 1024, 11], [4096, 4096, 11]]
    pre_trained = PreTrained(exp_fc_neurons_width[0], activations_width, epochs=used_epoch)

    pre_trained_training, pre_trained_testing = data_loader_getter(28, batch_size=used_batch, three_channel=True, val=True)

    exp_fc_neurons_depth = [[256, 11], [256, 256, 256, 256, 11], [256, 256, 256, 256, 256, 256, 256, 256, 11]]
    activations_depth = [[nn.ReLU()], [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                         [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]]


    parameters_list = [(ai, bi) for ai, bi in zip(exp_fc_neurons_depth, activations_depth)]

    print("Width")

    width_accuracies, width_accuracies_per_epoch, width_times = \
        series_cnn_hyper_parameter_choice(pre_trained, "layer_sizes", exp_fc_neurons_width,
                                          pre_trained_training, pre_trained_testing)
    print("Depth")

    depth_accuracies, depth_accuracies_per_epoch, depth_times = \
        series_cnn_hyper_parameter_choice(pre_trained, ["layer_sizes", "activations"],
                                          parameters_list, pre_trained_training, pre_trained_testing)

    plot_hyper_parameter_choice("Network Depth", [1, 4, 8],
                                depth_accuracies, depth_accuracies_per_epoch, depth_times,
                                f=f_4, ax=ax_4, row=1, scale="linear", validation=True)

    ax_4[4].set_title(f"Network depth dependence")
    ax_4[3].set_title(f"Hyper-parameter choice")

    plot_hyper_parameter_choice("Network width", [256, 1024, 4096],
                                width_accuracies, width_accuracies_per_epoch, width_times,
                                f=f_4, ax=ax_4, row=2, scale="log", validation=True)

    ax_4[7].set_title(f"Network width dependence")

    f_4.tight_layout()
    with PdfPages(f"code_dir/hw_3/analysis_files/cnn.pdf") as line_pdf:
        line_pdf.savefig(f_4)
