import copy
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Union

from torch import optim, nn

import numpy as np
from medmnist import OrganAMNIST
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from code_dir.linear_logistic_regression.run import get_color, grid
from code_dir.MLP_CNN_classification.convolution_nn import PreTrained, train_model
from code_dir.MLP_CNN_classification.multilayer_perceptron import MLP
from code_dir.timer import Timer


def hyper_parameter_choice(model: MLP, hyper_parameter: str, parameter_list,
                           x_train_data, y_train_data, x_test_data, y_test_data):
    test_accuracies = []
    accuracies_per_epoch = []
    times = []
    timing = Timer(False)
    for parameter in parameter_list:
        model_copy = copy.deepcopy(model)
        model_copy.__setattr__(hyper_parameter, parameter)
        timing.start()
        model_copy.mpl_gradient_descent(x_train_data, y_train_data, x_test_data, y_test_data)
        timing.stop()
        times.append(timing.elapsed_time)
        test_accuracies.append(model_copy.test_accuracy_per_epoch)
        accuracies_per_epoch.append(model_copy.accuracy_per_epoch)

    return test_accuracies, accuracies_per_epoch, times


def run_with_hyperparameter(model: MLP, hyper_parameter: Union[str, List[str]], parameter,
                            x_train_data, y_train_data, x_test_data, y_test_data) -> Tuple[float, List[float], float]:
    model_copy = copy.deepcopy(model)
    if isinstance(hyper_parameter, str):
        model_copy.__setattr__(hyper_parameter, parameter)
    else:
        for hyper_name, single_parameter in zip(hyper_parameter, parameter):
            model_copy.__setattr__(hyper_name, single_parameter)

    timing = Timer(False)
    timing.start()
    model_copy.mpl_gradient_descent(x_train_data, y_train_data, x_test_data, y_test_data)
    timing.stop()

    elapsed_time = timing.elapsed_time
    accuracy_per_epoch = model_copy.accuracy_per_epoch
    test_accuracy = model_copy.test_accuracy_per_epoch

    return test_accuracy, accuracy_per_epoch, elapsed_time


def par_hyper_parameter_choice(model: MLP, hyper_parameter: Union[str, List[str]], parameter_list,
                               x_train_data, y_train_data, x_test_data, y_test_data):
    results = []

    with ProcessPoolExecutor(max_workers=18) as executor:
        futures = [
            executor.submit(run_with_hyperparameter, model, hyper_parameter, parameter,
                            x_train_data, y_train_data, x_test_data, y_test_data)
            for parameter in parameter_list
        ]

        for future in futures:
            results.append(future.result())

    test_accuracies = [result[0] for result in results]
    accuracies_per_epoch = [result[1] for result in results]
    times = [result[2] for result in results]

    return np.array(test_accuracies), np.array(accuracies_per_epoch), np.array(times)


def plot_hyper_parameter_choice(parameter_name, parameter_list, test_accuracies, accuracies_per_epoch, times, f=None,
                                ax=None, color_skeme="plasma", row=0, scale="log", validation=True):
    if validation:
        val = "Validation"
    else:
        val = "Testing"

    for dex, (accuracy_per_epoch, test_accuracy) in enumerate(zip(accuracies_per_epoch, test_accuracies)):
        ax[0 + row * 3].plot(np.arange(len(accuracy_per_epoch)), accuracy_per_epoch, marker='o', markersize=2,
                             linestyle='-',
                             linewidth=1,
                             color=get_color(dex, len(times), color_skeme=color_skeme),
                             label=f"{parameter_name}={parameter_list[dex]}")

        ax[1 + row * 3].plot(np.arange(len(accuracy_per_epoch)), test_accuracy, marker='o', markersize=2,
                             linestyle='-',
                             linewidth=1,
                             color=get_color(dex, len(times), color_skeme=color_skeme),
                             label=f"{parameter_list[dex]}")

    # ax[2 + row * 3].plot(parameter_list, times, marker='o', markersize=5, linestyle='-', linewidth=2,
    #                      color=point_color, label=label)

    ax2 = ax[2 + row * 3].twinx()
    ax[2 + row * 3].bar([f"{par}" for par in parameter_list], times, color="skyblue", width=0.2,
                        label='Training time', zorder=1)
    ax2.bar([f"{par}" for par in parameter_list], test_accuracies[:, -1], color="red", width=0.1,
            label=f'{val} accuracy', zorder=1)
    ax[0 + row * 3].legend(fontsize=5)
    ax[1 + row * 3].legend(fontsize=5)
    ax[2 + row * 3].legend(loc="upper left", fontsize=5)
    ax2.legend(loc="upper right", fontsize=5)

    ax[0 + row * 3].set_xlabel("Epochs")
    ax[1 + row * 3].set_xlabel("Epochs")
    ax[2 + row * 3].set_xlabel(parameter_name)

    # ax[2 + row * 3].set_xscale(scale)

    ax[0 + row * 3].set_ylabel("Training accuracy")
    ax[1 + row * 3].set_ylabel(f"{val} accuracy")
    ax[2 + row * 3].set_ylabel("Time (second)")
    ax2.set_ylabel(f"{val} accuracy")
    ax[2 + row * 3].set_ylim(0, np.max(times) * 1.1)
    ax2.set_ylim(0, np.max(test_accuracies[:, -1]) * 1.1)

    ax[2 + row * 3].set_title(f"Optimum {parameter_name}= {parameter_list[np.argmax(test_accuracies[:, -1])]}")

    for axs in ax:
        grid(axs)

    f.tight_layout()


def mean_std_getter(dataset):
    total_sum = 0.0
    total_squared_sum = 0.0
    num_pixels = 0
    for img, _ in dataset:
        img = img.view(-1)
        total_sum += img.sum().item()
        total_squared_sum += (img ** 2).sum().item()
        num_pixels += img.numel()

    mean = total_sum / num_pixels
    std = ((total_squared_sum / num_pixels) - (mean ** 2)) ** 0.5
    return mean, std


def repeat_channels(x):
    """Repeat single-channel data to create a 3-channel tensor."""
    return x.repeat(3, 1, 1)

def data_loader_getter(size, batch_size=500, three_channel=False, val=False):
    data_transform = ToTensor()
    train_data = OrganAMNIST(split="train", download=True, transform=data_transform, target_transform=None, size=size)
    if val:
        test_data = OrganAMNIST(split="val", download=True, transform=data_transform, target_transform=None, size=size)
    else:
        test_data = OrganAMNIST(split="test", download=True, transform=data_transform, target_transform=None, size=size)
    combined_data = ConcatDataset([train_data, test_data])

    all_mean, all_std = mean_std_getter(combined_data)
    train_mean, train_std = all_mean, all_std
    test_mean, test_std = all_mean, all_std

    if three_channel:
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(repeat_channels),
                                             transforms.Normalize((train_mean,) * 3, (train_std,) * 3)])
        test_data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(repeat_channels),
             transforms.Normalize((test_mean,) * 3, (test_std,) * 3)])
    else:
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((train_mean,), (train_std,))])
        test_data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((test_mean,), (test_std,))])

    train_data.transform = data_transform
    test_data.transform = test_data_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def cnn_run_with_hyperparameter(pre_trained: PreTrained, hyper_parameter: Union[str, List[str]], parameter,
                                pre_trained_training, pre_trained_testing) -> Tuple[float, List[float], float]:
    model_copy = copy.deepcopy(pre_trained)
    if isinstance(hyper_parameter, str):
        model_copy.__setattr__(hyper_parameter, parameter)
    else:
        for hyper_name, single_parameter in zip(hyper_parameter, parameter):
            model_copy.__setattr__(hyper_name, single_parameter)

    resnet_optimizer = optim.SGD(model_copy.resnet.fc.parameters(), lr=0.001, momentum=0.75)
    criterion = nn.CrossEntropyLoss()

    timing = Timer(False)
    timing.start()
    train_model(model_copy, model_copy.resnet, criterion, resnet_optimizer, pre_trained_training,
                pre_trained_testing, device="cpu")
    timing.stop()

    elapsed_time = timing.elapsed_time
    accuracy_per_epoch = model_copy.accuracy_per_epoch
    test_accuracy = model_copy.test_accuracy_per_epoch

    return test_accuracy, accuracy_per_epoch, elapsed_time


def cnn_hyper_parameter_choice(model: PreTrained, hyper_parameter: Union[str, List[str]], parameter_list,
                               pre_trained_training, pre_trained_testing):
    results = []

    with ProcessPoolExecutor(max_workers=18) as executor:
        futures = [
            executor.submit(cnn_run_with_hyperparameter, model, hyper_parameter, parameter,
                            pre_trained_training, pre_trained_testing)
            for parameter in parameter_list
        ]

        for future in futures:
            results.append(future.result())

    test_accuracies = [result[0] for result in results]
    accuracies_per_epoch = [result[1] for result in results]
    times = [result[2] for result in results]

    return np.array(test_accuracies), np.array(accuracies_per_epoch), np.array(times)


def series_cnn_hyper_parameter_choice(model: PreTrained, hyper_parameter: Union[str, List[str]], parameter_list,
                                      pre_trained_training, pre_trained_testing):
    test_accuracies = []
    accuracies_per_epoch = []
    times = []
    timing = Timer(False)
    criterion = nn.CrossEntropyLoss()
    for parameter in parameter_list:
        if isinstance(hyper_parameter, str):
            model_copy = PreTrained(parameter, model.activations, epochs=model.epochs)
            # model_copy.__setattr__(hyper_parameter, parameter)
        else:
            model_copy = PreTrained(parameter[0], parameter[1], epochs=model.epochs)
        resnet_optimizer = optim.SGD(model_copy.resnet.fc.parameters(), lr=0.001, momentum=0.75)
        timing.start()
        train_model(model_copy, model_copy.resnet, criterion, resnet_optimizer, pre_trained_training,
                    pre_trained_testing)
        timing.stop()
        times.append(timing.elapsed_time)
        test_accuracies.append(model_copy.test_accuracy_per_epoch)
        accuracies_per_epoch.append(model_copy.accuracy_per_epoch)

    return np.array(test_accuracies), np.array(accuracies_per_epoch), np.array(times)
