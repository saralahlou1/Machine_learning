import numpy as np
import torchvision.transforms as transforms
from medmnist import OrganAMNIST

from code_dir.MLP_CNN_classification.config import xp


class DataClassNN:
    train_data: xp.ndarray
    test_data: xp.ndarray
    val_data: xp.ndarray
    train_labels: xp.ndarray
    test_labels: xp.ndarray
    val_labels: xp.ndarray

    def __init__(self, size=28, normalize=True):
        for data_type in ["train", "test", "val"]:
            self.data_initialization(data_type, size, normalize)


    def data_initialization(self, data_type, size, normalize):
        data_transform = transforms.Compose([transforms.ToTensor()])
        self.__setattr__(f"{data_type}_data", OrganAMNIST(split=data_type, download=True, transform=data_transform, size=size))
        if normalize:
            sum_for_mean = 0
            squared_sum = 0
            N = 0
            for data, _ in self.__getattribute__(f"{data_type}_data"):
                sum_for_mean += data.sum(dim=[0, 1, 2])
                squared_sum += (data ** 2).sum(dim=[0, 1, 2])
                N += data.numel()
            mean = sum_for_mean / N
            std = np.sqrt((squared_sum / N) - (mean ** 2))
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,)),
                                                 transforms.Lambda(lambda x: x.view(-1))])
            self.__getattribute__(f"{data_type}_data").transform = data_transform

        else:
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
            self.__getattribute__(f"{data_type}_data").transform = data_transform

        all_data = []
        all_labels = []
        for data, label in self.__getattribute__(f"{data_type}_data"):
            all_data.append(data.numpy())
            all_labels.append(label[0])


        self.__setattr__(f"{data_type}_data", xp.array(all_data))
        self.__setattr__(f"{data_type}_labels", xp.array(all_labels))

        num_labels = self.__getattribute__(f"{data_type}_labels").size

        y_train_one_hot = xp.zeros((num_labels, int(xp.max(self.__getattribute__(f"{data_type}_labels"))) + 1))
        y_train_one_hot[xp.arange(num_labels), self.__getattribute__(f"{data_type}_labels")] = 1

        self.__setattr__(f"{data_type}_labels", y_train_one_hot)




