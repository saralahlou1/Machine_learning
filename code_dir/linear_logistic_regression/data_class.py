from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
# import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo



def sort_from_reference(reference_array, *args):
    return zip(*sorted(zip(reference_array, *args)))


class UCIdata(ABC):
    target_to_classify: str

    def __init__(self, data_id, target_to_classify):
        all_data = fetch_ucirepo(id=data_id)
        self.all_data_features = all_data.data.features
        self.all_data_targets = all_data.data.targets
        self.features = []
        self.str_features = []
        for feature in self.all_data_features.keys():
            if self.all_data_features[feature].dtypes in ["float64", "int64"]:
                self.features.append(feature)
            else:
                self.str_features.append(feature)
        self.targets = self.all_data_targets.keys()
        self.target_to_classify = target_to_classify


    def nan_num_print(self):
        for feature in self.features:
            num_nans = self.all_data_features[feature].isna().sum()
            if num_nans > 0:
                print(f"{feature} has {num_nans} nans")

        for target in self.targets:
            num_nans = self.all_data_targets[target].isna().sum()
            if num_nans > 0:
                print(f"{target} has {num_nans} nans")

    def standarize(self):
        scale = StandardScaler()
        scaled_features = scale.fit_transform(self.all_data_features.select_dtypes(include='number'))
        return pd.DataFrame(scaled_features, columns=self.all_data_features.select_dtypes(include='number').columns)

    @abstractmethod
    def correlation_to_name(self):
        pass

class Temperature(UCIdata):

    def __init__(self, data_id, target_to_classify):
        super().__init__(data_id, target_to_classify)
        self.large_outlier_handler()


    def nan_handler(self):
        oldIndices = self.all_data_features.index
        self.all_data_features = self.all_data_features.dropna(how="any")
        newIndices = self.all_data_features.index
        removedIndices = list(set(oldIndices) - set(newIndices))
        print(f"The removed NaN indices are: {removedIndices}")
        print(f"The new size of the data set is: {np.shape(self.all_data_targets)}")
        self.all_data_targets = self.all_data_targets.drop(removedIndices)

    def large_outlier_handler(self, feature='Distance', threshold=1, max_value=0.8):
        self.all_data_features[feature][self.all_data_features[feature] > threshold] = max_value

    def plot_all_feature(self):
        row, col = 4, len(self.features) // 4 + 1
        f, ax = plt.subplots(row, col, figsize=(25, 15))
        axs_flat = ax.flatten()
        for dex, feature in enumerate(self.features):
            axs_flat[dex].plot(self.all_data_features[feature],
                               self.all_data_targets[self.target_to_classify], "ro")
            axs_flat[dex].set_xlabel(feature)
            axs_flat[dex].set_ylabel(self.target_to_classify)
            axs_flat[dex].set_title(f"{self.target_to_classify}")
        f.tight_layout()
        plt.show()

    def correlation_to_name(self):
        numeric_features = self.all_data_features.select_dtypes(include='number')
        combined_df2 = pd.concat([numeric_features, self.all_data_targets], axis=1)
        correlation_matrix2 = combined_df2.corr()
        feature_columns2 = numeric_features.columns
        target_columns2 = self.all_data_targets.columns
        relevant_correlations2 = correlation_matrix2.loc[feature_columns2, target_columns2]
        reference_array = list(relevant_correlations2[self.target_to_classify].to_numpy())
        feature_name_array = relevant_correlations2.index.tolist()
        reference_array, feature_name_array = sort_from_reference(reference_array, feature_name_array)
        return reference_array, feature_name_array

    # def categorical_hist(self):
    #     combined_temp_df = pd.concat([self.all_data_features, self.all_data_targets], axis=1)
    #     f, ax = plt.subplots(1, 3, figsize=(14, 5))
    #     axs_flat = ax.flatten()
    #     for dex, categorical in enumerate(self.str_features):
    #         sns.histplot(data=combined_temp_df, x=self.target_to_classify, hue=categorical, multiple="stack",
    #                      palette='tab10', ax=axs_flat[dex])
    #         axs_flat[dex].set_title(f'{categorical}')
    #     f.tight_layout()
    #     plt.show()


class Diabetes(UCIdata):

    def __init__(self, data_id, target_to_classify):
        super().__init__(data_id, target_to_classify)

    def plot_target(self):
        f, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.hist(self.all_data_targets[self.target_to_classify], bins=3)
        ax.set_title(f'{self.target_to_classify}')
        plt.show()

    def plot_feature_hist(self):
        scaled_features = self.standarize()
        scaled_combined_df = pd.concat([scaled_features, self.all_data_targets], axis=1)
        row, col = 4, len(self.features) // 4 + 1
        f, ax = plt.subplots(row, col, figsize=(16, 9))
        axs_flat = ax.flatten()
        for dex, feature in enumerate(self.features):
            axs_flat[dex].hist(scaled_combined_df[scaled_combined_df[self.target_to_classify] == 0][feature],
                               bins=15, color='blue', alpha=0.7, label='Label = 0', edgecolor='black')
            axs_flat[dex].hist(scaled_combined_df[scaled_combined_df[self.target_to_classify] == 1][feature],
                               bins=15, color='red', alpha=0.7, label='Label = 1', edgecolor='black')
            axs_flat[dex].set_title(f'{feature}')
            axs_flat[dex].legend()
            axs_flat[dex].grid(axis='y')
        f.tight_layout()
        plt.show()

    def correlation_to_name(self):
        feature_list, correlations = [], []
        for feature in self.all_data_features.columns:
            feature_list.append(feature)
            f = self.all_data_features[feature].to_numpy()
            corr, p_value = pointbiserialr(self.all_data_targets.to_numpy().ravel(), f.ravel())
            correlations.append(corr)
        reference_array, feature_name_array = sort_from_reference(correlations, feature_list)
        return reference_array, feature_name_array
