
from torch.utils.data import Dataset as TDataset
import pandas as pd
import torch
import numpy as np
import os

# Use the Kalman filter to compute the state vector for each time window.
class KalmanFilterSSM:
    def __init__(self, input_dim, state_dim):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.transition_matrix = torch.eye(state_dim, dtype=torch.float32)
        self.transition_cov = torch.eye(state_dim, dtype=torch.float32) * 0.1
        self.emission_matrix = torch.randn(input_dim, state_dim, dtype=torch.float32)
        self.emission_cov = torch.eye(input_dim, dtype=torch.float32) * 0.1
        self.initial_state_mean = torch.zeros(state_dim, dtype=torch.float32)
        self.initial_state_cov = torch.eye(state_dim, dtype=torch.float32) * 0.1

    def filter(self, observations):
        T = observations.shape[0]
        filtered_means = []
        state_mean = self.initial_state_mean
        state_cov = self.initial_state_cov

        for t in range(T):

            predicted_state_mean = self.transition_matrix @ state_mean
            predicted_state_cov = self.transition_matrix @ state_cov @ self.transition_matrix.T + self.transition_cov

            observation = observations[t]
            innovation = observation.unsqueeze(1) - self.emission_matrix.T @ predicted_state_mean.unsqueeze(1)
            innovation_cov = self.emission_matrix.T @ predicted_state_cov @ self.emission_matrix + self.emission_cov

            kalman_gain = predicted_state_cov @ self.emission_matrix @ torch.inverse(innovation_cov)
            state_mean = predicted_state_mean.unsqueeze(1) + kalman_gain @ innovation
            state_mean = state_mean.squeeze(1)
            state_cov = predicted_state_cov - kalman_gain @ self.emission_matrix.T @ predicted_state_cov

            filtered_means.append(state_mean)

        return torch.stack(filtered_means)



class Dataset(TDataset):
    def __init__(self, mode, win_size, data_name):
        super(Dataset, self).__init__()
        assert mode in ('train', 'test')
        self.mode = mode
        self.win_size = win_size
        Flag = True
        if data_name == "EO":
            print("load data", data_name)
            folder = '../EO'
        elif data_name == "WADI":
            print("load data", data_name)
            folder = '../WADI/'
        elif data_name == "Swat":
            print("load data", data_name)
            folder = '../SWAT/'
        elif data_name == "PSM":
            print("load data", data_name)
            folder = '../PSM'
        else:
            print("load data", data_name)
            folder = '../HAI'

        loader = []
        if mode == "train":
            print('---load training data----')
            if data_name == "EO":
                file = 'train/EO_train'
            elif data_name == "WADI":
                file = 'train/train'
            elif data_name == "Swat":
                file = 'train/train_normal'
            elif data_name == "PSM":
                file = 'train'
            else:
                file = 'train'
            # data = np.array(pd.read_csv(os.path.join(folder, f'{file}.csv')).fillna(0))

        else:
            if Flag:
                print('---load testing data----')
                if data_name == "EO":
                    file = 'test/EO_test'# EO
                elif data_name == "WADI":
                    file = 'test/test'# WADI
                elif data_name == "Swat":
                    file = 'test/test_attach' # Swat
                elif data_name == "PSM":
                    file = 'test'  # PSM
                else:
                    file = 'test' # HAI
            else:
                print('=====load noise testing data====')

                if data_name == "EO":
                    file = '../test_spike_noisy_c2'
                elif data_name == "WADI":
                    file = '../test_spike_noisy_c2'
                elif data_name == "Swat":
                    file = '../test_gaussian_noisy_c2'
                elif data_name == "PSM":
                    file = 'test_spike_noisy_c2'
                else:
                    file = 'test_spike_noisy_c2'

        df = pd.read_csv(os.path.join(folder, f'{file}.csv')).fillna(0)
        data = np.array(df)
        self.sensors_name = df.columns.tolist()


        data = torch.from_numpy(data)
        mean_data = torch.mean(data, dim=(-2, -1), keepdim=True)
        std_data = torch.std(data, dim=(-2, -1), keepdim=True)

        print(data.shape)

        data = (data - mean_data) / (std_data + 1e-8)

        for i in range(0, len(data) - win_size + 1, win_size):
            loader.append(data[i:i + win_size])

        self.loader = loader

    def get_sensors_name(self):
        """Method to get the sensor names"""
        return self.sensors_name

    def __len__(self):
        if self.mode == 'train':
            return len(self.loader)
        return len(self.loader)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.loader[index]
        return self.loader[index]