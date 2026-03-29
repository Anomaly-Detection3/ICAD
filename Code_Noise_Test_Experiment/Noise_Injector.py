import numpy as np
import pandas as pd
from typing import List, Optional


class NoiseInjector:
    """
    A class to inject various types of noise into time series data for robustness testing.
    """

    def __init__(self, train_csv_path: str):
        """
        Initializes the NoiseInjector with a path to the clean training data CSV.

        """
        try:
            self.train_df = pd.read_csv(train_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data file not found at: {train_csv_path}")

        self.channel_stats = {
            'mean': self.train_df.mean(),
            'std': self.train_df.std()
        }
        print("NoiseInjector initialized. Training data statistics calculated.")

    def inject_gaussian_noise(self,
                              target_df: pd.DataFrame,
                              noisy_channels: List[str],
                              noise_level: float = 0.5) -> pd.DataFrame:
        """
        Injects Gaussian noise into specified channels of a target DataFrame.

        Args:
            target_df (pd.DataFrame): The DataFrame to inject noise into.
            noisy_channels (List[str]): A list of column names to add noise to.
            noise_level (float): A multiplier for the standard deviation of the noise.
                                 noise_std = noise_level * channel_std_from_train.

        Returns:
            pd.DataFrame: A new DataFrame with Gaussian noise added.
        """
        noisy_df = target_df.copy()

        for channel in noisy_channels:
            if channel not in noisy_df.columns:
                print(f"Warning: Channel '{channel}' not found in target DataFrame. Skipping.")
                continue

            channel_std = self.channel_stats['std'].get(channel, 0)
            if channel_std == 0:
                print(f"Warning: Std for channel '{channel}' is 0. Cannot add noise. Skipping.")
                continue

            noise_std = noise_level * channel_std

            noise = np.random.normal(loc=0.0, scale=noise_std, size=len(noisy_df))
            noisy_df[channel] += noise

        return noisy_df

    def inject_spike_noise(self,
                           target_df: pd.DataFrame,
                           noisy_channels: List[str],
                           spike_ratio: float = 0.01,
                           spike_magnitude: float = 5.0) -> pd.DataFrame:
        """
        Injects random spike noise (outliers) into specified channels.

        Spikes are created by adding/subtracting a multiple of the channel's
        standard deviation from the original value at random locations.

        Args:
            target_df (pd.DataFrame): The DataFrame to inject noise into.
            noisy_channels (List[str]): A list of column names to add noise to.
            spike_ratio (float): The proportion of data points to turn into spikes (e.g., 0.01 for 1%).
            spike_magnitude (float): How many standard deviations away the spike should be.

        Returns:
            pd.DataFrame: A new DataFrame with spike noise added.
        """
        noisy_df = target_df.copy()
        num_points_to_spike = int(spike_ratio * len(noisy_df))

        for channel in noisy_channels:
            if channel not in noisy_df.columns:
                print(f"Warning: Channel '{channel}' not found in target DataFrame. Skipping.")
                continue


            channel_std = self.channel_stats['std'].get(channel, 0)
            if channel_std == 0:
                print(f"Warning: Std for channel '{channel}' is 0. Cannot add spikes. Skipping.")
                continue


            spike_indices = np.random.choice(noisy_df.index, size=num_points_to_spike, replace=False)

            spike_values = spike_magnitude * channel_std * np.random.choice([-1, 1], size=num_points_to_spike)

            noisy_df.loc[spike_indices, channel] += spike_values

        return noisy_df



TEST_CSV = '../EO/EO_test.csv'

# 1. Initialize the injector with clean training data
injector = NoiseInjector(train_csv_path=TEST_CSV)

# 2. Load the target data
test_df = pd.read_csv(TEST_CSV)

# --- Experiment 1: Gaussian Noise on a few channels ---
np.random.seed(41)
num_channels_to_corrupt = 4
all_channels = list(test_df.columns)

channels_to_corrupt = np.random.choice(all_channels, size=num_channels_to_corrupt, replace=False).tolist()
# print(f"Selected channels for Gaussian noise: {channels_to_corrupt}")

#
# test_df_gaussian_noisy = injector.inject_gaussian_noise(
#     target_df=test_df,
#     noisy_channels=channels_to_corrupt,
#     noise_level=0.5
# )
#
# test_df_gaussian_noisy.to_csv('../test_gaussian_noisy_C4.csv', index=False)
# print("Saved test data with Gaussian noise to 'test_gaussian_noisy.csv'")


# --- Experiment 2: Spike Noise (Outliers) ---

spike_channels = channels_to_corrupt  # or choose different ones
print(f"\nSelected channels for Spike noise: {spike_channels}")

test_df_spike_noisy = injector.inject_spike_noise(
    target_df=test_df,
    noisy_channels=spike_channels,
    spike_ratio=0.10,  # Corrupt 2% of the data points in these channels
    spike_magnitude=5.0  # Spikes will be 5 standard deviations from the original
)

test_df_spike_noisy.to_csv('../test_spike_noisy_c4.csv', index=False)
print("Saved test data with Spike noise to 'test_spike_noisy.csv'")
