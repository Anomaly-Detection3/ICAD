import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy


def calculate_entropy(signal):

    probability_distribution, bin_edges = np.histogram(signal, bins=20, density=True)
    probability_distribution = probability_distribution[probability_distribution > 0]
    return entropy(probability_distribution)


def detect_trend_with_entropy(time_series):
    diffs = np.diff(time_series, axis=0)
    dif_len = diffs.shape[1]
    diffs1, diffs2 = diffs[:, :dif_len//2], diffs[:, dif_len//2:]

    def M(diff_value):
        row_mean = np.mean(diff_value, axis=1)
        row_std = np.var(diff_value, axis=1)
        M_statics = row_mean/row_std
        M_statics = np.mean(M_statics)
        return M_statics

    a = diffs.flatten()
    combined_entropy = calculate_entropy(a)
    diff1_s, diffs2_s = M(diffs1), M(diffs2)
    if diff1_s > 0 and diffs2_s > 0:
        trend = "Monotonic Increase"
    elif diff1_s < 0 and diffs2_s < 0:
        trend = "Monotonic Decrease"
    elif diff1_s > 0 and diffs2_s < 0:
        trend = "Inverse U-shaped Trend"
    else:
        trend = "U-shaped Trend"

    return trend, combined_entropy



#---------test----------#
time = np.random.randint(3, 15, size=(15, 25))

trend, combined_entropy = detect_trend_with_entropy(time)

print(f" trend pattern: {trend}")
print(f"Joint Information Entropy: {combined_entropy:.4f}")

trend_all = []
combined_entropy_all = []
trend_all.append(trend)
combined_entropy_all.append(combined_entropy)
pd.DataFrame(trend_all, columns=['trend']).to_csv("../trend.csv")
pd.DataFrame(combined_entropy_all, columns=['entropy']).to_csv("../entropy.csv")
