import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

matplotlib.rcParams['mathtext.fontset'] = 'stix'


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def calculate_entropy(signal, j):

    probability_distribution, bin_edges = np.histogram(signal, bins=20, density=True)

    #colors = plt.cm.plasma(np.linspace(0, 1, 10))
    cmap = plt.get_cmap("nipy_spectral")
    colors = [cmap(i/20) for i in range(20)]
    # plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], probability_distribution, width=np.diff(bin_edges), color=colors)
    plt.title('Diiference Histogram',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("../D_H"+str(j)+"value.pdf", dpi=500, bbox_inches='tight')
    plt.show()
    probability_distribution = probability_distribution[probability_distribution > 0]
    return entropy(probability_distribution)


def detect_trend_with_entropy(time_series, j):
    diffs = np.diff(time_series, axis=0)
    dif_len = diffs.shape[1]
    diffs1, diffs2 = diffs[:, :dif_len//2], diffs[:, dif_len//2:]

    def M(diff_value):
        row_mean = np.mean(diff_value, axis=1)
        row_std = np.var(diff_value, axis=1)
        M_statics = row_mean/row_std
        M_statics = np.mean(M_statics)
        return M_statics

    combined_entropy = calculate_entropy(diffs.flatten(), j)
    diff1_s, diffs2_s = M(diffs1), M(diffs2)
    if diff1_s > 0 and diffs2_s > 0:
        trend = "Monotonic Increase"
    elif diff1_s < 0 and diffs2_s < 0:
        trend = "Monotonic Decrease"
    elif diff1_s > 0 and diffs2_s < 0:
        trend = "Inverse U-shaped"
    else:
        trend = "U-shaped"

    return trend, combined_entropy




data = pd.read_csv("../test.csv")
data = np.array(data)

win_size = 50
time = []
for i in range(0, len(data) - win_size + 1, win_size):
    time.append(data[i:i + win_size])
time = np.array(time)
trend_all = []
combined_entropy_all = []
for j in range(time.shape[0]):
    trend, combined_entropy = detect_trend_with_entropy(time[j, :, :], j)
    trend_all.append(trend)
    combined_entropy_all.append(combined_entropy)

    plt.plot(time[j, :, :])
    plt.xlabel('Time',fontsize=16)
    plt.ylabel("Value",fontsize=16)
    line1 = f"The Trend Pattern is {trend},"
    line2 = f"and Information Entropy is  {round(combined_entropy, 3)}."
    plt.title(f"{line1}\n{line2}", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("../D_H"+str(j)+"value.pdf",dpi=500, bbox_inches='tight')
    plt.show()
    print(f"United trend pattern: {trend}")
    print(f"Joint Information Entropy: {combined_entropy:.4f}")

# pd.DataFrame(trend_all, columns=['trend']).to_csv("D:/trend.csv")
# pd.DataFrame(combined_entropy_all, columns=['entropy']).to_csv("D:/entropy.csv")



