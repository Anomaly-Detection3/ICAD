import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
import warnings
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')


train_data = pd.read_csv(r"../Datasets/EO_train.csv")
test_data = pd.read_csv(r"../Datasets/EO_test.csv")


def get_distribution_difference(train_data, test_data):

    diff_stats = []
    common_cols = list(set(train_data.columns) & set(test_data.columns))
    for col in common_cols:
        stat, p_value = ks_2samp(train_data[col], test_data[col])
        diff_stats.append((col, stat, p_value))

    diff_stats.sort(key=lambda x: x[1], reverse=True)
    return diff_stats



def plot_combined_visualization(train_data, test_data, top_n=2, filename='domain_difference_visualization_compact.png'):

    print("Calculating feature distribution differences...")
    diff_stats = get_distribution_difference(train_data, test_data)

    fig = plt.figure(figsize=(16, 8))

    width_ratios = [2] + [1] * top_n
    gs = GridSpec(1, 1 + top_n, width_ratios=width_ratios)


    print("Performing t-SNE dimensionality reduction...")
    ax1 = fig.add_subplot(gs[0])

    if train_data.shape[1] > 50:
        print("Many features detected, using PCA to reduce to 50 dimensions first...")
        pca = PCA(n_components=50)
        train_data_reduced = pca.fit_transform(train_data)
        test_data_reduced = pca.transform(test_data)
    else:
        train_data_reduced = train_data
        test_data_reduced = test_data

    combined_data = np.vstack([train_data_reduced, test_data_reduced])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(combined_data)

    train_tsne = tsne_result[:len(train_data)]
    test_tsne = tsne_result[len(train_data):]

    ax1.scatter(train_tsne[:, 0], train_tsne[:, 1], c='blue', alpha=0.6,
                label='Training Set', s=30, edgecolors='none')
    ax1.scatter(test_tsne[:, 0], test_tsne[:, 1], c='red', alpha=0.6,
                label='Test Set', s=30, edgecolors='none')

    ax1.set_title('Data Distribution after t-SNE', fontsize=14)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)


    print(f"Drawing Top-{top_n} feature boxplots...")
    top_features = [x[0] for x in diff_stats[:top_n]]

    for i, feature in enumerate(top_features):
        ax = fig.add_subplot(gs[i + 1])
        box_data = [train_data[feature], test_data[feature]]

        box = ax.boxplot(box_data, labels=['Training', 'Test'], patch_artist=True, widths=0.7)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'{feature}', fontsize=14)
        if i == 0:
            ax.set_ylabel('Feature Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Domain Difference Visualization between Training and Test Sets', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {filename}")



print("Starting visualization generation...")
plot_combined_visualization(train_data, test_data, top_n=2)
print("Visualization complete!")
