import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import os
from scipy.signal import find_peaks
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="h5py not installed, hdf5 features will not be supported.")



def analyze_cluster_trends(windows, cluster_labels, n_clusters, analysis_params=None):

    """
    analysis_params: dict, optional
        Dictionary with keys like "prominence_factor", "slope_threshold",
        "amplitude_thresh_low", "amplitude_thresh_mid".
    """

    cluster_trends_map = {}
    params = {
        "prominence_factor": 0.1,
        "slope_threshold": 0.001,
        "amplitude_thresh_low": 0.6,
        "amplitude_thresh_mid": 1.1,
    }
    if analysis_params:
        params.update(analysis_params)

    for cluster_id in range(n_clusters):
        current_cluster_windows = windows[cluster_labels == cluster_id]
        if len(current_cluster_windows) == 0:
            cluster_trends_map[cluster_id] = f"cluster {cluster_id}: No window allocated to this cluster"
            continue

        avg_window = np.mean(current_cluster_windows, axis=0)
        n_timesteps, n_features = avg_window.shape
        overall_trends = []

        for feature_idx in range(n_features):
            feature_data = avg_window[:, feature_idx]
            x = np.arange(len(feature_data))
            slope, _ = np.polyfit(x, feature_data, 1)

            std_dev = np.std(feature_data)
            prominence_value = std_dev * params["prominence_factor"]
            if prominence_value < 0.001 and std_dev > 0:
                prominence_value = 0.001
            elif std_dev == 0:
                prominence_value = 0.01

            peaks, _ = find_peaks(feature_data, prominence=prominence_value)
            valleys, _ = find_peaks(-feature_data, prominence=prominence_value)

            slope_th = params["slope_threshold"]

            trend = "Unknown"
            if abs(slope) < slope_th:
                trend = "Steady"
            elif len(peaks) == 0 and len(valleys) == 0:
                trend = "Monotonic Increase" if slope > 0 else "Monotonic Decrease"
            elif len(peaks) == 1 and len(valleys) == 0:
                peak_pos = peaks[0] / len(feature_data) if len(feature_data) > 0 else 0
                if peak_pos < 0.3:
                    trend = "Rapid Increase Followed by Gradual Decline"
                elif peak_pos > 0.7:
                    trend = "Gradual Increase Followed by Rapid Decline"
                else:
                    trend = "Increase Followed by Decrease"
            elif len(peaks) == 0 and len(valleys) == 1:
                valley_pos = valleys[0] / len(feature_data) if len(feature_data) > 0 else 0
                if valley_pos < 0.3:
                    trend = "Rapid Decrease Followed by Gradual Recovery"
                elif valley_pos > 0.7:
                    trend = "Gradual Decrease Followed by Rapid Recovery"
                else:
                    trend = "Decrease Followed by Increase"
            elif len(peaks) + len(valleys) >= 2:
                if slope > slope_th:
                    trend = "Fluctuating Increase"
                elif slope < -slope_th:
                    trend = "Fluctuating Decrease"
                else:
                    trend = "Fluctuating Stability"
            else:
                trend = "Overall Increase" if slope > 0 else "Overall Decrease"
            overall_trends.append(trend)


        trend_counts = {}
        for t in overall_trends:
            trend_counts[t] = trend_counts.get(t, 0) + 1
        sorted_trends = sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)
        main_trends_list = sorted_trends[:min(3, len(sorted_trends))]
        trend_description = f"Cluster {cluster_id} ({len(current_cluster_windows)} windows):"

        if not main_trends_list:
            trend_description += " No clear dominant trends identified."
        else:
            trend_description += " Dominant trends are:"
            for i_trend, (trend_item, count) in enumerate(main_trends_list):
                percentage = count / n_features * 100
                if i_trend > 0:
                    trend_description += ","
                trend_description += f" {trend_item}({percentage:.1f}%)"

        if len(current_cluster_windows) > 0:
            avg_std_per_window_per_feature = np.mean(np.std(current_cluster_windows, axis=1), axis=1)
            avg_cluster_std = np.mean(avg_std_per_window_per_feature)
        else:
            avg_cluster_std = 0

        if avg_cluster_std < params["amplitude_thresh_low"]:
            amplitude = "Small variation amplitude"
        elif avg_cluster_std < params["amplitude_thresh_mid"]:
            amplitude = "Medium variation amplitude"
        else:
            amplitude = "Large variation amplitude"

        trend_description += f". {amplitude}."
        cluster_trends_map[cluster_id] = trend_description

    return cluster_trends_map



class TimeSeriesTrendDataset(TorchDataset):
    def __init__(self, mode, win_size, data_name,
                 k_range=(2, 11), analysis_params=None,
                 random_state=42, n_jobs=-1, save_artifacts=True, artifacts_dir="clustering_artifacts"):
        super(TimeSeriesTrendDataset, self).__init__()
        assert mode in ('train', 'test')

        self.win_size = win_size

        self.items = []

        if data_name == "EO":
            print("load data", data_name)
            folder = '../EO'
        elif data_name == "WADI":
            print("load data", data_name)
            folder = '../WADI'
        elif data_name == "Swat":
            print("load data", data_name)
            folder = '../SWAT'
        elif data_name == "PSM":
            print("load data", data_name)
            folder = '../PSM'
        else:
            print("load data", data_name)
            folder = '../HAI'

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
        else:
            print('---load testing data----')
            if data_name == "EO":
                file = 'test/EO_test'
            elif data_name == "WADI":
                file = 'test/test'
            elif data_name == "Swat":
                file = 'test/test_attach'
            elif data_name == "PSM":
                file = 'test'
            else:
                file = 'test'

        df = pd.read_csv(os.path.join(folder, f'{file}.csv')).fillna(0)
        raw_data_np = np.array(df)



        num_total_timesteps, n_features = raw_data_np.shape
        print(f"Loaded data shape: ({num_total_timesteps}, {n_features})")


        mean_val = np.mean(raw_data_np, axis=0, keepdims=True)
        std_val = np.std(raw_data_np, axis=0, keepdims=True)
        std_val[std_val == 0] = 1e-8
        scaled_full_data_np = (raw_data_np - mean_val) / std_val
        print("Data scaled globally.")


        windows_list = []
        for i in range(0, num_total_timesteps - self.win_size + 1, self.win_size):
            windows_list.append(scaled_full_data_np[i: i + self.win_size, :])

        if not windows_list:
            print("Warning: No windows created. Check win_size, and data length.")
            return


        X_windows_np = np.stack(windows_list, axis=0)
        n_windows_created = X_windows_np.shape[0]
        print(f"Created {n_windows_created} windows of shape ({self.win_size}, {n_features})")


        print("\n--- Determining Optimal K using Silhouette Score (DTW metric) ---")
        dtw_distance_matrix = cdist_dtw(X_windows_np, n_jobs=n_jobs, verbose=0)
        print("  DTW distance matrix calculated for silhouette score.")

        current_k_range = range(min(k_range), min(max(k_range) + 1, n_windows_created))  # Ensure K is not > n_samples
        silhouette_scores = []
        best_k = -1
        max_silhouette_score = -1

        print(f"  Testing K values in range: {list(current_k_range)}")
        for k_val in current_k_range:
            if k_val < 2: continue
            temp_model = TimeSeriesKMeans(n_clusters=k_val, metric="dtw", max_iter=15,
                                          random_state=random_state, verbose=0, n_jobs=n_jobs)
            temp_labels = temp_model.fit_predict(X_windows_np)

            if len(np.unique(temp_labels)) < 2:

                current_score = -1.0
            else:
                try:
                    current_score = silhouette_score(dtw_distance_matrix, temp_labels, metric="precomputed")
                    # print(f"    K = {k_val}, Silhouette Score = {current_score:.4f}")
                except ValueError:

                    current_score = -1.0
            silhouette_scores.append(current_score)
            if current_score > max_silhouette_score:
                max_silhouette_score = current_score
                best_k = k_val

        if best_k == -1 and silhouette_scores:
            best_k_idx = np.argmax(silhouette_scores)
            if best_k_idx < len(current_k_range) and list(current_k_range)[best_k_idx] >= 2:
                best_k = list(current_k_range)[best_k_idx]
            else:
                best_k = min(3, n_windows_created - 1) if n_windows_created > 2 else (
                    2 if n_windows_created == 2 else 1)
                if best_k < 2 and n_windows_created >= 2:
                    best_k = 2
                elif n_windows_created < 2:
                    print("Not enough windows to perform clustering. Setting K=1.")
                    best_k = 1
            max_silhouette_score = silhouette_scores[best_k_idx] if silhouette_scores else -1

        if n_windows_created < 2:
            print("Not enough windows for meaningful clustering. Assigning all to cluster 0 with a default trend.")
            best_k = 1
            final_cluster_labels = np.zeros(n_windows_created, dtype=int)
            cluster_trend_descriptions = {0: "Cluster 0: Insufficient data, default trend. Small variation."}
        elif best_k < 2 and n_windows_created >= 2:
            print(f"Warning: best_k determined as {best_k}, which is too low for clustering. Setting to 2.")
            best_k = 2

            final_model = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=25,
                                           random_state=random_state, verbose=0, n_jobs=n_jobs)
            final_cluster_labels = final_model.fit_predict(X_windows_np)
            cluster_trend_descriptions = analyze_cluster_trends(X_windows_np, final_cluster_labels, best_k,
                                                                analysis_params)
        else:
            print(f"\nOptimal K based on Silhouette Score: {best_k} (Score: {max_silhouette_score:.4f})")
            final_model = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", max_iter=25,
                                           random_state=random_state, verbose=0, n_jobs=n_jobs)
            final_cluster_labels = final_model.fit_predict(X_windows_np)
            print("Final clustering finished.")
            cluster_trend_descriptions = analyze_cluster_trends(X_windows_np, final_cluster_labels, best_k,
                                                                analysis_params)

        print("\n--- Generated Cluster Trend Descriptions ---")
        for cid, desc in cluster_trend_descriptions.items():
            print(desc)


        for i in range(n_windows_created):
            window_tensor = torch.from_numpy(X_windows_np[i]).float()
            assigned_cluster = final_cluster_labels[i]
            trend_text = cluster_trend_descriptions.get(assigned_cluster, "Trend not found for cluster.")
            self.items.append((window_tensor, trend_text))

        print(f"\nTimeSeriesTrendDataset initialized. Number of items: {len(self.items)}")

        if save_artifacts:
            if not os.path.exists(artifacts_dir):
                os.makedirs(artifacts_dir)


            if silhouette_scores and len(list(current_k_range)) == len(silhouette_scores):
                plt.figure(figsize=(8, 5))
                plt.plot(list(current_k_range), silhouette_scores, marker='o')
                plt.title('Silhouette Score for Different Values of K (DTW)')
                plt.xlabel('Number of Clusters (K)')
                plt.ylabel('Silhouette Score')
                plt.xticks(list(current_k_range))
                plt.grid(True)
                plt.savefig(os.path.join(artifacts_dir, "silhouette_scores_plot.png"))
                plt.close()


            if n_windows_created > 1:
                print("  Generating t-SNE plot...")
                perplexity_val = min(30.0, n_windows_created - 1.0)
                if n_windows_created <= perplexity_val: perplexity_val = max(5.0, n_windows_created - 1.0)
                if n_windows_created <= 5: perplexity_val = max(1.0, n_windows_created - 1.0)

                if perplexity_val > 0:
                    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=random_state,
                                metric="precomputed", n_iter=300, learning_rate='auto', init='random', n_jobs=n_jobs)
                    tsne_results = tsne.fit_transform(dtw_distance_matrix)

                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=final_cluster_labels,
                                          cmap=plt.get_cmap("tab10", best_k), s=60, alpha=0.8)
                    plt.title(f't-SNE visualization (Optimal K={best_k}, DTW distance)')
                    plt.xlabel('t-SNE Component 1');
                    plt.ylabel('t-SNE Component 2')
                    # Legend
                    handles = []
                    legend_labels = []
                    active_labels_viz = sorted(list(np.unique(final_cluster_labels)))
                    cmap_norm_factor_viz = (best_k - 1) if best_k > 1 else 1
                    for label_val in active_labels_viz:
                        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=plt.get_cmap("tab10", best_k)(
                                                      label_val / cmap_norm_factor_viz if cmap_norm_factor_viz > 0 else 0),
                                                  markersize=10))
                        legend_labels.append(f'Cluster {label_val}')
                    if handles: plt.legend(handles, legend_labels, title="Clusters")
                    plt.grid(True)
                    plt.savefig(os.path.join(artifacts_dir, "tsne_cluster_visualization.png"))
                    plt.close()
                else:
                    print("Skipping t-SNE due to insufficient samples for perplexity.")


            print("  Saving window tensors and trend descriptions to CSV...")
            window_trend_details_list_for_csv = []
            for i in range(len(self.items)):

                original_window_index = i
                _window_tensor_ignored, trend_desc_for_window = self.items[i]

                window_info_csv = {
                    'Original_Window_Index': original_window_index,

                    'Cluster_Trend_Description': trend_desc_for_window
                }
                window_trend_details_list_for_csv.append(window_info_csv)

            if window_trend_details_list_for_csv:
                window_trends_df_csv = pd.DataFrame(window_trend_details_list_for_csv)
                csv_filename_window_trends_ds = os.path.join(artifacts_dir, "dataset_window_trend_details.csv")
                window_trends_df_csv.to_csv(csv_filename_window_trends_ds, index=False)
                print(f"  Detailed window trend information saved to {csv_filename_window_trends_ds}")


            print(f"Artifacts saved to {artifacts_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# --- Example Usage ---
if __name__ == '__main__':

    dummy_data_rows = 200
    dummy_data_features = 5
    dummy_np_data = np.random.rand(dummy_data_rows, dummy_data_features)
    for i in range(dummy_data_features):
        if i % 3 == 0:
            dummy_np_data[:, i] += np.linspace(0, 5, dummy_data_rows)
        elif i % 3 == 1:
            dummy_np_data[:, i] += np.linspace(3, 0, dummy_data_rows)


    dummy_csv_path = "dummy_timeseries_data.csv"
    pd.DataFrame(dummy_np_data).to_csv(dummy_csv_path, header=False, index=False)
    print(f"Dummy CSV created at: {dummy_csv_path}")


    custom_analysis_params = {
        "prominence_factor": 0.15,
        "slope_threshold": 0.02,
        "amplitude_thresh_low": 0.5,
        "amplitude_thresh_mid": 1.0,
    }

    print("\n--- Creating Training Dataset ---")
    ts_dataset = TimeSeriesTrendDataset(
        csv_path=dummy_csv_path,
        win_size=25,
        k_range=(2, 6),
        analysis_params=custom_analysis_params,
        random_state=42,
        n_jobs=-1,
        save_artifacts=True,
        artifacts_dir="my_clustering_results"
    )


    if len(ts_dataset) > 0:
        print(f"\n--- Accessing items from TimeSeriesTrendDataset (first 3) ---")
        for i in range(min(3, len(ts_dataset))):
            window_tensor, trend_description = ts_dataset[i]
            print(f"\nItem {i}:")
            print(f"  Window Tensor Shape: {window_tensor.shape}")
            print(f"  Trend Description: {trend_description}")


        from torch.utils.data import DataLoader

        ts_dataloader = DataLoader(ts_dataset, batch_size=4, shuffle=True)

        print(f"\n--- Iterating through DataLoader (first batch) ---")
        for batch_idx, (window_batch, trend_batch) in enumerate(ts_dataloader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Window Batch Shape: {window_batch.shape}")
            print(f"  Trend Batch (list of strings, length): {len(trend_batch)}")

            if batch_idx == 0:
                break
    else:
        print("Dataset is empty, cannot demonstrate item access or DataLoader.")

    os.remove(dummy_csv_path)
    print(f"\nDummy CSV cleaned up: {dummy_csv_path}")
