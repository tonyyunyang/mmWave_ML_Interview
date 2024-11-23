import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_signatures(signatures_path, min_n_time_points):
    signatures = []
    filenames = []
    
    for file in os.listdir(signatures_path):
        if not file.endswith(".npy"):
            continue
        
        signature = np.load(os.path.join(signatures_path, file))
        signature = signature[:min_n_time_points, :]
        signatures.append(signature)
        filenames.append(file)
    
    return np.array(signatures), filenames

def find_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters, silhouette_scores

def plot_results(data_2d, labels, scores, method_name, score_type='Silhouette', start_n_clusters=2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    n_clusters_range = range(start_n_clusters, len(scores) + start_n_clusters)
    ax1.plot(n_clusters_range, scores, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel(f'{score_type} Score')
    ax1.set_title(f'{method_name} Method:\n{score_type} Score vs Number of Clusters')
    ax1.grid(True)
    
    scatter = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    ax2.set_xlabel('First Component')
    ax2.set_ylabel('Second Component')
    ax2.set_title(f'{method_name} Method:\nSignature Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()