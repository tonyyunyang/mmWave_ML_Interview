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


def analyze_clusters(clusters, save_to_file=True):
    """
    Print and optionally save detailed analysis of the clusters
    """
    analysis_text = []
    analysis_text.append("Detailed Cluster Analysis:\n")
    
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, files in sorted_clusters:
        cluster_info = [
            f"\nCluster {cluster_id}:",
            f"Number of samples: {len(files)}",
            "Files in this cluster:",
            "----------------"
        ]
        # Add all files in the cluster
        cluster_info.extend(sorted(files))  # Sort files for better readability
        cluster_info.append("----------------\n")
        
        # Join with newlines and add to main analysis
        analysis_text.extend(cluster_info)
    
    # Print to console
    print('\n'.join(analysis_text))
    
    # Optionally save to file
    if save_to_file:
        with open('cluster_analysis.txt', 'w') as f:
            f.write('\n'.join(analysis_text))
        print(f"\nDetailed analysis has been saved to 'cluster_analysis.txt'")
    
    return analysis_text

# Additional function to get files in a specific cluster
def get_cluster_files(clusters, cluster_id):
    """
    Get all files belonging to a specific cluster
    """
    if cluster_id in clusters:
        files = sorted(clusters[cluster_id])  # Sort for consistent output
        print(f"\nFiles in Cluster {cluster_id} ({len(files)} files):")
        print("----------------")
        for file in files:
            print(file)
        print("----------------")
    else:
        print(f"Cluster {cluster_id} not found")