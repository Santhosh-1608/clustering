import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# Simulating data for example purpose
np.random.seed(0)
X = np.random.rand(200, 2) * 100  # Replace this with your actual dataset

# KMeans Clustering
def kmeans_clustering(X, n_clusters=5, max_iter=300):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
    
    # Initial clusters (random initialization)
    initial_labels = kmeans.fit_predict(X)
    initial_centers = kmeans.cluster_centers_

    # Final clusters after convergence
    final_labels = kmeans.labels_
    final_centers = kmeans.cluster_centers_

    # Calculate error rate (inertia)
    error_rate = kmeans.inertia_

    return initial_labels, initial_centers, final_labels, final_centers, error_rate

# Agglomerative Clustering
def agglomerative_clustering(X, n_clusters=5):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(X)
    return labels

# KMeans Clustering Results
initial_labels, initial_centers, final_labels, final_centers, error_rate = kmeans_clustering(X)

# Agglomerative Clustering Results
hierarchical_labels = agglomerative_clustering(X)

# Plotting the clusters for KMeans
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', marker='o')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering Final Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plotting the clusters for Agglomerative Clustering
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap='viridis', marker='o')
plt.title('Agglomerative Clustering Final Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Plotting Dendrogram for Agglomerative Clustering
plt.figure(figsize=(16, 8))  # Adjusting the figure size for clarity
Z = linkage(X, method='ward')
dendrogram = shc.dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Distance", fontsize=12)
plt.title("Hierarchical Clustering Dendrogram (Truncated)")
plt.tight_layout()
plt.show()

# Print Cluster Information
print("KMeans Clustering:")
print("Initial Cluster Centers:\n", initial_centers)
print("Final Cluster Centers:\n", final_centers)
print("Error Rate (Inertia):", error_rate)

# Cluster Sizes for KMeans
kmeans_cluster_sizes = {i: sum(final_labels == i) for i in range(len(final_centers))}
print("\nCluster Sizes (KMeans Clustering):")
for cluster, size in kmeans_cluster_sizes.items():
    print(f"Cluster {cluster + 1}: {size} points")

# Cluster Sizes for Agglomerative Clustering
hierarchical_cluster_sizes = {i: sum(hierarchical_labels == i) for i in range(5)}
print("\nCluster Sizes (Agglomerative Clustering):")
for cluster, size in hierarchical_cluster_sizes.items():
    print(f"Cluster {cluster + 1}: {size} points")

# Cophenetic Correlation Coefficient for Hierarchical Clustering
cophenetic_corr, _ = shc.cophenet(Z, pdist(X))
print("Cophenetic Correlation Coefficient for Hierarchical Clustering:", cophenetic_corr)

# Silhouette Scores
kmeans_silhouette = silhouette_score(X, final_labels)
hierarchical_silhouette = silhouette_score(X, hierarchical_labels)

print("\nSilhouette Score (KMeans):", kmeans_silhouette)
print("Silhouette Score (Agglomerative Clustering):", hierarchical_silhouette)
