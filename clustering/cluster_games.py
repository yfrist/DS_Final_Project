import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage, dendrogram


def load_data(file_path):
    """
    Load CSV data into a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def scale_data(df):
    """
    Standardize all columns in the DataFrame.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    return pd.DataFrame(scaled_array, columns=df.columns)


def log_transform(df, exclude_cols=[]):
    df_log = df.copy()
    for col in df.columns:
        if col not in exclude_cols and (df[col] > 0).all():
            df_log[col] = np.log1p(df[col])
    return df_log


def overlay_kde_background(ax, data, x='pca1', y='pca2'):
    sns.kdeplot(
        data=data, x=x, y=y,
        ax=ax, fill=True, cmap='Blues', thresh=0.05, alpha=0.5, bw_adjust=0.8
    )


def refined_plot_clusters(df, cluster_col, title, filename, max_points_per_cluster=None):
    if max_points_per_cluster is not None:
        df = df.groupby(cluster_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_points_per_cluster), random_state=42)
        )

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    overlay_kde_background(ax, df)

    sns.scatterplot(
        data=df,
        x='pca1',
        y='pca2',
        hue=cluster_col,
        palette='tab10',
        s=70,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7,
        ax=ax
    )

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("PCA1", fontsize=13)
    plt.ylabel("PCA2", fontsize=13)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.show()


def perform_kmeans(df_scaled, k=None):
    """
    Apply K-Means clustering and return cluster assignments.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(df_scaled)


def perform_hierarchical(df_scaled, k=None):
    """
    Apply Agglomerative clustering and return cluster assignments.
    """
    hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
    return hier.fit_predict(df_scaled)


def apply_pca(df_scaled):
    """
    Reduce to 2D using PCA.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    return pd.DataFrame(pca_result, columns=['pca1', 'pca2'])


def plot_dendrogram(df_scaled, num_clusters=4):
    """
    Generate and save a dendrogram from hierarchical clustering.
    """
    plt.figure(figsize=(12, 8))
    linked = linkage(df_scaled, method='ward')
    dendrogram(
        linked,
        truncate_mode=None,
        color_threshold=linked[-(num_clusters - 1), 2],  # cut at the level that gives `num_clusters` clusters
        show_leaf_counts=False,
        no_labels=True
    )
    plt.title("Clustering Dendrogram", fontsize=16, weight='bold')
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("plots/dendrogram.png")
    plt.show()


def plot_truncated_dendrogram(df_scaled, p=30):
    plt.figure(figsize=(12, 6))
    linked = linkage(df_scaled, method='ward')
    dendrogram(
        linked,
        truncate_mode='lastp',
        p=p,
        show_leaf_counts=True,
        color_threshold=linked[-7, 2]  # assume 8 clusters
    )
    plt.title(f"Truncated Dendrogram (Last {p} Clusters)", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig("plots/dendrogram_truncated.png")
    plt.show()


def describe_clusters(df, cluster_col):
    """
    Print and save the mean of each feature per cluster.
    """
    numeric_df = df.select_dtypes(include='number')
    cluster_profile = numeric_df.groupby(df[cluster_col]).mean().round(2)

    # Print to console
    print(f"\n=== Cluster Profiles for {cluster_col} ===")
    print(cluster_profile)

    # Save to file
    output_path = f"data/{cluster_col}_profiles.csv"
    try:
        cluster_profile.to_csv(output_path)
        print(f"Cluster profile saved to {output_path}")
    except Exception as e:
        print(f"Error saving cluster profile: {e}")


def ensure_output_dirs():
    """
    Ensure necessary output folders exist.
    """
    os.makedirs("plots", exist_ok=True)


def annotated_pca_plot(df, method_label="KMeans"):
    """
    Plot PCA scatter with annotated cluster labels using same style as refined_plot_clusters.
    """

    # df['cluster_label'] = df['kmeans_cluster'].map(cluster_labels)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    overlay_kde_background(ax, df, x='pca1', y='pca2')  # Same background as refined_plot_clusters

    sns.scatterplot(
        data=df,
        x='pca1',
        y='pca2',
        hue='cluster_label',
        palette='tab10',  # match style
        s=70,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7,
        ax=ax
    )

    plt.title(f"{method_label} Clustering of Chess Games (Annotated)", fontsize=16, weight='bold')
    plt.xlabel("PCA1", fontsize=13)
    plt.ylabel("PCA2", fontsize=13)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plots/{method_label.lower()}_clusters_annotated.png", dpi=300)
    plt.show()


cluster_labels_kmeans = {
    0: "Balanced, Well-Developed Games",
    1: "Quick Decisive Blitzes",
    2: "Long Incremental Imbalanced Games",
    3: "Stretched Positional Battles"
}
cluster_labels_hier = {
    0: "Standard Mixed-Pace Games",
    1: "Drawn-Out, Imbalanced Blitzes",
    2: "Tactical Openings, Stable Tempo",
    3: "Quick Clean Wins or Errors"
}

def choose_optimal_k(df_scaled, max_k=10):
    """
    Compare silhouette scores and inertia to suggest an optimal k.
    """
    silhouette_scores = []
    inertias = []
    ks = range(2, max_k+1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        silhouette_scores.append(silhouette_score(df_scaled, labels))
        inertias.append(kmeans.inertia_)

    # Plot both together
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia', color='tab:blue')
    ax1.plot(ks, inertias, 'bo-', label='Inertia')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette Score', color='tab:green')
    ax2.plot(ks, silhouette_scores, 'go-', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('Elbow and Silhouette Method Combined', fontsize=16, weight='bold')
    plt.grid(True)
    fig.tight_layout()
    plt.savefig("plots/combined_k_selection.png")
    plt.show()

    best_k = ks[silhouette_scores.index(max(silhouette_scores))]
    print(f"Suggested k based on silhouette score: {best_k}")
    return best_k


def plot_ordered_similarity(df_scaled, labels, sample_size=500):
    if len(df_scaled) > sample_size:
        sampled_indices = df_scaled.sample(sample_size, random_state=42).index
        df_scaled = df_scaled.loc[sampled_indices]
        labels = labels[sampled_indices]

    # Sort by cluster label
    sorted_idx = np.argsort(labels)
    ordered_data = df_scaled.iloc[sorted_idx]
    similarity = cosine_similarity(ordered_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title("Cosine Similarity Matrix (Cluster Ordered)", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig("plots/similarity_matrix_ordered.png")
    plt.show()


def compare_clusterings(labels1, labels2):
    ari = adjusted_rand_score(labels1, labels2)
    print(f"Adjusted Rand Index between KMeans and Hierarchical: {ari:.4f}")
    return ari


def apply_tsne(df_scaled):
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    tsne_result = tsne.fit_transform(df_scaled)
    return pd.DataFrame(tsne_result, columns=['tsne1', 'tsne2'])


def remove_outliers_tsne(df, columns=['tsne1', 'tsne2'], contamination=0.01):
    """
    Use Isolation Forest to remove outliers in t-SNE space.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    is_inlier = clf.fit_predict(df[columns]) == 1
    return df[is_inlier].reset_index(drop=True)


def plot_tsne_decision_boundaries(df, cluster_col='kmeans_cluster', method_label='KMeans'):
    """
    Plot t-SNE space with KNN-based decision boundaries.
    """
    X = df[['tsne1', 'tsne2']].values
    y = df[cluster_col].values

    # Fit KNN classifier
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X, y)

    # Create mesh grid
    h = 0.5  # step size
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict over mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(sns.color_palette("tab10", len(np.unique(y))))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # Overlay scatter points
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=y,
        palette='tab10', edgecolor='black', alpha=0.8, s=60
    )

    plt.title(f"t-SNE with {method_label} Decision Boundaries", fontsize=16, weight='bold')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plots/tsne_{method_label}_boundaries.png", dpi=300)
    plt.show()


def describe_clusters_annotated(df, cluster_col):
    stats = df.groupby(cluster_col).agg(['mean', 'median', 'std', 'count'])
    print(f"\n=== Annotated Summary for {cluster_col} ===")
    print(stats.round(2))
    stats.to_csv(f"data/{cluster_col}_annotated_summary.csv")
    return stats


def main():
    ensure_output_dirs()

    file_path = 'data/cleaned_chess_games_clusters.csv'
    df_raw = load_data(file_path)
    if df_raw is None:
        return

    # Apply log transform before scaling
    df_log = log_transform(df_raw)
    df_scaled = scale_data(df_log)

    # Visualize raw PCA before clustering/tsne
    df_pca = apply_pca(df_scaled)
    refined_plot_clusters(df_pca, cluster_col=None, title="PCA Projection (No Clusters)",
                          filename="pca_preclustering.png")

    # t-SNE projection
    tsne_df = apply_tsne(df_scaled)
    df_tsne = pd.DataFrame(tsne_df, columns=['tsne1', 'tsne2'])

    # Find optimal k using t-SNE space
    best_k = choose_optimal_k(df_tsne, max_k=10)

    # Apply clustering on t-SNE space
    df_tsne['kmeans_cluster'] = perform_kmeans(df_tsne, k=best_k)
    df_tsne['hierarchical_cluster'] = perform_hierarchical(df_tsne, k=best_k)

    # Evaluate clustering agreement
    compare_clusterings(df_tsne['kmeans_cluster'], df_tsne['hierarchical_cluster'])

    # Plot KMeans decision boundaries
    df_clean_k = remove_outliers_tsne(df_tsne[['tsne1', 'tsne2', 'kmeans_cluster']], columns=['tsne1', 'tsne2'], contamination=0.01)
    plot_tsne_decision_boundaries(df_clean_k, cluster_col='kmeans_cluster', method_label='KMeans')

    # Plot Hierarchical decision boundaries
    df_clean_h = df_tsne[['tsne1', 'tsne2', 'hierarchical_cluster']].copy()
    df_clean_h = remove_outliers_tsne(df_clean_h, columns=['tsne1', 'tsne2'], contamination=0.01)
    df_clean_h_renamed = df_clean_h.rename(columns={'hierarchical_cluster': 'kmeans_cluster'})
    plot_tsne_decision_boundaries(df_clean_h_renamed, cluster_col='kmeans_cluster', method_label='Hierarchical')

    # Refined scatter with KDE
    df_plot = df_tsne.rename(columns={'tsne1': 'pca1', 'tsne2': 'pca2'})
    refined_plot_clusters(df_plot, 'kmeans_cluster', "K-Means on t-SNE (Refined)", "kmeans_tsne_refined.png", max_points_per_cluster=300)
    refined_plot_clusters(df_plot, 'hierarchical_cluster', "Hierarchical on t-SNE (Refined)", "hierarchical_tsne_refined.png", max_points_per_cluster=300)

    # Plot dendrograms from t-SNE space
    plot_dendrogram(df_tsne[['tsne1', 'tsne2']], num_clusters=best_k)
    plot_truncated_dendrogram(df_tsne[['tsne1', 'tsne2']], p=30)

    # Plot similarity matrix (KMeans)
    plot_ordered_similarity(df_tsne[['tsne1', 'tsne2']], df_tsne['kmeans_cluster'], sample_size=500)

    # Annotated K-Means and Hierarchical plots
    df_plot['cluster_label'] = df_tsne['kmeans_cluster'].map(cluster_labels_kmeans)
    annotated_pca_plot(df_plot, method_label="KMeans")
    df_plot['cluster_label'] = df_tsne['hierarchical_cluster'].map(cluster_labels_hier)
    annotated_pca_plot(df_plot, method_label="Hierarchical")

    # Cluster profiles from original data
    df_raw['kmeans_cluster'] = df_tsne['kmeans_cluster']
    df_raw['hierarchical_cluster'] = df_tsne['hierarchical_cluster']
    describe_clusters(df_raw, 'kmeans_cluster')
    describe_clusters(df_raw, 'hierarchical_cluster')

    # Annotated summaries
    describe_clusters_annotated(df_raw, 'kmeans_cluster')
    describe_clusters_annotated(df_raw, 'hierarchical_cluster')

if __name__ == "__main__":
    main()
