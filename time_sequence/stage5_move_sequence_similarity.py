import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# === 1. Ensure output directory exists ===
output_dir = 'stage5_analysis'
os.makedirs(output_dir, exist_ok=True)

# === 2. Load data ===
df = pd.read_csv("data/chess_games.csv")

# === 3. Pick move column ===
move_col = 'moves' if 'moves' in df.columns else 'opening_moves'
print(f"Using column '{move_col}' for move sequences.")

# === 4. Clean & prep ===
df = df[df[move_col].notna()]
df[move_col] = df[move_col].astype(str).str.strip()
df = df[df[move_col] != '']
df['opening_sequence'] = df[move_col].str.split().str[:8].str.join(' ')
sequences = df['opening_sequence']
print(f"Sequences to vectorize: {len(sequences)}")

# === 5. TF-IDF on char n-grams ===
vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000)
X = vec.fit_transform(sequences)

# === 6. Dimensionality reduction ===
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)

# === 7. Fast clustering ===
n_clusters = 5
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
labels = kmeans.fit_predict(X_reduced)
df['opening_cluster'] = labels

# === 8. Save top-5 sequences per cluster ===
sample = (
    df.groupby('opening_cluster')['opening_sequence']
      .apply(lambda s: s.value_counts().head(5))
      .reset_index(name='count')
)
sample.to_csv(os.path.join(output_dir, "opening_clusters_sample.csv"), index=False)

# === 9. Plot cluster sizes ===
counts = df['opening_cluster'].value_counts().sort_index()
plt.figure(figsize=(8,5))
counts.plot(kind='bar')
plt.title('Number of Games per Opening Cluster')
plt.xlabel('Cluster')
plt.ylabel('Games')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'opening_cluster_counts.png'))
plt.show()

print(f"✅ Done! Results saved in '{output_dir}/'.")
