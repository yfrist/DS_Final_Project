import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import os

# === Ensure communities_analysis folder exists ===
output_folder = 'communities_analysis'
os.makedirs(output_folder, exist_ok=True)

# Load your cleaned chess games CSV
df = pd.read_csv("data/cleaned_chess_games_prediction.csv")

# Only keep games with a clear winner (ignore draws/other outcomes)
df = df[df['winner'].isin(['White', 'Black'])]

# Build the directed player graph: loser → winner
G = nx.DiGraph()
for idx, row in df.iterrows():
    white = row['white_id']
    black = row['black_id']
    winner = row['winner']
    if winner == 'White':
        G.add_edge(black, white)  # black lost to white
    elif winner == 'Black':
        G.add_edge(white, black)  # white lost to black

# Compute centrality measures
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)

# Community detection (label propagation, using undirected version for community)
communities = nx.community.label_propagation_communities(G.to_undirected())
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

# Visualize the network (top N players by degree centrality)
N = 500  # Change this for more/fewer nodes
top_players = [x[0] for x in Counter(degree_centrality).most_common(N)]
subG = G.subgraph(top_players)

plt.figure(figsize=(14, 9))
pos = nx.spring_layout(subG, k=0.5, seed=42)
colors = [community_map.get(node, 0) for node in subG.nodes()]
nx.draw(subG, pos, with_labels=True, node_color=colors, cmap=plt.cm.tab20,
        node_size=500, edge_color="#aaa", arrowsize=15, font_size=9)
plt.title("Top Player Network with Detected Communities", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_player_network_communities.png"))
plt.close()

# Centrality rankings output table
centrality_df = pd.DataFrame({
    'player_id': list(degree_centrality.keys()),
    'degree_centrality': list(degree_centrality.values()),
    'pagerank': [pagerank.get(k, 0) for k in degree_centrality.keys()],
    'betweenness': [betweenness.get(k, 0) for k in degree_centrality.keys()],
    'community': [community_map.get(k, -1) for k in degree_centrality.keys()],
})
centrality_df = centrality_df.sort_values('pagerank', ascending=False)

# Show top 10 influential players
print(centrality_df.head(10))

# Save centrality rankings to CSV
centrality_df.to_csv(os.path.join(output_folder, "player_centrality_ranking.csv"), index=False)
