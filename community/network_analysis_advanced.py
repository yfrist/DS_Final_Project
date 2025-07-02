import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
from networkx.algorithms import bipartite
from networkx.algorithms.components import strongly_connected_components


# === Ensure communities_analysis folder exists ===
output_folder = 'communities_analysis'
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv("data/cleaned_chess_games_prediction.csv")
df = df[df['winner'].isin(['White', 'Black'])]

# ===== 1. TEMPORAL NETWORK EVOLUTION =====
if 'year' not in df.columns:
    num_periods = 4
    df['year'] = pd.qcut(df['game_id'], num_periods, labels=[2019, 2020, 2021, 2022])

for period in sorted(df['year'].unique()):
    dfi = df[df['year'] == period]
    G = nx.DiGraph()
    for _, row in dfi.iterrows():
        if row['winner'] == 'White':
            G.add_edge(row['black_id'], row['white_id'])
        else:
            G.add_edge(row['white_id'], row['black_id'])
    if len(G) > 0:
        largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
        subG = G.subgraph(largest_cc)
        plt.figure(figsize=(9, 7))
        pos = nx.spring_layout(subG, k=0.6, seed=42)
        nx.draw(subG, pos, node_size=120, with_labels=False, edge_color="#999", alpha=0.7)
        title = f"Player Network, Year {period}"
        plt.title(title, fontsize=16)
        plt.tight_layout()
        filename = os.path.join(output_folder, f"temporal_network_{period}.png")
        plt.savefig(filename)
        plt.close()

# ===== 2. EDGE WEIGHTS: RIVALRIES & INTENSITY =====
rivalry_graph = nx.DiGraph()
for _, row in df.iterrows():
    if row['winner'] == 'White':
        u, v = row['black_id'], row['white_id']
    else:
        u, v = row['white_id'], row['black_id']
    if rivalry_graph.has_edge(u, v):
        rivalry_graph[u][v]['weight'] += 1
    else:
        rivalry_graph.add_edge(u, v, weight=1)

edge_weights = nx.get_edge_attributes(rivalry_graph, 'weight')
N = 20
top_edges = [edge for edge, _ in sorted(edge_weights.items(), key=lambda x: -x[1])[:N]]
top_nodes = set([u for u, v in top_edges] + [v for u, v in top_edges])
subG = rivalry_graph.subgraph(top_nodes)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(subG, k=0.6, seed=99)
weights = [subG[u][v]['weight'] * 0.7 for u, v in subG.edges()]
nx.draw(subG, pos, with_labels=True, node_size=500, width=weights, edge_color='b', arrowsize=15)
title = "Top Rivalries (Edge Thickness = Games Played)"
plt.title(title, fontsize=16)
plt.tight_layout()
filename = os.path.join(output_folder, "top_rivalries.png")
plt.savefig(filename)
plt.close()

# ===== 3. WIN STREAKS & INFLUENCE CHAINS =====
# (No plot, just text output. If you want a plot, let me know!)

# Check for cycles
if nx.is_directed_acyclic_graph(rivalry_graph):
    longest_path = nx.dag_longest_path(rivalry_graph)
    with open(os.path.join(output_folder, "longest_win_chain.txt"), "w") as f:
        f.write(" -> ".join(longest_path) + f"\nChain length: {len(longest_path)}\n")
else:
    # Instead, report the largest strongly connected component (largest group of mutually reachable players)
    scc = max(strongly_connected_components(rivalry_graph), key=len)
    with open(os.path.join(output_folder, "longest_win_chain.txt"), "w") as f:
        f.write("Graph contains cycles, so no DAG longest path.\n")
        f.write(f"Largest strongly connected component (group of players all connected by paths):\n")
        f.write(", ".join(scc) + f"\nSize: {len(scc)}\n")

# ===== 4. NETWORK METRICS VS. RATING =====
pagerank = nx.pagerank(rivalry_graph)
degree_cent = nx.degree_centrality(rivalry_graph)
ratings = defaultdict(list)
for _, row in df.iterrows():
    ratings[row['white_id']].append(row['white_rating'])
    ratings[row['black_id']].append(row['black_rating'])
avg_rating = {k: sum(v) / len(v) for k, v in ratings.items()}

centrality_rating_df = pd.DataFrame({
    'player_id': list(pagerank.keys()),
    'pagerank': [pagerank[k] for k in pagerank.keys()],
    'degree_centrality': [degree_cent[k] for k in pagerank.keys()],
    'avg_rating': [avg_rating.get(k, 0) for k in pagerank.keys()]
})

plt.figure(figsize=(7, 5))
plt.scatter(centrality_rating_df['avg_rating'], centrality_rating_df['pagerank'], alpha=0.6)
plt.xlabel("Average Player Rating")
plt.ylabel("PageRank Centrality")
plt.title("Is High Rating Related to Influence?", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "pagerank_vs_rating.png"))
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(centrality_rating_df['avg_rating'], centrality_rating_df['degree_centrality'], alpha=0.6)
plt.xlabel("Average Player Rating")
plt.ylabel("Degree Centrality")
plt.title("Is High Rating Related to Network Degree?", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "degree_centrality_vs_rating.png"))
plt.close()

centrality_rating_df.to_csv(os.path.join(output_folder, "centrality_vs_rating.csv"), index=False)

# ===== 5. PLAYER–OPENING BIPARTITE NETWORK =====

bipartite_G = nx.Graph()
player_nodes = set()
opening_nodes = set()

for _, row in df.iterrows():
    # Use a string as an opening ID (customize if you have a better name/code)
    opening_id = f"moves:{row['opening_moves']}_freq:{row['opening_freq']:.3f}"
    # Add edges for both white and black player to this opening
    bipartite_G.add_edge(row['white_id'], opening_id)
    bipartite_G.add_edge(row['black_id'], opening_id)
    player_nodes.add(row['white_id'])
    player_nodes.add(row['black_id'])
    opening_nodes.add(opening_id)

# Draw a bipartite network (show only top N players for readability)
N = 30
top_players = [p for p, _ in Counter([n for n in player_nodes]).most_common(N)]
subG = bipartite_G.subgraph(top_players + list(opening_nodes)[:N])

plt.figure(figsize=(13, 8))
pos = nx.spring_layout(subG, k=0.8, seed=42)
color_map = ['#1f78b4' if n in top_players else '#33a02c' for n in subG.nodes()]
nx.draw(subG, pos, node_size=280, with_labels=False, node_color=color_map, alpha=0.8)
# Make player nodes slightly larger and labeled
nx.draw_networkx_labels(subG, pos, labels={n: n for n in top_players}, font_size=8, font_color='black')
plt.title("Player–Opening Bipartite Network", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "player_opening_bipartite.png"))
plt.close()

# ===== 6. OPENING-TO-OPENING TRANSITION NETWORK =====
# We'll use consecutive games by each player, and connect opening_id of game N → opening_id of game N+1
# (You can use a more meaningful opening name/code if available)
transition_graph = nx.DiGraph()
df_sorted = df.sort_values(by=['white_id', 'game_id'])  # assuming 'game_id' is chronological

for player in pd.unique(df['white_id'].tolist() + df['black_id'].tolist()):
    # Find all games for this player as white
    player_games = df[df['white_id'] == player].sort_values('game_id')
    prev_opening = None
    for _, row in player_games.iterrows():
        opening_id = f"moves:{row['opening_moves']}_freq:{row['opening_freq']:.3f}"
        if prev_opening is not None:
            if transition_graph.has_edge(prev_opening, opening_id):
                transition_graph[prev_opening][opening_id]['weight'] += 1
            else:
                transition_graph.add_edge(prev_opening, opening_id, weight=1)
        prev_opening = opening_id

# Show top N most frequent transitions
edge_weights = nx.get_edge_attributes(transition_graph, 'weight')
N = 30
top_edges = sorted(edge_weights.items(), key=lambda x: -x[1])[:N]
top_nodes = set()
for (u, v), _ in top_edges:
    top_nodes.add(u)
    top_nodes.add(v)
subG = transition_graph.subgraph(top_nodes)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(subG, k=0.8, seed=44)
weights = [subG[u][v]['weight'] * 0.7 for u, v in subG.edges()]
nx.draw(subG, pos, with_labels=True, node_size=600, width=weights, edge_color='#d62728', arrowsize=13, alpha=0.8)
plt.title("Opening-to-Opening Transition Network (White Games)", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "opening_to_opening_transitions.png"))
plt.close()
