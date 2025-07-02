# ♟️ Online Chess Data Science Project

This repository contains an end-to-end analysis of online chess games using methods from data science and machine learning. Each module in the project is self-contained and can be run independently. This README provides instructions for running each section and locating the generated outputs.

For a full overview of the project’s goals, methodology, and findings, see the full report:
[**Online Chess Games: A Multi-Faceted Data Science Exploration (PDF)**](Online%20Chess%20Games_%20A%20Multi-Faceted%20Data%20Science%20Exploration.pdf)


---

## ⚙️ Setup

Before running any scripts, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 1. Statistics

**Script:**  
`statistics/code/chess_statistical_analysis.py`

**How to Run:**  
```bash
python statistics/code/chess_statistical_analysis.py
```

**Outputs:**  
- Plots: `statistics/plots/`  
- Summary tables: `statistics/data/`

---

## 2. Prediction

**Script:**  
`prediction/main.py`

**How to Run:**  
```bash
python prediction/main.py
```

**Outputs:**  
- Evaluation plots, metrics: `prediction/results/`  
- Trained models: `prediction/models/`  
- CatBoost logs: `prediction/catboost_info/`

---

## 3. Clustering

**Scripts:**  
- `clustering/cluster_games.py` (main)  
- `clustering/data_cleaner.py` (preprocessing)

**How to Run:**  
```bash
python clustering/cluster_games.py
```

**Outputs:**  
- Cluster visualizations and evaluation metrics: `clustering/plots/`  
- Cluster profiles and summaries: `clustering/data/`

---

## 4. Community Detection

**Scripts:**  
- `community/network_analysis_basic.py`  
- `community/network_analysis_advanced.py`  
- `community/data_cleaner.py`

**How to Run:**  
```bash
python community/network_analysis_basic.py
python community/network_analysis_advanced.py
```

**Outputs:**  
All results, plots, and centrality metrics are saved to:  
`community/communities_analysis/`

---

## 5. Time Sequence Analysis

**Scripts:**  
- `time_sequence/stage5_time_control_analysis.py`  
- `time_sequence/stage5_extra_plots.py`  
- `time_sequence/stage5_move_sequence_similarity.py`

**How to Run:**  
```bash
python time_sequence/stage5_time_control_analysis.py
python time_sequence/stage5_extra_plots.py
python time_sequence/stage5_move_sequence_similarity.py
```

**Outputs:**  
All visualizations and result files are saved in:  
`time_sequence/stage5_analysis/`

---

## 📁 Project Structure

<details>
<summary>Click to expand full file tree</summary>

```
/
├── clustering
│   ├── cluster_games.py
│   ├── data
│   │   ├── chess_games.csv
│   │   ├── cleaned_chess_games_clusters.csv
│   │   ├── hierarchical_cluster_annotated_summary.csv
│   │   ├── hierarchical_cluster_profiles.csv
│   │   ├── kmeans_cluster_annotated_summary.csv
│   │   └── kmeans_cluster_profiles.csv
│   ├── data_cleaner.py
│   └── plots
│       ├── combined_k_selection.png
│       ├── dendrogram.png
│       ├── dendrogram_truncated.png
│       ├── hierarchical_clusters_annotated.png
│       ├── hierarchical_tsne_refined.png
│       ├── kmeans_clusters_annotated.png
│       ├── kmeans_tsne_refined.png
│       ├── pca_preclustering.png
│       ├── similarity_matrix_ordered.png
│       ├── tsne_Hierarchical_boundaries.png
│       └── tsne_KMeans_boundaries.png
├── community
│   ├── communities_analysis
│   │   ├── centrality_vs_rating.csv
│   │   ├── degree_centrality_vs_rating.png
│   │   ├── longest_win_chain.txt
│   │   ├── opening_to_opening_transitions.png
│   │   ├── pagerank_vs_rating.png
│   │   ├── player_centrality_ranking.csv
│   │   ├── player_opening_bipartite.png
│   │   ├── temporal_network_2019.png
│   │   ├── temporal_network_2020.png
│   │   ├── temporal_network_2021.png
│   │   ├── temporal_network_2022.png
│   │   ├── top_player_network_communities.png
│   │   └── top_rivalries.png
│   ├── data
│   │   ├── chess_games.csv
│   │   └── cleaned_chess_games_prediction.csv
│   ├── data_cleaner.py
│   ├── network_analysis_advanced.py
│   ├── network_analysis_basic.py
│   └── Stage3_Explained.pdf
├── file_tree.py
├── prediction
│   ├── catboost_info/
│   ├── data/
│   ├── data_cleaner.py
│   ├── main.py
│   ├── models/
│   └── results/
├── requirements.txt
├── statistics
│   ├── code/
│   ├── data/
│   └── plots/
└── time_sequence
    ├── data/
    ├── stage5_analysis/
    ├── stage5_extra_plots.py
    ├── stage5_move_sequence_similarity.py
    └── stage5_time_control_analysis.py
```

</details>

---

## 📝 Notes

- Make sure the input CSVs are present in each module’s `data/` folder before running scripts.
- Figures are saved as `.png` for inclusion in slides or reports.
- Processed CSVs can be used for further analysis or visual inspection.

---
