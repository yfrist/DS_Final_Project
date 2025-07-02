# ------------------- Imports & Config ------------------- #
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from math import ceil
from pathlib import Path
# ------------------- Imports & Config ------------------- #


# ------------------- Constants ------------------- #
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "chess_games.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "plots"
# ------------------- Constants ------------------- #


# ------------------- Helpers ------------------- #
def normalize_victory_status(series: pd.Series) -> pd.Series:
    """Standardize messy 'victory_status' strings to canonical forms."""
    canonical = {
        "draw": "Draw",
        "mate": "Mate",
        "out of time": "Out of Time",
        "resign": "Resign",
    }
    cleaned = (
        series.astype(str)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.lower()
              .map(canonical)
    )
    if cleaned.isna().any():
        bad = cleaned[cleaned.isna()].unique()
        raise ValueError(f"Unexpected victory_status values: {bad}")
    return cleaned


def save_plot(fig, name: str, folder: Path, save_figs=True):
    """Save or show a matplotlib figure."""
    if save_figs:
        folder.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_opening_fullname_distribution(df, eco_col='opening_fullname', top_n=10, save=False):
    full_count = df[eco_col].value_counts()
    full_top = full_count.nlargest(top_n)
    total_games = full_count.sum()

    colors = plt.cm.tab10.colors if top_n <= 10 else plt.cm.get_cmap('tab20', top_n).colors
    numbered_labels = [f"{i+1}. {name}" for i, name in enumerate(full_top.index)]
    legend_handles = [mpatches.Patch(color=colors[i], label=label) for i, label in enumerate(numbered_labels)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(1, top_n + 1), full_top.values, color=colors[:top_n])

    # Add percentage label inside each bar (smaller, black font)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height / full_top.sum()) * 100  # for full names
        ax.text(bar.get_x() + bar.get_width() / 2, height - height * 0.1, f"{pct:.1f}%",
                ha='center', va='bottom', fontsize=8, color='black')

    ax.set_title(f"Top {top_n} Full Opening Names")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Opening Index")
    ax.set_xticks(range(1, top_n + 1))
    ax.set_xticklabels([str(i) for i in range(1, top_n + 1)])

    ax.legend(handles=legend_handles, title="Opening Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save:
        save_plot(fig, "opening_fullname_dist", FIGURES_DIR, save)
    else:
        plt.show()


def plot_opening_shortname_distribution(df, short_col='opening_shortname', top_n=10, save=False):
    short_counts = df[short_col].value_counts()
    short_top = short_counts.nlargest(top_n)
    total_games = short_counts.sum()

    percentage = short_top.sum() / total_games * 100

    colors = plt.cm.tab10.colors if top_n <= 10 else plt.cm.get_cmap('tab20', top_n).colors
    numbered_labels = [f"{i+1}. {name}" for i, name in enumerate(short_top.index)]
    legend_handles = [mpatches.Patch(color=colors[i], label=label) for i, label in enumerate(numbered_labels)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(1, top_n + 1), short_top.values, color=colors[:top_n])

    # Add percentage label inside each bar (smaller, black font)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height / short_top.sum()) * 100  # for full names
        ax.text(bar.get_x() + bar.get_width() / 2, height - height * 0.1, f"{pct:.1f}%",
                ha='center', va='bottom', fontsize=8, color='black')

    ax.set_title(f"Top {top_n} Opening Short Names")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Opening Index")
    ax.set_xticks(range(1, top_n + 1))
    ax.set_xticklabels([str(i) for i in range(1, top_n + 1)])

    ax.legend(handles=legend_handles, title="Short Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save:
        save_plot(fig, "opening_shortname_dist", FIGURES_DIR, save)
    else:
        plt.show()


def opening_response(df, shortname_col='opening_shortname', response_col='opening_response', save_figs=True):
    """
    Plots a grid of pie charts showing response frequencies for each opening short name that has responses.
    """
    # ... (previous code remains the same until the end) ...
    
    response_colors = {
        'Accepted': '#4CAF50',  # green
        'Declined': '#FF9800',  # orange
        'Refused': '#F44336',  # red
    }


    # Filter for games with non-null response and group by shortname
    filtered_df = df[[shortname_col, response_col]].dropna()
    openings_with_responses = filtered_df.groupby(shortname_col)[response_col].value_counts(normalize=True).unstack().fillna(0)

    n_openings = len(openings_with_responses)
    cols = 4
    rows = ceil(n_openings / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (opening, row) in enumerate(openings_with_responses.iterrows()):
        labels = row.index
        sizes = row.values

        # Filter out 0% responses
        nonzero_mask = sizes > 0
        labels = labels[nonzero_mask]
        sizes = sizes[nonzero_mask]

        colors = [response_colors[label] for label in labels]
        axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(opening, fontsize=10)

    # Hide any extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Opening Response Distribution by Short Opening Name", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_figs:
        save_path = FIGURES_DIR / "opening_responses.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
# ------------------- Helpers ------------------- #


# ------------------- Methods ------------------- #
def victory_status(save_figs=True):
    vict_cnt = df["victory_status"].value_counts().sort_index()
    vict_pct = (vict_cnt / len(df) * 100).round(2)

    if save_figs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        vict_pct.rename("percent").to_csv(OUTPUT_DIR / "victory_status_percent.csv", header=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(vict_cnt, labels=vict_cnt.index, autopct="%1.1f%%")
    ax.set_title("Game Outcomes by Victory Status")
    save_plot(fig, "victory_status", FIGURES_DIR, save_figs)


def wins_by_colour(save_figs=True):
    decisive = df[df["winner"].isin(["White", "Black"])]
    wcnt = decisive["winner"].value_counts().sort_index()
    wpct = (wcnt / len(decisive) * 100).round(2)

    if save_figs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        wpct.rename("percent").to_csv(OUTPUT_DIR / "winner_colour_percent.csv", header=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(wcnt, labels=wcnt.index, autopct="%1.1f%%")
    ax.set_title("Winner Colour – Decisive Games")
    save_plot(fig, "winner_colour", FIGURES_DIR, save_figs)


def game_length(save_figs=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["turns"], bins=30, edgecolor="black")
    ax.set(title="Distribution of Game Turns", xlabel="Turns", ylabel="Frequency")
    save_plot(fig, "games_length_hist", FIGURES_DIR, save_figs)
    
    df["game_type"] = df["victory_status"].apply(lambda x: "Draw" if x == "Draw" else "Decisive")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot([
        df[df["game_type"] == "Draw"]["turns"],
        df[df["game_type"] == "Decisive"]["turns"]
    ], tick_labels=["Draws", "Decisive"])
    ax.set_title("Game Length by Outcome")
    ax.set_ylabel("Turns")
    save_plot(fig, "turns_boxplot", FIGURES_DIR, save_figs)


def opening_moves(save_figs=True):
    df['opening_fullname'] = df['opening_fullname'].str.strip().str.capitalize()
    df['opening_shortname'] = df['opening_shortname'].str.strip().str.capitalize()
    plot_opening_fullname_distribution(df, top_n=10, save=save_figs)
    plot_opening_shortname_distribution(df, top_n=10, save=save_figs)
# ------------------- Methods ------------------- #


if __name__ == "__main__":
    """
    analyze_chess_games.py
    -----------------------
    Exploratory data analysis of chess game records:
        - Victory status distribution
        - Winner color breakdown
        - Game length stats (histogram + boxplot)
        - Most popular openings (full and short names)
        - Opening response frequencies

    Outputs:
        • Figures in 'figures/'
        • Summary tables in 'output/'
    """
    # ------------------- Load & Clean Data ------------------- #
    df = pd.read_csv(DATA_FILE)
    df["victory_status"] = normalize_victory_status(df["victory_status"])
    
    # if save_figs == True figs are saved, else they are shown
    save_figs = True
    
    # ------------------- 1. Victory Status ------------------- #
    victory_status(save_figs)

    # ------------------- 2. Winner Color (Decisive Only) ------------------- #
    wins_by_colour(save_figs)

    # ------------------- 3. Game Length Distribution ------------------- #
    game_length(save_figs)

    # ------------------- 4. Opening Popularity ------------------- #
    opening_moves(save_figs)

    # ------------------- 5. Opening Response Frequency ------------------- #
    opening_response(df, save_figs=save_figs)

    # ------------------- Completion Message ------------------- #
    if save_figs:
        print("\nAll figures saved to:", FIGURES_DIR.resolve())
        print("All tables saved to:", OUTPUT_DIR.resolve())
