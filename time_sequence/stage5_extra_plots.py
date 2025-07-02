import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Output folder ===
output_dir = 'stage5_analysis'
os.makedirs(output_dir, exist_ok=True)

# === 2. Load data ===
df = pd.read_csv("data/chess_games.csv")  # change path if needed


# === 3. Group definitions and order for plots ===
def map_time_control(tc):
    try:
        mins, inc = tc.split("+")
        mins = int(mins)
        inc = int(inc)
    except:
        return "Unknown"
    total_time = mins + inc * 40 // 60  # Approximate total game time (incl. increment)
    if total_time <= 5:
        return "Bullet"
    elif total_time <= 10:
        return "Blitz"
    elif total_time <= 25:
        return "Rapid"
    elif total_time < 90:
        return "Classical"
    else:
        return "Other"


order = ["Blitz", "Rapid", "Classical", "Other"]

df['time_control_group'] = df['time_increment'].apply(map_time_control)

# === 4. WIN RATE BY TIME CONTROL GROUP ===
win_rate = df[df['winner'].isin(['White', 'Black', 'Draw'])].groupby(
    ['time_control_group', 'winner']).size().unstack().fillna(0)
win_rate = win_rate.div(win_rate.sum(axis=1), axis=0)
win_rate = win_rate.reindex(order)
win_rate.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#fff6b8', '#b8daff', '#bababa'])
plt.title('Win Rate by Time Control Group')
plt.xlabel('Time Control Group')
plt.ylabel('Fraction of Games')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'win_rate_by_time_control_group.png'))
plt.show()

# === 5. AVERAGE GAME LENGTH (TURNS) BY TIME CONTROL GROUP ===
avg_turns = df.groupby('time_control_group')['turns'].mean().reindex(order)
avg_turns.plot(kind='bar', color='slateblue')
plt.title('Average Game Length (Turns) by Time Control Group')
plt.xlabel('Time Control Group')
plt.ylabel('Average Turns')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'avg_game_length_by_time_control_group.png'))
plt.show()

# === 6. DISTRIBUTION OF GAME LENGTHS FOR TOP 3 TIME CONTROLS ===
popular_tc = df['time_increment'].value_counts().head(3).index
subset = df[df['time_increment'].isin(popular_tc)]
plt.figure(figsize=(10, 6))
sns.boxplot(data=subset, x='time_increment', y='turns')
plt.title('Game Length Distribution for Top 3 Time Controls')
plt.xlabel('Time Increment')
plt.ylabel('Turns')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'game_length_distribution_top3.png'))
plt.show()

# === 7. MOST POPULAR OPENINGS BY TIME CONTROL GROUP ===
if 'opening_fullname' in df.columns:
    for group in df['time_control_group'].unique():
        group_df = df[df['time_control_group'] == group]
        top_openings = group_df['opening_fullname'].value_counts().head(5)
        if not top_openings.empty:
            plt.figure(figsize=(8, 4))
            top_openings.plot(kind='bar')
            plt.title(f'Most Popular Openings in {group}')
            plt.xlabel('Opening')
            plt.ylabel('Games')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'popular_openings_{group}.png'))
            plt.close()

# === 8. OUTCOME TYPES BY TOP 10 TIME CONTROLS ===
top_n = df['time_increment'].value_counts().head(10).index
subset = df[df['time_increment'].isin(top_n)]
outcome_types = subset.groupby(['time_increment', 'victory_status']).size().unstack(fill_value=0)
outcome_types.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Outcome Types by Most Popular Time Controls')
plt.xlabel('Time Increment')
plt.ylabel('Number of Games')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'outcome_types_top10_timecontrols.png'))
plt.show()

print(f"✅ All plots saved in '{output_dir}/'")
