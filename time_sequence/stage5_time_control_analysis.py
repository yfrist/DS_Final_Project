import pandas as pd
import matplotlib.pyplot as plt
import os

# === Ensure output directory exists ===
output_dir = 'stage5_analysis'
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv("data/chess_games.csv")


# === Utility: Map Time Control to Group ===
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


df['time_control_group'] = df['time_increment'].apply(map_time_control)

# === Plot 1: Grouped Outcomes by Time Control Group ===
grouped_counts = df.groupby(['time_control_group', 'victory_status']).size().unstack(fill_value=0)
order = ["Bullet", "Blitz", "Rapid", "Classical", "Other", "Unknown"]
grouped_counts = grouped_counts.reindex(order).dropna(how='all')  # Ensure fixed order, drop all-NaN rows

plt.figure(figsize=(8, 6))
grouped_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Game Outcomes by Time Control Group')
plt.xlabel('Time Control Group')
plt.ylabel('Number of Games')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_control_grouped_outcomes.png'))
plt.show()

# === Plot 2: Fraction "Out of Time" by Time Control Group ===
out_of_time_fraction = grouped_counts.apply(lambda x: x.get('Out of Time', 0) / x.sum() if x.sum() else 0, axis=1)
out_of_time_fraction.plot(kind='bar', color='steelblue')
plt.title('Fraction of "Out of Time" Games by Time Control Group')
plt.xlabel('Time Control Group')
plt.ylabel('Fraction Out of Time')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'out_of_time_fraction_grouped.png'))
plt.show()

# === Plot 3: Top 8 Most Popular Time Increments ===
top_n = df['time_increment'].value_counts().head(8).index
subset = df[df['time_increment'].isin(top_n)]
outcome_counts = subset.groupby(['time_increment', 'victory_status']).size().unstack(fill_value=0)

outcome_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Game Outcomes by Most Popular Time Controls')
plt.xlabel('Time Increment')
plt.ylabel('Number of Games')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_control_top8_outcomes.png'))
plt.show()
