import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(data):
    """
    Clean the data by removing duplicates, handling missing values, and formatting columns.
    """
    # Drop duplicates based on game_id
    data = data.drop_duplicates(subset=['game_id'])

    # Drop rows where critical columns are missing
    critical_columns = ['game_id', 'winner']
    data = data.dropna(subset=critical_columns)

    # Clean and preprocess each column
    # game_id: Ensure integer and drop rows with invalid values
    data['game_id'] = pd.to_numeric(data['game_id'], errors='coerce')
    data = data.dropna(subset=['game_id'])
    data['game_id'] = data['game_id'].astype(int)

    # rated: Convert to binary (0: False, 1: True)
    data['rated'] = data['rated'].astype(bool).astype(int)

    # turns: Convert to integer and handle missing values
    data['turns'] = pd.to_numeric(data['turns'], errors='coerce').fillna(data['turns'].median()).astype(int)

    # victory_status: Convert to categorical, fill missing with 'Unknown'
    data['victory_status'] = data['victory_status'].fillna('Unknown').astype('category')

    # winner: Convert to categorical
    data['winner'] = data['winner'].astype('category')

    # Split time increment and fill missing values in one step
    time_df = data['time_increment'].str.split('+', expand=True).astype(float, errors='ignore')
    time_df = time_df.fillna(0)
    time_df.columns = ['base_time', 'increment']
    data[['base_time', 'increment']] = time_df

    # white_rating and black_rating: Convert to integer and handle missing values
    data['white_rating'] = pd.to_numeric(data['white_rating'], errors='coerce').fillna(data['white_rating'].median()).astype(int)
    data['black_rating'] = pd.to_numeric(data['black_rating'], errors='coerce').fillna(data['black_rating'].median()).astype(int)

    # Feature Engineering: Rating difference
    data['rating_diff'] = data['white_rating'] - data['black_rating']

    # opening_code: Standardize format and handle missing values
    data['opening_code'] = data['opening_code'].fillna('Unknown').str.upper()

    # Frequency Encoding for opening_code
    opening_freq = data['opening_code'].value_counts(normalize=True)
    data['opening_freq'] = data['opening_code'].map(opening_freq)

    # Target Encoding for opening_code based on win rate (fraction of white wins)
    win_rate = data.groupby('opening_code')['winner'].apply(lambda x: (x == 'White').mean())
    data['opening_win_rate'] = data['opening_code'].map(win_rate)
    data['opening_win_rate'] = data['opening_win_rate'].fillna(data['opening_win_rate'].mean())


    # Drop the original opening_code column
    data = data.drop(columns=['opening_code'])

    # opening_moves: Convert to integer and fill missing with median
    data['opening_moves'] = pd.to_numeric(data['opening_moves'], errors='coerce').fillna(data['opening_moves'].median()).astype(int)

    # Drop non-predictive or redundant columns
    data = data.drop(columns=['moves', 'opening_fullname',
                              'opening_shortname', 'opening_response', 'opening_variation', 'time_increment'])

    print("Data cleaned and preprocessed for prediction successfully.")
    return data

def filter_data(data):
    """
    Filter the data to include only relevant columns.
    """
    relevant_columns = ['game_id', 'rated', 'turns', 'victory_status', 'winner',
                        'base_time', 'increment', 'white_rating', 'black_rating', 'white_id', 'black_id',
                        'rating_diff', 'opening_moves', 'opening_freq', 'opening_win_rate']
    data = data[relevant_columns]
    print("Data filtered successfully for prediction.")
    return data

def save_data(data, output_path):
    """
    Save the cleaned data to a new CSV file.
    """
    try:
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Load data
    file_path = "data/chess_games.csv"
    data = load_data(file_path)

    if data is not None:
        # Clean data
        cleaned_data = clean_data(data)

        # Filter data
        filtered_data = filter_data(cleaned_data)

        # Save cleaned data
        save_data(filtered_data, "data/cleaned_chess_games_prediction.csv")
    else:
        print("No data to process.")
