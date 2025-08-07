import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict
from dotenv import load_dotenv

# --- Elo Calculation Logic ---

K_FACTOR = 32  # A standard K-factor for Elo calculations
DEFAULT_ELO = 1500 # Starting Elo for all players

def get_expected_score(player_elo, opponent_elo):
    """Calculates the expected score for a player based on Elo ratings."""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

def update_elo(winner_elo, loser_elo):
    """Updates the Elo ratings for a winner and loser."""
    expected_win = get_expected_score(winner_elo, loser_elo)
    expected_loss = get_expected_score(loser_elo, winner_elo)

    new_winner_elo = winner_elo + K_FACTOR * (1 - expected_win)
    new_loser_elo = loser_elo + K_FACTOR * (0 - expected_loss)
    
    return new_winner_elo, new_loser_elo

# --- Main Script Logic ---

def calculate_surface_elo(data_path, output_path):
    """
    Calculates surface-specific Elo ratings for players and adds them to the dataset.

    Args:
        data_path (str): Path to the combined, raw CSV data.
        output_path (str): Path to save the new CSV with Elo features.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # Ensure data is sorted chronologically, which is essential for Elo calculation
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    print("Data sorted by date.")

    # Initialize Elo ratings
    player_elos = defaultdict(lambda: {'Hard': DEFAULT_ELO, 'Clay': DEFAULT_ELO, 'Grass': DEFAULT_ELO})

    # Create new columns to store the Elo ratings at the time of the match
    df['winner_surface_elo'] = np.nan
    df['loser_surface_elo'] = np.nan

    print("Calculating surface-specific Elo ratings for each match...")
    for index, row in df.iterrows():
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        surface = row['surface']

        if surface not in ['Hard', 'Clay', 'Grass']:
            continue
            
        winner_current_elo = player_elos[winner_id][surface]
        loser_current_elo = player_elos[loser_id][surface]

        df.at[index, 'winner_surface_elo'] = winner_current_elo
        df.at[index, 'loser_surface_elo'] = loser_current_elo

        new_winner_elo, new_loser_elo = update_elo(winner_current_elo, loser_current_elo)

        player_elos[winner_id][surface] = new_winner_elo
        player_elos[loser_id][surface] = new_loser_elo

    df_elo = df.dropna(subset=['winner_surface_elo', 'loser_surface_elo']).copy()
    
    print(f"\nSaving data with Elo features to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_elo.to_csv(output_path, index=False)
    
    print("Process complete!")
    print(f"Total rows in new file: {len(df_elo)}")


if __name__ == '__main__':
    # Load environment variables from a .env file
    load_dotenv()

    # --- Read Configuration from Environment ---
    input_file = os.getenv('COMBINED_DATA_PATH')
    output_file = os.getenv('ELO_DATA_PATH')

    # --- Validate Configuration ---
    if not input_file or not output_file:
        print("Error: Required file path variables not found in your .env file.")
        print("Please ensure COMBINED_DATA_PATH and ELO_DATA_PATH are set.")
    else:
        # If paths are found, run the main function
        calculate_surface_elo(data_path=input_file, output_path=output_file)
