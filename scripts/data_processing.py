import pandas as pd
import os
from dotenv import load_dotenv

def filter_elo_data(input_path, output_path):
    """
    Loads the dataset with Elo ratings and filters it to keep only the
    essential columns for modeling.
    
    Args:
        input_path (str): Path to the input CSV file with Elo data.
        output_path (str): Path to save the filtered output CSV file.
    """
    print(f"Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"\nError: Input file not found at '{input_path}'.")
        print("Please ensure the path in your .env file is correct.")
        return

    # Define the columns to keep for the model
    columns_to_keep = [
        'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'winner_hand', 'loser_hand',
        'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
        'surface', 'tourney_level', 'best_of',
        # It's important to keep the new Elo columns as well!
        'winner_surface_elo', 'loser_surface_elo' 
    ]

    # Check if all required columns exist in the dataframe
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        print(f"\nError: The following required columns are missing from the input file: {missing_cols}")
        return

    # Filter the DataFrame
    print("Filtering DataFrame to keep essential columns...")
    df_filtered = df[columns_to_keep]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the filtered DataFrame to the new CSV file
    df_filtered.to_csv(output_path, index=False)

    print(f"\nFiltered data successfully saved to: {output_path}")


if __name__ == '__main__':
    # Load environment variables from a .env file
    load_dotenv()

    # --- Read Configuration from Environment ---
    # Get the file paths from the .env file.
    # If a variable is not found, os.getenv() will return None.
    input_file = os.getenv('INPUT_ELO_DATA_PATH')
    output_file = os.getenv('FILTERED_ELO_DATA_PATH')

    # --- Validate Configuration ---
    # Check if the variables were found in the .env file.
    # If not, print an error and exit the script.
    if not input_file or not output_file:
        print("Error: Required file path variables not found in your .env file.")
        print("Please ensure INPUT_ELO_DATA_PATH and FILTERED_ELO_DATA_PATH are set.")
    else:
        # If paths are found, run the main function
        filter_elo_data(input_path=input_file, output_path=output_file)
