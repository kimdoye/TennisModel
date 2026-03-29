import pandas as pd
import glob
import os
from dotenv import load_dotenv

def combine_csv_files(input_path, output_file):
    """
    Combines all CSV files in a given directory into a single CSV file.

    Args:
        input_path (str): The directory path containing the CSV files.
        output_file (str): The name of the combined output CSV file.
    """
    print(f"Searching for CSV files in: {input_path}")
    # Use glob to find all files in the directory that start with 'atp_matches_'
    csv_files = glob.glob(os.path.join(input_path, 'atp_matches_*.csv'))
    
    if not csv_files:
        print(f"No CSV files matching the pattern 'atp_matches_*.csv' were found in the directory: {input_path}")
        return

    print(f"Found {len(csv_files)} files to combine.")
    
    df_list = []
    for file in sorted(csv_files):
        print(f"Reading file: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {file}. Error: {e}")
            
    if not df_list:
        print("No dataframes to combine. Exiting.")
        return

    print("\nCombining all DataFrames...")
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print("\nProcess complete!")
    print(f"Total rows in combined file: {len(combined_df)}")
    print(f"Combined file saved as: {output_file}")


if __name__ == '__main__':
    # Load environment variables from a .env file
    load_dotenv()

    # --- Read Configuration from Environment ---
    input_dir = os.getenv('RAW_DATA_DIRECTORY')
    output_filename = os.getenv('COMBINED_DATA_PATH')

    # --- Validate Configuration ---
    if not input_dir or not output_filename:
        print("Error: Required file path variables not found in your .env file.")
        print("Please ensure RAW_DATA_DIRECTORY and COMBINED_DATA_PATH are set.")
    else:
        # If paths are found, run the main function
        combine_csv_files(input_path=input_dir, output_file=output_filename)
