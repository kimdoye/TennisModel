import requests
from bs4 import BeautifulSoup
import sqlite3
import os
import re
from datetime import datetime
from dotenv import load_dotenv

ATP_RANKINGS_URL = 'https://tennisabstract.com/reports/atp_elo_ratings.html'

def initialize_database(db_path):
    """
    Creates the database file and the players table if they don't exist.
    This function is for one-time setup or verification.
    
    Args:
        db_path (str): The path to the SQLite database file.
    """
    print(f"Initializing database at {db_path}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the players table with a clear schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            name TEXT PRIMARY KEY,
            age NUMERIC,
            rank INTEGER,
            elo NUMERIC,
            hard_elo NUMERIC,
            clay_elo NUMERIC,
            grass_elo NUMERIC,
            last_updated TEXT


        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def scrape_atp_rankings(url):
    """
    Scrapes the ATP rankings page and returns a list of player data.

    Args:
        url (str): The URL of the ATP singles rankings page.

    Returns:
        list: A list of tuples, where each tuple contains player data.
              Returns an empty list if scraping fails.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Fetching data from {url}...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    player_rows = soup.select('table#reportable tbody tr')

    if not player_rows:
        print("Could not find playerrows. Website structure may have changed.")
        return []

    player_data = []
    print(f"Found {len(player_rows)} players. Parsing data...")
    for row in player_rows:
        cells = row.find_all('td')
        try:
            player_info = {
                'name': cells[1].find('a').text,
                'age': float(cells[2].text),
                'elo': float(cells[3].text),
                'hard_elo': float(cells[6].text),
                'clay_elo': float(cells[8].text),
                'grass_elo': float(cells[10].text),
                'rank': int(cells[15].text)

            }
            player_data.append(player_info)
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Skipping a row due to parsing error: {e}")
            continue
            
    return player_data


def update_players_in_db(db_path, players):
    """
    Updates the database with the latest Elo ratings for each player.

    Args:
        db_path (str): The path to the SQLite database file.
        players (list): A list of player data dictionaries.
    """
    if not players:
        print("No player data provided to update. Aborting.")
        return

    print(f"Connecting to database to update Elo ratings for {len(players)} players...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    updated_count = 0
    for player in players:
        # We use the player's name as the key to update records.
        cursor.execute('''
            INSERT OR REPLACE INTO players (
                name,
                age,
                rank,
                elo,
                hard_elo,
                clay_elo,
                grass_elo,
                last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            player['name'],
            player['age'],
            player['rank'],
            player['elo'],
            player['hard_elo'],
            player['clay_elo'],
            player['grass_elo'],
            update_time
        ))
        if cursor.rowcount > 0:
            updated_count += 1

    conn.commit()
    conn.close()
    print("Database Elo update complete.")
    print(f"Successfully updated {updated_count} existing player records.")

def remove_all_whitespace_py(text):
    """
    A Python function that removes all types of whitespace from a string,
    including standard spaces, non-breaking spaces, tabs, etc.
    
    Args:
        text (str): The input string.
        
    Returns:
        str: The string with all whitespace removed, or the original input if not a string.
    """
    if not isinstance(text, str):
        return text
    
    # \s is a regex pattern that matches any whitespace character.
    # This is more robust than trying to replace each type of space individually.
    return re.sub(r'\s+', '', text)

def clean_column_whitespace_advanced(db_path, table_name, column_name):
    """
    Removes ALL whitespace (including non-breaking spaces) from a column
    by registering a custom Python function in SQLite.

    Args:
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table to update.
        column_name (str): The name of the text column to clean.
    """
    print(f"Connecting to {db_path} for advanced cleaning of '{column_name}'...")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Register our Python function as a custom SQL function named 'REMOVE_WHITESPACE'
        # The '1' means the function takes one argument.
        conn.create_function("REMOVE_WHITESPACE", 1, remove_all_whitespace_py)
        
        cursor = conn.cursor()

        # Now, use our custom function directly in the SQL command
        sql_command = f"UPDATE {table_name} SET {column_name} = REMOVE_WHITESPACE({column_name});"
        
        print(f"Executing advanced cleaning command: {sql_command}")
        cursor.execute(sql_command)
        
        changes = conn.total_changes
        
        conn.commit()
        conn.close()
        
        print("\nAdvanced cleaning complete!")
        print(f"{changes} rows were checked/updated in the '{column_name}' column.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


    
if __name__ == '__main__':

    load_dotenv()
    db_path = os.getenv('DB_PATH')

    clean_column_whitespace_advanced(db_path=db_path, table_name='players', column_name='name')

    # 1. Ensure the database and table exist
    '''initialize_database(db_path=db_path)
    
    # 2. Scrape the latest player data from the web
    scraped_players = scrape_atp_rankings(url=ATP_RANKINGS_URL)
    
    # 3. Update the database with the scraped data
    if scraped_players:
        update_players_in_db(db_path=db_path, players=scraped_players)'''
