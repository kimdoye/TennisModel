import pandas as pd
import joblib
import sqlite3
import os
import argparse
from dotenv import load_dotenv

def get_player_stats(db_path, player_name):
    """
    Retrieves the latest stats for a single player from the database.
    """
    try:
        conn = sqlite3.connect(db_path)
        player_stats = pd.read_sql_query(f"SELECT * FROM players WHERE name = ?", conn, params=(player_name,)).iloc[0]
        conn.close()
        return player_stats
    except (IndexError, sqlite3.OperationalError):
        return None

def predict_match(db_path, model_path, player1_name, player2_name, surface, tourney_level, best_of):
    """
    Predicts the outcome of a future tennis match using the trained model
    and the player database.
    """
    print("--- Tennis Match Predictor ---")
    if not os.path.exists(model_path) or not os.path.exists(db_path):
        print("Error: Model or database file not found. Please check paths in your .env file.")
        return

    model_pipeline = joblib.load(model_path)
    
    print(f"Fetching stats for {player1_name} and {player2_name}...")
    p1_stats = get_player_stats(db_path, player1_name)
    p2_stats = get_player_stats(db_path, player2_name)

    if p1_stats is None or p2_stats is None:
        missing_player = player1_name if p1_stats is None else player2_name
        print(f"Error: Could not find '{missing_player}' in the database. Please run your update scripts.")
        return

    # --- Construct DataFrame for Prediction ---
    p1_elo_col = f'{surface.lower()}_elo'
    p2_elo_col = f'{surface.lower()}_elo'

    data = {
        'p1_age': p1_stats['age'],
        'p1_ht': p1_stats.get('ht', 185),
        'p1_hand': p1_stats.get('hand', 'R'),
        'p1_rank': p1_stats['rank'],
        'p1_rank_points': p1_stats.get('rank_points', 0),
        'p1_surface_elo': p1_stats.get(p1_elo_col, 1500),
        
        'p2_age': p2_stats['age'],
        'p2_ht': p2_stats.get('ht', 185),
        'p2_hand': p2_stats.get('hand', 'R'),
        'p2_rank': p2_stats['rank'],
        'p2_rank_points': p2_stats.get('rank_points', 0),
        'p2_surface_elo': p2_stats.get(p2_elo_col, 1500),
        
        'surface': surface,
        'tourney_level': tourney_level,
        'best_of': best_of
    }
    
    match_df = pd.DataFrame([data])
    
    print("\nMatch Data:")
    print(f"  {player1_name}: Rank {p1_stats['rank']}, Elo ({surface}) {data['p1_surface_elo']:.0f}")
    print(f"  {player2_name}: Rank {p2_stats['rank']}, Elo ({surface}) {data['p2_surface_elo']:.0f}")

    # --- Make Prediction ---
    print("\nPredicting outcome...")
    prediction = model_pipeline.predict(match_df)
    prediction_proba = model_pipeline.predict_proba(match_df)

    winner_name = player1_name if prediction[0] == 1 else player2_name
    confidence = prediction_proba[0][prediction[0]]

    print("-" * 40)
    print(f"  Predicted Winner: {winner_name}")
    print(f"  Confidence: {confidence:.2%}")
    print("-" * 40)


if __name__ == '__main__':
    # Load environment variables from a .env file in the project root
    load_dotenv()
    #python predict_match.py "NovakDjokovic" "CarlosAlcaraz" --surface Clay --level M --best_of 5
    # --- Read Configuration from Environment ---
    # This makes the script portable and easy to configure.
    db_path_from_env = os.getenv('DB_PATH')
    model_path_from_env = os.getenv('MODEL_PATH')

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Predict the outcome of a future tennis match.")
    
    parser.add_argument("player1", type=str, help="Name of Player 1.")
    parser.add_argument("player2", type=str, help="Name of Player 2.")
    parser.add_argument("--surface", type=str, default="Hard", choices=["Hard", "Clay", "Grass"], help="Court surface.")
    parser.add_argument("--level", type=str, default="G", help="Tournament Level (e.g., G, M, A).")
    parser.add_argument("--best_of", type=int, default=5, help="Best of 3 or 5 sets.")
    
    args = parser.parse_args()
    
    predict_match(
        db_path=db_path_from_env,
        model_path=model_path_from_env,
        player1_name=args.player1,
        player2_name=args.player2,
        surface=args.surface,
        tourney_level=args.level,
        best_of=args.best_of
    )
