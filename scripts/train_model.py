import os
import argparse
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb # UPDATED: Import xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Handles missing values
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report

from dotenv import load_dotenv

def train_model(data_path, model_output_path):
    """
    Trains a Logistic Regression classifier on the processed tennis data,
    handling missing values with imputation.

    Args:
        data_path (str): The path to the processed CSV file.
        model_output_path (str): The path to save the trained model.
    """
    # --- 1. Data Loading and Preparation ---
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    print("Preparing data for binary classification...")
    # This logic reframes the data for a binary prediction task.
    df_p1 = pd.DataFrame()
    df_p1['p1_age'] = df['winner_age']
    df_p1['p1_ht'] = df['winner_ht']
    df_p1['p1_hand'] = df['winner_hand']
    df_p1['p1_rank'] = df['winner_rank']
    df_p1['p1_rank_points'] = df['winner_rank_points']
    df_p1['p2_age'] = df['loser_age']
    df_p1['p2_ht'] = df['loser_ht']
    df_p1['p2_hand'] = df['loser_hand']
    df_p1['p2_rank'] = df['loser_rank']
    df_p1['p2_rank_points'] = df['loser_rank_points']
    df_p1['surface'] = df['surface']
    df_p1['tourney_level'] = df['tourney_level']
    df_p1['best_of'] = df['best_of']
    df_p1['p1_wins'] = 1

    df_p2 = pd.DataFrame()
    df_p2['p1_age'] = df['loser_age']
    df_p2['p1_ht'] = df['loser_ht']
    df_p2['p1_hand'] = df['loser_hand']
    df_p2['p1_rank'] = df['loser_rank']
    df_p2['p1_rank_points'] = df['loser_rank_points']
    df_p2['p2_age'] = df['winner_age']
    df_p2['p2_ht'] = df['winner_ht']
    df_p2['p2_hand'] = df['winner_hand']
    df_p2['p2_rank'] = df['winner_rank']
    df_p2['p2_rank_points'] = df['winner_rank_points']
    df_p2['surface'] = df['surface']
    df_p2['tourney_level'] = df['tourney_level']
    df_p2['best_of'] = df['best_of']
    df_p2['p1_wins'] = 0

    df_model = pd.concat([df_p1, df_p2], ignore_index=True).sample(frac=1, random_state=42)

    # --- 2. Define Features, Target, and Split Data (80/10/10) ---
    print("Splitting data into training (80%), validation (10%), and testing (10%) sets...")
    features = [col for col in df_model.columns if col != 'p1_wins']
    X = df_model[features]
    y = df_model['p1_wins']

    # Step 1: Split off the Test set (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Step 2: Split the remaining 90% into Training (80%) and Validation (10%)
    validation_size_fraction = 0.1 / 0.9 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size_fraction, random_state=42, stratify=y_train_val
    )
    
    print(f"\nData split sizes:")
    print(f"Training set:   {len(X_train)} rows (~{len(X_train)/len(df_model):.0%})")
    print(f"Validation set: {len(X_val)} rows (~{len(X_val)/len(df_model):.0%})")
    print(f"Testing set:    {len(X_test)} rows (~{len(X_test)/len(df_model):.0%})\n")
    
    # --- 3. Preprocessing with Imputation ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Pipeline for numeric features: impute missing values with the median, then scale.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: impute missing values with the most frequent, then one-hot encode.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 4. Model Training ---
    print("Training Logistic Regression model...")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Model Evaluation ---
    print("\n--- Model Evaluation on Validation Set ---")
    y_val_pred = model_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    print("\n--- Final Model Evaluation on Test Set ---")
    y_test_pred = model_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # --- 6. Model Saving ---
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    print(f"\nSaving trained model pipeline to {model_output_path}...")
    joblib.dump(model_pipeline, model_output_path)
    print("Model saved successfully.")

if __name__ == '__main__':    
    load_dotenv()

    data_path = os.getenv('HISTORICAL_DATA_PATH') # This will be None if not found
    model_path = os.getenv('MODEL_PATH')

    
    train_model(data_path=data_path, model_output_path=model_path)
