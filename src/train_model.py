import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_cleaning import load_and_clean_data
from feature_selection import select_features, perform_eda

def train_and_evaluate_models(filepath, n_features=10):
    """
    Pipeline to load data, select features, train models, and save the best one.
    """
    print("--- Starting ML Pipeline ---")
    df = load_and_clean_data(filepath)
    
    target_col = 'Bankrupt?'
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/"))
    perform_eda(df, output_dir=output_dir)
    selected_features = select_features(df, target_col=target_col, n_features=n_features, output_dir=output_dir)
    
    X = df[selected_features]
    y = df[target_col]
    
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    # We will pick Random Forest as the best model by default for this use-case,
    # as tree-based models generally perform better on tabular data like financial ratios.
    best_model = rf_model
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "bankruptcy_model.pkl")
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'features': selected_features
    }, model_path)
    
    print(f"Model successfully saved to {model_path}")
    
    return lr_model, rf_model, X_test_scaled, y_test, selected_features

if __name__ == "__main__":
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/company_bankruptcy.csv"))
    train_and_evaluate_models(filepath, n_features=10)
