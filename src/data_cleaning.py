import pandas as pd
import numpy as np
import os

def load_and_clean_data(filepath):
    """
    Loads dataset, inspects structure, handles missing values, and checks data types.
    """
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        print("Creating dummy dataset for initial testing...")
        return create_dummy_data()

    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")
    
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values before cleaning: {missing_count}")
    
    # Handle missing values by replacing with median for numerical columns
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Check data types
    print("Data types summary:")
    print(df.dtypes.value_counts())
    
    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant columns: {len(constant_cols)} columns")
        df.drop(columns=constant_cols, inplace=True)
    
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def create_dummy_data():
    """Generates a dummy dataset that mimics the Company Bankruptcy Prediction dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 95
    
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    X = np.random.randn(n_samples, n_features)
    
    columns = [f"Feature_{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "Bankrupt?", y)
    
    feature_importance = np.random.randn(n_features)
    for i in range(n_features):
        df[f"Feature_{i+1}"] += y * feature_importance[i] * 0.5
        
    for col in columns[:5]:
        mask = np.random.rand(n_samples) < 0.05
        df.loc[mask, col] = np.nan
        
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    print("Dummy dataset created successfully.")
    return df

if __name__ == "__main__":
    filepath = "../dataset/company_bankruptcy.csv"
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), filepath))
    df_clean = load_and_clean_data(filepath)
    print("Data cleaning module test complete.")
