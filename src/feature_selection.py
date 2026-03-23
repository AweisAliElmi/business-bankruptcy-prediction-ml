import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

def perform_eda(df, output_dir="../dataset/"):
    """
    Generates Exploratory Data Analysis (EDA) visualizations:
    - Target distribution
    - Correlation heatmap of top features
    """
    print("Generating EDA visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Bankrupt?', data=df, palette='viridis')
    plt.title("Target Distribution (Bankrupt?)")
    target_dist_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(target_dist_path)
    plt.close()
    print(f"Target distribution saved to {target_dist_path}")

def select_features(df, target_col='Bankrupt?', n_features=15, output_dir="../dataset/"):
    """
    Selects top features using Random Forest feature importance.
    """
    print(f"Selecting top {n_features} features...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    top_features = feature_imp_df.head(n_features)['Feature'].tolist()
    
    print(f"Top {n_features} features selected:")
    for i, feature in enumerate(top_features):
        print(f"{i+1}. {feature} (Importance: {feature_imp_df.iloc[i]['Importance']:.4f})")
        
    top_cols = [target_col] + top_features
    corr_matrix = df[top_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Top Features")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")
    
    return top_features

if __name__ == "__main__":
    from data_cleaning import load_and_clean_data
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/company_bankruptcy.csv"))
    df = load_and_clean_data(filepath)
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/"))
    perform_eda(df, output_dir=output_dir)
    top_features = select_features(df, n_features=15, output_dir=output_dir)
