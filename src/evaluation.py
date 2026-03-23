import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a model using various metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc = 0.5 
        
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("-" * 30)

def perform_sanity_checks():
    """
    Runs 3 sample predictions using the saved model.
    """
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/bankruptcy_model.pkl"))
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return
        
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']
    features = data['features']
    
    print("\n--- Performing Sanity Checks ---")
    np.random.seed(99)
    samples = np.random.randn(3, len(features))
    
    # Modify the last sample to increase the likelihood of being HIGH RISK
    samples[2] = samples[2] * 2.5 + 1.0 
    
    df_samples = pd.DataFrame(samples, columns=features)
    scaled_samples = scaler.transform(df_samples)
    
    probs = model.predict_proba(scaled_samples)[:, 1]
    preds = model.predict(scaled_samples)
    
    for i in range(3):
        risk_level = "HIGH RISK" if preds[i] == 1 else "LOW RISK"
        print(f"Sample {i+1}:")
        
        feature_dict = dict(zip(features[:3], np.round(samples[i][:3], 2)))
        print(f"Input features subset: {feature_dict}...")
        print(f"Prediction: {risk_level}")
        print(f"Probability: {probs[i]:.2f}\n")

if __name__ == "__main__":
    from train_model import train_and_evaluate_models
    
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/company_bankruptcy.csv"))
    print("Training models to evaluate...")
    lr_model, rf_model, X_test_scaled, y_test, _ = train_and_evaluate_models(filepath, n_features=10)
    
    evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    perform_sanity_checks()
