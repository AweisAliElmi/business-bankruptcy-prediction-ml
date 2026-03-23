import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from train_model import train_and_evaluate_models
import os

filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/company_bankruptcy.csv"))
print("Training models to generate graphs...")
lr_model, rf_model, X_test_scaled, y_test, _ = train_and_evaluate_models(filepath, n_features=10)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/"))
os.makedirs(output_dir, exist_ok=True)

# 1. ROC Curve
plt.figure(figsize=(8, 6))
# Random Forest
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# Logistic Regression
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, color='orange', linestyle='--', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Incorrectly flagged Safe companies)')
plt.ylabel('True Positive Rate (Correctly caught Bankruptcies)')
plt.title('ROC Curve Comparison: RF vs Logistic Regression')
plt.legend(loc="lower right")
roc_path = os.path.join(output_dir, 'ROC_Curve_Comparison.png')
plt.savefig(roc_path)
plt.close()
print(f"ROC Curve saved to {roc_path}")

# 2. Confusion Matrix (Random Forest)
rf_pred = rf_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe (0)", "Bankrupt (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix')
cm_path = os.path.join(output_dir, 'RF_Confusion_Matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Confusion Matrix saved to {cm_path}")
