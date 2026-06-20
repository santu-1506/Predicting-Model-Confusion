import torch
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(split_name):
    # Load from the extracted PyTorch tensors and convert to NumPy arrays for scikit-learn
    features = torch.load(f"data/extracted/{split_name}_features.pt", weights_only=True).numpy()
    labels = torch.load(f"data/extracted/{split_name}_labels.pt", weights_only=True).numpy()
    return features, labels

def main():
    print("Loading extracted features...")
    cal_features, cal_labels = load_data("calibration")
    test_features, test_labels = load_data("test")
    
    print(f"Calibration data: {cal_features.shape}")
    print(f"Test data: {test_features.shape}")

    print("Initializing Gradient Boosting Classifier...")
    # We use a Gradient Boosting Classifier as recommended for the Meta-Model
    meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=1)

    print("Training meta-model on calibration data...")
    meta_model.fit(cal_features, cal_labels)

    print("Evaluating meta-model on test data...")
    test_preds = meta_model.predict(test_features)
    test_probs = meta_model.predict_proba(test_features)[:, 1] # Probability of "trust" (is_correct=1)

    # Calculate metrics
    acc = accuracy_score(test_labels, test_preds)
    # Note: Zero_division parameter avoids warnings if precision or recall is 0
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    # Try-except block for ROC-AUC in case all labels are 1 or 0
    try:
        roc_auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        roc_auc = float('nan')

    print("\n--- Meta-Model Evaluation on Test Set ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("-----------------------------------------\n")

    # Save the model
    save_path = "models/meta_model.pkl"
    joblib.dump(meta_model, save_path)
    print(f"Meta-model saved to {save_path}")

if __name__ == "__main__":
    main()
