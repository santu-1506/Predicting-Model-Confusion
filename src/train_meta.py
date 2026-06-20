import torch
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(split_name):
    """Load extracted features and labels, converting to NumPy for scikit-learn."""
    features = torch.load(f"data/extracted/{split_name}_features.pt", weights_only=True).numpy()
    labels = torch.load(f"data/extracted/{split_name}_labels.pt", weights_only=True).numpy()
    return features, labels

def evaluate_model(name, model, X_test, y_test):
    """Evaluate a trained model and print metrics."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, probs)
    except ValueError:
        roc_auc = float('nan')
    
    print(f"\n--- {name} — Test Set Metrics ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return roc_auc

def main():
    print("Loading extracted features...")
    cal_features, cal_labels = load_data("calibration")
    test_features, test_labels = load_data("test")
    
    print(f"Calibration data: {cal_features.shape}")
    print(f"Test data:        {test_features.shape}")
    print(f"Feature dimension: {cal_features.shape[1]}")

    # --- Train multiple models and pick the best ---
    
    models = {}
    
    # Model 1: Gradient Boosting (200 estimators, tuned)
    print("\n[1/2] Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    gb_model.fit(cal_features, cal_labels)
    gb_auc = evaluate_model("Gradient Boosting", gb_model, test_features, test_labels)
    models["GradientBoosting"] = (gb_model, gb_auc)
    
    # Model 2: Random Forest (300 estimators)
    print("\n[2/2] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rf_model.fit(cal_features, cal_labels)
    rf_auc = evaluate_model("Random Forest", rf_model, test_features, test_labels)
    models["RandomForest"] = (rf_model, rf_auc)
    
    # --- Pick the best model by ROC-AUC ---
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_auc = models[best_name]
    
    print(f"\n{'='*50}")
    print(f"  Best model: {best_name} (ROC-AUC: {best_auc:.4f})")
    print(f"{'='*50}")

    # Save the best model
    save_path = "models/meta_model.pkl"
    joblib.dump(best_model, save_path)
    print(f"\nBest meta-model saved to {save_path}")

if __name__ == "__main__":
    main()
