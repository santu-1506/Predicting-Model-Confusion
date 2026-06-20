import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from data import get_dataloaders
from train_base import SimpleCNN

def calculate_selective_accuracies(scores, is_correct, coverages):
    """
    Given an array of scores (confidence or trust) and is_correct binary labels,
    calculate the accuracy when retaining only the top 'coverage' proportion of samples.
    """
    # Sort from highest score to lowest
    sorted_indices = np.argsort(scores)[::-1]
    sorted_is_correct = is_correct[sorted_indices]
    
    accuracies = []
    total_samples = len(scores)
    
    for cov in coverages:
        num_to_keep = int(cov * total_samples)
        if num_to_keep == 0:
            accuracies.append(0.0)
            continue
            
        kept_correct = sorted_is_correct[:num_to_keep]
        acc = kept_correct.mean()
        accuracies.append(acc)
        
    return accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataloaders
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(batch_size=128)

    # Initialize model and load weights
    base_model = SimpleCNN(num_classes=10).to(device)
    base_model.load_state_dict(torch.load("models/base_model.pt", map_location=device))
    base_model.eval()
    print("Loaded base model.")

    # Load meta model
    meta_model = joblib.load("models/meta_model.pkl")
    print("Loaded meta model.")

    # Get test features and labels to run through meta model
    test_features = torch.load("data/extracted/test_features.pt", weights_only=True).numpy()
    test_labels = torch.load("data/extracted/test_labels.pt", weights_only=True).numpy() # This is is_correct

    # 1. Get Trust Scores from Meta Model
    print("Calculating Meta-Model trust scores...")
    trust_scores = meta_model.predict_proba(test_features)[:, 1]

    # 2. Get Raw Confidence (MCP) from Base Model
    print("Calculating Base Model raw confidences (MCP)...")
    mcp_scores = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = base_model(inputs)
            probs = softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
            mcp_scores.append(max_probs.cpu().numpy())
            
    mcp_scores = np.concatenate(mcp_scores, axis=0)

    # Evaluate across different coverages (100% down to 50%)
    coverages = np.linspace(1.0, 0.5, 50)
    
    mcp_accuracies = calculate_selective_accuracies(mcp_scores, test_labels, coverages)
    trust_accuracies = calculate_selective_accuracies(trust_scores, test_labels, coverages)

    # 3. Plotting
    print("Generating Risk-Coverage plot...")
    os.makedirs("outputs/plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(coverages * 100, mcp_accuracies, label='Baseline (Raw Softmax Confidence)', color='red', linestyle='--')
    plt.plot(coverages * 100, trust_accuracies, label='Meta-Model (Trust Score)', color='blue', linewidth=2)
    
    plt.gca().invert_xaxis() # Reverse x-axis so coverage goes from 100% -> 50%
    plt.title('Selective Prediction: Risk-Coverage Curve')
    plt.xlabel('Coverage (%) - How much of the dataset we retain')
    plt.ylabel('Selective Accuracy - Accuracy on retained data')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    save_path = "outputs/plots/risk_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    main()
