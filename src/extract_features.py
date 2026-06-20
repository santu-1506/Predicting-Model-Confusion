import os
import torch
import numpy as np
from torch.nn.functional import softmax
from data import get_dataloaders
from train_base import SimpleCNN

def extract_features(model, dataloader, device):
    """
    Extract rich features from the base model for each sample in the dataloader.
    
    For each image, we extract:
      - max_softmax:       Maximum softmax probability (MCP)
      - entropy:           Shannon entropy of the softmax distribution
      - top1_top2_margin:  Difference between the top-1 and top-2 softmax probabilities
      - embedding_norm:    L2 norm of the 256-dim hidden embedding
      - embedding_mean:    Mean of the embedding values
      - embedding_std:     Standard deviation of the embedding values
      - predicted_class:   The predicted class index (integer)
      - embedding (256d):  Raw embedding from the second-to-last layer
    
    Total feature dimension: 7 handcrafted + 256 embedding = 263
    
    Returns:
        features: numpy array of shape (N, 263)
        is_correct: numpy array of shape (N,) — 1 if base model got it right, 0 if wrong
    """
    model.eval()
    all_features = []
    all_is_correct = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get logits and hidden embedding from base model
            logits, hidden = model(inputs, return_embedding=True)
            
            # Compute softmax probabilities
            probs = softmax(logits, dim=1)
            
            # --- Handcrafted uncertainty features ---
            
            # 1. Max softmax probability (MCP)
            max_probs, predicted = probs.max(dim=1)
            
            # 2. Shannon entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)  # add epsilon to avoid log(0)
            entropy = -torch.sum(probs * log_probs, dim=1)
            
            # 3. Top1 - Top2 margin
            top2_probs, _ = probs.topk(2, dim=1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            
            # 4. Embedding statistics
            emb_norm = torch.norm(hidden, dim=1)
            emb_mean = hidden.mean(dim=1)
            emb_std = hidden.std(dim=1)
            
            # 5. Predicted class (as float for concatenation)
            pred_class = predicted.float()
            
            # --- Combine all features ---
            # Stack handcrafted features: (batch, 7)
            handcrafted = torch.stack([
                max_probs, entropy, margin,
                emb_norm, emb_mean, emb_std, pred_class
            ], dim=1)
            
            # Concatenate with raw embedding: (batch, 7 + 256 = 263)
            combined = torch.cat([handcrafted, hidden], dim=1)
            
            # Check correctness
            is_correct = (predicted == labels).long()
            
            all_features.append(combined.cpu())
            all_is_correct.append(is_correct.cpu())
            
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_is_correct = torch.cat(all_is_correct, dim=0)
    
    return all_features, all_is_correct

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataloaders
    print("Loading datasets...")
    _, calibration_loader, test_loader = get_dataloaders(batch_size=128)

    # Initialize model and load weights
    model = SimpleCNN(num_classes=10).to(device)
    model_path = "models/base_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Please train the base model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded base model from {model_path}")

    # Create directory for extracted features
    os.makedirs("data/extracted", exist_ok=True)

    # Feature names for reference
    feature_names = [
        "max_softmax", "entropy", "top1_top2_margin",
        "embedding_norm", "embedding_mean", "embedding_std",
        "predicted_class",
    ] + [f"emb_{i}" for i in range(256)]
    print(f"Extracting {len(feature_names)} features per sample:")
    print(f"  Handcrafted: {feature_names[:7]}")
    print(f"  Raw embedding: 256 dimensions")

    print("\nExtracting features for calibration set...")
    cal_features, cal_is_correct = extract_features(model, calibration_loader, device)
    
    torch.save(cal_features, "data/extracted/calibration_features.pt")
    torch.save(cal_is_correct, "data/extracted/calibration_labels.pt")
    
    cal_correct_pct = cal_is_correct.float().mean().item() * 100
    print(f"  Saved: features {cal_features.shape}, labels {cal_is_correct.shape}")
    print(f"  Base model accuracy on calibration: {cal_correct_pct:.2f}%")

    print("\nExtracting features for test set...")
    test_features, test_is_correct = extract_features(model, test_loader, device)
    
    torch.save(test_features, "data/extracted/test_features.pt")
    torch.save(test_is_correct, "data/extracted/test_labels.pt")
    
    test_correct_pct = test_is_correct.float().mean().item() * 100
    print(f"  Saved: features {test_features.shape}, labels {test_is_correct.shape}")
    print(f"  Base model accuracy on test: {test_correct_pct:.2f}%")

if __name__ == "__main__":
    main()
