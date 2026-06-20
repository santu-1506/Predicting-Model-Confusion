import os
import torch
from data import get_dataloaders
from train_base import SimpleCNN

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_is_correct = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # return_embedding=True returns logits and the intermediate hidden state
            logits, hidden = model(inputs, return_embedding=True)
            
            # Check correctness
            _, predicted = logits.max(1)
            is_correct = (predicted == labels).long()
            
            all_features.append(hidden.cpu())
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
    # We only care about calibration and test for meta-model training
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

    print("Extracting features for calibration set...")
    cal_features, cal_is_correct = extract_features(model, calibration_loader, device)
    
    torch.save(cal_features, "data/extracted/calibration_features.pt")
    torch.save(cal_is_correct, "data/extracted/calibration_labels.pt")
    print(f"Saved calibration features: {cal_features.shape}, labels: {cal_is_correct.shape}")

    print("Extracting features for test set...")
    test_features, test_is_correct = extract_features(model, test_loader, device)
    
    torch.save(test_features, "data/extracted/test_features.pt")
    torch.save(test_is_correct, "data/extracted/test_labels.pt")
    print(f"Saved test features: {test_features.shape}, labels: {test_is_correct.shape}")

if __name__ == "__main__":
    main()
