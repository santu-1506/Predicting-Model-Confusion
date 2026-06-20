"""
Step 10 — Interactive demo showing individual predictions with trust scores.
Shows sample images with: true label, base prediction, confidence, trust score, and accept/defer decision.
"""
import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.nn.functional import softmax
from torchvision import datasets, transforms

from train_base import SimpleCNN

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Inverse normalization to display images properly
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])

def unnormalize(img_tensor):
    """Convert a normalized tensor back to displayable image."""
    img = img_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    base_model = SimpleCNN(num_classes=10).to(device)
    base_model.load_state_dict(torch.load("models/base_model.pt", map_location=device))
    base_model.eval()
    print("Loaded base model.")
    
    # Load meta model
    meta_model = joblib.load("models/meta_model.pkl")
    print("Loaded meta model.")
    
    # Load test dataset (with transforms for model, raw for display)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Set seed for reproducibility, then pick diverse examples
    np.random.seed(42)
    
    # We want to find specific categories of examples:
    # 1. Correct + High Trust (ideal)
    # 2. Wrong + Low Trust (meta-model caught it)
    # 3. Wrong + High Trust (dangerous — meta-model missed it)
    # 4. Correct + Low Trust (overly cautious)
    
    examples = {
        'correct_high_trust': [],
        'wrong_low_trust': [],
        'wrong_high_trust': [],
        'correct_low_trust': [],
    }
    
    print("Scanning test set for interesting examples...")
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_tensor, true_label = test_dataset[idx]
            inp = img_tensor.unsqueeze(0).to(device)
            
            logits, hidden = base_model(inp, return_embedding=True)
            probs = softmax(logits, dim=1)
            max_prob, predicted = probs.max(dim=1)
            
            pred_class = predicted.item()
            confidence = max_prob.item()
            is_correct = (pred_class == true_label)
            
            # Compute meta-model features (same as extract_features.py)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=1).item()
            top2 = probs.topk(2, dim=1)[0]
            margin = (top2[0, 0] - top2[0, 1]).item()
            emb_norm = torch.norm(hidden, dim=1).item()
            emb_mean = hidden.mean(dim=1).item()
            emb_std = hidden.std(dim=1).item()
            
            handcrafted = [confidence, entropy, margin, emb_norm, emb_mean, emb_std, float(pred_class)]
            features = np.array(handcrafted + hidden.cpu().squeeze().tolist()).reshape(1, -1)
            
            trust_score = meta_model.predict_proba(features)[0, 1]
            
            # Determine the trust threshold (use 0.5 for demo clarity)
            high_trust = trust_score >= 0.5
            
            entry = {
                'idx': idx,
                'img': img_tensor,
                'true_label': true_label,
                'pred_class': pred_class,
                'confidence': confidence,
                'trust_score': trust_score,
                'is_correct': is_correct,
            }
            
            if is_correct and high_trust and len(examples['correct_high_trust']) < 3:
                examples['correct_high_trust'].append(entry)
            elif not is_correct and not high_trust and len(examples['wrong_low_trust']) < 3:
                examples['wrong_low_trust'].append(entry)
            elif not is_correct and high_trust and len(examples['wrong_high_trust']) < 3:
                examples['wrong_high_trust'].append(entry)
            elif is_correct and not high_trust and len(examples['correct_low_trust']) < 3:
                examples['correct_low_trust'].append(entry)
            
            # Check if we have enough
            all_found = all(len(v) >= 2 for v in examples.values())
            if all_found:
                break
    
    # Collect all examples in display order
    display_order = [
        ('correct_high_trust', '✅ Correct + High Trust (Ideal)', '#27ae60'),
        ('wrong_low_trust',    '✅ Wrong + Low Trust (Caught!)', '#3498db'),
        ('wrong_high_trust',   '⚠️ Wrong + High Trust (Missed!)', '#e74c3c'),
        ('correct_low_trust',  '🟡 Correct + Low Trust (Cautious)', '#f39c12'),
    ]
    
    # Build the figure — 2 examples per category, 4 categories = 8 images
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Demo: Meta-Model Trust Score in Action', fontsize=18, fontweight='bold', y=1.02)
    
    for col, (key, title, color) in enumerate(display_order):
        for row in range(min(2, len(examples[key]))):
            ax = axes[row, col]
            entry = examples[key][row]
            
            img = unnormalize(entry['img'])
            ax.imshow(img)
            ax.axis('off')
            
            true_name = CLASSES[entry['true_label']]
            pred_name = CLASSES[entry['pred_class']]
            conf = entry['confidence']
            trust = entry['trust_score']
            decision = "ACCEPT" if trust >= 0.5 else "DEFER"
            decision_color = '#27ae60' if decision == 'ACCEPT' else '#e74c3c'
            
            info = f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2f} | Trust: {trust:.2f}"
            ax.set_title(info, fontsize=10, fontweight='bold')
            
            # Add decision badge
            ax.text(0.5, -0.08, decision, transform=ax.transAxes, fontsize=13,
                    fontweight='bold', ha='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=decision_color, alpha=0.9))
            
            if row == 0:
                ax.text(0.5, 1.35, title, transform=ax.transAxes, fontsize=11,
                        fontweight='bold', ha='center', color=color)
    
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    save_path = "assets/demo_predictions.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    # Print summary
    print("\n--- Demo Summary ---")
    for key, title, _ in display_order:
        print(f"\n{title}:")
        for entry in examples[key][:2]:
            true_name = CLASSES[entry['true_label']]
            pred_name = CLASSES[entry['pred_class']]
            decision = "ACCEPT" if entry['trust_score'] >= 0.5 else "DEFER"
            print(f"  True: {true_name:>10} | Pred: {pred_name:>10} | "
                  f"Conf: {entry['confidence']:.3f} | Trust: {entry['trust_score']:.3f} | {decision}")

if __name__ == "__main__":
    main()
