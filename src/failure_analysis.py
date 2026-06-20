"""
Step 9 — Break down failure types into the four quadrants:
  High trust + correct   → ideal case
  High trust + wrong     → dangerous case (minimize this!)
  Low trust  + correct   → overly cautious
  Low trust  + wrong     → meta-model did its job
"""
import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def main():
    print("Loading test data and models...")
    
    # Load test features and labels
    test_features = torch.load("data/extracted/test_features.pt", weights_only=True).numpy()
    test_labels = torch.load("data/extracted/test_labels.pt", weights_only=True).numpy()  # is_correct
    
    # Load meta model and get trust scores
    meta_model = joblib.load("models/meta_model.pkl")
    trust_scores = meta_model.predict_proba(test_features)[:, 1]
    
    # Use median trust score as threshold (splits data roughly in half)
    threshold = np.median(trust_scores)
    print(f"Trust score threshold (median): {threshold:.4f}")
    
    # Categorize into 4 quadrants
    high_trust = trust_scores >= threshold
    low_trust = trust_scores < threshold
    correct = test_labels == 1
    wrong = test_labels == 0
    
    quadrants = {
        "High Trust + Correct\n(Ideal ✅)":         np.sum(high_trust & correct),
        "High Trust + Wrong\n(Dangerous ⚠️)":      np.sum(high_trust & wrong),
        "Low Trust + Correct\n(Overly Cautious)":   np.sum(low_trust & correct),
        "Low Trust + Wrong\n(Caught by Meta ✅)":   np.sum(low_trust & wrong),
    }
    
    total = len(test_labels)
    
    print("\n--- Failure Type Breakdown ---")
    for name, count in quadrants.items():
        label = name.replace('\n', ' ')
        print(f"  {label}: {count} ({100*count/total:.1f}%)")
    print(f"  Total: {total}")
    print(f"  Base model accuracy: {100*np.mean(test_labels):.2f}%")
    
    # --- Visualization 1: Quadrant bar chart ---
    os.makedirs("assets", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    ax = axes[0]
    names = list(quadrants.keys())
    counts = list(quadrants.values())
    colors = ['#27ae60', '#e74c3c', '#f39c12', '#3498db']
    
    bars = ax.barh(names, counts, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Failure Type Breakdown', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{count} ({100*count/total:.1f}%)', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, max(counts) * 1.35)
    
    # Pie chart
    ax2 = axes[1]
    short_names = ['Ideal ✅', 'Dangerous ⚠️', 'Overly Cautious', 'Caught ✅']
    wedges, texts, autotexts = ax2.pie(
        counts, labels=short_names, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax2.set_title('Distribution of Prediction Types', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = "assets/failure_breakdown.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    # --- Visualization 2: Trust score distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(trust_scores[correct], bins=50, alpha=0.6, label='Correct predictions',
            color='#27ae60', edgecolor='white', linewidth=0.5)
    ax.hist(trust_scores[wrong], bins=50, alpha=0.6, label='Wrong predictions',
            color='#e74c3c', edgecolor='white', linewidth=0.5)
    
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.3f})')
    ax.set_xlabel('Meta-Model Trust Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Trust Score Distribution — Correct vs Wrong Predictions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    
    save_path2 = "assets/trust_distribution.png"
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path2}")

if __name__ == "__main__":
    main()
