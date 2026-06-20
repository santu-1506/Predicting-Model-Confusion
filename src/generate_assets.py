"""Generate visual assets for README and Steps.md documentation."""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

os.makedirs("assets", exist_ok=True)

# --- 1. Training Curve ---
epochs = list(range(1, 11))
losses = [1.7320, 1.4054, 1.2394, 1.1294, 1.0460, 0.9808, 0.9321, 0.8910, 0.8568, 0.8180]
accs   = [36.10, 48.90, 55.50, 59.85, 63.12, 65.62, 67.37, 68.88, 70.29, 71.44]

fig, ax1 = plt.subplots(figsize=(10, 5))

color_loss = '#e74c3c'
color_acc  = '#2ecc71'

ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Loss', color=color_loss, fontsize=13, fontweight='bold')
ax1.plot(epochs, losses, color=color_loss, linewidth=2.5, marker='o', markersize=6, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_ylim(0.5, 2.0)

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color=color_acc, fontsize=13, fontweight='bold')
ax2.plot(epochs, accs, color=color_acc, linewidth=2.5, marker='s', markersize=6, label='Training Accuracy')
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.set_ylim(30, 80)

fig.suptitle('Base Model Training — SimpleCNN on CIFAR-10', fontsize=15, fontweight='bold', y=0.98)
ax1.set_xticks(epochs)
ax1.grid(True, linestyle=':', alpha=0.4)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

plt.tight_layout()
plt.savefig('assets/training_curve.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: assets/training_curve.png")


# --- 2. Meta-Model Comparison ---
fig, ax = plt.subplots(figsize=(9, 4))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
gb_vals = [0.7830, 0.8320, 0.8898, 0.8599, 0.8284]
rf_vals = [0.7827, 0.8132, 0.9213, 0.8639, 0.8310]

x = np.arange(len(metrics))
width = 0.32

bars1 = ax.bar(x - width/2, gb_vals, width, label='Gradient Boosting', color='#3498db', edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest ★', color='#e67e22', edgecolor='white', linewidth=0.8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Meta-Model Comparison on Test Set', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.7, 1.0)
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle=':', alpha=0.4)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

plt.tight_layout()
plt.savefig('assets/meta_model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: assets/meta_model_comparison.png")


# --- 3. Feature extraction summary ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

feature_data = [
    ['max_softmax', 'Highest softmax probability', 'Confidence signal'],
    ['entropy', 'Shannon entropy of softmax', 'Uncertainty signal'],
    ['top1_top2_margin', 'Gap between top-1 and top-2', 'Ambiguity signal'],
    ['embedding_norm', 'L2 norm of hidden layer', 'Activation strength'],
    ['embedding_mean', 'Mean of hidden layer', 'Central tendency'],
    ['embedding_std', 'Std dev of hidden layer', 'Activation spread'],
    ['predicted_class', 'Predicted class index', 'Class identity'],
    ['embedding[0:255]', '256-dim hidden layer output', 'Raw representation'],
]

table = ax.table(
    cellText=feature_data,
    colLabels=['Feature', 'Description', 'Purpose'],
    loc='center',
    cellLoc='left',
    colWidths=[0.28, 0.42, 0.30]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for j in range(3):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(feature_data) + 1):
    color = '#ecf0f1' if i % 2 == 0 else '#ffffff'
    for j in range(3):
        table[i, j].set_facecolor(color)

ax.set_title('263 Meta-Model Features Extracted Per Sample', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('assets/feature_table.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: assets/feature_table.png")

print("\nAll assets generated successfully!")
