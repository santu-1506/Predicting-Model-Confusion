# Step 1 - Load and Split the Dataset

## Goal

Load the CIFAR-10 dataset and split it into three separate parts:

```text
base_train   -> used to train the base image classifier
calibration  -> used to train the meta-model / trust model
test         -> used only for final evaluation
```

The important rule is that the base model must never train on the calibration or test data.

## What to Create

Create this file:

```text
src/data.py
```

This file should handle:

1. Downloading/loading CIFAR-10.
2. Applying basic image transforms.
3. Splitting the original CIFAR-10 training set into:
   - `40,000` images for `base_train`
   - `10,000` images for `calibration`
4. Keeping the CIFAR-10 test set as the untouched final `test` set.
5. Returning PyTorch `DataLoader` objects for all three sets.

## Install Required Libraries

Create a `requirements.txt` file with:

```text
torch
torchvision
scikit-learn
pandas
numpy
matplotlib
tqdm
joblib
```

Then install them:

```bash
pip install -r requirements.txt
```

## Suggested Code Structure

Inside `src/data.py`, create a function like:

```python
def get_dataloaders(batch_size=128, num_workers=2):
    """
    Returns:
        base_train_loader
        calibration_loader
        test_loader
    """
```

The split should look like this:

```python
base_train_size = 40000
calibration_size = 10000

base_train, calibration = random_split(
    full_train_dataset,
    [base_train_size, calibration_size],
    generator=torch.Generator().manual_seed(42),
)
```

Use a fixed random seed, such as `42`, so the split is reproducible.

## Expected Output

When Step 1 is complete, you should be able to run a quick check and see:

```text
Base-train images: 40000
Calibration images: 10000
Test images: 10000
```

## Why This Step Matters

The calibration set is used later to teach the meta-model when the base model is likely to be correct or wrong. If the base model trains on the calibration set, the meta-model will learn from unrealistic results and the final trust score will not be reliable.

---

# Step 2 - Train the Base Model

## Goal

Train a small CNN base model exclusively on the `base_train` dataset. The base model must never see the `calibration` or `test` datasets during training.

## What to Create

Create `src/train_base.py` containing:
1. A basic CNN architecture for CIFAR-10 (e.g., 2-3 conv layers, followed by fully connected layers).
2. A training loop that runs for a few epochs (e.g., 10 epochs).
3. Code to save the trained model weights to `models/base_model.pt`.
![alt text](image.png)

---

# Step 3 - Extract Features & Generate Meta-Model Data

## Goal

Run the trained base model on the `calibration` and `test` datasets to see where it succeeds and where it fails. We will extract its internal features (embeddings) and record whether its predictions were correct or incorrect. This information will form the dataset used to train our trust/meta-model.

## What to Create

Create `src/extract_features.py` containing:
1. Code to load the trained `models/base_model.pt`.
2. A function that runs the model over the `calibration` and `test` dataloaders.
3. For each image, save:
   - The intermediate features/embeddings from the base model (e.g., from the second-to-last layer).
   - A binary label `is_correct` (1 if the base model's prediction was right, 0 if wrong).
4. Save the extracted features and labels into a format suitable for training the meta-model (e.g., NumPy `.npy` files or PyTorch `.pt` files in a `data/extracted/` directory).

## Why This Step Matters

The meta-model needs to learn the patterns of when the base model is likely to be confused. By looking at the base model's internal embeddings and whether it ultimately got the answer right or wrong on the calibration set, the meta-model can predict "trust scores" for new, unseen data in the test set.