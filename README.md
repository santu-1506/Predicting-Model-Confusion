# Predicting Model Confusion

A meta-model that predicts when a base AI classifier is likely to be wrong.

## Project Idea

Most machine learning classifiers give a prediction and a confidence score, but they can still be confidently wrong. This project builds a second small model, called a meta-model, whose job is to estimate whether the base model's prediction should be trusted.

The system has two parts:

1. **Base model**: a normal classifier that predicts the class of an input.
2. **Meta-model**: a smaller binary classifier that predicts whether the base model's prediction is likely correct or wrong.

Example:

```text
Input image: dog
Base model prediction: cat
Base model confidence: 88%
Meta-model trust score: 31%
Final decision: defer to human
```

## Recommended Setup

Use CIFAR-10 image classification for the first version.

```text
Dataset: CIFAR-10
Base model: small CNN or ResNet-18
Meta-model: GradientBoostingClassifier, RandomForestClassifier, or XGBoost
Language: Python
Libraries: PyTorch, torchvision, scikit-learn, pandas, numpy, matplotlib
```

Why CIFAR-10:

- Small and easy to train.
- Standard image classification benchmark.
- Results are easy to visualize.
- Confidence, entropy, margin, and embedding features are simple to extract.

## Core Methodology

Use a three-way split:

```text
Base-train set     -> trains the base classifier
Calibration set    -> creates training data for the meta-model
Test set           -> final evaluation only
```

This split is important. The meta-model must not be trained on the same samples that trained the base model, because the base model may look too confident on its own training data.

Recommended CIFAR-10 split:

```text
40,000 training images     -> base model training
10,000 calibration images  -> meta-model training
10,000 test images         -> final evaluation
```

## Final Workflow

### Step 1: Load and Split the Dataset

Load CIFAR-10 using `torchvision.datasets.CIFAR10`.

Create three datasets:

```text
base_train
calibration
test
```

The base model trains only on `base_train`.

The calibration set is used only after the base model is trained.

The test set stays untouched until the final evaluation.

### Step 2: Train the Base Model

Train a normal image classifier.

Recommended first model:

```text
Small CNN
```

Better version if time allows:

```text
ResNet-18 adapted for CIFAR-10
```

Save the trained base model.

Track:

```text
training loss
training accuracy
calibration accuracy
test accuracy
```

Important rule:

```text
Do not train the base model on calibration or test data.
```

### Step 3: Run Base Model on Calibration Set

For every calibration sample, collect:

```text
input image
true label
base model predicted class
base model probability distribution
whether the base prediction is correct
second-to-last-layer embedding
```

The target label for the meta-model is:

```text
1 = base model prediction was correct
0 = base model prediction was wrong
```

During this step, ground truth labels are used only to create the meta-model training labels.

### Step 4: Extract Meta-Model Features

For each calibration sample, compute:

```text
max_softmax_probability
entropy
top1_top2_margin
embedding_norm
embedding_mean
embedding_std
predicted_class_id
```

Optional image-level features:

```text
image_brightness
image_contrast
image_blur_score
```

Optional uncertainty feature:

```text
Monte Carlo dropout variance
```

Example meta-model row:

```text
[confidence, entropy, margin, embedding_norm, embedding_mean, embedding_std] -> is_correct
```

Example:

```text
[0.93, 0.21, 0.71, 14.8, 0.12, 0.44] -> 1
[0.58, 1.67, 0.04, 7.3, 0.03, 0.91] -> 0
```

### Step 5: Train the Meta-Model

Train a small binary classifier.

Recommended first choice:

```text
GradientBoostingClassifier from scikit-learn
```

Other good options:

```text
RandomForestClassifier
LogisticRegression
XGBoost
Small 2-layer neural network
```

The meta-model input is the extracted feature vector.

The meta-model output is a trust score:

```text
0.95 = base model is probably correct
0.10 = base model is probably wrong
```

### Step 6: Evaluate on the Test Set

Run the trained base model on the untouched test set.

Extract the same features used for calibration.

Run the trained meta-model on those features.

Measure:

```text
Base model accuracy
Meta-model accuracy
Meta-model precision
Meta-model recall
Meta-model F1-score
Meta-model ROC-AUC
```

The meta-model is not replacing the base classifier. It estimates whether the base classifier should be trusted.

### Step 7: Compare Against Simple Baselines

Compare the meta-model's trust score against simple uncertainty baselines:

```text
Raw confidence
Entropy
Top1-top2 margin
```

At minimum, compare against raw confidence.

This is important because the project must show that the meta-model is better than simply trusting the softmax confidence score.

### Step 8: Build Risk-Coverage Curve

This is the most important evaluation.

Sort test samples by trust score from highest to lowest.

Then measure accuracy after accepting only the most trusted predictions.

Example result table:

```text
Coverage    Meaning                         Accuracy
100%        accept all predictions           78%
90%         reject riskiest 10%              82%
80%         reject riskiest 20%              86%
70%         reject riskiest 30%              89%
60%         reject riskiest 40%              92%
```

This shows whether the system can improve reliability by deferring uncertain predictions.

Main claim:

```text
When the model rejects the riskiest predictions, accuracy on accepted predictions increases.
```

### Step 9: Analyze Failure Types

Create a breakdown of predictions:

```text
High trust + correct    -> ideal accepted cases
High trust + wrong      -> dangerous failures
Low trust + correct     -> overly cautious rejections
Low trust + wrong       -> successfully flagged failures
```

The most important category is:

```text
High confidence + wrong
```

These are the scary cases where the base model looks confident but makes a mistake.

### Step 10: Build a Small Demo

The demo should show individual examples from the test set.

For each sample, display:

```text
image
true label
base model prediction
base model confidence
meta-model trust score
final decision: accept or defer
```

Example:

```text
True label: dog
Base prediction: cat
Base confidence: 88%
Meta trust score: 31%
Decision: defer to human
```

## Suggested Project Structure

```text
model-confusion-predictor/
  README.md
  requirements.txt
  src/
    data.py
    train_base.py
    extract_features.py
    train_meta.py
    evaluate.py
    demo.py
  notebooks/
    exploration.ipynb
  models/
    base_model.pt
    meta_model.pkl
  outputs/
    plots/
    tables/
```

## Implementation Order

Build the project in this order:

1. Create dataset loading and three-way split.
2. Train the base CNN on the base-train split.
3. Save and reload the base model.
4. Extract calibration features from the trained base model.
5. Train the meta-model using calibration features.
6. Extract test features.
7. Evaluate base model and meta-model on test data.
8. Generate selective accuracy and risk-coverage plots.
9. Add example visualizations for accepted and deferred predictions.
10. Prepare final report and demo.

## Key Metrics

Base model metrics:

```text
accuracy
loss
confusion matrix
```

Meta-model metrics:

```text
accuracy
precision
recall
F1-score
ROC-AUC
```

Selective prediction metrics:

```text
coverage
selective accuracy
risk
risk-coverage curve
```

Definitions:

```text
coverage = percentage of predictions accepted
risk = error rate among accepted predictions
selective accuracy = accuracy among accepted predictions
```

## Expected Final Outputs

The final project should produce:

```text
trained base classifier
trained meta-model
base model accuracy report
meta-model classification report
risk-coverage curve
selective accuracy table
confidence baseline comparison
example predictions with accept/defer decisions
```

## Final Presentation Structure

Use this structure for the project presentation:

1. **Problem**
   - AI models can be confidently wrong.
   - We need a way to flag unreliable predictions.

2. **Idea**
   - Train a base classifier.
   - Train a second model to predict whether the base classifier is likely correct.

3. **Method**
   - CIFAR-10 dataset.
   - Base CNN classifier.
   - Calibration set for meta-model training.
   - Features: confidence, entropy, margin, embedding statistics.

4. **Evaluation**
   - Base accuracy.
   - Meta-model ROC-AUC.
   - Risk-coverage curve.
   - Comparison with raw confidence baseline.

5. **Results**
   - Show selective accuracy improvement when deferring risky predictions.

6. **Demo**
   - Show accepted predictions.
   - Show deferred predictions.
   - Show high-confidence wrong cases.

7. **Conclusion**
   - A lightweight meta-model can help identify when another AI model may be unreliable.

## Important Clarification

During training:

```text
Ground truth labels are used to determine whether the base model was correct.
```

During inference:

```text
Ground truth is unavailable.
The meta-model predicts trust using only the input-derived features and base model signals.
```

This distinction is important for explaining the project correctly.

