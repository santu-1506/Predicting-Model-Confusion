# Predicting Model Confusion

### A meta-model that learns when to trust (and when to second-guess) a classifier

This repo builds a small "second opinion" model that sits on top of a normal image classifier. The base model still does the actual job of predicting cat vs dog vs whatever. The meta-model's only job is to look at how the base model behaved on a given input and decide: _should we trust this prediction, or should we flag it for a human to check?_

This README is split into two parts.

- **Part 1** is the why. It explains the problem in plain terms and walks through the research (mainly Tsiligkaridis, 2020) that backs up why this approach actually works.
- **Part 2** is the how. It's the full build, step by step, from raw dataset to a working demo.

---

## PART 1 — The Problem and the Theory

### 1.1 The core problem

A classifier doesn't just output a label, it also outputs a confidence number (usually from softmax). The natural instinct is to trust the prediction when the confidence is high and doubt it when confidence is low. The problem is, that instinct is wrong more often than people expect.

Neural networks are known to be **poorly calibrated**. They can be 95% confident about a wrong answer just as easily as a right one. This isn't a minor quirk, it's a well documented failure mode, and it's exactly the gap this project is trying to close.

### 1.2 Why raw softmax confidence (MCP) isn't good enough

The most common shortcut people use is **Maximum Class Probability (MCP)**: just take the highest softmax score and treat it as "how sure the model is." The paper explains why this is a flawed baseline:

- Softmax inflates confidence by default, models tend to push probabilities toward 0 or 1 even when they have no business being that certain.
- Because of this, MCP gives high scores even on inputs the model is about to get wrong, which makes it bad at actually _separating_ the correct predictions from the failures. That separation is the entire point of failure prediction, so a baseline that blurs it isn't useful on its own.

This is the exact reason your project plan tells you to compare the meta-model against raw confidence as a baseline. If your meta-model can't beat plain softmax confidence, it isn't adding anything.

### 1.3 A better signal: True Class Probability (TCP)

The paper builds on earlier work (Corbiere et al., 2019) that proposes a different target: instead of asking "how confident was the model in whatever it predicted," ask "how much probability mass did the model actually put on the _correct_ class." This is called **True Class Probability (TCP)**.

TCP is a much cleaner signal than MCP because it's tied to ground truth during training. The catch is obvious: at test time you don't know the true label, so you can't compute TCP directly. The whole trick is training a separate small network (or in your case, a separate small classifier) to _estimate_ what the TCP would have been, using only signals that are available at inference time, like confidence, entropy, margin between top classes, and embedding statistics.

That's exactly your meta-model. It's a TCP estimator, just using gradient boosting / random forest instead of a neural confidence head.

### 1.4 What the paper actually contributes (Dirichlet networks)

The paper's main contribution is on the base model side, not the meta-model side. Instead of training a normal classifier with cross-entropy, it trains what's called an **Information Aware Dirichlet (IAD) network**. Rather than the network outputting a single probability per class, it outputs _concentration parameters_ of a Dirichlet distribution, basically a distribution over possible probability distributions. This lets the model express not just "what's my best guess" but "how much do I actually know here."

The loss function is built so that:

- the correct class concentration goes up when the model gets it right,
- and critically, **incorrect classes are actively discouraged from getting high concentration**, which keeps the model from being overconfident on inputs it's about to mess up.

The payoff, shown empirically across Fashion-MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet: when you plot the TCP scores for correct vs incorrect predictions, IAD networks separate the two groups much more cleanly than a standard cross-entropy network. Errors cluster near zero confidence instead of bleeding into the high-confidence region. That separation is the entire game in failure prediction. The cleaner it is, the easier it becomes for any meta-model (constrained network, gradient boosting, whatever) to tell correct from incorrect.

### 1.5 What this means for your project

You're not training a Dirichlet network here, you're using a normal CNN/ResNet-18 base model, which is the right call for a first version (Dirichlet networks are a deeper rabbit hole and a CNN gets you a working pipeline much faster). But the paper still matters for your project in three concrete ways:

1. **It validates the approach.** A peer-reviewed lab (MIT Lincoln Lab) is doing the exact same two-stage idea: base model + separate confidence/meta layer, with a documented track record of beating raw softmax confidence on the same kind of metrics you're using.
2. **It tells you what "good" looks like.** The paper reports AUROC in the high 80s to low 90s and AUPRC-Error around 50 to 75 depending on dataset, for CIFAR-10 specifically, around 90-94 AUROC. That's your ballpark for what a working meta-model should land near. If you're way below that, something's off in your features or your calibration split.
3. **It gives you an upgrade path.** Once your first version works (CNN + GradientBoosting meta-model), the next step described in the paper is exactly that: making the base model itself uncertainty-aware (Dirichlet style), so the meta-model has cleaner signal to work with in the first place.

### 1.6 The bigger framing: selective prediction

What you're building is part of a broader idea called **selective prediction** (sometimes "learning to defer" or "human-AI deferral"). The system is allowed to abstain. Instead of forcing an answer on every input, it can say "I'm not confident here, send this to a human or a more expensive model." This matters in any high-stakes setting (medical imaging, autonomous driving, fraud detection) where a wrong-but-confident answer is far more dangerous than an honest "I don't know."

The risk-coverage curve (Part 2, Step 8) is the standard way to prove a selective prediction system actually works: as you reject more low-trust predictions, accuracy on what's left should keep climbing. That single curve is usually the headline result in any paper or project on this topic.

---

## PART 2 — The Build, Step by Step

### 2.1 Setup

```text
Dataset: CIFAR-10
Base model: small CNN (first version) or ResNet-18 (upgrade)
Meta-model: GradientBoostingClassifier, RandomForestClassifier, or XGBoost
Language: Python
Libraries: PyTorch, torchvision, scikit-learn, pandas, numpy, matplotlib
```

CIFAR-10 is the right starting dataset, it's small, standard, and every feature you need (confidence, entropy, margin, embeddings) is trivial to extract from it.

### 2.2 The three-way split (this is the part people get wrong)

```text
Base-train set     -> trains the base classifier        (40,000 images)
Calibration set    -> trains the meta-model              (10,000 images)
Test set            -> final evaluation only              (10,000 images)
```

The reason this split exists: if you train the meta-model on the same data the base model was trained on, the base model looks artificially confident and correct on almost everything (it's basically memorized it). The meta-model would learn a fantasy version of "when to trust the model" that falls apart on real unseen data. The calibration set has to be data the base model has never touched during training.

### 2.3 Step-by-step workflow

**Step 1 — Load and split**
Load CIFAR-10 with `torchvision.datasets.CIFAR10`, split into `base_train`, `calibration`, `test`. Keep these three completely separate for the rest of the pipeline.

**Step 2 — Train the base model**
Train a small CNN first (get the pipeline working end to end), then upgrade to ResNet-18 adapted for CIFAR-10 if you have time. Only train on `base_train`. Track training loss, training accuracy, calibration accuracy, and test accuracy. Never let the base model see calibration or test data during training.

**Step 3 — Run the base model on the calibration set**
For every calibration image, record:

```text
input image, true label, predicted class, full probability distribution,
whether the prediction was correct, second-to-last-layer embedding
```

The correctness flag (1 = correct, 0 = wrong) becomes the label for the meta-model.

**Step 4 — Extract meta-model features**
For each calibration sample, compute:

```text
max_softmax_probability, entropy, top1_top2_margin,
embedding_norm, embedding_mean, embedding_std, predicted_class_id
```

Optional extras: image brightness/contrast/blur, or Monte Carlo dropout variance if you want a richer uncertainty signal.

**Step 5 — Train the meta-model**
Feed the feature vectors into a GradientBoostingClassifier (good first choice), with the correctness flag as the target. Output is a trust score between 0 and 1: closer to 1 means "probably trust this," closer to 0 means "probably don't."

**Step 6 — Evaluate on the test set**
Run the base model on the untouched test set, extract the same features, run them through the trained meta-model. Measure base model accuracy plus meta-model accuracy, precision, recall, F1, and ROC-AUC.

**Step 7 — Compare against baselines**
This is the step that proves your meta-model is worth having. Compare its trust score against raw confidence, entropy alone, and the top1-top2 margin alone. At minimum, beat raw confidence, that's the MCP baseline the paper in Part 1 also benchmarks against.

**Step 8 — Risk-coverage curve (the headline result)**
Sort test samples by trust score, then measure accuracy as you progressively reject the riskiest predictions:

```text
Coverage    Meaning                    Accuracy
100%        accept all                  78%
90%         reject riskiest 10%         82%
80%         reject riskiest 20%         86%
70%         reject riskiest 30%         89%
60%         reject riskiest 40%         92%
```

The claim you're proving: as you defer more of the risky predictions to a human, accuracy on what's left climbs. This is the core selective prediction result.

**Step 9 — Break down failure types**

```text
High trust + correct   -> the ideal case
High trust + wrong     -> the dangerous case (this is what you most want to minimize)
Low trust + correct    -> overly cautious, not dangerous but a bit wasteful
Low trust + wrong      -> the meta-model did its job
```

"High trust + wrong" is the category to obsess over. Every improvement to your features or your model should be judged by whether it shrinks this bucket.

**Step 10 — Build a small demo**
For a handful of test samples, show: image, true label, base prediction, base confidence, meta trust score, final decision (accept/defer). This is the easiest way to make the project feel real in a presentation, people get it instantly when they see "model said cat, was actually a dog, and the meta-model correctly flagged it."

### 2.4 Project structure

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

### 2.5 Build order

1. Dataset loading and three-way split
2. Train base CNN on `base_train`
3. Save and reload the base model
4. Extract calibration features
5. Train the meta-model
6. Extract test features
7. Evaluate base model and meta-model on test data
8. Generate selective accuracy and risk-coverage plots
9. Add example visualizations (accepted vs deferred predictions)
10. Final report and demo

### 2.6 Key metrics, all in one place

```text
Base model:        accuracy, loss, confusion matrix
Meta-model:         accuracy, precision, recall, F1, ROC-AUC
Selective prediction: coverage, risk, selective accuracy, risk-coverage curve
```

Definitions, since these get mixed up easily:

```text
coverage             = % of predictions accepted
risk                 = error rate among accepted predictions
selective accuracy   = accuracy among accepted predictions
```

### 2.7 The important rule to remember when explaining this project

During **training**, ground truth is used to label whether the base model was correct, that's how the meta-model learns. During **inference**, there is no ground truth available. The meta-model only ever sees confidence, entropy, margin, and embedding stats, it has never seen the true label at prediction time. If you get asked about this in a viva or interview, this is the line that shows you actually understand selective prediction instead of just running the code.

---

## References (for the literature grounding in Part 1)

- Tsiligkaridis, T. (2020). _Failure Prediction by Confidence Estimation of Uncertainty-Aware Dirichlet Networks._ MIT Lincoln Laboratory.
- Corbiere, C., Thome, N., Bar-Hen, A., Cord, M., Perez, P. (2019). _Addressing Failure Prediction by Learning Model Confidence._ NeurIPS.
- Hendrycks, D., Gimpel, K. (2017). _A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks._ ICLR. (this is the MCP baseline referenced throughout)
- Jiang, H., Kim, B., Guan, M., Gupta, M. (2018). _To Trust or Not to Trust a Classifier._ NeurIPS. (Trust Score method)
- Sensoy, M., Kaplan, L., Kandemir, M. (2018). _Evidential Deep Learning to Quantify Classification Uncertainty._ NeurIPS.
- Malinin, A., Gales, M. (2019). _Reverse KL-Divergence Training of Prior Networks._ NeurIPS.
