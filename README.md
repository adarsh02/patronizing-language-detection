# Patronizing and Condescending Language (PCL) Detection

Binary classification of paragraphs as containing patronizing or condescending language (PCL) directed at vulnerable groups, using RoBERTa-base as the backbone model.

---

## Task Description

Given a paragraph and a vulnerable-group keyword (e.g. *homeless*, *refugee*, *women*), the model predicts whether the paragraph contains patronizing or condescending language towards that group. This is a binary classification task — the positive class (PCL) corresponds to annotation scores of 2, 3, or 4 in the original dataset, where the score reflects the degree of annotator agreement that PCL is present.

- **Positive class**: annotation score ≥ 2 (~9.5% of the dataset)
- **Negative class**: annotation score ≤ 1 (~90.5% of the dataset)
- **Evaluation metric**: F1 score on the positive class

---

## Dataset

The dataset used is [DontPatronizeMe](https://github.com/Perez-AlmendrosC/dontpatronizeme)

```
data/
├── dontpatronizeme_pcl.tsv          # Main dataset with paragraphs and labels
├── train_semeval_parids-labels.csv  # Official train split par_ids + multi-labels
├── dev_semeval_parids-labels.csv    # Official dev split par_ids + multi-labels
├── task4_test.tsv                   # Official test set (no labels)
└── dontpatronizeme_categories.tsv   # Multi-label PCL category annotations
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Training all six ablation experiments takes approximately 3–4 hours on a single GPU.

---

## Approach

The proposed approach builds on RoBERTa-base with four modifications and one additional architectural component, each targeting a specific weakness of the baseline.

### 1. Weighted Random Sampling
At a 9:1 class ratio, a random batch of 16 contains on average only 1.5 positive examples. A `WeightedRandomSampler` rebalances batches to approximately 50/50 at the data level, before any loss is computed.

### 2. Focal Loss
Replaces standard binary cross-entropy to down-weight easy negatives dynamically:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

with γ=2 and α set to the observed positive class frequency.

### 3. Concatenated Pooling (CLS ⊕ mean ⊕ max)
Replaces CLS-only pooling with a concatenation of three complementary strategies. Max-pooling is particularly suited to detecting single high-signal tokens that are characteristic of subtle PCL.

### 4. Layer-wise Learning Rate Decay
Applies a decayed learning rate per transformer layer (`lr_i = lr_base × 0.9^(11-i)`), preserving general linguistic representations in lower layers while allowing upper layers to adapt fully to the task.

### 5. Keyword-conditioned Gating
The vulnerable-group keyword is encoded separately through the same RoBERTa encoder and used to compute a per-token scalar gate. This conditions feature extraction on the specific group being discussed, capturing the group-conditional nature of patronising language. The only parameters added are fewer than 1,600 on top of a 124M parameter model.

---

## Results

### Ablation Study

| Configuration | Test F1 | Δ |
|---|---|---|
| RoBERTa baseline (no modifications) | 0.5464 | — |
| + Weighted random sampling | 0.5808 | +0.034 |
| + Focal Loss (γ=2) | 0.5650 | −0.016 |
| + Concatenated pooling | 0.5602 | −0.005 |
| + Layer-wise LR decay | 0.5690 | +0.009 |
| + Keyword-conditioned gating (full model) | **0.5874** | +0.018 |

### Best Model — Official Dev Set

| Metric | Value |
|---|---|
| F1 (positive class) | 0.5874 |
| Precision | 0.5484 |
| Recall | 0.6330 |
| Average Precision | 0.6027 |
| False Positive Rate | 0.0549 |
| False Negative Rate | 0.3668 |

### Baseline Comparison

| Model | Dev F1 |
|---|---|
| SemEval official baseline (1 epoch) | 0.48 |
| Improved baseline (6 epochs, tuned LR) | 0.5464 |
| Best model (full approach) | 0.5874 |

---

## Submission Files

After running the full notebook, the submission files are, these files are inside /BestModel:

- `dev.txt` — one prediction per line (0 or 1) for the official dev set
- `test.txt` — one prediction per line (0 or 1) for the official test set

---

## References

- Pérez-Almendros et al. (2020). *Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities*. COLING.
