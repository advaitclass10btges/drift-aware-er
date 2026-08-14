# Boundary-Aware Drift Detection for Entity Resolution

## 1. Problem Definition

Entity resolution systems operate on continuously arriving data streams.
Over time, changes in entity attributes, naming conventions, missing values,
and record generation processes cause distributional drift.

Traditional drift detectors monitor changes in feature distributions or
prediction statistics. However, entity resolution decisions are determined
by a classification boundary separating matches and non-matches.

This work investigates whether instability of this decision boundary provides
an earlier and more meaningful indicator of ER drift.


---

# 2. Core Hypothesis

Entity resolution degradation is preceded by instability in the decision region.

The proposed hypothesis:

Drift
causes
similarity-space perturbation

which causes

decision boundary expansion

which causes

matching instability.


---

# 3. Synthetic Drift Generation

Three controlled drift scenarios are evaluated:

## Title Drift

Simulates changes in product/entity descriptions.

Examples:
- formatting changes
- token rearrangement
- lexical variation


## Naming Drift

Simulates changes in naming conventions.

Examples:
- abbreviations
- spelling variation
- alternative naming patterns


## Missingness Drift

Simulates degradation caused by increasing missing attributes.


---

# 4. Similarity Stream Construction

Each incoming record pair is transformed into a similarity vector.

The stream contains:

- matching pairs
- non-matching pairs
- similarity scores
- temporal ordering


The stream is evaluated sequentially to simulate production ER systems.


---

# 5. Boundary Representation

Instead of monitoring only score distributions,
the method models the uncertainty region around the classification threshold.

Boundary mass represents the proportion of samples approaching
the decision boundary.

Higher boundary mass indicates increased uncertainty.


---

# 6. Boundary Instability Metric

For stable and drift periods:

Boundary instability is measured as:

Increase =
BoundaryMass(drift)
-
BoundaryMass(stable)


A positive increase indicates expansion of the uncertain decision region.


---

# 7. Boundary-Aware Drift Detector

The detector combines:

## Mean statistics

Detect global similarity shifts.


## Variability statistics

Detect increased uncertainty.


## Entropy

Measure disorder in similarity assignments.


## Boundary mass

Capture decision-region instability.


The combined signal is evaluated against adaptive thresholds.


---

# 8. Baseline Comparison

The proposed method is compared against:

## ADWIN

Adaptive window-based distribution drift detector.


## Page-Hinkley

Sequential mean-change detector.


Evaluation metrics:

- Detection probability
- False alarm rate
- Detection delay


---

# 9. Statistical Validation

To evaluate robustness:

300 block-bootstrap repetitions are performed.

Configuration:

- repetitions: 300
- block size: 50
- fixed seed: 20260812


Confidence intervals are reported for:

- detection probability
- false alarm rate
- detection delay


Effect sizes are measured using Cliff's Delta.


---

# 10. Feature Ablation

The contribution of each boundary component is evaluated.

Variants:

- Boundary Only
- Entropy Only
- Mean Only
- Std Only
- Full model
- Removed feature variants


This identifies which signals contribute most to reliable detection.


---

# 11. Boundary Evolution Analysis

Boundary mass evolution is visualized before and after drift.

The analysis demonstrates that drift events correspond
with measurable changes in the decision surface.


---

# 12. Decision Stability Analysis

The effect of drift on ER quality is measured using:

- F1 score
- Precision
- Recall


The analysis connects:

Boundary instability

to

Decision instability.


---

# 13. Reproducibility

The repository provides:

- pinned dependencies
- deterministic random seeds
- experiment scripts
- generated result files
- automated validation tests


All statistical experiments can be reproduced from the provided environment.

---

# Boundary Instability Hypothesis

## Motivation

Entity resolution systems make decisions through confidence
thresholds. Therefore, model degradation does not necessarily
appear as a large global shift in score statistics.

Instead, degradation can manifest as increased uncertainty
around the decision boundary.

We define decision-boundary instability as the increase in
probability mass near the matching threshold.

## Boundary Mass

For confidence scores s:

B(t) = P(|s - τ| < ε)

where:

- s represents entity matching confidence,
- τ represents the decision threshold,
- ε represents the boundary neighbourhood width.

A rising boundary mass indicates that more candidate pairs
are becoming ambiguous.

## Multivariate Boundary-Aware Detection

The detector represents each score window using:

1. Mean score
2. Score variance
3. Boundary mass
4. Score entropy

The detector compares the current window representation
against a stable reference distribution using standardized
distance.

This allows detection of drift caused by changes in
decision ambiguity rather than only global score movement.

## Real Dataset Validation

The Amazon-Google entity resolution benchmark was used to
evaluate whether boundary instability exists in real-world
matching scenarios.

Analysis showed that ambiguous decisions concentrate in
high-confidence regions where:

- entropy increases,
- boundary mass increases,
- score distributions become less separable.

This supports the hypothesis that entity resolution degradation
is closely related to decision-boundary instability.