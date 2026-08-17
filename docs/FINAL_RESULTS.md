# Drift-Aware Entity Resolution
## Final Experimental Summary

## Overview

This work proposes a Boundary-Aware Drift Detection framework for Entity Resolution (ER) systems.

Unlike conventional drift detectors that monitor generic statistical changes, the proposed approach focuses on instability of the ER decision boundary, where small score distribution changes can alter matching decisions.

---

# Core Methodology

## 1. Entity Resolution Score Stream

Each entity pair is transformed into a matching score:

$$
s_i \in [0,1]
$$

forming a temporal score stream.

The score stream represents the operational behavior of the ER system.

---

## 2. Boundary Calibration

A decision threshold is calibrated from stable score distributions.

The calibrated threshold defines the operational decision region:

$$
\hat y_i =
\begin{cases}
1,& s_i \geq \tau\\
0,& otherwise
\end{cases}
$$

---

## 3. Boundary Instability Measurement

Two complementary measures are used.

### Boundary Mass Instability

Measures redistribution of scores near the decision boundary.

$$
\Delta M =
|M_{drift}-M_{stable}|
$$

---

### Decision Flip Rate

Measures actual ER decisions affected by drift.

$$
DFR =
\frac{1}{N}
\sum I(\hat y_{stable}\neq\hat y_{drift})
$$

This directly quantifies decision-boundary instability.

---

# Experimental Validation

## 1. Controlled Synthetic Drift

Evaluated on:

- Title degradation
- Naming drift
- Missingness drift

Results:

| Shift | Detection Rate | Mean Delay |
|---|---:|---:|
| Title | 98.67% | 64.7 |
| Naming | 85.67% | 210.7 |
| Missingness | 86.67% | 209.7 |

---

# 2. Amazon-Google Entity Resolution Benchmark

A real product matching dataset was evaluated under controlled score compression.

Results:

- Detection rate: 100%
- False alarm rate: 1.92%

Across all degradation severities, the detector successfully identified score instability.

---

# 3. DBLP-Scholar Cross-Domain Validation

A second real ER benchmark was evaluated to test generalization.

Results:

| Severity | Detection | Delay | Decision Flip Rate |
|---|---:|---:|---:|
| 0.95 | 100% | 93 | 11.38% |
| 0.90 | 100% | 18 | 19.27% |
| 0.85 | 100% | 18 | 25.50% |
| 0.80 | 100% | 18 | 27.74% |
| 0.75 | 100% | 18 | 28.49% |

At severe degradation, nearly 30% of ER decisions changed due to drift-induced boundary instability.

---

# Main Contributions

1. Boundary-aware drift detection specifically designed for Entity Resolution systems.

2. Decision Boundary Instability metric that quantifies operational impact rather than only distributional change.

3. Cross-domain validation on:
   - Synthetic drift scenarios
   - Amazon-Google product matching
   - DBLP-Scholar academic matching

4. Reproducible experimental pipeline with statistical validation and ablation analysis.