# Drift-Aware Entity Resolution

Research project exploring adaptive threshold calibration
for streaming entity resolution under distribution shift.

Pipeline:

Dataset
 -> Feature Extraction
 -> Static Matcher
 -> Similarity Score Stream
 -> Drift Detection
 -> Calibration
 -> Evaluation


---

## Novel Contribution: Decision-Boundary Instability Detection

Traditional drift detectors monitor global statistics such as mean
or variance shifts. However, entity resolution systems make
threshold-based decisions, where degradation often appears as
increasing ambiguity near the decision boundary.

This work introduces a boundary-aware drift detection framework
that monitors:

- Mean score distribution
- Score variance
- Boundary mass around the matching threshold
- Score entropy

The detector is evaluated through:

1. Controlled drift simulations:
   - title degradation
   - naming perturbation
   - missingness shift

2. Statistical validation:
   - 300 block-bootstrap repetitions
   - bootstrap confidence intervals
   - paired permutation testing
   - Cliff's delta effect sizes

3. Real-world validation:
   - Amazon-Google entity resolution benchmark

Results show that decision-boundary instability provides a
more informative signal of ER degradation than global score
statistics alone.