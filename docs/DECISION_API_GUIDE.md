# BlinkThink Decision API & Routing Guide

## How System 1 (BlinkEngine) Works
System 1 performs **0 autoregressive token decoding steps**. It constructs $N$ cyclic permutations of candidate options in a single batched prefill pass:
$$\sigma_b(j) = ((j - 1 + b) \bmod K) + 1$$

Candidate logits are gathered at the prefill boundary, calibrated via validation-tuned temperature $T^*$, and remapped back to canonical options:
$$\bar{\mathbf{p}} = \frac{1}{N} \sum_{b=0}^{N-1} \mathbf{p}^{(b)}$$

## Confidence Metric Formulations
Confidence $\mathcal{S}_1(\mathbf{x})$ can be configured using one of three metrics:
1. **Predictive Margin (`margin`):** $M = \bar{p}_{(1)} - \bar{p}_{(2)}$ *(Recommended default for failure detection)*
2. **Maximum Softmax Probability (`msp`):** $\max_k \bar{p}_k$
3. **Composite (`composite`):** $M \cdot \sqrt{\mathcal{C}_R \cdot \max(0, 1 - \mathcal{D})}$

## Tuning Routing Thresholds ($\tau^*$)
Use `BlinkThinkTuner` to find the knee-point threshold balancing latency and accuracy:

```python
from blinkthink import BlinkThinkRouter, BlinkThinkTuner

tuner = BlinkThinkTuner(router)
report = tuner.fit(val_dataset, n_blinks=2, objective="balanced")
print(f"Optimal Threshold: {report.optimal_threshold:.2f}")
```
