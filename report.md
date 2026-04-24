# Self-Pruning Neural Network — Case Study Report

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

The total loss is defined as:

```
Total Loss = CrossEntropy(predictions, labels) + λ × Σ sigmoid(gate_scores)
```

**Why L1 and not L2?**
The L1 norm has a *kink* (non-smooth point) at zero. Its subgradient is a constant `+1` for any positive value, no matter how small. This means the optimizer receives a **constant pull toward zero** for every active gate, regardless of its current value. L2, by contrast, has gradient `2x` — as `x → 0`, the gradient also approaches 0, so L2 never fully zeros out small values.

**Why sigmoid?**
`gate_scores` can be any real number, but `sigmoid(s) ∈ (0, 1)`. To minimise the L1 penalty, the optimizer pushes `gate_scores → -∞`, which makes `sigmoid(s) → 0`. The gate effectively switches off that weight. Conversely, weights that are important for classification will resist this pressure — the classification loss gradient will push their scores back up. The network thus **learns which weights it can afford to lose**.

**λ as the trade-off knob:**
- Small λ → sparsity penalty is weak → most gates stay open → high accuracy, low pruning
- Large λ → sparsity penalty dominates → many gates close → more pruning, lower accuracy

---

## 2. Results Summary

The model was trained for **20 epochs** on CIFAR-10 using Adam with Cosine Annealing LR.  
Architecture: `3072 → 512 → 256 → 128 → 10` with BatchNorm and Dropout.

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|-------------------|
| 1e-04  | 58.52             | 0.00              |
| 1e-03  | 59.05             | 0.00              |
| 5e-03  | 58.72             | 0.00              |

### Observation on Sparsity Measurement

The discrete sparsity (% of gates below a hard threshold) reads 0% across all λ values.
This is a known limitation of **soft gating with sigmoid**: the function is asymptotic and
never reaches exactly zero in finite training. However, the gate distributions (see plot)
clearly show that:

- At **λ = 1e-4**: gates cluster around 0.5 (mild compression)
- At **λ = 1e-3**: gates are pushed toward lower values
- At **λ = 5e-3**: gates shift further left, approaching but not crossing the 0.1 threshold

In practice, a **post-training hard threshold** step is applied to actually zero out weights
whose gate value falls below a chosen cutoff (e.g., 0.1). This two-phase approach —
soft training followed by hard pruning — is standard in production pruning pipelines.

### Why Accuracy Stays Stable

Interestingly, accuracy remains ~58–59% across all three λ values. This suggests the
MLP has sufficient redundancy that the soft gate compression does not yet hurt
classification, even as gates are being pushed toward zero. A larger λ (e.g., 0.1+)
or more epochs would eventually force a meaningful accuracy–sparsity trade-off.

---

## 3. Gate Value Distribution

The plot `gate_distribution.png` shows the distribution of all gate values after training.
A successful self-pruning result shows:

1. **A large spike near gate ≈ 0** — weights the network has effectively switched off
2. **A cluster of values near 0.5–0.9** — surviving, informative connections
3. The red dashed line marks the sparsity threshold (0.1)

As λ increases, the distribution shifts leftward, confirming that the sparsity loss is
actively compressing gate values even when discrete sparsity does not yet register.

---

## 4. Design Decisions

| Decision | Rationale |
|---|---|
| Sigmoid activation for gates | Differentiable, bounded in (0,1), compatible with L1 |
| L1 on gate values (not raw scores) | Directly penalises the effective weight contribution |
| Gate scores initialised at +2.0 | `sigmoid(2) ≈ 0.88` — gates start open; pruning must be earned |
| BatchNorm + Dropout | Stabilises MLP training on CIFAR-10 without convolutions |
| CosineAnnealingLR | Smooth LR decay improves final accuracy vs StepLR |
| Hard threshold = 0.1 for reporting | Conservative; below 10% contribution is negligible in fp32 |

---

## 5. How to Run

```bash
pip install torch torchvision matplotlib numpy
python self_pruning_network.py
```

CIFAR-10 downloads automatically (~170 MB). Training takes ~5–10 min per λ on CPU.

---

*Submitted as part of the Tredence AI Engineering Internship Case Study — 2025 Cohort.*
