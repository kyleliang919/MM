# Min-Min Algorithm Validation Metrics Guide

## Overview
During training with `minmin_mode=on`, the algorithm automatically validates every N steps (controlled by `minmin_validate_every`) to ensure it's working correctly.

## Key Validation Metrics

### 1. **Gradient Alignment** (Range: -1 to 1)
- **What it measures**: Cosine similarity between small-batch and large-batch gradients
- **Interpretation**:
  - **Close to 1.0**: Gradients are well-aligned (GOOD) → Large batch gradient similar to small batch
  - **Close to 0.0**: Gradients are orthogonal → Different optimization directions
  - **Close to -1.0**: Gradients are opposite (BAD) → Conflicting directions
- **Min-Min's goal**: Keep alignment reasonably high (0.7+) while still exploring
- **Typical value**: 0.6-0.9 is healthy

### 2. **Gradient Norm Ratio** (ℓ/s, typically > 1)
- **What it measures**: Ratio of large-batch gradient norm to small-batch gradient norm
- **Interpretation**:
  - **Ratio > 1**: Large batch gradient is stronger (GOOD) → Indicates large batch is more stable
  - **Ratio ≈ 1**: Similar magnitude
  - **Ratio < 1**: Small batch gradient is stronger (watch out) → May indicate instability
- **Min-Min's goal**: Ratio should be consistently > 1.2, showing large batch stability
- **Why it matters**: Indicates whether the larger batch provides better gradient estimates
- **Typical value**: 1.5-3.0

### 3. **Perturbation Magnitude** 
- **What it measures**: L2 norm of change in parameters (how far we moved)
- **Interpretation**:
  - **Too small** (< 0.001): Perturbation has minimal effect
  - **Reasonable** (0.01-0.1): Good exploration
  - **Too large** (> 0.5): May be escaping the optimization region
- **Min-Min's goal**: Controlled perturbation that explores without diverging
- **Typical value**: 0.01-0.05 for language models

### 4. **Small Batch Loss** vs **Large Batch Loss**
- **What it measures**: Loss values at small and large batches respectively
- **Interpretation**:
  - **Large loss > Small loss**: Normal, small batch is noisier
  - **Similar loss**: May need larger batch size for better distinction
  - **Drastically different**: May indicate batch distribution mismatch
- **Min-Min's goal**: Large batch should be more reliable, small batch more exploratory
- **Typical pattern**: Large loss should be 10-30% higher than small loss

### 5. **Gradient Norms** (Small vs Large)
- **What it measures**: Magnitude of gradients from each batch
- **Interpretation**:
  - **Decreasing over time**: Learning is converging
  - **Stable**: Good optimization
  - **Increasing**: May indicate divergence or hard phase
- **Min-Min's goal**: Both should decrease over training, with large > small
- **Watch for**: Sudden spikes indicating loss landscape changes

## Example Output

```
Step 00250 | Min-Min Validation Metrics:
  Gradient Alignment: 0.7324 (closer to 1.0 = better alignment)
  Gradient Norm Ratio: 1.8421 (>1 = large batch more stable)
  Perturbation Magnitude: 0.0234
  Perturbation Efficiency: 4.2667
  Batch Size Ratio (ℓ/s): 8.0
  Small Grad Norm: 0.0456
  Large Grad Norm: 0.0839
  Small Batch Loss: 4.5234
  Large Batch Loss: 4.6123
```

### Reading this example:
✅ **Good signs:**
- Gradient alignment = 0.73 (healthy, > 0.6)
- Gradient norm ratio = 1.84 (large batch 1.84x stronger, good stability)
- Perturbation magnitude = 0.023 (controlled exploration)
- Batch ratio = 8.0 (8x more data in large batch, good)
- Large grad norm > small grad norm (expected)

⚠️ **Things to watch:**
- Alignment < 0.5 → Gradients diverging, may need smaller ρ
- Norm ratio < 1.0 → Small batch is stronger, unusual
- Perturbation > 0.5 → Exploration too aggressive
- Large loss << small loss → Batch distribution very different

## Interpreting Trends Over Time

### Healthy Training:
```
Step 250:  alignment=0.73, norm_ratio=1.84, pert_mag=0.023
Step 500:  alignment=0.75, norm_ratio=1.92, pert_mag=0.021
Step 750:  alignment=0.77, norm_ratio=1.98, pert_mag=0.019
```
✅ Alignment increasing, norm ratio stable/increasing, perturbation decreasing

### Concerning Pattern 1: Diverging Gradients
```
Step 250:  alignment=0.73, norm_ratio=1.84
Step 500:  alignment=0.45, norm_ratio=0.92
Step 750:  alignment=0.12, norm_ratio=0.35
```
❌ Action: Reduce `minmin_inner_step_size` (ρ)

### Concerning Pattern 2: Unstable Large Batch
```
Step 250:  norm_ratio=1.84
Step 500:  norm_ratio=0.98
Step 750:  norm_ratio=0.45
```
❌ Action: Increase `minmin_large_batch_size` (b_ℓ)

## Configuration Adjustments Based on Metrics

| Symptom | Cause | Fix |
|---------|-------|-----|
| alignment < 0.5 | Perturbation too aggressive | Decrease ρ |
| alignment > 0.95 | Perturbation too conservative | Increase ρ |
| norm_ratio < 1.2 | Large batch not stable | Increase b_ℓ |
| norm_ratio > 5.0 | Small batch too noisy | Increase b_s |
| pert_mag > 0.5 | Step size too large | Decrease ρ |
| pert_mag < 0.001 | Step size too small | Increase ρ |

## How to Run with Validation

```bash
# Enable Min-Min with default validation (every 250 steps)
python mm/base_train.py minmin_mode=on

# Custom validation frequency
python mm/base_train.py minmin_mode=on minmin_validate_every=100

# Disable validation (faster training)
python mm/base_train.py minmin_mode=on minmin_validate_every=-1

# All custom parameters
python mm/base_train.py \
    minmin_mode=on \
    minmin_inner_step_size=0.1 \
    minmin_outer_lr=0.01 \
    minmin_small_batch_size=32 \
    minmin_large_batch_size=256 \
    minmin_validate_every=250
```

## Viewing Metrics in Wandb

When using W&B, all validation metrics are logged automatically with the prefix `minmin/`:
- `minmin/grad_alignment`
- `minmin/grad_norm_ratio`
- `minmin/perturbation_magnitude`
- `minmin/small_batch_loss`
- `minmin/large_batch_loss`
- `minmin/small_grad_norm`
- `minmin/large_grad_norm`

Create custom charts in W&B dashboard to track these over time.

## Performance Impact

- **No validation** (minmin_validate_every=-1): Minimal overhead
- **Validation every 250 steps**: ~2-5% training slowdown
- **Validation every 100 steps**: ~5-10% training slowdown

Choose validation frequency based on your needs vs compute budget.
