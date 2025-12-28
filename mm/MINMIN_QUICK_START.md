"""
Min-Min Integration into base_train.py - Quick Start Guide
"""

# To use the Min-Min algorithm in base_train.py:

# 1. Enable Min-Min at runtime:
python mm/base_train.py use_minmin=True

# 2. Or set custom hyperparameters:
python mm/base_train.py \
    use_minmin=True \
    minmin_inner_step_size=0.1 \
    minmin_outer_lr=0.01 \
    minmin_small_batch_size=32 \
    minmin_large_batch_size=256

# Key Parameters:
# ρ (rho, minmin_inner_step_size): Magnitude of perturbation in inner step
#   - Typical range: 0.05 - 0.2
#   - Controls how far to explore in gradient direction
#   - Higher = more exploration, lower = closer to original point

# η (eta, minmin_outer_lr): Learning rate for outer (main) optimization step
#   - Typical range: 0.001 - 0.1
#   - Controls step size for large-batch gradient

# b_s (minmin_small_batch_size): Batch size for inner minimization
#   - Smaller batch for faster computation, more noise
#   - Typical: 32-64

# b_ℓ (minmin_large_batch_size): Batch size for outer minimization
#   - Larger batch for stable gradient estimate
#   - Typical: 256-512
#   - Usually b_ℓ >> b_s for good exploration-exploitation tradeoff

# Algorithm Flow (per training iteration):
# 
# 1. INNER STEP (Small Batch - Exploration):
#    - Sample small batch B_s of size b_s
#    - Compute gradient: g_s = ∇L(θ_t; B_s)
#    - Normalize: g̃_s = g_s / ||g_s||_2
#    - Perturb: θ̃_t = θ_t - ρ * g̃_s
#
# 2. OUTER STEP (Large Batch - Exploitation):
#    - Sample large batch B_ℓ of size b_ℓ (disjoint from B_s)
#    - Compute gradient at perturbed point: g_ℓ = ∇L(θ̃_t; B_ℓ)
#    - Update: θ_{t+1} = θ_t - η * g_ℓ

# Example commands:

# Minimal settings (fast, exploratory):
python mm/base_train.py use_minmin=True depth=12 num_iterations=1000

# Conservative settings (stable):
python mm/base_train.py \
    use_minmin=True \
    minmin_inner_step_size=0.05 \
    minmin_outer_lr=0.005 \
    depth=12

# Aggressive settings (strong exploration):
python mm/base_train.py \
    use_minmin=True \
    minmin_inner_step_size=0.2 \
    minmin_large_batch_size=512 \
    depth=12

# Distributed training with Min-Min:
torchrun --nproc_per_node=8 mm/base_train.py \
    use_minmin=True \
    minmin_inner_step_size=0.1 \
    depth=12

# Compare with standard training (baseline):
python mm/base_train.py use_minmin=False depth=12 num_iterations=1000

# Troubleshooting:
# 
# If training diverges:
#   - Reduce minmin_inner_step_size (ρ)
#   - Reduce minmin_outer_lr (η)
#   - Increase minmin_large_batch_size (b_ℓ)
#
# If training is too slow:
#   - Increase minmin_small_batch_size (b_s)
#   - Increase minmin_large_batch_size (b_ℓ)
#   - Reduce model size
#
# If validation loss doesn't improve:
#   - Increase minmin_inner_step_size (ρ) for more exploration
#   - Try different batch size ratios
#   - Adjust learning rate schedule

# Expected improvements with Min-Min:
# - Better generalization (smoother loss landscape)
# - More stable convergence
# - Potentially faster convergence (depends on hyperparameters)
# - More robust to batch noise

# Integration Details:
# - Enabled via use_minmin=True flag (default: False)
# - Uses standard Muon/Adam optimizers for outer step
# - Supports gradient accumulation
# - Works with mixed precision (bfloat16)
# - Compatible with DDP (distributed training)
# - Checkpoints save Min-Min config in user_config
