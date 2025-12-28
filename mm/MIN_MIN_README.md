# Min-Min Algorithm Implementation

Two-Stage Minimization with Small and Large Batches

Based on: "Why is second order gradient useless" paper

## Algorithm Overview

The Min-Min algorithm performs two minimization steps per iteration:

### Step 1: Inner Minimization (Small Batch)
```
Compute small-batch gradient: g_s = ∇_θ L(θ_t; B_s)
Normalize: g̃_s = g_s / ||g_s||_2
Perturbation: δ_t = -ρ * g̃_s
Perturbed point: θ̃_t = θ_t + δ_t
```

### Step 2: Outer Minimization (Large Batch)
```
Compute large-batch gradient at perturbed point: g_ℓ = ∇_θ L(θ̃_t; B_ℓ)
Parameter update: θ_{t+1} = θ_t - η * g_ℓ
```

## Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Inner step size | ρ | 0.1 | Magnitude of perturbation in inner step |
| Outer learning rate | η | 0.01 | Learning rate for outer update |
| Small batch size | b_s | 32 | Batch size for inner minimization |
| Large batch size | b_ℓ | 256 | Batch size for outer minimization |

## Installation

The implementation is located in `/home/kylel/ARBP/mm/`:

```python
from mm.minmin_optimizer import MinMinOptimizer, minmin_training_step
from mm.minmin_utils import MinMinBatchSampler
```

## Usage

### Basic Usage with Custom Optimizer

```python
import torch
from mm.minmin_optimizer import MinMinOptimizer
from mm.minmin_utils import MinMinBatchSampler

# Setup
model = YourModel()
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
device = torch.device('cuda')

# Create Min-Min optimizer
minmin_opt = MinMinOptimizer(
    model=model,
    base_optimizer=base_optimizer,
    inner_step_size=0.1,      # ρ
    outer_lr=0.01,             # η
    small_batch_size=32,       # b_s
    large_batch_size=256,      # b_ℓ
)

# Create batch sampler for small/large batch pairs
batch_sampler = MinMinBatchSampler(
    train_loader=train_loader,
    small_batch_size=32,
    large_batch_size=256,
)

# Training loop
for epoch in range(num_epochs):
    for _ in range(num_steps):
        # Get small and large batch
        small_batch, large_batch = batch_sampler.get_pair()
        
        # Perform Min-Min step
        loss = minmin_training_step(
            model=model,
            optimizer=minmin_opt,
            small_batch=small_batch,
            large_batch=large_batch,
            loss_fn=loss_fn,
        )
        
        # Update learning rate schedule
        scheduler.step()
```

### Lightweight Integration into Existing Loop

```python
from mm.minmin_example import minmin_step_with_existing_loader

for i in range(0, len(train_loader) - 1, 2):
    batch1 = train_loader[i]
    batch2 = train_loader[i + 1]
    
    loss = minmin_step_with_existing_loader(
        model=model,
        batch1=batch1,
        batch2=batch2,
        loss_fn=loss_fn,
        optimizer=optimizer,
        inner_step_size=0.1,
        device=device,
    )
```

### With Learning Rate Scheduler

```python
from mm.minmin_optimizer import MinMinScheduler

scheduler = MinMinScheduler(
    optimizer=minmin_opt,
    initial_inner_step=0.1,
    initial_outer_lr=0.01,
    schedule_type="cosine",  # or "linear", "exponential", "constant"
    total_steps=total_training_steps,
)

for step in range(total_training_steps):
    # Training step
    loss = minmin_training_step(...)
    
    # Update schedule
    scheduler.step()
    
    # Get current learning rates
    inner_lr, outer_lr = scheduler.get_last_lr()
```

## Files

### Core Implementation
- **`minmin_optimizer.py`**: Main optimizer class and training functions
  - `MinMinOptimizer`: Two-stage optimizer wrapper
  - `minmin_training_step()`: Single step function
  - `minmin_validation()`: Validation without Min-Min
  - `MinMinScheduler`: Learning rate scheduler

### Utilities
- **`minmin_utils.py`**: Helper functions and utilities
  - `normalize_gradient()`: Normalize to unit norm
  - `perturbation_step()`: Compute perturbation
  - `MinMinBatchSampler`: Sample small/large batch pairs
  - `learning_rate_schedule()`: Various LR schedules

### Examples
- **`minmin_example.py`**: Complete usage examples

## Theory

### Why Min-Min?

1. **Inner Minimization**: Finding gradient on small batch and normalizing it explores the loss landscape with a unit-norm direction. This is more robust to batch noise.

2. **Perturbation**: Moving by distance ρ in the negative gradient direction samples a neighboring point. This allows escaping sharp minima.

3. **Outer Minimization**: Evaluating gradient at the perturbed point on a larger batch provides a more stable estimate that's less affected by small-batch noise.

4. **Combined Effect**: The algorithm effectively combines small-batch exploration (inner) with large-batch exploitation (outer), balancing sharpness and generalization.

### Related to SAM

Min-Min is similar to Sharpness Aware Minimization (SAM) but:
- Uses normalized gradient (unit norm) instead of fixed ε
- Operates on different batch sizes for inner/outer steps
- Can be more computationally efficient

## Performance Tips

1. **Batch Sizes**: Start with b_s << b_ℓ (e.g., 32 vs 256)
   - Smaller small-batch for exploratory perturbation
   - Larger large-batch for stable gradient estimate

2. **Step Sizes**: 
   - ρ (inner_step_size): Control exploration magnitude (0.05-0.2)
   - η (outer_lr): Control main optimization (standard LR values)

3. **Scheduling**:
   - Cosine schedule often works well for both ρ and η
   - Can decay ρ faster than η to focus on optimization later

4. **Batch Creation**:
   - Ensure b_s and b_ℓ don't overlap (disjoint batches)
   - Use `MinMinBatchSampler` to automatically handle this

## Experimental Results

The Min-Min algorithm is designed to:
- Improve generalization by escaping sharp minima
- Be more efficient than full second-order methods
- Work with standard optimizers (SGD, Adam, etc.)
- Scale to large models and datasets

## References

- Paper: "Why is second order gradient useless"
- Related: Sharpness Aware Minimization (SAM) - Foret et al., 2020

## Common Issues

### Issue: Out of memory
**Solution**: Reduce `large_batch_size` or reduce model size

### Issue: Training loss doesn't decrease
**Solution**: 
- Increase `inner_step_size` (ρ) to explore more
- Check that batches are being created correctly
- Verify loss function is working

### Issue: High variance in losses
**Solution**:
- Increase `large_batch_size` for more stable gradient estimates
- Use learning rate schedule with decay
- Consider batch normalization or gradient clipping

## Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now training will print detailed information
```

Check batch sampler:

```python
sampler = MinMinBatchSampler(train_loader, 32, 256)
small_batch, large_batch = sampler.get_pair()

print(f"Small batch shape: {small_batch[0].shape}")
print(f"Large batch shape: {large_batch[0].shape}")
```

## Contributing

To extend or modify the implementation:

1. Modify `minmin_optimizer.py` for core algorithm changes
2. Add utilities to `minmin_utils.py` for helper functions
3. Add examples to `minmin_example.py` for new use cases
4. Update tests in `tests/` directory

## License

Same as parent project
