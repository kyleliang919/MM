"""
Example usage of Min-Min algorithm with base_train.py

This shows how to integrate the Min-Min two-stage minimization algorithm
into the existing training pipeline.
"""

import torch
from mm.minmin_optimizer import MinMinOptimizer, minmin_training_step, minmin_validation, MinMinScheduler
from mm.minmin_utils import MinMinBatchSampler, learning_rate_schedule


def integrate_minmin_into_training(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    loss_fn,
    device: torch.device,
    num_epochs: int = 10,
    inner_step_size: float = 0.1,  # rho
    outer_lr: float = 0.01,  # eta
    small_batch_size: int = 32,
    large_batch_size: int = 256,
):
    """
    Example training loop using Min-Min algorithm.
    
    Args:
        model: neural network model
        train_loader: training data loader
        val_loader: validation data loader
        loss_fn: loss function
        device: torch device
        num_epochs: number of training epochs
        inner_step_size: step size for inner minimization (rho)
        outer_lr: learning rate for outer minimization (eta)
        small_batch_size: size of small batch for inner step (b_s)
        large_batch_size: size of large batch for outer step (b_l)
    """
    
    # Create base optimizer (we use it as wrapper by MinMinOptimizer)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=outer_lr)
    
    # Create Min-Min optimizer
    minmin_opt = MinMinOptimizer(
        model=model,
        base_optimizer=base_optimizer,
        inner_step_size=inner_step_size,
        outer_lr=outer_lr,
        small_batch_size=small_batch_size,
        large_batch_size=large_batch_size,
    )
    
    # Create batch sampler that provides small/large batch pairs
    batch_sampler = MinMinBatchSampler(
        train_loader=train_loader,
        small_batch_size=small_batch_size,
        large_batch_size=large_batch_size,
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * num_epochs // (small_batch_size + large_batch_size)
    scheduler = MinMinScheduler(
        optimizer=minmin_opt,
        initial_inner_step=inner_step_size,
        initial_outer_lr=outer_lr,
        schedule_type="cosine",
        total_steps=total_steps,
    )
    
    # Training loop
    model.train()
    step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0
        
        while True:
            try:
                # Get small and large batch pair
                small_batch, large_batch = batch_sampler.get_pair()
                
                # Move to device
                small_x, small_y = small_batch
                large_x, large_y = large_batch
                
                small_x = small_x.to(device)
                small_y = small_y.to(device)
                large_x = large_x.to(device)
                large_y = large_y.to(device)
                
                # Perform Min-Min step
                loss = minmin_training_step(
                    model=model,
                    optimizer=minmin_opt,
                    small_batch=(small_x, small_y),
                    large_batch=(large_x, large_y),
                    loss_fn=loss_fn,
                )
                
                # Update learning rate schedule
                scheduler.step()
                
                epoch_loss += loss
                num_steps += 1
                step += 1
                
                # Logging
                if step % 100 == 0:
                    avg_loss = epoch_loss / num_steps
                    inner_lr, outer_lr = scheduler.get_last_lr()
                    print(f"Step {step:5d} | Loss: {avg_loss:.6f} | "
                          f"Inner LR: {inner_lr:.6f} | Outer LR: {outer_lr:.6f}")
            
            except StopIteration:
                break
        
        # Validation
        val_loss = minmin_validation(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss/num_steps:.6f} | "
              f"Val Loss: {val_loss:.6f}")
    
    return model


# Alternative: Lightweight integration into existing optimizer loop
def minmin_step_with_existing_loader(
    model: torch.nn.Module,
    batch1,
    batch2,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    inner_step_size: float = 0.1,
    device: torch.device = None,
):
    """
    Minimal Min-Min step that works with existing training loops.
    
    Can be dropped into any PyTorch training loop that already has:
    - model, optimizer, loss function
    - two consecutive batches from data loader
    
    Args:
        model: neural network
        batch1: first batch (small batch for inner step)
        batch2: second batch (large batch for outer step)
        loss_fn: loss function
        optimizer: any PyTorch optimizer
        inner_step_size: rho parameter
        device: torch device
    
    Example usage:
    ```
    for i, (batch1, batch2) in enumerate(zip(train_loader, train_loader)):
        loss = minmin_step_with_existing_loader(
            model, batch1, batch2, loss_fn, optimizer,
            inner_step_size=0.1, device=device
        )
    ```
    """
    x1, y1 = batch1
    x2, y2 = batch2
    
    if device is not None:
        x1, y1 = x1.to(device), y1.to(device)
        x2, y2 = x2.to(device), y2.to(device)
    
    # Save original parameters
    original_params = [p.data.clone() for p in model.parameters()]
    
    # ===== Inner step (small batch) =====
    optimizer.zero_grad()
    logits1 = model(x1)
    loss1 = loss_fn(logits1, y1)
    loss1.backward()
    
    # Perturb parameters: p = p - rho * g / ||g||
    for p in model.parameters():
        if p.grad is not None:
            g_norm = torch.norm(p.grad, p=2)
            if g_norm > 1e-8:
                p.data = p.data - inner_step_size * (p.grad / g_norm)
    
    # ===== Outer step (large batch at perturbed point) =====
    optimizer.zero_grad()
    logits2 = model(x2)
    loss2 = loss_fn(logits2, y2)
    loss2.backward()
    optimizer.step()
    
    return loss2.item()


if __name__ == "__main__":
    # Example setup
    print("Min-Min Algorithm Integration Example")
    print("=" * 50)
    print("\nAlgorithm: Two-Stage Minimization with Small/Large Batches")
    print("\n1. Inner Step (Small Batch):")
    print("   - Compute gradient on small batch")
    print("   - Normalize gradient: g_s / ||g_s||_2")
    print("   - Perturb: θ_tilde = θ + δ, where δ = -ρ * g_s / ||g_s||_2")
    print("\n2. Outer Step (Large Batch at perturbed point):")
    print("   - Compute gradient on large batch at θ_tilde")
    print("   - Update: θ_{t+1} = θ - η * g_ℓ")
    print("\n" + "=" * 50)
    print("\nKey Parameters:")
    print("  ρ (rho, inner_step_size): perturbation magnitude")
    print("  η (eta, outer_lr): main learning rate")
    print("  b_s (small_batch_size): inner batch size")
    print("  b_ℓ (large_batch_size): outer batch size")
    print("\n" + "=" * 50)
