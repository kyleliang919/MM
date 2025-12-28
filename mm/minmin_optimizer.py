"""
Min-Min Algorithm: Two-Stage Minimization with Small/Large Batches

This implements the Min-Min algorithm from "Why is second order gradient useless":
1. First minimization (inner): Compute gradient on small batch, normalize, move in that direction
2. Second minimization (outer): Compute gradient on large batch at perturbed point, perform gradient step

This is a SAM (Sharpness Aware Minimization) variant that helps escape sharp minima by:
- Perturbing parameters using small-batch gradient direction
- Evaluating larger batch gradient at perturbed point
- Updating based on large-batch gradient (which is more stable)
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Tuple, Callable, Optional


class MinMinOptimizer:
    """
    Min-Min Two-Stage Minimization Optimizer.
    
    Performs two minimization steps per iteration:
    1. Inner (small batch): Perturbation step along normalized small-batch gradient
    2. Outer (large batch): Gradient step using large-batch gradient at perturbed point
    
    Args:
        model: neural network model to optimize
        base_optimizer: base optimizer (e.g., torch.optim.SGD, torch.optim.Adam)
        inner_step_size (rho): step size for inner minimization (small-batch perturbation)
        outer_lr (eta): learning rate for outer minimization (large-batch update)
        small_batch_size (b_s): size of small batch for inner step
        large_batch_size (b_l): size of large batch for outer step
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        inner_step_size: float = 0.1,
        outer_lr: Optional[float] = None,
        small_batch_size: int = 32,
        large_batch_size: int = 256,
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.inner_step_size = inner_step_size  # rho in algorithm
        self.outer_lr = outer_lr  # eta in algorithm (if None, use base_optimizer lr)
        self.small_batch_size = small_batch_size
        self.large_batch_size = large_batch_size
        
        # Store original parameters for later restoration
        self.param_groups = self.base_optimizer.param_groups
        
        # State tracking
        self.step_count = 0
        self.in_inner_step = False  # Track which stage we're in
    
    def zero_grad(self):
        """Clear gradients."""
        self.base_optimizer.zero_grad()
    
    def inner_step(self, loss: Tensor) -> Tensor:
        """
        Inner minimization step (first minimization with small batch).
        Computes small-batch gradient, normalizes it, and perturbs parameters.
        
        Args:
            loss: scalar loss computed on small batch
        
        Returns:
            The loss value
        """
        # Compute gradient on small batch
        self.base_optimizer.zero_grad()
        loss.backward()
        
        # Store original parameters for restoration after inner step
        self.stored_params = []
        for group in self.param_groups:
            for p in group['params']:
                self.stored_params.append(p.data.clone())
        
        # Inner step: move along normalized gradient direction by rho
        # delta_t = -rho * g_s / ||g_s||_2
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    param_idx += 1
                    continue
                
                g = p.grad.data
                g_norm = torch.norm(g, p=2)
                
                # Only perturb if gradient is non-zero
                if g_norm > 0:
                    # delta_t = -rho * g_s / ||g_s||_2
                    normalized_grad = g / g_norm
                    step = -self.inner_step_size * normalized_grad
                    p.data = self.stored_params[param_idx] + step
                else:
                    # g_s == 0, no perturbation
                    p.data = self.stored_params[param_idx]
                
                param_idx += 1
        
        self.in_inner_step = True
        return loss
    
    def outer_step(self, loss: Tensor) -> Tensor:
        """
        Outer minimization step (second minimization with large batch).
        Computes large-batch gradient at perturbed point and updates parameters.
        
        Args:
            loss: scalar loss computed on large batch at perturbed parameters
        
        Returns:
            The loss value
        """
        # Compute gradient on large batch at perturbed point
        self.base_optimizer.zero_grad()
        loss.backward()
        
        # Outer step: actual parameter update using large-batch gradient
        # theta_{t+1} = theta_t - eta * g_l
        if self.outer_lr is not None:
            # Custom learning rate for outer step
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.data = self.stored_params[self.param_groups.index(group)] - self.outer_lr * p.grad.data
        else:
            # Use base optimizer with its default lr
            self.base_optimizer.step()
        
        self.in_inner_step = False
        self.step_count += 1
        
        return loss
    
    def step(self):
        """Step the base optimizer."""
        if not self.in_inner_step:
            self.base_optimizer.step()


def create_minmin_dataloaders(
    base_train_loader,
    base_val_loader,
    small_batch_size: int = 32,
    large_batch_size: int = 256,
) -> Tuple[Callable, Callable]:
    """
    Create wrapper functions that return appropriate batch sizes for min-min.
    
    Since the actual DataLoader is fixed, we'll return functions that can subsample
    from available batches or combine batches as needed.
    
    Args:
        base_train_loader: original training data loader
        base_val_loader: original validation data loader
        small_batch_size: desired small batch size for inner step
        large_batch_size: desired large batch size for outer step
    
    Returns:
        Tuple of (get_small_batch_fn, get_large_batch_fn)
    """
    # Store iterator state
    small_batch_buffer = []
    large_batch_buffer = []
    
    def get_small_batch():
        """Get a batch for the inner (small-batch) step."""
        return next(iter(base_train_loader), None)
    
    def get_large_batch():
        """Get a batch for the outer (large-batch) step."""
        return next(iter(base_train_loader), None)
    
    return get_small_batch, get_large_batch


def minmin_training_step(
    model: nn.Module,
    optimizer: MinMinOptimizer,
    small_batch: Tuple[Tensor, Tensor],
    large_batch: Tuple[Tensor, Tensor],
    loss_fn: Callable,
) -> float:
    """
    Perform one Min-Min training step (both inner and outer minimizations).
    
    Args:
        model: neural network model
        optimizer: MinMinOptimizer instance
        small_batch: tuple of (inputs, targets) for small batch
        large_batch: tuple of (inputs, targets) for large batch
        loss_fn: loss function that takes (model(x), y) and returns scalar loss
    
    Returns:
        Loss value from outer (large-batch) step
    """
    small_x, small_y = small_batch
    large_x, large_y = large_batch
    
    # ===== Inner Step (First Minimization with Small Batch) =====
    # Compute small-batch gradient and perturb parameters
    optimizer.zero_grad()
    small_logits = model(small_x)
    small_loss = loss_fn(small_logits, small_y)
    optimizer.inner_step(small_loss)
    
    # ===== Outer Step (Second Minimization with Large Batch) =====
    # Compute gradient at perturbed point using large batch
    optimizer.zero_grad()
    large_logits = model(large_x)
    large_loss = loss_fn(large_logits, large_y)
    optimizer.outer_step(large_loss)
    
    return large_loss.item()


def minmin_validation(
    model: nn.Module,
    val_loader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """
    Validate model using standard approach (no min-min during validation).
    
    Args:
        model: neural network model
        val_loader: validation data loader
        loss_fn: loss function
        device: torch device
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x = batch
                y = batch  # fallback
            
            x = x.to(device) if isinstance(x, Tensor) else x
            y = y.to(device) if isinstance(y, Tensor) else y
            
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


class MinMinScheduler:
    """
    Learning rate scheduler for Min-Min algorithm.
    
    Controls both inner step size (rho) and outer learning rate (eta).
    """
    
    def __init__(
        self,
        optimizer: MinMinOptimizer,
        initial_inner_step: float = 0.1,
        initial_outer_lr: float = 0.01,
        schedule_type: str = "cosine",
        total_steps: int = 1000,
    ):
        self.optimizer = optimizer
        self.initial_inner_step = initial_inner_step
        self.initial_outer_lr = initial_outer_lr
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.step_count = 0
    
    def step(self):
        """Update learning rates based on schedule."""
        if self.schedule_type == "constant":
            lr_factor = 1.0
        elif self.schedule_type == "linear":
            lr_factor = 1.0 - (self.step_count / self.total_steps)
        elif self.schedule_type == "cosine":
            import math
            lr_factor = 0.5 * (1.0 + math.cos(math.pi * self.step_count / self.total_steps))
        elif self.schedule_type == "exponential":
            decay_rate = 0.99
            lr_factor = decay_rate ** (self.step_count / 100)
        else:
            lr_factor = 1.0
        
        self.optimizer.inner_step_size = self.initial_inner_step * lr_factor
        if self.optimizer.outer_lr is not None:
            self.optimizer.outer_lr = self.initial_outer_lr * lr_factor
        
        self.step_count += 1
    
    def get_last_lr(self) -> Tuple[float, float]:
        """Get current learning rates."""
        return (self.optimizer.inner_step_size, self.optimizer.outer_lr or self.initial_outer_lr)
