"""
Min-Min Utilities: Helper functions for two-stage minimization with small/large batches.
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Callable, Optional, Dict, Any
import math


def normalize_gradient(gradient: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize a gradient tensor to unit norm.
    
    Args:
        gradient: gradient tensor to normalize
        eps: small value for numerical stability
    
    Returns:
        Normalized gradient with L2 norm = 1
    """
    grad_norm = torch.norm(gradient, p=2)
    if grad_norm > eps:
        return gradient / grad_norm
    else:
        return gradient


def perturbation_step(
    params: list,
    gradients: list,
    inner_step_size: float,
    eps: float = 1e-8,
) -> list:
    """
    Compute perturbation for inner minimization step.
    
    Implements: delta_t = -rho * g_s / ||g_s||_2
    
    Args:
        params: list of parameter tensors
        gradients: list of gradient tensors
        inner_step_size: rho (step size for perturbation)
        eps: numerical stability constant
    
    Returns:
        List of perturbed parameter tensors
    """
    perturbed_params = []
    
    for p, g in zip(params, gradients):
        if g is None:
            perturbed_params.append(p.clone())
            continue
        
        # Normalize gradient
        g_norm = torch.norm(g, p=2)
        
        if g_norm > eps:
            # delta = -rho * g / ||g||_2
            normalized_g = g / g_norm
            perturbation = -inner_step_size * normalized_g
            perturbed_p = p + perturbation
        else:
            # If gradient is zero, don't perturb
            perturbed_p = p.clone()
        
        perturbed_params.append(perturbed_p)
    
    return perturbed_params


def apply_perturbation(
    model: nn.Module,
    perturbation: list,
) -> None:
    """
    Apply perturbation to model parameters.
    
    Args:
        model: neural network model
        perturbation: list of perturbed parameter tensors
    """
    param_idx = 0
    for p in model.parameters():
        if param_idx < len(perturbation):
            p.data = perturbation[param_idx].data
            param_idx += 1


def restore_parameters(
    model: nn.Module,
    original_params: list,
) -> None:
    """
    Restore model parameters to original values.
    
    Args:
        model: neural network model
        original_params: list of original parameter tensors
    """
    param_idx = 0
    for p in model.parameters():
        if param_idx < len(original_params):
            p.data = original_params[param_idx].data
            param_idx += 1


def save_parameters(model: nn.Module) -> list:
    """
    Save current model parameters.
    
    Args:
        model: neural network model
    
    Returns:
        List of cloned parameter tensors
    """
    return [p.data.clone() for p in model.parameters()]


def compute_batch_gradient(
    model: nn.Module,
    batch: Tuple[Tensor, Tensor],
    loss_fn: Callable,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Compute gradient for a batch.
    
    Args:
        model: neural network model
        batch: tuple of (inputs, targets)
        loss_fn: loss function
        device: torch device
    
    Returns:
        Scalar loss value
    """
    x, y = batch
    
    if device is not None:
        x = x.to(device) if isinstance(x, Tensor) else x
        y = y.to(device) if isinstance(y, Tensor) else y
    
    logits = model(x)
    loss = loss_fn(logits, y)
    return loss


def minmin_step_with_closure(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    small_batch: Tuple[Tensor, Tensor],
    large_batch: Tuple[Tensor, Tensor],
    loss_fn: Callable,
    inner_step_size: float,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Perform one complete Min-Min step using PyTorch optimizer closure.
    
    This version works with optimizers that support closure (like LBFGS).
    
    Args:
        model: neural network model
        optimizer: PyTorch optimizer
        small_batch: tuple of (inputs, targets) for small batch
        large_batch: tuple of (inputs, targets) for large batch
        loss_fn: loss function
        inner_step_size: step size for perturbation (rho)
        device: torch device
    
    Returns:
        Dictionary with 'small_loss' and 'large_loss' keys
    """
    # ===== Inner Step =====
    # Save original parameters
    original_params = save_parameters(model)
    
    # Forward on small batch
    optimizer.zero_grad()
    small_loss = compute_batch_gradient(model, small_batch, loss_fn, device)
    small_loss.backward()
    
    # Extract gradients
    small_grads = [p.grad.clone() if p.grad is not None else None 
                   for p in model.parameters()]
    
    # Compute perturbation: delta = -rho * g / ||g||_2
    small_batch_x, small_batch_y = small_batch
    if device is not None:
        small_batch_x = small_batch_x.to(device) if isinstance(small_batch_x, Tensor) else small_batch_x
        small_batch_y = small_batch_y.to(device) if isinstance(small_batch_y, Tensor) else small_batch_y
    
    for p, g in zip(model.parameters(), small_grads):
        if g is None or torch.norm(g) < 1e-8:
            continue
        
        # Normalize gradient and apply perturbation
        g_normalized = g / (torch.norm(g) + 1e-8)
        p.data = p.data - inner_step_size * g_normalized
    
    # ===== Outer Step =====
    optimizer.zero_grad()
    large_loss = compute_batch_gradient(model, large_batch, loss_fn, device)
    large_loss.backward()
    
    # Update using optimizer
    optimizer.step()
    
    return {
        'small_loss': small_loss.item(),
        'large_loss': large_loss.item(),
    }


class MinMinBatchSampler:
    """
    Sampler that provides small and large batches for Min-Min algorithm.
    
    Ensures that small and large batches are disjoint (don't overlap).
    """
    
    def __init__(
        self,
        train_loader,
        small_batch_size: int,
        large_batch_size: int,
        shuffle: bool = True,
    ):
        self.train_loader = train_loader
        self.small_batch_size = small_batch_size
        self.large_batch_size = large_batch_size
        self.shuffle = shuffle
        self.iterator = None
        self.buffer = []
    
    def get_pair(self) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Get a pair of (small_batch, large_batch).
        
        Returns:
            Tuple of ((small_inputs, small_targets), (large_inputs, large_targets))
        """
        # Ensure we have enough samples
        while len(self.buffer) < (self.small_batch_size + self.large_batch_size):
            try:
                if self.iterator is None:
                    self.iterator = iter(self.train_loader)
                batch = next(self.iterator)
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, None
                self.buffer.append((x, y))
            except StopIteration:
                # Restart iterator
                self.iterator = None
                if len(self.buffer) == 0:
                    raise
        
        # Split buffer into small and large batches
        small_batch = (
            torch.cat([self.buffer[i][0] for i in range(self.small_batch_size)]),
            torch.cat([self.buffer[i][1] for i in range(self.small_batch_size)])
            if self.buffer[0][1] is not None else None,
        )
        
        large_batch = (
            torch.cat([self.buffer[self.small_batch_size + i][0] 
                      for i in range(self.large_batch_size)]),
            torch.cat([self.buffer[self.small_batch_size + i][1] 
                      for i in range(self.large_batch_size)])
            if self.buffer[0][1] is not None else None,
        )
        
        # Remove used samples from buffer
        self.buffer = self.buffer[self.small_batch_size + self.large_batch_size:]
        
        return small_batch, large_batch
    
    def reset(self):
        """Reset the sampler."""
        self.iterator = None
        self.buffer = []


def compute_sharpness(
    model: nn.Module,
    loss: Tensor,
    batch_size: int = 100,
) -> float:
    """
    Compute measure of loss landscape sharpness.
    
    Uses Hessian eigenvalue as proxy for sharpness.
    
    Args:
        model: neural network model
        loss: loss tensor
        batch_size: number of samples for estimation
    
    Returns:
        Estimated sharpness (related to max Hessian eigenvalue)
    """
    # Simple sharpness estimation using gradient magnitude
    # In practice, this is faster than computing actual Hessian eigenvalues
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grad_norm_sq = sum(g.pow(2).sum() for g in grads if g is not None)
    
    return grad_norm_sq.item()


def learning_rate_schedule(
    initial_lr: float,
    step: int,
    total_steps: int,
    schedule_type: str = "cosine",
    warmup_steps: int = 0,
) -> float:
    """
    Compute learning rate with schedule.
    
    Args:
        initial_lr: initial learning rate
        step: current step
        total_steps: total training steps
        schedule_type: type of schedule ("constant", "linear", "cosine", "exponential")
        warmup_steps: number of warmup steps
    
    Returns:
        Learning rate for current step
    """
    # Warmup phase
    if step < warmup_steps and warmup_steps > 0:
        return initial_lr * (step / warmup_steps)
    
    # Main schedule
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    
    if schedule_type == "constant":
        return initial_lr
    elif schedule_type == "linear":
        return initial_lr * (1.0 - progress)
    elif schedule_type == "cosine":
        return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule_type == "exponential":
        return initial_lr * (0.1 ** progress)
    else:
        return initial_lr
