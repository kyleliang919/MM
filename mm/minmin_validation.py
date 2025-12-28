"""
Min-Min Algorithm Validation Metrics

Tracks whether Min-Min is actually working by measuring:
1. Gradient alignment: angle between small-batch and large-batch gradients
2. Perturbation effectiveness: how much the perturbation changes the gradient
3. Loss landscape smoothness: curvature at perturbation point
4. Gradient norm ratio: stability indicator
"""

import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, List
import math
import copy


def compute_gradient_alignment(
    grad1: List[Tensor],
    grad2: List[Tensor],
    eps: float = 1e-8,
) -> float:
    """
    Compute cosine similarity between two gradient vectors.
    
    Values closer to 1.0 mean gradients are aligned.
    Values closer to -1.0 mean gradients are opposite.
    
    Args:
        grad1: First gradient (list of parameter gradients)
        grad2: Second gradient (list of parameter gradients)
        eps: numerical stability
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Flatten and concatenate all gradients
    g1_flat = torch.cat([g.flatten() for g in grad1 if g is not None])
    g2_flat = torch.cat([g.flatten() for g in grad2 if g is not None])
    
    # Compute cosine similarity
    dot_product = torch.dot(g1_flat, g2_flat)
    norm1 = torch.norm(g1_flat)
    norm2 = torch.norm(g2_flat)
    
    cosine_sim = dot_product / (norm1 * norm2 + eps)
    return cosine_sim.item()


def compute_gradient_norm_ratio(
    grad_small: List[Tensor],
    grad_large: List[Tensor],
    eps: float = 1e-8,
) -> float:
    """
    Compute ratio of large-batch to small-batch gradient norms.
    
    Helps detect if there's high variance (small grad norm) or stable estimate (large grad norm).
    
    Args:
        grad_small: Small-batch gradient
        grad_large: Large-batch gradient
        eps: numerical stability
    
    Returns:
        Ratio of norms (typically > 1 indicates large batch is more stable)
    """
    norm_small = torch.cat([g.flatten() for g in grad_small if g is not None]).norm()
    norm_large = torch.cat([g.flatten() for g in grad_large if g is not None]).norm()
    
    return (norm_large / (norm_small + eps)).item()


def compute_loss_at_perturbation(
    model: torch.nn.Module,
    original_params: List[Tensor],
    perturbed_params: List[Tensor],
    batch: Tuple[Tensor, Tensor],
    loss_fn,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Compute loss at original and perturbed points.
    
    Useful for measuring loss landscape curvature.
    
    Args:
        model: neural network
        original_params: parameter values at original point
        perturbed_params: parameter values at perturbed point
        batch: (inputs, targets)
        loss_fn: loss function
        device: torch device
    
    Returns:
        Tuple of (loss_at_original, loss_at_perturbed)
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    
    # Compute loss at original point
    for p, p_orig in zip(model.parameters(), original_params):
        p.data = p_orig.data
    
    with torch.no_grad():
        logits_orig = model(x)
        loss_orig = loss_fn(logits_orig, y).item()
    
    # Compute loss at perturbed point
    for p, p_pert in zip(model.parameters(), perturbed_params):
        p.data = p_pert.data
    
    with torch.no_grad():
        logits_pert = model(x)
        loss_pert = loss_fn(logits_pert, y).item()
    
    return loss_orig, loss_pert


def compute_perturbation_magnitude(
    original_params: List[Tensor],
    perturbed_params: List[Tensor],
) -> float:
    """
    Compute L2 norm of perturbation.
    
    Shows how far we moved from original point.
    
    Args:
        original_params: original parameter values
        perturbed_params: perturbed parameter values
    
    Returns:
        L2 norm of total parameter change
    """
    diff = torch.cat([
        (p_pert - p_orig).flatten()
        for p_orig, p_pert in zip(original_params, perturbed_params)
    ])
    return torch.norm(diff).item()


def compute_sharpness_measure(
    model: torch.nn.Module,
    batch: Tuple[Tensor, Tensor],
    loss_fn,
    device: torch.device,
    perturbation_size: float = 0.01,
) -> float:
    """
    Estimate loss landscape sharpness using finite differences.
    
    Computes: sharpness ≈ ||∇²L|| (Hessian norm approximation)
    
    Args:
        model: neural network
        batch: (inputs, targets)
        loss_fn: loss function
        device: torch device
        perturbation_size: size of perturbation for finite difference
    
    Returns:
        Sharpness estimate (higher = sharper)
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    
    # Get gradients at current point
    model.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    
    grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
             for p in model.parameters()]
    
    # Estimate Hessian-vector products using finite differences
    hvp_norms = []
    
    for p_idx, (p, g) in enumerate(zip(model.parameters(), grads)):
        if g.norm() < 1e-8:
            continue
        
        # Normalize direction
        direction = g / (g.norm() + 1e-8)
        
        # Perturb in positive direction
        p.data = p.data + perturbation_size * direction
        model.zero_grad()
        logits_pert = model(x)
        loss_pert = loss_fn(logits_pert, y)
        loss_pert.backward()
        
        grad_pert = p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        
        # Restore parameters
        p.data = p.data - perturbation_size * direction
        
        # Hessian-vector product ≈ (∇f(θ + εd) - ∇f(θ)) / ε
        hvp = (grad_pert - g) / (perturbation_size + 1e-8)
        hvp_norms.append(hvp.norm().item())
    
    # Return average HVP norm as sharpness measure
    if hvp_norms:
        return sum(hvp_norms) / len(hvp_norms)
    else:
        return 0.0


def validate_minmin_step(
    model: torch.nn.Module,
    original_params: List[Tensor],
    perturbed_params: List[Tensor],
    small_grad: List[Tensor],
    large_grad: List[Tensor],
    small_loss: float,
    large_loss: float,
    small_batch_size: int,
    large_batch_size: int,
    inner_step_size: float,
) -> Dict[str, float]:
    """
    Comprehensive validation of Min-Min step effectiveness.
    
    Args:
        model: neural network
        original_params: parameters before perturbation
        perturbed_params: parameters after perturbation
        small_grad: gradient from small batch
        large_grad: gradient from large batch at perturbed point
        small_loss: loss on small batch
        large_loss: loss on large batch at perturbed point
        small_batch_size: size of small batch
        large_batch_size: size of large batch
        inner_step_size: magnitude of perturbation (ρ)
    
    Returns:
        Dictionary of validation metrics
    """
    metrics = {}
    
    # 1. Gradient alignment
    alignment = compute_gradient_alignment(small_grad, large_grad)
    metrics['grad_alignment'] = alignment
    
    # 2. Gradient norm ratio (stability indicator)
    norm_ratio = compute_gradient_norm_ratio(small_grad, large_grad)
    metrics['grad_norm_ratio'] = norm_ratio
    
    # 3. Perturbation magnitude
    perturbation = compute_perturbation_magnitude(original_params, perturbed_params)
    metrics['perturbation_magnitude'] = perturbation
    
    # 4. Loss change from small batch
    metrics['small_batch_loss'] = small_loss
    
    # 5. Loss at perturbed point
    metrics['large_batch_loss'] = large_loss
    
    # 6. Expected vs actual perturbation size
    expected_perturbation = inner_step_size
    metrics['perturbation_efficiency'] = expected_perturbation / (perturbation + 1e-8)
    
    # 7. Batch size ratio (exploration vs exploitation)
    metrics['batch_size_ratio'] = large_batch_size / small_batch_size
    
    # 8. Gradient norm estimates
    small_grad_norm = torch.cat([g.flatten() for g in small_grad if g is not None]).norm().item()
    large_grad_norm = torch.cat([g.flatten() for g in large_grad if g is not None]).norm().item()
    metrics['small_grad_norm'] = small_grad_norm
    metrics['large_grad_norm'] = large_grad_norm
    
    return metrics


def format_minmin_metrics(metrics: Dict[str, float]) -> str:
    """
    Format Min-Min validation metrics for logging.
    
    Args:
        metrics: dictionary of metrics from validate_minmin_step
    
    Returns:
        Formatted string for printing
    """
    lines = [
        "Min-Min Validation Metrics:",
        f"  Gradient Alignment: {metrics['grad_alignment']:.4f} (closer to 1.0 = better alignment)",
        f"  Gradient Norm Ratio: {metrics['grad_norm_ratio']:.4f} (>1 = large batch more stable)",
        f"  Perturbation Magnitude: {metrics['perturbation_magnitude']:.6f}",
        f"  Perturbation Efficiency: {metrics['perturbation_efficiency']:.4f}",
        f"  Batch Size Ratio (ℓ/s): {metrics['batch_size_ratio']:.1f}",
        f"  Small Grad Norm: {metrics['small_grad_norm']:.6f}",
        f"  Large Grad Norm: {metrics['large_grad_norm']:.6f}",
        f"  Small Batch Loss: {metrics['small_batch_loss']:.6f}",
        f"  Large Batch Loss: {metrics['large_batch_loss']:.6f}",
    ]
    return "\n".join(lines)


def should_validate_minmin(step: int, validate_every: int = 250) -> bool:
    """
    Check if we should run Min-Min validation at this step.
    
    Args:
        step: current training step
        validate_every: run validation every N steps
    
    Returns:
        True if validation should run
    """
    return step > 0 and step % validate_every == 0


def evaluate_minmin_validation(
    model: torch.nn.Module,
    val_loader,
    small_accum_steps: int,
    large_accum_steps: int,
    inner_step_size: float,
    autocast_ctx,
) -> Dict[str, float]:
    """
    Run Min-Min-style validation without applying the outer update.

    Uses a small-batch gradient to perturb, then evaluates large-batch loss
    at the perturbed point, and restores the original parameters.
    """
    batch_iter = iter(val_loader)

    val_model = copy.deepcopy(model)
    val_model.train()

    # Small batch loss + grads
    val_model.zero_grad(set_to_none=True)
    small_loss = torch.tensor(0.0, device=val_model.get_device())
    for _ in range(small_accum_steps):
        x, y = next(batch_iter)
        with autocast_ctx:
            loss = val_model(x, y)
        small_loss = small_loss + loss
    small_loss = small_loss / small_accum_steps
    small_loss.backward()

    orig_params = [p.data.clone() for p in val_model.parameters()]

    # Per-parameter perturbation
    for p in val_model.parameters():
        if p.grad is not None:
            g_norm = torch.norm(p.grad, p=2)
            if g_norm > 1e-8:
                normalized_grad = p.grad / g_norm
                p.data = p.data - inner_step_size * normalized_grad

    # Large batch loss at perturbed point (no update)
    large_batches = []
    with torch.no_grad():
        large_loss = torch.tensor(0.0, device=val_model.get_device())
        for _ in range(large_accum_steps):
            x, y = next(batch_iter)
            large_batches.append((x, y))
            with autocast_ctx:
                loss = val_model(x, y)
            large_loss = large_loss + loss
        large_loss = large_loss / large_accum_steps

    # Restore original params and clear grads
    for p, p_orig in zip(val_model.parameters(), orig_params):
        p.data = p_orig.data
    val_model.zero_grad(set_to_none=True)

    # Large batch loss at original params (same batches)
    with torch.no_grad():
        large_loss_orig = torch.tensor(0.0, device=val_model.get_device())
        for x, y in large_batches:
            with autocast_ctx:
                loss = val_model(x, y)
            large_loss_orig = large_loss_orig + loss
        large_loss_orig = large_loss_orig / large_accum_steps

    return {
        "small_loss": small_loss.item(),
        "large_loss": large_loss.item(),
        "large_loss_orig": large_loss_orig.item(),
        "large_loss_delta": (large_loss - large_loss_orig).item(),
    }
