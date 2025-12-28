"""
Min-Min Algorithm Package

Two-Stage Minimization with Small/Large Batches
From: "Why is second order gradient useless"

Key Components:
- MinMinOptimizer: Main optimizer class
- minmin_training_step: Single training step function
- MinMinBatchSampler: Utility for sampling small/large batch pairs
- Learning rate schedulers and utilities
"""

from .minmin_optimizer import (
    MinMinOptimizer,
    minmin_training_step,
    minmin_validation,
    MinMinScheduler,
    create_minmin_dataloaders,
)

from .minmin_utils import (
    normalize_gradient,
    perturbation_step,
    apply_perturbation,
    restore_parameters,
    save_parameters,
    compute_batch_gradient,
    minmin_step_with_closure,
    MinMinBatchSampler,
    compute_sharpness,
    learning_rate_schedule,
)

__all__ = [
    # Optimizer
    'MinMinOptimizer',
    'MinMinScheduler',
    
    # Training functions
    'minmin_training_step',
    'minmin_validation',
    'create_minmin_dataloaders',
    
    # Utilities
    'normalize_gradient',
    'perturbation_step',
    'apply_perturbation',
    'restore_parameters',
    'save_parameters',
    'compute_batch_gradient',
    'minmin_step_with_closure',
    'MinMinBatchSampler',
    'compute_sharpness',
    'learning_rate_schedule',
]

__version__ = "1.0.0"
__author__ = "Min-Min Implementation"
