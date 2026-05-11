import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD
import itertools

class SparseSGDM(SGD):
    """
    Implements Stochastic Gradient Descent with Momentum (SGDM)
    supporting gradient masking for sparse fine-tuning.
    This optimizer zeros out specific weight updates based on a binary mask
    to facilitate task-localized sparse fine-tuning.
    """
    def __init__(self, params, lr=0.001, momentum=0.9, weight_decay=0.0, masks=None):
        """
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor (standard SGDM).
            weight_decay: L2 penalty.
            masks: Dictionary mapping parameter objects to binary masks (1 to update, 0 to freeze).
        """
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.masks = masks

    def set_masks(self, masks):
        """
        Updates the masks used by the optimizer during training rounds.
        """
        self.masks = masks

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with gradient masking.
        """
        # Apply the gradient masks before the standard SGD update logic
        if self.masks is not None:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None and p in self.masks:
                        # Multiply gradient by mask in-place: grad = grad * mask
                        # This zeros out updates for 'least-sensitive' parameters.
                        p.grad.mul_(self.masks[p])

        # Call the parent SGD step to handle momentum and weight updates
        return super().step(closure)

@torch.no_grad()
def compute_fisher_sensitivity(model, dataloader, criterion, device, num_batches=10):
    """
    Computes the diagonal Fisher Information Matrix (FIM) as a proxy for parameter sensitivity.
    Sensitivity is defined as the empirical expected value of the squared gradients.

    Args:
        model: The neural network to evaluate.
        dataloader: Data source for sensitivity calibration.
        criterion: Loss function used to compute gradients.
        device: Hardware device (CPU/GPU) to perform computations on.
        num_batches: Number of data batches to average over for calibration.

    Returns:
        A dictionary mapping each trainable parameter to its sensitivity score tensor.
    """
    model.eval()  # Set to evaluation mode; note that gradients can still be computed.

    # 1. Initialize sensitivity score accumulators using a dictionary comprehension.
    # Only includes parameters that require gradients to save memory.
    sensitivity = {p: torch.zeros_like(p) for p in model.parameters() if p.requires_grad}

    # 2. Iterate through a limited number of batches using islice for efficiency.
    for inputs, targets in itertools.islice(dataloader, num_batches):
        inputs, targets = inputs.to(device), targets.to(device)

        # 3. Explicitly enable gradient calculation for the backward pass.
        with torch.enable_grad():
            model.zero_grad()  # Clear previous gradients before backpropagation.
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        # 4. Accumulate the square of the gradients.
        # Diagonal Fisher Information is approximately E[gradient^2].
        for p, score in sensitivity.items():
            if p.grad is not None:
                # Use in-place addition to minimize memory allocation.
                score.add_(p.grad.pow(2))

    # 5. Normalize scores by the number of batches to get the average sensitivity.
    # These scores help identify 'least-sensitive' parameters for sparse fine-tuning.
    return {p: s / num_batches for p, s in sensitivity.items()}


import torch


def calibrate_masks(model, strategy='least_sensitive', sparsity_ratio=0.1, sensitivity_scores=None):
    """
    Calibrates gradient masks based on various selection strategies for sparse fine-tuning. [cite: 69, 77]

    Args:
        model: The neural network model.
        strategy: Selection rule ('least_sensitive', 'most_sensitive', 'low_magnitude', 'high_magnitude', 'random'). [cite: 78]
        sparsity_ratio: Fraction of total parameters to update (Mask=1).
        sensitivity_scores: Pre-computed Fisher Information scores (required for sensitivity strategies). [cite: 53]
    """
    masks = {}
    params = [p for p in model.parameters() if p.requires_grad]

    # 1. Collect all evaluation values globally across all layers
    if strategy in ['least_sensitive', 'most_sensitive']:
        if sensitivity_scores is None: raise ValueError("Sensitivity scores required for this strategy.")
        all_values = torch.cat([s.view(-1) for s in sensitivity_scores.values()])
    elif strategy in ['low_magnitude', 'high_magnitude']:
        all_values = torch.cat([p.data.abs().view(-1) for p in params])
    elif strategy == 'random':
        all_values = torch.randn(sum(p.numel() for p in params))  # Generate random scores
    else:
        raise NotImplementedError(f"Strategy {strategy} not supported.")

    # 2. Determine the threshold for the top/bottom K elements
    num_params = all_values.numel()
    k = int(num_params * sparsity_ratio)

    # Sort or find the K-th value to define the boundary
    if strategy in ['least_sensitive', 'low_magnitude', 'random']:
        threshold = torch.kthvalue(all_values, k).values.item()
    else:  # most_sensitive or high_magnitude
        threshold = torch.kthvalue(all_values, num_params - k).values.item()

    # 3. Generate binary masks for each parameter
    for p in params:
        if strategy == 'least_sensitive':
            masks[p] = (sensitivity_scores[p] <= threshold).float()
        elif strategy == 'most_sensitive':
            masks[p] = (sensitivity_scores[p] >= threshold).float()
        elif strategy == 'low_magnitude':
            masks[p] = (p.data.abs() <= threshold).float()
        elif strategy == 'high_magnitude':
            masks[p] = (p.data.abs() >= threshold).float()
        elif strategy == 'random':
            # For random, we just generate a mask with the correct ratio
            masks[p] = (torch.rand_like(p) <= sparsity_ratio).float()

    return masks