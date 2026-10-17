"""
E4: Auxiliary boundary-heatmap prediction head.

Attaches a lightweight single-channel sigmoid output to the P3-level feature map.
The target is a thin band along annotated box edges.

This head is **train-time only** — it provides gradient signal to sharpen
boundary features but is bypassed/ignored at inference (zero inference cost).
Includes dynamic on-GPU target generation from augmented batch bounding boxes,
ensuring targets perfectly align with mosaic/mixup augmentations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class BoundaryHead(nn.Module):
    """Auxiliary boundary heatmap prediction head.

    Predicts a single-channel heatmap from the P3-level feature map.
    The prediction is supervised with a combined BCE + Dice loss.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()
        )
        self.latest_heatmap = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use gradient checkpointing to save VRAM
        if x.requires_grad:
            self.latest_heatmap = cp.checkpoint(self.branch, x, use_reentrant=False)
        else:
            self.latest_heatmap = self.branch(x)
        return x # Return input unmodified to seamlessly pass through to YOLO Detect head

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined BCE + Dice loss."""
        # Ensure pred and target shapes match exactly
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')
            
        bce = F.binary_cross_entropy(pred, target)
        dice = dice_loss(pred, target)
        
        total = self.alpha * bce + (1 - self.alpha) * dice
        return total, bce, dice


class InterceptWrapper(nn.Module):
    """Wraps a YOLO layer to intercept its output and feed it to BoundaryHead."""
    def __init__(self, original_module: nn.Module, boundary_head: nn.Module):
        super().__init__()
        self.original = original_module
        self.boundary = boundary_head
        # Proxy standard Ultralytics attributes
        self.i = getattr(original_module, 'i', -1)
        self.f = getattr(original_module, 'f', -1)
        self.type = getattr(original_module, 'type', 'InterceptWrapper')
        
    def forward(self, x):
        out = self.original(x)
        # We only need to run the boundary branch during training
        if self.training:
            self.boundary(out)
        return out


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary heatmaps."""
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def generate_boundary_targets_tensor(
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    thickness: int = 3,
) -> torch.Tensor:
    """Dynamically generate boundary heatmaps on GPU from normalized boxes.
    
    Parameters
    ----------
    bboxes : torch.Tensor
        Normalized boxes [cx, cy, w, h], shape (N, 4).
    batch_idx : torch.Tensor
        Batch indices for each box, shape (N,).
    batch_size, height, width : int
        Dimensions of the output tensor.
    thickness : int
        Thickness of the boundary band in pixels (on the feature map scale).
    """
    device = bboxes.device
    targets = torch.zeros((batch_size, 1, height, width), device=device)
    
    if len(bboxes) == 0:
        return targets
        
    # Convert normalized cx,cy,w,h to absolute pixel coordinates
    cx = bboxes[:, 0] * width
    cy = bboxes[:, 1] * height
    w = bboxes[:, 2] * width
    h = bboxes[:, 3] * height
    
    x1 = (cx - w / 2).long()
    y1 = (cy - h / 2).long()
    x2 = (cx + w / 2).long()
    y2 = (cy + h / 2).long()
    
    x1 = torch.clamp(x1, 0, width - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    x2 = torch.clamp(x2, 0, width - 1)
    y2 = torch.clamp(y2, 0, height - 1)
    
    # Half thickness
    t = max(1, thickness // 2)
    
    # Draw boxes
    for i in range(len(bboxes)):
        b = int(batch_idx[i].item())
        x1_i, y1_i, x2_i, y2_i = x1[i], y1[i], x2[i], y2[i]
        
        # Top edge
        targets[b, 0, max(0, y1_i-t):min(height, y1_i+t+1), x1_i:x2_i] = 1.0
        # Bottom edge
        targets[b, 0, max(0, y2_i-t):min(height, y2_i+t+1), x1_i:x2_i] = 1.0
        # Left edge
        targets[b, 0, y1_i:y2_i, max(0, x1_i-t):min(width, x1_i+t+1)] = 1.0
        # Right edge
        targets[b, 0, y1_i:y2_i, max(0, x2_i-t):min(width, x2_i+t+1)] = 1.0
        
    return targets
