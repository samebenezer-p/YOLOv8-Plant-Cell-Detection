"""
E3: Lightweight bidirectional multi-scale fusion neck.

Replaces the default PAN-FPN neck in YOLOv8 with a more efficient
bidirectional feature fusion network (inspired by BiFPN / GFPN concepts)
to improve multi-scale feature aggregation for detecting xylem cells of
varying sizes.

Implemented using dynamic injection to seamlessly swap the stock YOLOv8
neck with this custom architecture before training. Includes gradient
checkpointing to stay within 4GB VRAM limits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class FastNormalizedFusion(nn.Module):
    """Fast normalized fusion with learnable weights."""
    def __init__(self, num_inputs: int = 2, epsilon: float = 1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        # Relu ensures weights are positive
        w = F.relu(self.weights)
        weight_sum = torch.sum(w) + self.epsilon
        
        out = features[0] * (w[0] / weight_sum)
        for i in range(1, len(features)):
            out += features[i] * (w[i] / weight_sum)
            
        return out


class BidirectionalFusionBlock(nn.Module):
    """Single bidirectional fusion block.

    Fuses features from P3, P4, P5 using top-down and bottom-up pathways
    with fast normalized fusion.
    """

    def __init__(self, channels: int = 256, epsilon: float = 1e-4):
        super().__init__()
        self.channels = channels
        
        # Top-down pathway
        self.td_fuser_p4 = FastNormalizedFusion(num_inputs=2, epsilon=epsilon)
        self.td_conv_p4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        self.td_fuser_p3 = FastNormalizedFusion(num_inputs=2, epsilon=epsilon)
        self.td_conv_p3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Bottom-up pathway
        self.bu_fuser_p4 = FastNormalizedFusion(num_inputs=3, epsilon=epsilon)
        self.bu_conv_p4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        self.bu_fuser_p5 = FastNormalizedFusion(num_inputs=2, epsilon=epsilon)
        self.bu_conv_p5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Top-down
        p4_td = self.td_conv_p4(self.td_fuser_p4(p4, self.upsample(p5)))
        p3_out = self.td_conv_p3(self.td_fuser_p3(p3, self.upsample(p4_td)))
        
        # Bottom-up
        p4_out = self.bu_conv_p4(self.bu_fuser_p4(p4, p4_td, self.downsample(p3_out)))
        p5_out = self.bu_conv_p5(self.bu_fuser_p5(p5, self.downsample(p4_out)))
        
        return p3_out, p4_out, p5_out


class FusionNeck(nn.Module):
    """Lightweight bidirectional multi-scale fusion neck.

    Replaces PAN-FPN with a bidirectional feature pyramid that performs
    both top-down and bottom-up feature fusion with learnable weights.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_repeats: int = 2,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_repeats = num_repeats
        
        # 1x1 convs to unify channel dimensions to `out_channels`
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False) 
            for in_channels in in_channels_list
        ])
        
        # Bidirectional fusion blocks
        self.blocks = nn.ModuleList([
            BidirectionalFusionBlock(channels=out_channels)
            for _ in range(num_repeats)
        ])

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # Unify channels
        p3 = self.lateral_convs[0](features[0])
        p4 = self.lateral_convs[1](features[1])
        p5 = self.lateral_convs[2](features[2])
        
        # Apply blocks iteratively
        for block in self.blocks:
            # Checkpoint the block forward pass to save VRAM (Memory-Safe mode)
            if p3.requires_grad:
                p3, p4, p5 = cp.checkpoint(block, p3, p4, p5, use_reentrant=False)
            else:
                p3, p4, p5 = block(p3, p4, p5)
            
        return p3, p4, p5


class _Select(nn.Module):
    """Helper module to extract a specific index from a tuple output.
    Used to seamlessly bridge our FusionNeck back into the standard YOLO sequential.
    """
    def __init__(self, index: int):
        super().__init__()
        self.index = index
        
    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return x[self.index]


def inject_fusion_neck(detection_model: nn.Module, num_repeats: int = 2, out_channels: int = 256) -> None:
    """Dynamically replace YOLOv8's standard neck with FusionNeck.

    Parameters
    ----------
    detection_model : nn.Module
        The ``DetectionModel`` (typically ``yolo.model.model``).
    num_repeats : int
        Number of bidirectional fusion blocks.
    out_channels : int
        Channel capacity of the fusion neck.
    """
    model_seq = detection_model.model
    # Identify backbone output channels by doing a dry-run inference on the backbone
    dummy = torch.zeros(1, 3, 64, 64)
    y = []
    
    # Run through backbone layers (0-9 in YOLOv8)
    for i in range(10):
        dummy = model_seq[i](dummy)
        if i in [4, 6, 9]:
            y.append(dummy)
            
    in_channels_list = [f.shape[1] for f in y]
    
    # Initialize our custom neck
    fusion_neck = FusionNeck(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        num_repeats=num_repeats
    )
    # Tell YOLO router that this module takes inputs from layers 4, 6, 9
    fusion_neck.f = [4, 6, 9]
    fusion_neck.i = 10
    
    # Bridge modules to extract the outputs of the neck
    sel0 = _Select(0)
    sel0.f, sel0.i = 10, 11
    
    sel1 = _Select(1)
    sel1.f, sel1.i = 10, 12
    
    sel2 = _Select(2)
    sel2.f, sel2.i = 10, 13
    
    # Update the Detect head to read from our bridge modules
    det = model_seq[-1]
    det.f = [11, 12, 13]
    det.i = 14
    
    # Before overwriting, make sure we adjust the Detect head channels if we changed them
    # The default YOLO detect head has lists of in_channels and intermediate convs.
    # It takes [256, 512, 1024] for 'n' model? Actually YOLOv8 uses different logic. 
    # Let's adjust the detect head's channel parameters dynamically.
    if hasattr(det, 'cv2') and hasattr(det, 'cv3'):
        for i in range(3):
            # YOLOv8 Detect uses cv2 and cv3 ModuleLists for bbox and cls
            # Replace the inner conv layer to accept our out_channels
            cv2_first = det.cv2[i][0]
            cv3_first = det.cv3[i][0]
            
            if hasattr(cv2_first, 'conv'):
                new_cv2 = nn.Conv2d(out_channels, cv2_first.conv.out_channels, 
                                   kernel_size=cv2_first.conv.kernel_size, stride=cv2_first.conv.stride, 
                                   padding=cv2_first.conv.padding, bias=(cv2_first.conv.bias is not None))
                cv2_first.conv = new_cv2
                
            if hasattr(cv3_first, 'conv'):
                new_cv3 = nn.Conv2d(out_channels, cv3_first.conv.out_channels, 
                                   kernel_size=cv3_first.conv.kernel_size, stride=cv3_first.conv.stride, 
                                   padding=cv3_first.conv.padding, bias=(cv3_first.conv.bias is not None))
                cv3_first.conv = new_cv3
            
    # Swap out layers 10 through 21 (the old neck) and insert our neck
    new_seq = nn.ModuleList([
        *model_seq[:10],
        fusion_neck,
        sel0, sel1, sel2,
        det
    ])
    
    # We must operate on the parent module to replace the Sequential/ModuleList properly
    # Ultralytics puts it in `model.model` which is actually an iterable ModuleList-like.
    for i in range(len(new_seq)):
        new_seq[i].i = i
        
    # YOLOv8 expects an iterable/Sequential for model.model
    detection_model.model = nn.Sequential(*new_seq)
    # The custom model.save list tells YOLO which layers' outputs to save for later use
    detection_model.save = [4, 6, 9, 10, 11, 12, 13]
