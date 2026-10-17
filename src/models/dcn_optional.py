"""
E7: Optional deformable convolution ablation module.

Replaces selected standard convolution layers in the YOLOv8 backbone or
neck with Deformable Convolution v2 (DCNv2) to improve the model's ability
to adapt receptive fields to irregularly shaped xylem cells.

This is an optional ablation to test whether learned geometric deformations
improve detection accuracy for non-rectangular cell shapes.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.ops


class DeformableConv2d(nn.Module):
    """Deformable Convolution v2 wrapper.

    Learns per-pixel offsets and modulation scalars to deform the regular
    convolution grid, adapting to object shape.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    stride : int
        Convolution stride.
    padding : int
        Convolution padding.
    groups : int
        Number of groups for grouped convolution.
    bias : bool
        Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Standard convolution weights and bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Offset prediction conv (2 * kernel_size^2 channels per group)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size * groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        # Mask prediction conv (kernel_size^2 channels per group)
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weight similarly to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offset and mask convs to zero so it initially behaves as a standard conv
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learned offsets.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map, shape (B, C_in, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map, shape (B, C_out, H', W').
        """
        # Predict offsets and mask
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        # Perform deformable convolution
        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )


def replace_conv_with_dcn(
    model: nn.Module,
    layer_indices: Optional[List[int]] = None,
) -> None:
    """Dynamically replace selected Conv layers with DeformableConv2d.

    Parameters
    ----------
    model : nn.Module
        The PyTorch YOLOv8 model instance (e.g. detector.model.model).
    layer_indices : list of int, optional
        Indices of backbone layers to replace. If None, replaces the 
        standard 3x3 conv inside the last two C2f blocks in the backbone 
        (layers 6 and 8 in YOLOv8n).
    """
    if layer_indices is None:
        layer_indices = [6, 8]
    
    # In Ultralytics YOLOv8, the sequence of layers is typically in model.model
    if not hasattr(model, "model") or not isinstance(model.model, nn.Sequential):
        print("WARNING: Model does not have a sequential 'model' attribute. Skipping DCN injection.")
        return

    for idx in layer_indices:
        if idx >= len(model.model):
            print(f"WARNING: Layer index {idx} out of range. Skipping.")
            continue
            
        layer = model.model[idx]
        
        # We target the Bottleneck modules inside a C2f block
        # YOLOv8 C2f contains an 'm' attribute which is an nn.ModuleList of Bottlenecks
        if type(layer).__name__ == "C2f" and hasattr(layer, "m"):
            for i, bottleneck in enumerate(layer.m):
                # Bottleneck typically has cv1 and cv2. cv2 is usually the 3x3 conv.
                # Let's replace the cv2 convolution if it's a 3x3 standard Conv.
                if hasattr(bottleneck, "cv2") and hasattr(bottleneck.cv2, "conv"):
                    orig_conv = bottleneck.cv2.conv
                    if isinstance(orig_conv, nn.Conv2d) and orig_conv.kernel_size == (3, 3):
                        dcn = DeformableConv2d(
                            in_channels=orig_conv.in_channels,
                            out_channels=orig_conv.out_channels,
                            kernel_size=orig_conv.kernel_size[0],
                            stride=orig_conv.stride[0],
                            padding=orig_conv.padding[0],
                            groups=orig_conv.groups,
                            bias=(orig_conv.bias is not None),
                        )
                        # Copy original weights
                        dcn.weight.data.copy_(orig_conv.weight.data)
                        if dcn.bias is not None and orig_conv.bias is not None:
                            dcn.bias.data.copy_(orig_conv.bias.data)
                        
                        # Replace
                        bottleneck.cv2.conv = dcn
                        print(f"Injected DCNv2 into layer {idx}, Bottleneck {i}, cv2.")
