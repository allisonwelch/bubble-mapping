# core/UNet.py
# This module implements U-Net, a deep learning architecture for image segmentation.
# U-Net excels at pixel-level prediction tasks (e.g., "which pixels belong to my target?").
# Key innovation: skip connections let the decoder use fine details from the encoder.

from typing import Iterable, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    ConvBlock: The basic building block of U-Net. Performs two rounds of 3x3 convolution,
    followed by batch normalization and optional dropout.

    Why 3x3 convolution?
    - 3x3 is the smallest kernel that "sees" neighbors in all 8 directions (like a pixel looking at
      its 8 neighbors on a grid).
    - Two 3x3 convs stacked together have the same reach as one 5x5 conv, but use fewer parameters
      and compute (more efficient).

    Batch Normalization (BN):
    - After convolutions, the outputs can have wildly different scales. BN normalizes each feature
      map to have mean~0 and std~1 within each batch, stabilizing learning and allowing faster
      training. Think of it as "resetting the dial" to a standard range.
    - eps=1e-3 and momentum=0.01: These are numerical stability tweaks. eps prevents division by zero;
      momentum controls how the running average is updated (0.01 = heavily weighted to old stats).
    - affine=True: Allows learnable shift (gamma) and scale (beta) after normalization—lets the network
      undo normalization if it wants.
    - track_running_stats=True: During training, BN uses batch statistics. During inference, it uses
      the accumulated running average (gives stable predictions on single images).

    Dropout:
    - A regularization trick: randomly zero out activations during training with probability p.
    - Why? Forces the network to not rely on any single neuron; prevents "co-adaptation" where neurons
      learn to compensate for each other (overfitting). Like a team where any member must be able to do
      the job alone.
    - Not applied during inference (dropout is disabled), so predictions use the full network.

    Two 3x3 convs instead of one:
    - More non-linearity (ReLU activations) between them = more expressive feature learning.
    - Stacking is key to deep learning: depth helps learn hierarchical features.
    """

    def __init__(
        self,
        in_channels: int,  # Number of input feature maps (e.g., 3 for RGB image, 64 for encoder layer)
        out_channels: int,  # Number of output feature maps after convolution
        dilation: int = 1,  # Dilation factor: how spread out the kernel looks (explained below)
        dropout: float = 0.0,  # Probability of dropping activations (0 = no dropout, 0.5 = 50% of neurons zeroed)
    ) -> None:
        super().__init__()

        # FIRST CONVOLUTION: 3x3 kernel with dilation
        # How convolution works (analogy):
        # Imagine sliding a 3x3 "window" (kernel) across the image. At each position, multiply
        # overlapping pixels by kernel weights and sum them up. This produces one output value.
        # Do this for all positions → new feature map. The kernel learns these weights during training.
        #
        # kernel_size=3: Classic choice. 3x3 is the smallest kernel seeing 8 neighbors.
        # padding=dilation: Add zeros around the edge to keep the output image same size as input.
        #   Without padding, output shrinks. padding=dilation ensures output shape = input shape.
        # dilation=dilation: Controls spacing of kernel weights. Dilation >1 means the kernel "skips" pixels.
        #   Example: dilation=2 means kernel sees positions (0,2,4) instead of (0,1,2) — sees a wider area
        #   with fewer weights. Useful for large-scale context without adding compute.
        # bias=True: Add a learnable offset term to each output (helps model fit data better).
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )

        # SECOND CONVOLUTION: Same settings as conv1
        # Why apply convolution twice? Two reasons:
        #   1. More layers = more non-linearity (ReLU in forward()) = richer feature learning.
        #   2. Two 3x3 convs have a "receptive field" of 5x5 (can see a 5x5 area), matching a single 5x5
        #      with fewer parameters. Efficiency win.
        # Input channels = out_channels (output of conv1 fed into conv2)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )

        # BATCH NORMALIZATION: Standardize feature map statistics
        # See class docstring for details. Only one BN here (applied after both convs).
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
        )

        # DROPOUT: Randomly zero activations during training (regularization)
        # self.do is either a Dropout layer (if dropout > 0) or an Identity layer (does nothing).
        # Identity is used when dropout=0 so we don't need an if-statement in forward().
        self.do = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DATA FLOW: Conv → ReLU → Conv → ReLU → BatchNorm → Dropout → Output

        # Apply first convolution and ReLU activation
        # ReLU (Rectified Linear Unit) = max(0, x): replaces negative values with 0.
        # Why ReLU? It's non-linear, allowing the network to learn complex patterns.
        # If all operations were linear (y = ax + b), the whole network would collapse to a single
        # linear function. Non-linearity = expressiveness.
        x = F.relu(self.conv1(x), inplace=False)

        # Apply second convolution and ReLU activation
        # Two ReLU activations = two "non-linear transformations" = more modeling capacity.
        x = F.relu(self.conv2(x), inplace=False)

        # Apply batch normalization to stabilize and accelerate training
        # Normalizes feature maps before dropout so magnitudes are consistent.
        x = self.bn(x)

        # Apply dropout (if enabled; otherwise is Identity layer, which does nothing)
        # Dropout is typically applied AFTER batch norm in modern architectures.
        x = self.do(x)

        return x


class AttentionGate2D(nn.Module):
    """
    Attention Gate: Learn which parts of a skip connection to use.

    Background: U-Net combines decoder features (upsampling) with encoder features (skip connections).
    Skip connections preserve fine details (like edges). But not all spatial locations are equally
    important for the final prediction. Attention gates learn to highlight relevant regions and
    suppress irrelevant ones.

    How it works (high level):
    - Two inputs: x (skip connection from encoder) and g (upsampled decoder).
    - x and g are spatially aligned (same resolution).
    - Network learns: "Which pixels in x should I emphasize or suppress?"
    - Output: x weighted by an attention map (0 = ignore, 1 = keep).

    Why use attention gates?
    1. Focus: The model learns to attend to decision-relevant parts (similar to human attention).
    2. Suppress clutter: Down-weight features that aren't useful for the task.
    3. Better gradients: Attention provides a "soft mask" that helps backprop through skip connections.
    """

    def __init__(
        self,
        in_channels_x: int,  # Channels in skip connection x (encoder feature map)
        in_channels_g: int,  # Channels in gating signal g (decoder feature map)
        inter_channels: Optional[int] = None,  # Bottleneck channels for dimensionality reduction
    ) -> None:
        super().__init__()

        # Default bottleneck size: 1/4 of g's channels
        # Reduces compute while learning attention. Smaller bottleneck = more compression.
        if inter_channels is None:
            inter_channels = max(1, in_channels_g // 4)

        # 1x1 convolution: projects skip connection x to bottleneck space
        # kernel_size=1 means no spatial mixing, just channel mixing (like a per-pixel fully-connected layer).
        # Why 1x1? Efficient way to change channel dimensions without enlarging computation.
        self.theta_x = nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, bias=True)

        # 1x1 convolution: projects gating signal g to bottleneck space
        # Both theta_x and phi_g map to inter_channels so they can be added (see forward).
        self.phi_g = nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, bias=True)

        # 1x1 convolution: projects attention signal to single channel (attention map)
        # Output has 1 channel, so it's a single attention weight per pixel.
        # Will be passed through sigmoid in forward() to get values in [0, 1].
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

    @staticmethod
    def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # Utility function: Resize src to match ref's spatial dimensions (height, width).
        # ref.shape[-2:] extracts the last two dimensions (H, W; skips batch and channel).
        # mode="bilinear": Interpolation method. Bilinear blends nearby pixels (smooth resizing).
        # align_corners=False: Controls how interpolation handles image corners (False = standard behavior).
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # ATTENTION GATE FORWARD PASS
        # Goal: Return x weighted by an attention map that highlights relevant regions.

        # Step 1: Project skip connection x to bottleneck space (inter_channels)
        # This reduces dimensionality, making the computation efficient.
        theta_x = self.theta_x(x)

        # Step 2: Project gating signal g to same bottleneck space
        # g represents context from deeper (upsampled) decoder layer.
        phi_g = self.phi_g(g)

        # Step 3: Resize if spatial dimensions don't match
        # Sometimes x and g have different resolutions despite being "aligned".
        # Resize g to match x's spatial size so they can be combined (element-wise addition).
        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = self._resize_like(phi_g, theta_x)

        # Step 4: Combine signals and apply non-linearity
        # theta_x + phi_g: Fuse the skip connection and gating signal.
        # ReLU: Non-linear combination (max(0, sum)) allows learning complex attention patterns.
        # Intuition: Where theta_x + phi_g is large/positive, attention is high.
        #           Where it's negative or zero, attention is low.
        f = F.relu(theta_x + phi_g, inplace=False)

        # Step 5: Project to single-channel attention map
        # psi has shape (N, 1, H, W): one value per spatial location.
        # Still unbounded (can be negative or large).
        psi = self.psi(f)

        # Step 6: Apply sigmoid to convert to attention weights in [0, 1]
        # Sigmoid(z) = 1 / (1 + exp(-z)): Squashes any value to (0, 1).
        # Interpretation: 0 = fully ignore, 1 = fully keep, 0.5 = neutral.
        # Example: if some pixels are not useful for segmentation, attention ~0 suppresses them.
        #          if some pixels are critical, attention ~1 amplifies them.
        rate = torch.sigmoid(psi)  # Shape: (N, 1, H, W), values in [0, 1]

        # Step 7: Element-wise multiply skip connection by attention weights
        # x * rate: Broadcast rate (1 channel) across all x channels.
        # Each spatial location is scaled by its attention weight.
        # Result: x is re-weighted; irrelevant regions are suppressed, relevant ones enhanced.
        return x * rate


class UNetAttention(nn.Module):
    """
    U-Net with Attention Gates: Full Encoder-Decoder Segmentation Model

    ARCHITECTURE OVERVIEW:
    U-Net is named for its "U" shape:
      - LEFT SIDE (Encoder): Progressively DOWNSAMPLES the image, extracting semantic features.
        Each layer sees larger regions (receptive field grows), learning "what" is in the image.
      - CENTER (Bottleneck): Deepest layer, highest semantic understanding, smallest spatial resolution.
      - RIGHT SIDE (Decoder): Progressively UPSAMPLES back to original resolution, refining spatial
        predictions using skip connections.
      - SKIP CONNECTIONS (u-shape "legs"): Copy encoder features to decoder layers at matching
        resolutions. Preserves fine details (edges, textures) that encoder downsampling would lose.

    Why this design?
    - Downsampling extracts high-level features but loses spatial detail.
    - Skip connections restore detail without re-computing the encoder.
    - Result: Rich semantic features + fine spatial precision = accurate pixel-wise predictions.

    CHANNEL GROWTH PATTERN:
    Layer_count (lc) controls network capacity. Channels grow as we go deeper (lc → 2*lc → 4*lc → ... → 16*lc)
    because deeper layers need more features to capture the increasing complexity.

    ATTENTION GATES:
    Standard U-Net concatenates skip features directly. Attention gates learn which skip features
    matter, suppressing irrelevant noise. This focuses gradients on decision-relevant regions.
    """

    def __init__(
        self,
        in_channels: int,  # Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        num_classes: int,  # Number of output classes/channels (e.g., 1 for binary segmentation)
        dilation_rate: int = 1,  # Dilation for encoder convolutions (see ConvBlock for explanation)
        layer_count: int = 64,  # Base channels; actual channels = layer_count * multiplier (1x, 2x, 4x, ...)
        dropout: float = 0.0,  # Dropout probability for regularization
        l2_weight: float = 1e-4,  # L2 regularization weight (for weight decay)
    ) -> None:
        super().__init__()
        lc = layer_count  # Abbreviation for readability

        # ==================== ENCODER (Downsampling Path) ====================
        # Goal: Extract hierarchical features, reducing spatial resolution while increasing semantic understanding.

        # ENCODER LEVEL 1: Input → 1*lc channels, 1/1 resolution
        # enc1: ConvBlock outputs (1*lc) features. Learns low-level patterns (edges, corners).
        self.enc1 = ConvBlock(in_channels, 1 * lc, dilation=dilation_rate, dropout=dropout)
        # pool1: MaxPool2d(2) reduces spatial dimensions by 2 (e.g., 256x256 → 128x128).
        # "Max" pooling: takes max value in each 2x2 window. Keeps strongest activations, discards weak ones.
        # Why downsample? Larger receptive field (context) with fewer parameters.
        self.pool1 = nn.MaxPool2d(2)

        # ENCODER LEVEL 2: Downsampled input → 2*lc channels, 1/2 resolution
        # Input: pooled features (half resolution). Learns mid-level patterns (textures, shapes).
        self.enc2 = ConvBlock(1 * lc, 2 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        # ENCODER LEVEL 3: Further downsampled → 4*lc channels, 1/4 resolution
        # Higher-level semantic features.
        self.enc3 = ConvBlock(2 * lc, 4 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        # ENCODER LEVEL 4: Further downsampled → 8*lc channels, 1/8 resolution
        # Deep semantic features.
        self.enc4 = ConvBlock(4 * lc, 8 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        # BOTTLENECK (Center): 1/16 resolution, 16*lc channels
        # Deepest layer. Combines all downsampled information into compressed semantic representation.
        # No pooling after this; this is where encoding ends.
        self.center = ConvBlock(8 * lc, 16 * lc, dilation=dilation_rate, dropout=dropout)

        # ==================== ATTENTION GATES ====================
        # Each gate learns which skip connection features to amplify/suppress.
        # (x_channels, g_channels): x = skip connection, g = gating signal (upsampled decoder).

        # att6: Attends to 8*lc skip (from enc4) using 16*lc signal (from center upsampling).
        self.att6 = AttentionGate2D(in_channels_x=8 * lc, in_channels_g=16 * lc)
        # att7: Attends to 4*lc skip (from enc3) using 8*lc signal.
        self.att7 = AttentionGate2D(in_channels_x=4 * lc, in_channels_g=8 * lc)
        # att8: Attends to 2*lc skip (from enc2) using 4*lc signal.
        self.att8 = AttentionGate2D(in_channels_x=2 * lc, in_channels_g=4 * lc)
        # att9: Attends to 1*lc skip (from enc1) using 2*lc signal.
        self.att9 = AttentionGate2D(in_channels_x=1 * lc, in_channels_g=2 * lc)

        # ==================== DECODER (Upsampling Path) ====================
        # Goal: Progressively upsample back to original resolution while fusing with encoder skip connections.

        # DECODER LEVEL 6: Center + upsampled → outputs 8*lc
        # Input channels: 16*lc (upsampled center) + 8*lc (attended skip from enc4) = 24*lc
        # Concatenation (dim=1) merges features along channel axis.
        self.dec6 = ConvBlock(in_channels=24 * lc, out_channels=8 * lc, dilation=1, dropout=dropout)

        # DECODER LEVEL 7: Upsampled + skip → outputs 4*lc
        # Input: 8*lc (upsampled dec6) + 4*lc (attended skip from enc3) = 12*lc
        self.dec7 = ConvBlock(in_channels=12 * lc, out_channels=4 * lc, dilation=1, dropout=dropout)

        # DECODER LEVEL 8: Upsampled + skip → outputs 2*lc
        # Input: 4*lc (upsampled dec7) + 2*lc (attended skip from enc2) = 6*lc
        self.dec8 = ConvBlock(in_channels=6 * lc, out_channels=2 * lc, dilation=1, dropout=dropout)

        # DECODER LEVEL 9: Upsampled + skip → outputs 1*lc (back to original resolution)
        # Input: 2*lc (upsampled dec8) + 1*lc (attended skip from enc1) = 3*lc
        self.dec9 = ConvBlock(in_channels=3 * lc, out_channels=1 * lc, dilation=1, dropout=dropout)

        # ==================== OUTPUT HEAD ====================
        # Final 1x1 convolution: maps 1*lc features to num_classes predictions (one per class).
        # kernel_size=1: No spatial mixing, pure channel transformation.
        # Output shape: (batch_size, num_classes, height, width)
        self.head = nn.Conv2d(1 * lc, num_classes, kernel_size=1, bias=True)

        # L2 weight decay: Regularization that encourages small weights (prevents overfitting).
        # Typically applied via optimizer parameter weight_decay.
        self.l2_weight = float(l2_weight) if l2_weight is not None else 0.0

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # Utility function: Resize x to match ref's spatial dimensions (height, width).
        # Used in decoder to ensure skip connections can be concatenated with upsampled features.
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FORWARD PASS: Image → Encoder → Bottleneck → Decoder → Segmentation Map
        # Variable naming: c = feature map at level, p = pooled (downsampled), u = upsampled

        # ==================== ENCODER PATH ====================
        # Extract features at progressively larger receptive fields, saving intermediates for skip connections.

        # Level 1: Input image → 1*lc feature maps (no pooling)
        # c1: Stores encoder output (used later as skip connection in decoder).
        c1 = self.enc1(x)
        # p1: Downsampled to 1/2 resolution, fed to next encoder level.
        p1 = self.pool1(c1)

        # Level 2: 1/2 resolution, 2*lc features
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        # Level 3: 1/4 resolution, 4*lc features
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        # Level 4: 1/8 resolution, 8*lc features
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        # Bottleneck (Center): 1/16 resolution, 16*lc features
        # This is the "deepest" encoding. No pooling after—decoder upsamples from here.
        c5 = self.center(p4)

        # ==================== DECODER PATH + ATTENTION GATES ====================
        # Progressively upsample back to original resolution, fusing encoder features via attention gates.

        # DECODER STAGE 6: Upsample bottleneck + fuse with enc4 skip
        # u6: Upsample center from 1/16 → 1/8 resolution
        # scale_factor=2.0: Double each dimension (H×2, W×2).
        u6 = F.interpolate(c5, scale_factor=2.0, mode="bilinear", align_corners=False)
        # Ensure u6 exactly matches c4 spatial dimensions (sometimes interpolation differs by 1 pixel)
        u6 = self._upsample_like(u6, c4)
        # a6: Apply attention gate to c4 skip, using u6 as gating signal
        # Output: c4 re-weighted by attention learned from u6.
        a6 = self.att6(c4, u6)
        # Concatenate upsampled decoder (u6, 16*lc) with attended skip (a6, 8*lc) → 24*lc
        # torch.cat(..., dim=1): concatenate along channel dimension (dim=1).
        # Then pass through dec6 to process and reduce channels from 24*lc → 8*lc.
        c6 = self.dec6(torch.cat([u6, a6], dim=1))

        # DECODER STAGE 7: Upsample level-6 + fuse with enc3 skip
        # Same pattern: upsample → attend skip → concatenate → process
        u7 = F.interpolate(c6, scale_factor=2.0, mode="bilinear", align_corners=False)
        u7 = self._upsample_like(u7, c3)
        a7 = self.att7(c3, u7)
        c7 = self.dec7(torch.cat([u7, a7], dim=1))

        # DECODER STAGE 8: Upsample level-7 + fuse with enc2 skip
        u8 = F.interpolate(c7, scale_factor=2.0, mode="bilinear", align_corners=False)
        u8 = self._upsample_like(u8, c2)
        a8 = self.att8(c2, u8)
        c8 = self.dec8(torch.cat([u8, a8], dim=1))

        # DECODER STAGE 9: Upsample level-8 + fuse with enc1 skip (back to original resolution)
        u9 = F.interpolate(c8, scale_factor=2.0, mode="bilinear", align_corners=False)
        u9 = self._upsample_like(u9, c1)
        # Final attention gate on original-resolution encoder features
        a9 = self.att9(c1, u9)
        c9 = self.dec9(torch.cat([u9, a9], dim=1))

        # ==================== OUTPUT: Sigmoid for Binary/Multi-Class Segmentation ====================
        # c9: (batch_size, 1*lc, H, W) features at original resolution
        # self.head: 1x1 conv reduces 1*lc → num_classes (e.g., 1 for binary, 5 for 5-class)
        # Sigmoid: Squashes each output to [0, 1], interpreted as class probability per pixel.
        #
        # Why sigmoid (not softmax)?
        # - Sigmoid treats each class independently (any combination of 0-1 per class is valid).
        # - Good for multi-label tasks (e.g., a pixel can be both "water" and "boundary").
        # - Softmax (used in classification) forces exactly one class per pixel.
        # - For binary segmentation (num_classes=1), sigmoid outputs P(pixel is target class).
        out = torch.sigmoid(self.head(c9))
        return out


def _normalize_num_classes(input_label_channels: Union[int, Iterable[int]]) -> int:
    """
    Normalize class specification into an integer count.

    Accepts two forms:
    1. int: e.g., 5 → output 5 classes
    2. iterable (list/tuple): e.g., [0, 1, 2, 3, 4] → count = 5 classes

    This flexibility lets users specify classes as either a count or explicit list of class IDs.
    """
    # Check if input is an iterable (list, tuple, etc.) but NOT a string or bytes
    if isinstance(input_label_channels, Iterable) and not isinstance(
        input_label_channels, (str, bytes)
    ):
        try:
            # Count items in the iterable
            return len(list(input_label_channels))
        except Exception:
            # Fallback: try to convert directly to int
            return int(input_label_channels)
    # Input is already an int (or single value); return as-is
    return int(input_label_channels)


def UNet(
    input_shape,  # Shape of input image, e.g., (256, 256, 3) for 256x256 RGB image
    input_label_channels: Union[int, Iterable[int]],  # Number of output classes
    dilation_rate: int = 1,  # Dilation for encoder convolutions (1 = no dilation, 2 = wider kernel)
    layer_count: int = 64,  # Base number of channels; network scales by (1x, 2x, 4x, ..., 16x)
    l2_weight: float = 1e-4,  # L2 regularization strength (weight decay)
    dropout: float = 0.0,  # Dropout probability for regularization (0 = none, 0.5 = 50%)
    weight_file: str = None,  # Optional path to saved model weights to load
    summary: bool = False,  # If True, print model architecture and parameter counts
) -> nn.Module:
    """
    Factory function: Create and configure a U-Net model for image segmentation.

    USAGE EXAMPLES:
    >>> model = UNet((256, 256, 3), num_classes=1)  # Binary segmentation of 256x256 RGB image
    >>> model = UNet((512, 512, 1), [0, 1, 2], layer_count=32)  # 3-class segmentation, grayscale, smaller model

    Args:
        input_shape: Image dimensions as tuple/list, e.g., (height, width, channels).
                    Only the last value (channels) is used; spatial dimensions are ignored.
                    Standard: (256, 256, 3) for RGB, (256, 256, 1) for grayscale.
        input_label_channels: Number of output classes (int) or list of class IDs (iterable).
                             e.g., 1 (binary), 5 (5-class), or [0, 1, 2] (3-class list).
        dilation_rate: Dilation factor for encoder convolutions. Larger dilation = wider receptive field
                      without more parameters. Good for wide-context tasks (e.g., dense prediction on large images).
        layer_count: Base channels. Actual channels scale: 1x, 2x, 4x, ..., 16x this value.
                    Larger = more capacity but more memory/compute. Default 64 is standard.
                    Use 32 for small memory, 128 for large datasets.
        l2_weight: L2 regularization (weight decay). Penalizes large weights to prevent overfitting.
                  Typical range: 1e-4 to 1e-3. Higher = stronger regularization.
        dropout: Dropout probability. Applied in ConvBlocks for regularization.
                Typical range: 0.0 to 0.5. Higher = more regularization but may hurt training.
        weight_file: Path to pre-trained weights file (.pt, .pth). If provided, model is initialized
                    with these weights (useful for transfer learning or inference with trained model).
        summary: If True, prints full model architecture, total parameters, and trainable parameters.
                Useful for debugging and understanding model size.

    Returns:
        UNetAttention: PyTorch nn.Module ready for training or inference.
    """
    # Extract number of input channels from input_shape (last dimension)
    # Example: input_shape = (256, 256, 3) → in_channels = 3
    in_channels = int(input_shape[-1])

    # Normalize num_classes: handle both int and iterable inputs
    # Example: input_label_channels = 5 → num_classes = 5
    #          input_label_channels = [0, 1, 2] → num_classes = 3 (length of list)
    num_classes = _normalize_num_classes(input_label_channels)

    # Instantiate the U-Net model with specified configuration
    model = UNetAttention(
        in_channels=in_channels,
        num_classes=num_classes,
        dilation_rate=dilation_rate,
        layer_count=layer_count,
        dropout=dropout,
        l2_weight=l2_weight,
    )

    # OPTIONAL: Load pre-trained weights if provided
    if weight_file:
        try:
            # torch.load: Deserialize PyTorch checkpoint file (saved model state or full model)
            # map_location="cpu": Force loading to CPU (useful if checkpoint was on GPU)
            state = torch.load(weight_file, map_location="cpu")

            # Handle different checkpoint formats:
            # Format 1: Dict with "state_dict" key (common in multi-framework training pipelines)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            # Format 2: Dict with just model weights (PyTorch standard)
            elif isinstance(state, dict):
                model.load_state_dict(state)
            # Format 3: Entire model object (less common, but supported)
            else:
                model = state
        except Exception as exc:
            # If loading fails, raise informative error
            raise RuntimeError(
                f"Failed to load PyTorch weights from '{weight_file}': {exc}"
            ) from exc

    # OPTIONAL: Print model summary (architecture and parameter counts)
    if summary:
        # Count total parameters in model
        # p.numel(): Number of elements in parameter tensor
        total = sum(p.numel() for p in model.parameters())
        # Count only trainable parameters (requires_grad=True)
        # Frozen parameters (e.g., batch norm) may have requires_grad=False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[UNET][MODEL] Architecture:")
        print(model)
        print(f"[UNET][MODEL] Total params: {total:,} - Trainable: {trainable:,}")

    return model
