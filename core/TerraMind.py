# core/TerraMind.py

"""
Thin wrapper around TerraTorch to build a TerraMind backbone plus decoder.

Design:
- Keeps imports optional and resilient; if mmseg or specific registries are
  not available, the code still runs when those decoders are not requested.
- Provides small helpers to derive default indices and band lists based on
  model size and modality, mirroring the TF-side configuration behavior.
"""

import torch
import torch.nn as nn

# TerraTorch: populate registries (safe to import if available)
try:
    import terratorch  # noqa: F401
    import terratorch.models  # noqa: F401
    import terratorch.models.backbones.terramind  # noqa: F401
except Exception:
    pass

# If UPerNet or other mmseg decoders are used, register them if available
try:
    import mmseg  # noqa: F401
    import terratorch.models.decoders.mmseg  # noqa: F401
except Exception:
    pass

def _tm_indices_for_size(size: str):
    """
    Recommended transformer layer taps for hierarchical decoders.

    Base uses [2, 5, 8, 11]; Large uses [5, 11, 17, 23].
    """
    size = (size or "base").lower()
    if size == "large":
        return [5, 11, 17, 23]
    # tiny/small/base -> default to base taps (12 layers total)
    return [2, 5, 8, 11]


def _bands_for_modality(modality: str, n_channels: int, explicit_bands=None):
    """
    Map input channels to TerraMind's pre-trained band names.

    If explicit_bands is provided, pass-through that.
    """
    if explicit_bands is not None:
        return explicit_bands

    m = (modality or "S2").upper()

    # 4-band PlanetScope / Aerial assumed order: [B, G, R, NIR]
    if m in ("PS", "AE"):
        return ["BLUE", "GREEN", "RED", "NIR_NARROW"]

    # S2: infer common subsets by channel count, else None -> all bands, just in case I dfont want all bands at one point lol
    if m == "S2":
        if n_channels == 3:
            return ["BLUE", "GREEN", "RED"]
        if n_channels == 4:
            return ["BLUE", "GREEN", "RED", "NIR_NARROW"]
        # 10–12 bands -> use all pretrained bands (no subset)
        return None

    # Fallback: no subset mapping
    return None


class TerraMind(nn.Module):
    """
    Build a TerraMind backbone and decoder via TerraTorch's factory.

    Notes:
        - Always reuses S2 embeddings (S2L2A) with optional subset band mapping.
        - Decoder is configurable; default is UperNetDecoder.
        - Keeps TerraTorch's ModelOutput to preserve rich keys (logits, etc.).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        modality: str = "S2",  # "S2", "PS", or "AE"
        tm_size: str = "large",  # "tiny" | "small" | "base" | "large"
        merge_method: str = "mean",  # mean | max | concat
        pretrained: bool = True,
        ckpt_path: str = None,
        indices_override=None,  # e.g. [5, 11, 17, 23]
        bands_override=None,  # e.g. ["BLUE","GREEN","RED","NIR_NARROW"]
        decoder: str = "UperNetDecoder",  # configurable decoder
        decoder_channels=256,  # int for UPerNet, list[int] for UNetDecoder
        decoder_kwargs: dict | None = None,  # extra decoder-specific args
        backbone: str | None = None,  # direct backbone name if desired
        rescale: bool = True,  # upsample logits to input size
    ):
        super().__init__()

        # 1Choose the TerraMind variant (or accept explicit backbone override)
        backbone_name = backbone or f"terramind_v1_{tm_size.lower()}"

        # Single input modality; reuse S2 embeddings with band subsets as needed, we matched AE and PS with histomatch
        modality = (modality or "S2").upper()
        backbone_modalities = ["S2L2A"]
        bands = _bands_for_modality(modality, in_channels, bands_override)
        backbone_bands = {"S2L2A": bands} if bands is not None else None

        # 3eck recipe ViT-like encoders to hierarchical decoders
        indices = indices_override or _tm_indices_for_size(tm_size)
        necks = [
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "SelectIndices", "indices": indices},
            {"name": "LearnedInterpolateToPyramidal"},
        ]

        # Build encoder+decoder model
        from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory  # lazy: only needed if TerraMind is actually used
        factory = EncoderDecoderFactory()
        extra = dict(decoder_kwargs or {})
        # always pass decoder_channels if provided
        if "decoder_channels" not in extra:
            extra["decoder_channels"] = decoder_channels

        self.model = factory.build_model(
            task="segmentation",
            backbone=backbone_name,
            decoder=decoder,
            num_classes=num_classes,
            necks=necks,
            rescale=rescale,
            # Backbone args
            backbone_pretrained=pretrained,
            backbone_modalities=backbone_modalities,
            backbone_merge_method=merge_method,
            backbone_bands=backbone_bands,
            backbone_ckpt_path=ckpt_path,
            # Decoder-specific kwargs
            **extra,
        )

    def forward(self, x):
        # Return TerraTorch's ModelOutput; training loop will interpret it. Whatever that is.
        return self.model(x)

