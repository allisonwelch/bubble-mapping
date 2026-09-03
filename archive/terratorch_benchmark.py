"""
terratorch_benchmark.py

Benchmark script for testing TerraTorch FOUNDATION MODEL backbones (DOFA, Prithvi, SatMAE)
against the DriftwoodMappingBenchmark framework.

This version assumes your input band order is:  B, G, R, NIR  (4 channels)

Adapters implemented:
- Prithvi: expects 6 bands (BLUE, GREEN, RED, NIR, SWIR1, SWIR2) in many variants.
          We create missing SWIR bands by mean-filling: [B,G,R,NIR,mean,mean]
- SatMAE: commonly expects RGB. We convert B,G,R,NIR -> R,G,B
- DOFA: supports arbitrary band counts but requires wavelengths at forward-time.
        Also many variants are patch16_224 -> crop to 224.

The script:
  1. Loads project config from config.configTerraMind (or configUnet/configSwinUnet)
  2. Uses the existing data pipeline (core.common.data) to load frames and create datasets
  3. Filters backbones to foundation models (dofa/prithvi/satmae) by default
  4. Trains each compatible backbone for a few epochs
  5. Records results to a CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from terratorch.models import EncoderDecoderFactory
from terratorch.registry import BACKBONE_REGISTRY

from core.common.data import get_all_frames, create_train_val_datasets


# -----------------------------
# Input adapters for foundation models
# -----------------------------
class PrithviMeanFill4to6(nn.Module):
    """
    Input order is B,G,R,NIR (4ch). Produce 6ch:
      [B, G, R, NIR, mean(B,G,R,NIR), mean(B,G,R,NIR)]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        if x.shape[1] != 4:
            raise ValueError(f"PrithviMeanFill4to6 expects 4 channels, got {int(x.shape[1])}")
        m = x.mean(dim=1, keepdim=True)
        return torch.cat([x, m, m], dim=1)


class SatMAE_BGRNIR_to_RGB(nn.Module):
    """
    Input order is B,G,R,NIR (4ch). Produce 3ch RGB in expected order:
      [R, G, B]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        if x.shape[1] < 3:
            raise ValueError(f"SatMAE_BGRNIR_to_RGB expects >=3 channels, got {int(x.shape[1])}")
        # B=0, G=1, R=2, NIR=3 -> RGB = [2,1,0]
        return x[:, [2, 1, 0], :, :]


class InputAdapterWrapper(nn.Module):
    """
    Wrap a TerraTorch model with an optional input adapter.
    Also accepts wavelength args (ignored unless the inner model uses them).
    """
    def __init__(self, model: nn.Module, adapter: Optional[nn.Module] = None):
        super().__init__()
        self.model = model
        self.adapter = adapter

    def forward(self, x: torch.Tensor, wavelengths=None, wave_list=None):
        if self.adapter is not None:
            x = self.adapter(x)
            # keep memory format friendly
            x = x.contiguous(memory_format=torch.channels_last)

        # If wavelengths are provided, try passing them through.
        if wavelengths is not None:
            try:
                return self.model(x, wavelengths=wavelengths)
            except TypeError:
                pass
            try:
                return self.model(x, wave_list=wavelengths)
            except TypeError:
                pass
            return self.model(x, wavelengths)

        if wave_list is not None:
            try:
                return self.model(x, wave_list=wave_list)
            except TypeError:
                return self.model(x, wave_list)

        return self.model(x)


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    # Accept BCHW or BHWC and normalize to BCHW
    if x.ndim == 4:
        # If likely BHWC
        if x.shape[-1] in (1, 2, 3, 4, 6, 8, 10, 12, 13) and x.shape[1] not in (
            1, 2, 3, 4, 6, 8, 10, 12, 13
        ):
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
    raise ValueError(f"Expected 4D tensor (BCHW/BHWC). Got shape: {tuple(x.shape)}")


# -----------------------------
# Model/runtime special-cases (e.g., DOFA)
# -----------------------------
def model_runtime_overrides(backbone_name: str) -> Dict[str, Any]:
    """
    Per-backbone runtime overrides derived from the backbone name (model defaults),
    not from your project config.

    - *_patchXX_* -> pad_multiple=XX
    - *_224 (suffix) -> crop input/mask to 224x224
    - DOFA -> needs_wavelengths=True
    """
    low = (backbone_name or "").lower()
    o: Dict[str, Any] = {"pad_multiple": None, "force_hw": None, "needs_wavelengths": False}

    m = re.search(r"patch(\d+)", low)
    if m:
        try:
            o["pad_multiple"] = int(m.group(1))
        except Exception:
            pass

    if low.endswith("_224"):
        o["force_hw"] = 224

    if "dofa" in low:
        o["needs_wavelengths"] = True

    return o


def default_wavelengths_um(modality: str, n_channels: int) -> List[float]:
    """
    Defaults for wavelength-requiring models (DOFA).
    For your AE/PS 4-band B,G,R,NIR -> use a generic [B,G,R,NIR] wavelengths (μm).
    """
    # For your workflow, keep it simple and deterministic.
    if n_channels == 4:
        # B,G,R,NIR
        return [0.48, 0.56, 0.64, 0.81]
    if n_channels == 3:
        # RGB
        return [0.64, 0.56, 0.48]
    return list(np.linspace(0.45, 0.90, n_channels).astype(np.float32))


def center_crop_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = x.shape
    if h == size and w == size:
        return x
    if h < size or w < size:
        raise ValueError(f"Cannot center-crop {h}x{w} to {size}x{size}")
    top = (h - size) // 2
    left = (w - size) // 2
    return x[:, :, top : top + size, left : left + size]


def center_crop_mask(y: torch.Tensor, size: int) -> torch.Tensor:
    if y.ndim == 3:
        _, h, w = y.shape
        if h == size and w == size:
            return y
        top = (h - size) // 2
        left = (w - size) // 2
        return y[:, top : top + size, left : left + size]
    if y.ndim == 4:
        _, _, h, w = y.shape
        if h == size and w == size:
            return y
        top = (h - size) // 2
        left = (w - size) // 2
        return y[:, :, top : top + size, left : left + size]
    raise ValueError(f"Unsupported mask shape: {tuple(y.shape)}")


def tt_forward(model: nn.Module, x: torch.Tensor, *, wavelengths: Optional[List[float]] = None) -> Any:
    """
    Robust forward that supports DOFA-style wavelength arguments.
    """
    if wavelengths is None:
        return model(x)

    try:
        return model(x, wavelengths=wavelengths)
    except TypeError:
        pass
    try:
        return model(x, wave_list=wavelengths)
    except TypeError:
        pass
    return model(x, wavelengths)


def batch_to_xy(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        return to_torch(x), to_torch(y)
    if isinstance(batch, dict):
        for xk in ("image", "x", "inputs"):
            if xk in batch:
                x = batch[xk]
                break
        else:
            raise KeyError("Could not find image/x key in dict batch.")
        for yk in ("mask", "y", "labels", "target"):
            if yk in batch:
                y = batch[yk]
                break
        else:
            raise KeyError("Could not find mask/y key in dict batch.")
        return to_torch(x), to_torch(y)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def autopad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    if multiple <= 1:
        return x, (0, 0, 0, 0)
    b, c, h, w = x.shape
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    pad_h = new_h - h
    pad_w = new_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return x, (pad_left, pad_right, pad_top, pad_bottom)


def crop_from_pad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = pad
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return x
    _, _, h, w = x.shape
    return x[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]


def extract_logits(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if hasattr(model_out, "logits"):
        val = getattr(model_out, "logits")
        if torch.is_tensor(val):
            return val
    if isinstance(model_out, dict):
        for k in ("logits", "pred", "out", "mask", "segmentation"):
            if k in model_out and torch.is_tensor(model_out[k]):
                return model_out[k]
        for v in model_out.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Could not extract logits from model output of type: {type(model_out)}")


def is_vit_like_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["prithvi", "satmae", "mae", "vit", "dofa", "decur", "clay", "galileo", "terramind"])


# -----------------------------
# Foundation-model input planning
# -----------------------------
def foundation_input_candidates(backbone_name: str, raw_in_channels: int) -> List[Tuple[int, Optional[nn.Module], str]]:
    """
    Returns candidate (effective_in_channels_for_backbone_build, adapter_module, note).

    raw input is always your loader: 4ch B,G,R,NIR.
    """
    low = backbone_name.lower()

    if raw_in_channels != 4:
        raise ValueError(f"This workflow expects raw 4-band input. Detected {raw_in_channels} channels.")

    # DOFA: keep 4ch, but will pass wavelengths at forward-time (handled elsewhere)
    if "dofa" in low:
        return [(4, None, "dofa_raw4")]

    # Prithvi: prefer 6ch build + mean-fill adapter; fallback try raw4 if registry variant supports it.
    if "prithvi" in low:
        return [
            (6, PrithviMeanFill4to6(), "prithvi_meanfill_4to6"),
            (4, None, "prithvi_raw4_fallback"),
        ]

    # SatMAE: prefer RGB 3ch build + reorder adapter; fallback try raw4 if variant supports it.
    if "satmae" in low:
        return [
            (3, SatMAE_BGRNIR_to_RGB(), "satmae_BGRNIR_to_RGB_4to3"),
            (4, None, "satmae_raw4_fallback"),
        ]

    # Default: no adapter
    return [(raw_in_channels, None, "raw")]


# -----------------------------
# Model building with fallbacks
# -----------------------------
def build_model_with_fallbacks(
    backbone_name: str,
    *,
    in_channels: int,
    num_classes: int,
    prefer_pretrained: bool,
) -> nn.Module:
    """
    Build segmentation model (prefer defaults).
    We only try generic neck/decoder combos + backbone_in_chans/in_channels.
    """
    factory = EncoderDecoderFactory()
    nlow = backbone_name.lower()

    # ---- neck candidates ----
    vit_base = [
        {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
        {"name": "ReshapeTokensToImage"},
    ]
    vit_pyr = vit_base + [{"name": "LearnedInterpolateToPyramidal"}]

    neck_candidates: List[Optional[List[Dict[str, Any]]]] = []
    if is_vit_like_name(backbone_name):
        neck_candidates.extend([vit_pyr, vit_base, None])
    else:
        neck_candidates.extend([None, vit_pyr, vit_base])

    # ---- decoders (defaults) ----
    decoder_candidates: List[str] = ["FCNDecoder", "UNetDecoder", "UperNetDecoder"]

    # ---- backbone kwargs attempts ----
    bb_kwargs_attempts: List[Dict[str, Any]] = []
    pretrained_order = [prefer_pretrained] + ([False] if prefer_pretrained else [])
    for pt in pretrained_order:
        bb_kwargs_attempts.append({"backbone_pretrained": pt, "backbone_in_chans": in_channels})
        bb_kwargs_attempts.append({"backbone_pretrained": pt, "backbone_in_channels": in_channels})
        bb_kwargs_attempts.append({"backbone_pretrained": pt})

    bb_kwargs_attempts.append({})

    last_exc: Optional[BaseException] = None
    for necks in neck_candidates:
        for decoder_name in decoder_candidates:
            for bb_kwargs in bb_kwargs_attempts:
                try:
                    model = factory.build_model(
                        task="segmentation",
                        backbone=backbone_name,
                        decoder=decoder_name,
                        necks=necks,
                        num_classes=num_classes,
                        rescale=True,
                        **bb_kwargs,
                    )
                    return model
                except Exception as e:
                    last_exc = e

    raise RuntimeError(f"Failed to build model for backbone={backbone_name}. Last error: {last_exc}") from last_exc


# -----------------------------
# Train / eval
# -----------------------------
@dataclass
class TrainSettings:
    epochs: int
    steps_per_epoch: int
    val_steps: int
    lr: float
    weight_decay: float
    amp: bool
    pad_multiple: int


def compute_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 1:
        y = y.float()
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        return nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y)
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:, 0]
    y = y.long()
    return nn.functional.cross_entropy(logits, y)


@torch.no_grad()
def eval_steps(
    model: nn.Module,
    val_iterable: Iterable,
    *,
    device: torch.device,
    settings: TrainSettings,
    num_classes: int,
    backbone_name: str,
    modality: str,
) -> float:
    model.eval()
    losses: List[float] = []

    over = model_runtime_overrides(backbone_name)
    pad_multiple = int(over.get("pad_multiple") or settings.pad_multiple)
    force_hw = over.get("force_hw", None)
    needs_wavelengths = bool(over.get("needs_wavelengths", False))

    it = iter(val_iterable)
    for _ in range(settings.val_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_iterable)
            batch = next(it)

        x, y = batch_to_xy(batch)
        x = ensure_nchw(x).float()
        y = to_torch(y)

        if force_hw is not None:
            x = center_crop_tensor(x, int(force_hw))
            y = center_crop_mask(y, int(force_hw))

        x = x.to(device)
        x = x.contiguous(memory_format=torch.channels_last)
        x, pad = autopad_to_multiple(x, multiple=pad_multiple)
        y = y.to(device)

        wavelengths = default_wavelengths_um(modality, int(x.shape[1])) if needs_wavelengths else None

        out = tt_forward(model, x, wavelengths=wavelengths)
        logits = extract_logits(out)
        logits = crop_from_pad(logits, pad)

        loss = compute_loss(logits, y, num_classes)
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def train_and_get_best_val(
    model: nn.Module,
    train_iterable: Iterable,
    val_iterable: Iterable,
    *,
    device: torch.device,
    settings: TrainSettings,
    num_classes: int,
    backbone_name: str,
    modality: str,
) -> Tuple[float, int]:
    model.train()
    model.to(device)
    model.to(memory_format=torch.channels_last)

    over = model_runtime_overrides(backbone_name)
    pad_multiple = int(over.get("pad_multiple") or settings.pad_multiple)
    force_hw = over.get("force_hw", None)
    needs_wavelengths = bool(over.get("needs_wavelengths", False))

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and settings.amp))

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(settings.epochs):
        model.train()
        it = iter(train_iterable)

        for _ in range(settings.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_iterable)
                batch = next(it)

            x, y = batch_to_xy(batch)
            x = ensure_nchw(x).float()
            y = to_torch(y)

            if force_hw is not None:
                x = center_crop_tensor(x, int(force_hw))
                y = center_crop_mask(y, int(force_hw))

            x = x.to(device)
            x = x.contiguous(memory_format=torch.channels_last)
            x, pad = autopad_to_multiple(x, multiple=pad_multiple)
            y = y.to(device)

            wavelengths = default_wavelengths_um(modality, int(x.shape[1])) if needs_wavelengths else None

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and settings.amp:
                with torch.cuda.amp.autocast():
                    out = tt_forward(model, x, wavelengths=wavelengths)
                    logits = extract_logits(out)
                    logits = crop_from_pad(logits, pad)
                    loss = compute_loss(logits, y, num_classes)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = tt_forward(model, x, wavelengths=wavelengths)
                logits = extract_logits(out)
                logits = crop_from_pad(logits, pad)
                loss = compute_loss(logits, y, num_classes)
                loss.backward()
                optimizer.step()

        val_loss = eval_steps(
            model,
            val_iterable,
            device=device,
            settings=settings,
            num_classes=num_classes,
            backbone_name=backbone_name,
            modality=modality,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

    return best_val, best_epoch


# -----------------------------
# Config loading
# -----------------------------
def load_config_from_project() -> Any:
    try:
        from config import configTerraMind as cfg  # type: ignore
        if hasattr(cfg, "Configuration"):
            return cfg.Configuration()
        return cfg
    except Exception:
        pass

    try:
        from config import configUnetxS2 as cfg  # type: ignore
        if hasattr(cfg, "Configuration"):
            return cfg.Configuration()
        return cfg
    except Exception:
        pass

    try:
        from config import configSwinUnet as cfg  # type: ignore
        if hasattr(cfg, "Configuration"):
            return cfg.Configuration()
        return cfg
    except Exception:
        pass

    raise ImportError("Could not import a config module (configTerraMind/configUnet/configSwinUnet).")


# -----------------------------
# Sweep
# -----------------------------
def sweep(
    *,
    conf: Any,
    out_csv: str,
    epochs: int,
    steps_per_epoch: int,
    val_steps: int,
    lr: float,
    weight_decay: float,
    only_terratorch_prefix: bool,
    include_substrings: Sequence[str],
    max_models: Optional[int],
    prefer_pretrained: bool,
    amp: bool,
    pad_multiple: int,
) -> str:
    set_global_seed(getattr(conf, "seed", None))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    import core.common.data as data_module
    data_module.config = conf

    frames = get_all_frames(conf)
    train_ds, val_ds, _ = create_train_val_datasets(frames)

    # Infer raw channels from the actual dataloader batch
    batch0 = next(iter(train_ds))
    x0, _ = batch_to_xy(batch0)
    x0 = ensure_nchw(x0)
    raw_in_channels = int(x0.shape[1])
    if raw_in_channels != 4:
        raise RuntimeError(f"Expected 4-channel raw input (B,G,R,NIR). Detected {raw_in_channels} channels.")

    modality = getattr(conf, "modality", "AE")
    num_classes = int(getattr(conf, "num_classes", 1))

    # registry list
    names = list(BACKBONE_REGISTRY)
    if only_terratorch_prefix:
        names = [n for n in names if n.startswith("terratorch_")]

    if include_substrings:
        low_subs = [s.lower() for s in include_substrings]
        names = [n for n in names if any(s in n.lower() for s in low_subs)]

    if max_models is not None:
        names = names[: max_models]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "status", "best_val_loss", "best_epoch", "params", "seconds", "notes"])

    settings = TrainSettings(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        lr=lr,
        weight_decay=weight_decay,
        amp=amp,
        pad_multiple=pad_multiple,
    )

    for idx, backbone in enumerate(names, start=1):
        t0 = time.time()
        status = "skip"
        best_val = float("nan")
        best_epoch = -1
        params = 0
        notes = ""

        try:
            # Try the candidate input strategies for this backbone.
            built = False
            last_err: Optional[BaseException] = None

            for eff_in_ch, adapter, note in foundation_input_candidates(backbone, raw_in_channels):
                try:
                    base_model = build_model_with_fallbacks(
                        backbone,
                        in_channels=eff_in_ch,
                        num_classes=num_classes,
                        prefer_pretrained=prefer_pretrained,
                    )
                    model = InputAdapterWrapper(base_model, adapter=adapter) if adapter is not None else base_model

                    # quick smoke test forward (uses DOFA wavelengths + 224 crops automatically)
                    _x, _y = batch_to_xy(batch0)
                    _x = ensure_nchw(_x).float()

                    over = model_runtime_overrides(backbone)
                    eff_pad_multiple = int(over.get("pad_multiple") or pad_multiple)
                    force_hw = over.get("force_hw", None)
                    needs_wavelengths = bool(over.get("needs_wavelengths", False))

                    if force_hw is not None:
                        _x = center_crop_tensor(_x, int(force_hw))

                    _x = _x.to(device).contiguous(memory_format=torch.channels_last)
                    _x, _pad = autopad_to_multiple(_x, multiple=eff_pad_multiple)

                    wavelengths = default_wavelengths_um(modality, int(_x.shape[1])) if needs_wavelengths else None

                    model = model.to(device).eval()
                    with torch.no_grad():
                        _ = tt_forward(model, _x, wavelengths=wavelengths)

                    # if we got here, this candidate works
                    built = True
                    notes = note
                    break
                except Exception as e:
                    last_err = e
                    continue

            if not built:
                status = "skip"
                notes = f"incompatible: {type(last_err).__name__}: {last_err}"
                raise RuntimeError(notes)

            params = count_params(model)

            # train
            status = "ok"
            best_val, best_epoch = train_and_get_best_val(
                model,
                train_ds,
                val_ds,
                device=device,
                settings=settings,
                num_classes=num_classes,
                backbone_name=backbone,
                modality=modality,
            )

        except torch.cuda.OutOfMemoryError:
            status = "oom"
            notes = "CUDA OOM"
        except Exception as e:
            if status != "ok":
                status = "skip"
            if not notes:
                notes = f"{type(e).__name__}: {e}"

        seconds = round(time.time() - t0, 2)
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([backbone, status, best_val, best_epoch, params, seconds, notes])

        print(
            f"[{idx}/{len(names)}] {backbone} -> {status}...{best_val} "
            f"(epoch {best_epoch}) | params={params} | {seconds}s | {notes}"
        )

        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out_csv


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="foundation_sweep_results.csv", help="Output CSV path")
    ap.add_argument("--epochs", type=int, default=8, help="Epochs per backbone")
    ap.add_argument("--steps-per-epoch", type=int, default=200, help="Train steps per epoch (batches)")
    ap.add_argument("--val-steps", type=int, default=50, help="Validation steps per epoch (batches)")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP on GPU")
    ap.add_argument("--pad-multiple", type=int, default=16, help="Auto-pad H/W to multiple (fallback if model name has no patchXX)")
    ap.add_argument("--include-all-sources", action="store_true", help="Include non-terratorch registry sources (timm/smp/etc)")
    ap.add_argument("--include", action="append", default=[], help="Substring filter; can repeat")
    ap.add_argument("--max-models", type=int, default=None, help="Limit number of backbones swept")
    ap.add_argument("--no-pretrained", action="store_true", help="Do not load pretrained weights")

    args = ap.parse_args()

    # Default to the three foundation-model families you requested
    if not args.include:
        args.include = ["dofa", "prithvi", "satmae"]

    conf = load_config_from_project()

    out_csv = sweep(
        conf=conf,
        out_csv=args.out,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        only_terratorch_prefix=(not args.include_all_sources),
        include_substrings=args.include,
        max_models=args.max_models,
        prefer_pretrained=(not args.no_pretrained),
        amp=(not args.no_amp),
        pad_multiple=args.pad_multiple,
    )

    print("\nWrote CSV:", out_csv)


if __name__ == "__main__":
    main()
