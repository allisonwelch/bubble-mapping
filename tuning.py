from __future__ import annotations

import gc
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import inspect  # NEW: to safely pass TerraMind kwargs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except Exception as _e:
    raise ImportError(
        "Optuna is required for tuning. Install with `pip install optuna`."
    ) from _e

from core.Swin_UNetPP import SwinUNet
from core.TerraMind import TerraMind
from core.UNet import UNet
from core.common.data import create_train_val_datasets, get_all_frames
from core.common.model_utils import (
    _as_probs_from_terratorch_logits_first,
    _ensure_nchw,
    _forward_with_autopad,
)
from core.common.console import print_search_space
from core.optimizers import get_optimizer  # project-level optimizer factory
from core.losses import tversky as _tversky  # Tversky loss (alpha+beta=1, alpha tuned)

from config.configUnetxAE import *

# -----------------------
# Utilities
# -----------------------
def _seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default(val, fallback):
    return val if val is not None else fallback


def _nan_to_num_torch(x: torch.Tensor, constant: float) -> torch.Tensor:
    return torch.where(
        torch.isfinite(x),
        x,
        torch.as_tensor(constant, dtype=x.dtype, device=x.device),
    )


def _sanitize_pair_xy(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    # Inputs/labels are safe to clamp in-place (no grad needed)
    x = _nan_to_num_torch(x, 0.0)#.clamp_(0.0, 1.0)
    y = _nan_to_num_torch(y, 0.0).clamp_(0.0, 1.0)
    return x, y


def _sanitize_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.to(dtype=torch.float32)
    w = _nan_to_num_torch(w, 1.0).clamp_(0.0, 1e6)
    return w


def _snap_hw_for_swin(
    H: int, W: int, patch_size: int, window_size: int, down_levels: int = 3
) -> Tuple[int, int]:
    need = window_size * (2 ** down_levels)
    t_h = int(np.ceil(H / patch_size))
    t_w = int(np.ceil(W / patch_size))
    t_h_adj = int(np.ceil(t_h / need)) * need
    t_w_adj = int(np.ceil(t_w / need)) * need
    H_adj = t_h_adj * patch_size
    W_adj = t_w_adj * patch_size
    return H_adj, W_adj


def _flatten_nchw_like(y: torch.Tensor) -> torch.Tensor:
    return y.reshape(y.shape[0], -1)


def _safe_dice(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y_true = y_true.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_pred = y_pred.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_true_f = _flatten_nchw_like(y_true)
    y_pred_f = _flatten_nchw_like(y_pred)
    inter = torch.sum(y_true_f * y_pred_f, dim=1)
    den = torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return dice.mean()


def _safe_iou(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y_true = y_true.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_pred = y_pred.to(dtype=torch.float32).clamp(0.0, 1.0)
    y_true_f = _flatten_nchw_like(y_true)
    y_pred_f = _flatten_nchw_like(y_pred)
    inter = torch.sum(y_true_f * y_pred_f, dim=1)
    union = torch.sum(y_true_f + y_pred_f, dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def _stop_dataloader_workers(dl: Optional[Iterable]) -> None:
    """
    Shutdown of DataLoader worker processes/threads.
    Safe to call on loaders with 0 workers or non-standard iterables.
    """
    if dl is None:
        return
    try:
        it = getattr(dl, "_iterator", None)
        if it is not None:
            shut = getattr(it, "_shutdown_workers", None)
            if callable(shut):
                shut()
            try:
                workers = getattr(it, "_workers", None)
                if workers:
                    for w in workers:
                        try:
                            w.terminate()
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                setattr(dl, "_iterator", None)
            except Exception:
                pass
    except Exception:
        pass


# -----------------------
# Logging helpers (concise / readable)
# -----------------------
def _phase_banner(model_key: str, phase: str, max_epochs: int, steps: int, val_steps: int, tune_batch: int):
    print(
        f"\n=== [{model_key.upper()}][{phase}] epochs={max_epochs}, "
        f"steps/epoch={steps}, val_steps={val_steps}, batch={tune_batch} ==="
    )


def _trial_banner(trial_num: int, model_key: str, hp: Dict[str, Any], data_hp: Dict[str, Any]):
    arch_bits = []
    if model_key == "unet":
        for k in ("dilation_rate", "layer_count", "l2_weight", "dropout"):
            if k in hp:
                arch_bits.append(f"{k}={hp[k]}")
    elif model_key == "swin":
        for k in ("C", "drop_path"):
            if k in hp:
                arch_bits.append(f"{k}={hp[k]}")
    else:
        for k in (
            "tm_decoder",
            "tm_decoder_channels",
            "tm_head_dropout",
            "tm_lr_backbone",
            "tm_lr_head_mult",
            "tm_weight_decay",
            "tm_freeze_backbone_epochs",
            "tm_size",
        ):
            if k in hp:
                arch_bits.append(f"{k}={hp[k]}")
    data_bits = (
        f"{data_hp['patch_h']}x{data_hp['patch_w']}, aug={data_hp['augmenter_strength']}, "
        f"minpos={data_hp['min_pos_frac']}"
    )
    opt_bits = f"opt={hp['optimizer']}" if "optimizer" in hp else ""
    if "learning_rate" in hp:
        opt_bits += f", lr={hp['learning_rate']:.3g}"
    if "weight_decay" in hp:
        opt_bits += f", wd={hp['weight_decay']:.2g}"
    sch_bits = f"sched={hp.get('scheduler','none')}"
    arch_str = "; ".join(arch_bits) if arch_bits else "-"
    print(f"\n[trial {trial_num}] {opt_bits}, {sch_bits} | arch: {arch_str} | data: {data_bits}")


def _epoch_line(epoch: int, best: float, cur: float, extra: str = ""):
    msg = f"  epoch {epoch:02d}: val_dice={cur:.4f} (best={best:.4f})"
    if extra:
        msg += f" | {extra}"
    print(msg)


# --------- Console color helpers ---------
class _C:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def _col(s: str, color: str) -> str:
    return f"{color}{s}{_C.RESET}"


def _format_val(v):
    if isinstance(v, float):
        if (abs(v) > 1e4) or (0 < abs(v) < 1e-4):
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _print_sxs_diff(
    prev_cfg: Dict[str, Any], curr_cfg: Dict[str, Any], prev_score: Optional[float], curr_score: float
) -> None:
    if prev_cfg is None:
        print("\n   baseline: no previous best to compare.")
        return

    keys = sorted(set(prev_cfg.keys()) | set(curr_cfg.keys()))
    rows = []
    for k in keys:
        a = prev_cfg.get(k, "-")
        b = curr_cfg.get(k, "-")
        rows.append((k, _format_val(a), _format_val(b), a != b))

    key_w = max(len(k) for k, *_ in rows) if rows else 0
    prev_w = max(len(p) for _, p, _, _ in rows) if rows else 0
    curr_w = max(len(c) for _, _, c, _ in rows) if rows else 0

    print("   config diff (prev  |  curr):")
    for k, prev_s, curr_s, changed in rows:
        k_disp = _col(k, _C.YELLOW) if changed else k
        k_pad = " " * (key_w - len(k))
        mark = "  *" if changed else ""
        print(f"     - {k_disp}:{k_pad}  {prev_s:>{prev_w}}  |  {curr_s:>{curr_w}}{mark}")

    if prev_score is not None:
        delta = curr_score - prev_score
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.5f}"
        color = _C.GREEN if delta > 0 else (_C.RED if delta < 0 else _C.CYAN)
        print(
            f"\n   score: prev={prev_score:>{prev_w}.5f} | "
            f"curr={curr_score:>{curr_w}.5f} ({_col(delta_str, color)})"
        )


# -----------------------
# SEARCH SPACE: HB is Hyperband, BO is Bayesian Optimization. HB is discrete/categorical, BO is continuous.
# -----------------------
def _optimizer_space_hb(trial: "optuna.Trial") -> Tuple[str, float, Optional[float]]:
    opt = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    wd = None
    if opt == "adamw":
        wd = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    return opt, lr, wd


def _schedule_space_hb(trial: "optuna.Trial") -> str:
    return trial.suggest_categorical("scheduler", ["none", "cosine", "onecycle"])


def _optimizer_space_bo(trial: "optuna.Trial", fixed_opt: str) -> Tuple[str, float, Optional[float]]:
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    wd = None
    if fixed_opt == "adamw":
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    return fixed_opt, lr, wd


def _unet_space_hb(trial: "optuna.Trial") -> Tuple[int, int, float, float]:
    dilation = trial.suggest_categorical("dilation_rate", [1, 2, 4])
    layer_cnt = trial.suggest_categorical("layer_count", [32, 64, 96])
    l2w = trial.suggest_categorical("l2_weight", [0.0, 1e-5, 1e-4])
    drp = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    return int(dilation), int(layer_cnt), float(l2w), float(drp)


def _swin_space_hb(trial: "optuna.Trial") -> Tuple[int, float]:
    # Respect architecture limits:
    # - If use_imagenet_weights=True --Y Swin-T only--> C=96 is required.
    # - Otherwise allow {64, 96}.
    use_imnet = False
    try:
        use_imnet = bool(getattr(config, "use_imagenet_weights", False))
    except Exception:
        use_imnet = False
    C_choices = [96] if use_imnet else [64, 96]
    C = trial.suggest_categorical("C", C_choices)
    drop_path = trial.suggest_categorical("drop_path", [0.0, 0.1, 0.2, 0.3])
    return int(C), float(drop_path)


def _data_space_hb(trial: "optuna.Trial") -> Tuple[int, int, float, float]:
    valid_sizes = [
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
    ]
    patch_h = trial.suggest_categorical("patch_h", valid_sizes)
    patch_w = patch_h
    aug_str = trial.suggest_categorical("augmenter_strength", [0.0, 0.5, 1.0])
    minpos = trial.suggest_categorical("min_pos_frac", [0.0, 0.01, 0.02])
    return int(patch_h), int(patch_w), float(aug_str), float(minpos)


# -------- TerraMind search spaces and helpers --------
def _tm_space_hb(trial: "optuna.Trial"):
    dec = trial.suggest_categorical("tm_decoder", ["UNetDecoder", "UperNetDecoder"])
    dec_ch = trial.suggest_categorical("tm_decoder_channels", [128, 192, 256, 384])
    head_do = trial.suggest_float("tm_head_dropout", 0.0, 0.2)
    freeze_ep = trial.suggest_categorical("tm_freeze_backbone_epochs", [0, 1, 3, 5])
    lr_backbone = trial.suggest_float("tm_lr_backbone", 1e-6, 3e-5, log=True)
    lr_head_mult = trial.suggest_categorical("tm_lr_head_mult", [5.0, 10.0])
    wd = trial.suggest_float("tm_weight_decay", 1e-6, 5e-4, log=True)
    return dec, dec_ch, head_do, freeze_ep, lr_backbone, lr_head_mult, wd


def _tm_space_bo(trial: "optuna.Trial", fixed):
    lr_backbone = trial.suggest_float("tm_lr_backbone", 1e-6, 3e-5, log=True)
    lr_head_mult = trial.suggest_categorical("tm_lr_head_mult", [5.0, 10.0])
    wd = trial.suggest_float("tm_weight_decay", 1e-6, 5e-4, log=True)
    return lr_backbone, lr_head_mult, wd


def _split_backbone_head_params(inner_model: nn.Module):
    """
    TerraTorch PixelWiseModel may not expose `.backbone`. Be defensive.
    Returns (bb_params, head_params).
    """
    inner = inner_model
    # If it is a wrapper that stores `model`, unwrap once
    if hasattr(inner, "model") and isinstance(inner.model, nn.Module):
        inner = inner.model

    bb_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    if hasattr(inner, "backbone") and isinstance(inner.backbone, nn.Module):
        bb_params = list(inner.backbone.parameters())
        bb_ids = {id(p) for p in bb_params}
        for p in inner_model.parameters():
            if id(p) not in bb_ids:
                head_params.append(p)
    else:
        # No explicit backbone handle - treat everything as "head"
        head_params = list(inner_model.parameters())

    return bb_params, head_params


def _make_tm_optimizer(model: TerraMind, lr_bb: float, lr_head_mult: float, wd: float, opt: str = "adamw"):
    bb_params, head_params = _split_backbone_head_params(model)
    if len(bb_params) == 0:
        groups = [{"params": head_params, "lr": lr_bb * float(lr_head_mult), "weight_decay": wd}]
    else:
        groups = [
            {"params": head_params, "lr": lr_bb * float(lr_head_mult), "weight_decay": wd},
            {"params": bb_params, "lr": lr_bb, "weight_decay": wd},
        ]
    return (torch.optim.AdamW if opt == "adamw" else torch.optim.Adam)(groups)


def _set_backbone_requires_grad(model: nn.Module, requires_grad: bool):
    inner = model
    if hasattr(inner, "model") and isinstance(inner.model, nn.Module):
        inner = inner.model
    if hasattr(inner, "backbone") and isinstance(inner.backbone, nn.Module):
        for p in inner.backbone.parameters():
            p.requires_grad = requires_grad


def _tm_raw_to_probs(
    raw: Union[torch.Tensor, Dict[str, Any], List[Any], tuple], num_classes: int = 1
) -> torch.Tensor:
    """
    Robustly extract logits from TerraTorch-style outputs and convert to probabilities.
    Out-of-place ops only (no in-place clamp) to keep autograd happy.
    """
    logits = None
    if isinstance(raw, torch.Tensor):
        logits = raw
    elif hasattr(raw, "logits"):
        logits = raw.logits  # typical ModelOutput
    elif isinstance(raw, dict):
        for k in ("logits", "pred", "y", "out", "output"):
            if k in raw:
                logits = raw[k]
                break
    elif isinstance(raw, (list, tuple)) and len(raw) > 0:
        # Many libs put logits at index 0
        logits = raw[0]

    if logits is None or not isinstance(logits, torch.Tensor):
        raise TypeError(
            "TerraMind forward did not return a Tensor or a dict/tuple containing a Tensor."
        )

    # binary vs multi-class
    if (num_classes is None) or (num_classes <= 1) or (logits.shape[1] == 1):
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=1)

    probs = probs.float()
    probs = torch.clamp(probs, 0.0, 1.0)
    return probs


# -----------------------
# TerraMind builder that is robust to different __init__ signatures
# -----------------------
def _build_terramind(
    conf,
    dec: Optional[str],
    dec_ch: Optional[int],
    head_do: Optional[float],
) -> TerraMind:
    kwargs: Dict[str, Any] = dict(
        in_channels=len(getattr(conf, "channel_list", [])),
        num_classes=int(getattr(conf, "num_classes", 1)),
        modality=getattr(conf, "modality", "S2"),
        tm_size=getattr(conf, "terramind_size", "base"),
        merge_method=getattr(conf, "terramind_merge_method", "mean"),
        pretrained=True,
    )
    sig = inspect.signature(TerraMind)
    if "decoder" in sig.parameters and dec is not None:
        kwargs["decoder"] = dec
    if "decoder_channels" in sig.parameters and dec_ch is not None:
        # UNetDecoder requires a list, other decoders expect an int
        if dec == "UNetDecoder":
            # Create a list with decreasing channels: [dec_ch, dec_ch//2, dec_ch//4, dec_ch//8]
            base_ch = int(dec_ch)
            kwargs["decoder_channels"] = [base_ch, base_ch // 2, base_ch // 4, base_ch // 8]
        else:
            kwargs["decoder_channels"] = int(dec_ch)
    if "head_dropout" in sig.parameters and head_do is not None:
        kwargs["head_dropout"] = float(head_do)
    return TerraMind(**kwargs)


# -----------------------
# Build / compile (optimizer + scheduler + loss)
# -----------------------
def _compile_with_optimizer(
    model: nn.Module,
    opt_name: str,
    lr: float,
    wd: Optional[float],
    scheduler_name: str,
    steps_per_epoch: int,
    max_epochs: int,
    conf,
    tversky_alpha: Optional[float] = None,
) -> Tuple[nn.Module, torch.optim.Optimizer, Any, Optional[Any], bool]:
    """
    Returns: model, optimizer, criterion, scheduler, scheduler_steps_per_batch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt_obj: Optional[torch.optim.Optimizer] = None

    # TerraMind: use tuned backbone/head LR & WD with param groups if available
    if isinstance(model, TerraMind) and all(
        hasattr(conf, a) for a in ("tm_lr_backbone", "tm_lr_head_mult", "tm_weight_decay")
    ):
        opt_obj = _make_tm_optimizer(
            model,
            lr_bb=float(getattr(conf, "tm_lr_backbone")),
            lr_head_mult=float(getattr(conf, "tm_lr_head_mult")),
            wd=float(getattr(conf, "tm_weight_decay")),
            opt=(opt_name or "adamw"),
        )
    else:
        # Try project optimizer factory
        try:
            opt_obj = get_optimizer(
                _default(opt_name, getattr(conf, "optimizer_fn", "adam")),
                _default(getattr(conf, "num_epochs", max_epochs), max_epochs),
                _default(getattr(conf, "num_training_steps", steps_per_epoch), steps_per_epoch),
                model,
                lr=float(lr),
                weight_decay=(float(wd) if wd is not None else None),
                clipnorm=None,
                internal_schedule=None,
            )
        except Exception:
            opt_obj = None

        # Fallback optimizer construction
        if opt_obj is None or isinstance(opt_obj, str):
            params = model.parameters()
            if (opt_name or "adam").lower() == "adamw":
                opt_obj = torch.optim.AdamW(
                    params, lr=float(lr), weight_decay=float(_default(wd, 1e-6))
                )
            else:
                opt_obj = torch.optim.Adam(
                    params,
                    lr=float(lr),
                    weight_decay=float(_default(wd, 0.0)),
                )

    # Tversky loss (alpha + beta = 1)
    alpha = float(tversky_alpha) if tversky_alpha is not None else 0.7
    beta = 1.0 - alpha

    def criterion(y_true, y_pred, w: Optional[torch.Tensor] = None):
        yt = _nan_to_num_torch(y_true.float(), 0.0).clamp(0.0, 1.0)
        yp = torch.clamp(_nan_to_num_torch(y_pred.float(), 0.5), 1e-6, 1.0 - 1e-6)
        # We ignore w here for simplicity; Tversky already focuses on imbalance via alpha/beta.
        return 1.0 - _tversky(yt, yp, alpha=alpha, beta=beta)

    # Scheduler setup
    scheduler = None
    step_per_batch = False
    scheduler_name = (scheduler_name or "none").lower()
    if scheduler_name == "onecycle":
        base_lrs = [float(g.get("lr", lr)) for g in opt_obj.param_groups]
        max_lr = base_lrs if len(base_lrs) > 1 else base_lrs[0]
        scheduler = OneCycleLR(
            opt_obj,
            max_lr=max_lr,
            steps_per_epoch=int(steps_per_epoch),
            epochs=int(max_epochs),
        )
        step_per_batch = True
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(opt_obj, T_max=int(max_epochs), eta_min=lr * 0.01)

    return model, opt_obj, criterion, scheduler, step_per_batch


# -----------------------
# Phase runner (HB / BO)
# -----------------------
def _run_phase(
    conf,
    model_key: str,
    phase: str,
    project_dir: str,
    tune_batch: int,
    max_epochs: int,
    steps: int,
    val_steps: int,
    executions_per_trial: int,
    overwrite: bool,
    hb_data_hp: Dict[str, Any],
) -> Tuple[Dict[str, Any], "optuna.Study"]:
    frames = get_all_frames(conf)

    # cache dataloaders for (patch_h, patch_w, aug, minpos, pos_ratio) combos (batch fixed)
    # Loader cache + restoration tracking
    dl_cache: Dict[Tuple[int, int, float, float, float], Tuple[Iterable, Iterable]] = {}
    _saved_conf_state: Dict[str, Any] = {}  # Track original conf values for final restoration

    def _get_loaders_for(
        ph: int,
        pw: int,
        aug: float,
        minpos: float,
        pos_ratio: Optional[float],
    ) -> Tuple[Iterable, Iterable]:
        key = (
            int(ph),
            int(pw),
            float(aug),
            float(minpos),
            float(pos_ratio) if pos_ratio is not None else -1.0,
        )
        if key in dl_cache:
            # CACHE HIT: also set conf.* to match cached values so dynamic sampling stays consistent
            conf.train_batch_size = int(tune_batch)
            conf.patch_size = (int(ph), int(pw))
            conf.augmenter_strength = float(aug)
            conf.min_pos_frac = float(minpos)
            conf.pos_ratio = pos_ratio
            return dl_cache[key]

        # Save original values ONCE (first call)
        if not _saved_conf_state:
            _saved_conf_state["train_batch_size"] = getattr(conf, "train_batch_size", None)
            _saved_conf_state["patch_size"] = getattr(conf, "patch_size", None)
            _saved_conf_state["augmenter_strength"] = getattr(conf, "augmenter_strength", None)
            _saved_conf_state["min_pos_frac"] = getattr(conf, "min_pos_frac", None)
            _saved_conf_state["pos_ratio"] = getattr(conf, "pos_ratio", None)

        # Set conf.* for this trial's sampled values
        conf.train_batch_size = int(tune_batch)
        conf.patch_size = (int(ph), int(pw))
        conf.augmenter_strength = float(aug)
        conf.min_pos_frac = float(minpos)
        conf.pos_ratio = pos_ratio  # may be None

        train_iter, val_iter, _ = create_train_val_datasets(frames)

        # DO NOT RESTORE conf.* immediately - keep sampled values active for the trial
        # Restoration will happen at end of phase

        dl_cache[key] = (train_iter, val_iter)

        if len(dl_cache) > 10:  # keep cache size small
            for k in list(dl_cache.keys())[:-10]:
                try:
                    tr_dl, va_dl = dl_cache[k]
                    _stop_dataloader_workers(tr_dl)
                    _stop_dataloader_workers(va_dl)
                except Exception:
                    pass
                del dl_cache[k]
                gc.collect()

        return dl_cache[key]

    def _restore_conf():
        """Restore original conf values after phase completes."""
        if _saved_conf_state:
            for k, v in _saved_conf_state.items():
                setattr(conf, k, v)
            _saved_conf_state.clear()

    # Study
    direction = "maximize"
    study_name = f"{model_key}_tuning_{phase}"
    pruner = HyperbandPruner() if phase == "HB" else None
    sampler = TPESampler(seed=_default(getattr(conf, "seed", None), 42))
    _ensure_dir(project_dir)
    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner, study_name=study_name, storage=None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    _phase_banner(model_key, phase, max_epochs, steps, val_steps, tune_batch)

    # Track best within this phase for nicer printing
    best_print = {"score": None, "cfg": None}

    def _single_execution(
        trial,
        train_iterable,
        val_iterable,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        scheduler,
        step_per_batch: bool,
        freeze_ep: int = 0,
    ) -> float:
        best_val_dice = -float("inf")
        patience = 20
        no_improve = 0
        was_frozen = None

        for epoch in range(1, max_epochs + 1):
            # Optional freeze schedule (no-op if backbone not exposed)
            if freeze_ep > 0:
                if epoch <= freeze_ep and was_frozen is not True:
                    _set_backbone_requires_grad(model, False)
                    was_frozen = True
                elif epoch == freeze_ep + 1 and was_frozen:
                    _set_backbone_requires_grad(model, True)
                    was_frozen = False

            model.train()
            train_it = iter(train_iterable)
            train_range = tqdm(range(steps), desc=f"Epoch {epoch}/{max_epochs} [Train]", leave=False)
            for _ in train_range:
                batch = next(train_it, None)
                if batch is None:
                    train_it = iter(train_iterable)
                    batch = next(train_it)

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, w = batch
                else:
                    x, y = batch
                    w = None

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x, y = _sanitize_pair_xy(x, y)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    # TerraMind path similar to training loop elsewhere
                    if model_key == "tm":
                        raw = _forward_with_autopad(model, x)
                        num_classes = int(getattr(conf, "num_classes", 1))
                        y_pred_full = _as_probs_from_terratorch_logits_first(
                            raw, num_classes=num_classes
                        )
                        y_pred_full = _ensure_nchw(y_pred_full).float()
                        if y_pred_full.shape[1] > 1:
                            cls_idx = int(getattr(conf, "metrics_class", 1))
                            cls_idx = max(0, min(cls_idx, y_pred_full.shape[1] - 1))
                            y_pred = y_pred_full[:, cls_idx: cls_idx + 1]
                        else:
                            y_pred = y_pred_full
                    else:
                        raw = model(x)  # Non-TM models
                        # Non-TM models expected to output probabilities already
                        y_pred = torch.as_tensor(raw, dtype=torch.float32, device=device)
                        y_pred = torch.clamp(y_pred, 0.0, 1.0)  # OUT-OF-PLACE
                    loss = criterion(y, y_pred, w)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise optuna.TrialPruned()

                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None and step_per_batch:
                    scheduler.step()

            # Validation
            model.eval()
            val_it = iter(val_iterable)
            val_dice_accum = 0.0
            with torch.no_grad():
                val_range = tqdm(range(val_steps), desc=f"Epoch {epoch}/{max_epochs} [Val]", leave=False)
                for _ in val_range:
                    vb = next(val_it, None)
                    if vb is None:
                        break
                    if isinstance(vb, (list, tuple)) and len(vb) == 3:
                        xv, yv, _wv = vb
                    else:
                        xv, yv = vb
                    xv = xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    xv, yv = _sanitize_pair_xy(xv, yv)

                    if model_key == "tm":
                        raw = _forward_with_autopad(model, xv)
                        num_classes = int(getattr(conf, "num_classes", 1))
                        yp_full = _as_probs_from_terratorch_logits_first(
                            raw, num_classes=num_classes
                        )
                        yp_full = _ensure_nchw(yp_full).float()
                        if yp_full.shape[1] > 1:
                            cls_idx = int(getattr(conf, "metrics_class", 1))
                            cls_idx = max(0, min(cls_idx, yp_full.shape[1] - 1))
                            yp = yp_full[:, cls_idx: cls_idx + 1]
                        else:
                            yp = yp_full
                    else:
                        raw = model(xv)
                        yp = torch.as_tensor(raw, dtype=torch.float32, device=device)
                        yp = torch.clamp(yp, 0.0, 1.0)

                    val_dice_accum += float(_safe_dice(yv, yp).detach().cpu().item())

            cur = val_dice_accum / max(1, val_steps)
            if cur > best_val_dice:
                best_val_dice = cur
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            _epoch_line(epoch, best_val_dice, cur)

            if scheduler is not None and not step_per_batch:
                scheduler.step()

        return best_val_dice

    # Objective
    def objective(trial: "optuna.Trial") -> float:
        # Sample HPs
        if model_key == "unet":
            if phase == "HB":
                dilation, layer_cnt, l2w, drp = _unet_space_hb(trial)
            else:
                dilation = int(fixed.get("dilation_rate", getattr(conf, "dilation_rate", 1)))
                layer_cnt = int(fixed.get("layer_count", getattr(conf, "layer_count", 64)))
                l2w = float(fixed.get("l2_weight", getattr(conf, "l2_weight", 0.0)))
                drp = float(fixed.get("dropout", getattr(conf, "dropout", 0.0)))
            opt, lr, wd = _optimizer_space_hb(trial) if phase == "HB" else _optimizer_space_bo(
                trial, fixed.get("optimizer", "adam")
            )
            sched = _schedule_space_hb(trial) if phase == "HB" else trial.suggest_categorical(
                "scheduler", ["none", "cosine", "onecycle"]
            )

        elif model_key == "swin":
            C, drop_path = _swin_space_hb(trial) if phase == "HB" else (None, None)
            opt, lr, wd = _optimizer_space_hb(trial) if phase == "HB" else _optimizer_space_bo(
                trial, fixed.get("optimizer", "adam")
            )
            sched = _schedule_space_hb(trial) if phase == "HB" else trial.suggest_categorical(
                "scheduler", ["none", "cosine", "onecycle"]
            )

        else:  # tm
            if phase == "HB":
                dec, dec_ch, head_do, freeze_ep, lr_backbone, lr_head_mult, wdtm = _tm_space_hb(trial)
                opt = trial.suggest_categorical("optimizer", ["adamw", "adam"])
                # use tm_* LR/WD as the actual learning rate/weight decay
                lr = lr_backbone
                wd = wdtm
                sched = _schedule_space_hb(trial)
            else:
                dec = fixed.get("tm_decoder", "UperNetDecoder")
                dec_ch = int(fixed.get("tm_decoder_channels", 256))
                head_do = float(fixed.get("tm_head_dropout", 0.0))
                freeze_ep = int(fixed.get("tm_freeze_backbone_epochs", 1))
                lr_backbone, lr_head_mult, wdtm = _tm_space_bo(trial, fixed)
                opt = fixed.get("optimizer", "adamw")
                lr = lr_backbone
                wd = wdtm
                sched = trial.suggest_categorical("scheduler", ["none", "cosine", "onecycle"])

        # Tversky loss alpha (beta = 1 - alpha)
        if phase == "HB":
            tv_alpha = trial.suggest_float("tversky_alpha", 0.55, 0.95)
        else:
            tv_alpha = float(fixed.get("tversky_alpha", 0.7))

        # Data HPs (class imbalance / sampling)
        patch_h = hb_data_hp["patch_h"]
        patch_w = hb_data_hp["patch_w"]
        aug = hb_data_hp["augmenter_strength"]

        if phase == "HB":
            minpos = trial.suggest_categorical("min_pos_frac", [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
            pos_ratio = trial.suggest_categorical("pos_ratio", [None, 0.25, 0.5, 0.75])
        else:
            minpos = hb_data_hp["min_pos_frac"]
            pos_ratio = hb_data_hp.get("pos_ratio", getattr(conf, "pos_ratio", None))

        # Print trial configuration summary
        print(f"\n{'='*80}")
        print(f"TRIAL {trial.number} CONFIGURATION ({model_key.upper()} / {phase})")
        print(f"{'='*80}")
        
        # Model architecture parameters
        print(f"\n[MODEL ARCHITECTURE]")
        if model_key == "unet":
            print(f"  dilation_rate    : {dilation}")
            print(f"  layer_count      : {layer_cnt}")
            print(f"  l2_weight        : {l2w}")
            print(f"  dropout          : {drp}")
        elif model_key == "swin":
            if phase == "HB":
                print(f"  C                : {C}")
                print(f"  drop_path        : {drop_path}")
            else:
                print(f"  C                : {hb_data_hp['fixed']['C']}")
                print(f"  drop_path        : {hb_data_hp['fixed']['drop_path']}")
            print(f"  patch_size       : {hb_data_hp['fixed'].get('patch_size', 'N/A')}")
            print(f"  window_size      : {hb_data_hp['fixed'].get('window_size', 'N/A')}")
        else:  # tm
            print(f"  tm_decoder       : {dec}")
            print(f"  tm_decoder_ch    : {dec_ch}")
            print(f"  tm_head_dropout  : {head_do}")
            print(f"  tm_freeze_epochs : {freeze_ep}")
            print(f"  tm_lr_backbone   : {lr_backbone:.2e}")
            print(f"  tm_lr_head_mult  : {lr_head_mult}")
            print(f"  tm_weight_decay  : {wdtm:.2e}")
        
        # Optimizer & scheduler
        print(f"\n[OPTIMIZER & SCHEDULER]")
        print(f"  optimizer        : {opt}")
        print(f"  learning_rate    : {lr:.2e}")
        if wd is not None:
            print(f"  weight_decay     : {wd:.2e}")
        print(f"  scheduler        : {sched}")
        
        # Loss function
        print(f"\n[LOSS FUNCTION]")
        print(f"  tversky_alpha    : {tv_alpha:.4f}")
        print(f"  tversky_beta     : {1.0 - tv_alpha:.4f}")
        
        # Data parameters
        print(f"\n[DATA PARAMETERS]")
        print(f"  patch_size       : {patch_h}x{patch_w}")
        print(f"  augmentation     : {aug}")
        print(f"  min_pos_frac     : {minpos}")
        print(f"  pos_ratio        : {pos_ratio}")
        print(f"  batch_size       : {tune_batch}")
        
        print(f"{'='*80}\n")

        # Propagate sampled architecture / TM-specific params into conf for this trial
        if model_key == "unet":
            conf.dilation_rate = int(dilation)
            conf.layer_count = int(layer_cnt)
            conf.l2_weight = float(l2w)
            conf.dropout = float(drp)
        elif model_key == "tm":
            conf.tm_decoder = dec
            conf.tm_decoder_channels = int(dec_ch)
            conf.tm_head_dropout = float(head_do)
            conf.tm_lr_backbone = float(lr_backbone)
            conf.tm_lr_head_mult = float(lr_head_mult)
            conf.tm_weight_decay = float(wdtm)

        # Build data loaders for this configuration
        train_iter, val_iter = _get_loaders_for(patch_h, patch_w, aug, minpos, pos_ratio)

        # Build model
        if model_key == "unet":
            # Try to pass layer_count, dropout, l2_weight if UNet supports them
            _in_ch = len(getattr(conf, "channel_list", []))
            _shape_in = [conf.train_batch_size, patch_h, patch_w, _in_ch]
            _shape_out = [_in_ch]
            _dil = getattr(conf, "dilation_rate", 1)
            
            # Introspect UNet signature to see what it accepts
            unet_sig = inspect.signature(UNet)
            unet_params = set(unet_sig.parameters.keys())
            
            # Try common names for these parameters
            layer_count_aliases = ["layer_count", "base_filters", "base_channels", "filters", "n_filters", "features", "start_filters"]
            dropout_aliases = ["dropout", "dropout_rate", "p_dropout"]
            l2_weight_aliases = ["l2_weight", "l2", "l2_reg", "l2_lambda", "weight_decay"]
            
            kwargs = {}
            not_passed = []
            
            # Try to pass layer_count
            layer_count_val = getattr(conf, "layer_count", None)
            if layer_count_val is not None:
                found = False
                for alias in layer_count_aliases:
                    if alias in unet_params:
                        kwargs[alias] = int(layer_count_val)
                        found = True
                        break
                if not found and ("kwargs" in unet_params or "**" in str(unet_sig)):
                    kwargs["layer_count"] = int(layer_count_val)
                    found = True
                if not found:
                    not_passed.append(f"layer_count={layer_count_val}")
            
            # Try to pass dropout
            dropout_val = getattr(conf, "dropout", None)
            if dropout_val is not None:
                found = False
                for alias in dropout_aliases:
                    if alias in unet_params:
                        kwargs[alias] = float(dropout_val)
                        found = True
                        break
                if not found and ("kwargs" in unet_params or "**" in str(unet_sig)):
                    kwargs["dropout"] = float(dropout_val)
                    found = True
                if not found:
                    not_passed.append(f"dropout={dropout_val}")
            
            # Try to pass l2_weight
            l2_weight_val = getattr(conf, "l2_weight", None)
            if l2_weight_val is not None:
                found = False
                for alias in l2_weight_aliases:
                    if alias in unet_params:
                        kwargs[alias] = float(l2_weight_val)
                        found = True
                        break
                if not found and ("kwargs" in unet_params or "**" in str(unet_sig)):
                    kwargs["l2_weight"] = float(l2_weight_val)
                    found = True
                if not found:
                    not_passed.append(f"l2_weight={l2_weight_val}")
            
            model = UNet(
                [conf.train_batch_size, patch_h, patch_w, len(getattr(conf, "channel_list", []))],
                [len(getattr(conf, "channel_list", []))],
                getattr(conf, "dilation_rate", 1),
                **kwargs
            )

        elif model_key == "swin":
            # Match training: channels_used has priority over channel_list
            in_ch = len(
                getattr(conf, "channels_used", getattr(conf, "channel_list", []))
            )

            if phase == "HB":
                c_val = int(C)
                dp_val = float(drop_path)
            else:
                c_val = int(hb_data_hp["fixed"]["C"])
                dp_val = float(hb_data_hp["fixed"]["drop_path"])

            ws_val = int(hb_data_hp["fixed"]["window_size"])
            ps_val = int(hb_data_hp["fixed"]["patch_size"])

            model = SwinUNet(
                h=patch_h,
                w=patch_w,
                ch=in_ch,
                c=c_val,
                patch_size=ps_val,
                window_size=ws_val,
                drop_path=dp_val,
            )

        else:  # tm
            model = _build_terramind(conf, dec, dec_ch, head_do)

        # Compile
        model, optimizer, criterion, scheduler, step_per_batch = _compile_with_optimizer(
            model, opt, lr, wd, sched, steps, max_epochs, conf, tversky_alpha=tv_alpha
        )

        # Optional: freeze backbone for first epochs (TM)
        freeze_for = int(freeze_ep) if (model_key == "tm" and freeze_ep is not None) else 0

        # Multiple executions per trial
        scores = []
        for _ in range(int(executions_per_trial)):
            score = _single_execution(
                trial, train_iter, val_iter, model, optimizer, criterion, scheduler, step_per_batch, freeze_for
            )
            scores.append(score)

        best_score = float(np.max(scores))
        return best_score

    # Run study
    tag = f"{model_key} tuning"
    fixed = hb_data_hp.get("fixed", {})
    n_trials = _default(getattr(conf, "tune_trials", None), 20)
    if phase == "HB":
        n_trials = _default(getattr(conf, "tune_trials_hb", None), n_trials)
    study.optimize(objective, n_trials=int(n_trials), gc_after_trial=True)

    # Save best
    best_params = study.best_params
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_json_path = os.path.join(project_dir, f"{stamp}_{model_key}_{phase}_best.json")
    with open(best_json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[{tag} {phase}] best saved to: {best_json_path}")

    # --- ensure loaders are shut down ---
    try:
        for _k, (tr_dl, va_dl) in list(dl_cache.items()):
            _stop_dataloader_workers(tr_dl)
            _stop_dataloader_workers(va_dl)
        dl_cache.clear()
    except Exception:
        pass
    gc.collect()
    
    # Restore original conf values after phase completes
    _restore_conf()

    return best_params, study


# -----------------------
# Chained tuner (HB -> BO)
# -----------------------
def _tune_chained(
    conf, model_type: str = "unet", executions_per_trial: int = 1, overwrite: bool = False
) -> Dict[str, Any]:
    """
    Two-stage tuning: HyperBand (broad) then Bayesian Optimization (refinement).
    Batch stays fixed; Tversky loss (alpha tuned, beta=1-alpha); clipnorm=1.0; AMP off for parity/stability.
    """
    global config
    config = conf

    print("\nStarting chained tuning (HyperBand + Bayesian).")
    _seed(_default(getattr(config, "seed", None), 42))

    # Budgets - use better defaults for tuning if None
    tune_epochs_raw = getattr(config, "tune_num_epochs", None)
    tune_epochs_bo_raw = getattr(config, "tune_num_epochs_bo", None)
    tune_steps_raw = getattr(config, "tune_steps_per_epoch", None)
    tune_val_steps_raw = getattr(config, "tune_validation_steps", None)
    
    # Use tuning-friendly defaults if None
    hb_epochs = _default(tune_epochs_raw, 10)  # Reduced from 20 for faster tuning
    bo_epochs = _default(tune_epochs_bo_raw, hb_epochs if tune_epochs_bo_raw is None else 15)
    steps = _default(tune_steps_raw, 200)  # Increased from 100 for better convergence signals
    val_steps = _default(tune_val_steps_raw, 100)  # Increased from 50
    
    tune_batch = _default(
        getattr(config, "tune_batch_size", None),
        max(8, _default(getattr(config, "train_batch_size", 8), 8)),
    )

    # WARN if using defaults for critical budgets
    if tune_epochs_raw is None or tune_steps_raw is None or tune_val_steps_raw is None:
        print("\n" + "="*80)
        print("⚠️  WARNING: Tuning budget parameters not set or None in config!")
        print(f"⚠️  tune_num_epochs={tune_epochs_raw} -> using {hb_epochs}")
        print(f"⚠️  tune_num_epochs_bo={tune_epochs_bo_raw} -> using {bo_epochs}")
        print(f"⚠️  tune_steps_per_epoch={tune_steps_raw} -> using {steps}")
        print(f"⚠️  tune_validation_steps={tune_val_steps_raw} -> using {val_steps}")
        print("⚠️  Set these in your config for explicit control!")
        print("="*80 + "\n")

    logs_dir = _default(getattr(config, "logs_dir", "./logs"), "./logs")
    key = (
        "unet" if model_type.lower() == "unet" else ("swin" if model_type.lower() == "swin" else "tm")
    )
    project_dir = os.path.join(logs_dir, f"{key}_tuning")
    _ensure_dir(project_dir)

    # HB data defaults
    hb_aug = _default(getattr(config, "augmenter_strength", None), 1.0)
    hb_minpos = _default(getattr(config, "min_pos_frac", None), 0.0)
    hb_posratio = getattr(config, "pos_ratio", None)

    if key == "swin":
        # Match training defaults: swin_patch_size=16, swin_window=7
        fixed_ps = _default(getattr(config, "swin_patch_size", None), 16)
        fixed_ws = _default(getattr(config, "swin_window", None), 7)

        # Respect Swin-T limits when using ImageNet weights
        use_imnet = bool(getattr(config, "use_imagenet_weights", False))
        if use_imnet:
            fixed_ps = 4
            fixed_ws = 7

        base_H, base_W = 384, 384
        hb_patch_h, hb_patch_w = _snap_hw_for_swin(
            base_H, base_W, fixed_ps, fixed_ws, down_levels=3
        )
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
            pos_ratio=hb_posratio,
            fixed=dict(patch_size=fixed_ps, window_size=fixed_ws),
        )

    elif key == "tm":
        hb_patch_h, hb_patch_w = config.patch_size if hasattr(config, "patch_size") else (256, 256)
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
            pos_ratio=hb_posratio,
            fixed={},
        )
    else:
        hb_patch_h, hb_patch_w = config.patch_size if hasattr(config, "patch_size") else (128, 128)
        hb_data_hp = dict(
            patch_h=hb_patch_h,
            patch_w=hb_patch_w,
            augmenter_strength=hb_aug,
            min_pos_frac=hb_minpos,
            pos_ratio=hb_posratio,
            fixed={},
        )

    # Phase 1: HB
    print_search_space(key, "HB", config, hb_data_hp)

    hb_best, study_hb = _run_phase(
        config,
        key,
        "HB",
        project_dir,
        tune_batch,
        hb_epochs,
        steps,
        val_steps,
        executions_per_trial,
        overwrite,
        hb_data_hp,
    )

    # Update sampling and loss params from HB best
    hb_minpos = float(hb_best.get("min_pos_frac", hb_minpos))
    hb_posratio = hb_best.get("pos_ratio", hb_posratio)
    tv_alpha_best = float(hb_best.get("tversky_alpha", getattr(config, "tversky_alpha", 0.7)))

    # Merge HB best and set up BO fixed settings
    if key == "unet":
        fixed = dict(
            optimizer=hb_best.get("optimizer", _default(getattr(config, "optimizer_fn", "adam"), "adam")),
            scheduler=hb_best.get("scheduler", "none"),
            dilation_rate=int(hb_best.get("dilation_rate", getattr(config, "dilation_rate", 1))),
            layer_count=int(hb_best.get("layer_count", getattr(config, "layer_count", 64))),
            l2_weight=float(hb_best.get("l2_weight", getattr(config, "l2_weight", 0.0))),
            dropout=float(hb_best.get("dropout", getattr(config, "dropout", 0.0))),
            tversky_alpha=tv_alpha_best,
        )
        bo_patch_h, bo_patch_w = int(hb_data_hp["patch_h"]), int(hb_data_hp["patch_w"])
    elif key == "swin":
        use_imnet = bool(getattr(config, "use_imagenet_weights", False))
        fixed_C = 96 if use_imnet else int(hb_best.get("C", getattr(config, "swin_base_channels", 96)))
        fixed = {
            "optimizer": hb_best.get(
                "optimizer", _default(getattr(config, "optimizer_fn", "adam"), "adam")
            ),
            "scheduler": hb_best.get("scheduler", "none"),
            "C": fixed_C,
            "patch_size": hb_data_hp["fixed"]["patch_size"],
            "window_size": hb_data_hp["fixed"]["window_size"],
            "drop_path": float(hb_best.get("drop_path", 0.1)),
            "tversky_alpha": tv_alpha_best,
        }
        bo_patch_h, bo_patch_w = int(hb_data_hp["patch_h"]), int(hb_data_hp["patch_w"])
    else:  # tm
        fixed = {
            "optimizer": hb_best.get(
                "optimizer", _default(getattr(config, "optimizer_fn", "adamw"), "adamw")
            ),
            "scheduler": hb_best.get("scheduler", "none"),
            "tm_decoder": hb_best.get("tm_decoder", "UperNetDecoder"),
            "tm_decoder_channels": int(hb_best.get("tm_decoder_channels", 256)),
            "tm_head_dropout": float(hb_best.get("tm_head_dropout", 0.0)),
            "tm_freeze_backbone_epochs": int(hb_best.get("tm_freeze_backbone_epochs", 1)),
            "tm_lr_backbone": float(hb_best.get("tm_lr_backbone", getattr(config, "tm_lr_backbone", 1e-5))),
            "tm_lr_head_mult": float(hb_best.get("tm_lr_head_mult", getattr(config, "tm_lr_head_mult", 10.0))),
            "tm_weight_decay": float(hb_best.get("tm_weight_decay", getattr(config, "tm_weight_decay", 1e-4))),
            "tversky_alpha": tv_alpha_best,
        }
        bo_patch_h, bo_patch_w = int(hb_data_hp["patch_h"]), int(hb_data_hp["patch_w"])

    # Phase 2: BO
    bo_data_hp = dict(
        patch_h=bo_patch_h,
        patch_w=bo_patch_w,
        augmenter_strength=hb_aug,
        min_pos_frac=hb_minpos,
        pos_ratio=hb_posratio,
        fixed=fixed,
    )

    print_search_space(key, "BO", config, bo_data_hp, hb_best=hb_best, fixed=fixed)

    bo_best, _ = _run_phase(
        config,
        key,
        "BO",
        project_dir,
        tune_batch,
        bo_epochs,
        steps,
        val_steps,
        executions_per_trial,
        overwrite,
        bo_data_hp,
    )

    # Merge final settings
    final = dict(fixed)
    if "learning_rate" in bo_best:
        final["learning_rate"] = float(bo_best["learning_rate"])
    if "weight_decay" in bo_best:
        final["weight_decay"] = float(bo_best["weight_decay"])
    if "scheduler" in bo_best:
        final["scheduler"] = bo_best["scheduler"]
    # TM-specific refined values (if present)
    for k in ("tm_lr_backbone", "tm_lr_head_mult", "tm_weight_decay"):
        if k in bo_best:
            final[k] = float(bo_best[k])
    final["patch_h"] = bo_patch_h
    final["patch_w"] = bo_patch_w
    final["augmenter_strength"] = float(hb_aug)
    final["min_pos_frac"] = float(hb_minpos)
    final["pos_ratio"] = hb_posratio
    final["tversky_alpha"] = float(tv_alpha_best)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_path = os.path.join(project_dir, f"{stamp}_{key}_tuning_FINAL_best_hparams.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print("\n" + "=" * 60)
    print(f"[{key} tuning] FINAL merged best saved to: {final_path}")
    print(json.dumps(final, indent=2))
    print("=" * 60 + "\n")
    return final


# -----------------------
# Public entry points
# -----------------------
def tune_UNet(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="unet")


def tune_SwinUNetPP(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="swin")


def tune_TerraMind(conf) -> Dict[str, Any]:
    return _tune_chained(conf, model_type="tm")
