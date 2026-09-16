"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_transformer import CNNTransformer
from models.tsm import TSM
from models.two_stream_transformer import TwoStreamTransformer
from utils import (
    FocalLoss,
    build_transforms,
    compute_class_weights,
    compute_prior_logits,
    compute_sample_weights,
    kfold_split,
    set_seed,
    split_train_val,
)
from torch.utils.data import WeightedRandomSampler

# Track B (open world) models — lazy imports inside build_model so Track A users
# don't need to install the extra transformers/huggingface dependency.


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        # Defaults preserve old `.pt` loading: a checkpoint saved before the
        # multi-backbone refactor has no `backbone` / `dropout` keys in its
        # cfg, so .get(...) falls back to "resnet18" / 0.0 — matching the
        # original architecture exactly.
        return CNNBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=str(cfg.model.get("backbone", "resnet18")),
            dropout=float(cfg.model.get("dropout", 0.0)),
        )
    if name == "cnn_lstm":
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 256)),
            lstm_num_layers=int(cfg.model.get("lstm_num_layers", 2)),
            lstm_dropout=float(cfg.model.get("lstm_dropout", 0.3)),
            feature_dropout=float(cfg.model.get("feature_dropout", 0.2)),
            head_dropout=float(cfg.model.get("head_dropout", 0.5)),
        )
    if name == "cnn_transformer":
        return CNNTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            spatial_tokens_side=int(cfg.model.get("spatial_tokens_side", 1)),
            d_model=int(cfg.model.get("d_model", 512)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 2.0)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
            drop_path=float(cfg.model.get("drop_path", 0.1)),
        )
    if name == "two_stream_transformer":
        return TwoStreamTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            base_channels=int(cfg.model.get("base_channels", 32)),
            spatial_attn_heads=int(cfg.model.get("spatial_attn_heads", 4)),
            d_model=int(cfg.model.get("d_model", 256)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            dim_feedforward=int(cfg.model.get("dim_feedforward", 1024)),
            dropout=float(cfg.model.get("dropout", 0.2)),
            head_dropout=float(cfg.model.get("head_dropout", 0.5)),
        )
    if name == "tsm":
        # TSM requires num_segments == dataset.num_frames (asserted in forward).
        return TSM(
            num_classes=num_classes,
            num_segments=int(cfg.model.get("num_segments", cfg.dataset.num_frames)),
            pretrained=pretrained,
            base_channels=int(cfg.model.get("base_channels", 64)),
            dropout=float(cfg.model.get("dropout", 0.5)),
            fold_div=int(cfg.model.get("fold_div", 8)),
            temporal_pool=str(cfg.model.get("temporal_pool", "mean")),
            resnet34=bool(cfg.model.get("resnet34", False)),
        )
    if name == "vivit":
        from models.vivit import ViViT

        return ViViT(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            image_size=int(cfg.model.get("image_size", 224)),
            patch_size=int(cfg.model.get("patch_size", 16)),
            tubelet_size=int(cfg.model.get("tubelet_size", 2)),
            embed_dim=int(cfg.model.get("embed_dim", 256)),
            spatial_depth=int(cfg.model.get("spatial_depth", 6)),
            temporal_depth=int(cfg.model.get("temporal_depth", 4)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
            dropout=float(cfg.model.get("dropout", 0.0)),
            attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.1)),
            layer_scale_init=float(cfg.model.get("layer_scale_init", 1.0)),
        )

    # Track B (open world) models. Prefixed with "b_" by convention.
    # Lazy import keeps the transformers dependency optional for Track A.
    if name == "b_videomae":
        from models.b_videomae import VideoMAEClassifier

        return VideoMAEClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_id=str(
                cfg.model.get("model_id", "MCG-NJU/videomae-base-finetuned-ssv2")
            ),
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
        )
    if name == "b_timesformer":
        from models.b_timesformer import TimeSformerClassifier

        return TimeSformerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_id=str(
                cfg.model.get("model_id", "facebook/timesformer-base-finetuned-ssv2")
            ),
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
        )
    if name in ("b_vjepa2", "b_vjepa2_1b"):
        # ViT-L (b_vjepa2) and ViT-g 1B (b_vjepa2_1b) share the same wrapper.
        # Only the defaults differ — model_id, image_size, gradient checkpointing.
        from models.b_vjepa2 import VJEPA2Classifier

        is_1b = name == "b_vjepa2_1b"
        default_model_id = (
            "facebook/vjepa2-vitg-fpc64-384-ssv2"
            if is_1b
            else "facebook/vjepa2-vitl-fpc16-256-ssv2"
        )
        default_image_size = 384 if is_1b else 256
        default_grad_ckpt = is_1b  # mandatory on ViT-g, optional on ViT-L

        # LoRA params are optional. If `model.use_lora` is absent or false,
        # the wrapper behaves exactly as before — old checkpoints stay loadable.
        lora_target = cfg.model.get("lora_target_modules", None)
        if lora_target is not None:
            lora_target = list(lora_target)  # OmegaConf ListConfig -> list

        return VJEPA2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_id=str(cfg.model.get("model_id", default_model_id)),
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            image_size=int(cfg.model.get("image_size", default_image_size)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            use_gradient_checkpointing=bool(
                cfg.model.get("use_gradient_checkpointing", default_grad_ckpt)
            ),
            use_lora=bool(cfg.model.get("use_lora", False)),
            lora_r=int(cfg.model.get("lora_r", 16)),
            lora_alpha=int(cfg.model.get("lora_alpha", 32)),
            lora_dropout=float(cfg.model.get("lora_dropout", 0.05)),
            lora_target_modules=lora_target,
            lora_bias=str(cfg.model.get("lora_bias", "none")),
        )

    if name == "b_mvitv2":
        # MViT / MViTv2 — hierarchical transformer, architecturally diverse
        # from V-JEPA 2 / VideoMAE / TimeSformer. `model_id` selects the
        # variant (Base ≈ 36M params, Large ≈ 213M).
        from models.b_mvitv2 import MViTv2Classifier

        return MViTv2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_id=str(
                cfg.model.get("model_id", "facebook/mvit-base-finetuned-ssv2")
            ),
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            image_size=int(cfg.model.get("image_size", 224)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            use_gradient_checkpointing=bool(
                cfg.model.get("use_gradient_checkpointing", False)
            ),
        )

    if name == "maxvit_temporal":
        # MaxViT-Tiny (Tu et al. ECCV 2022) wrapped as a per-frame encoder
        # with temporal mean pooling. Each block stacks MBConv + 7x7 block
        # (local) attention + grid (sparse-global) attention — strong CNN
        # inductive bias, trains from scratch. Track A default.
        from models.maxvit_temporal import MaxViTTemporal

        return MaxViTTemporal(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=float(cfg.model.get("dropout", 0.3)),
        )

    if name == "coatnet_temporal":
        # CoAtNet (Dai et al. NeurIPS 2021) via timm: 5-stage "conv first,
        # attention later" — MBConv (S1-S2) then relative-attention
        # transformer (S3-S4). Strong CNN prior + global receptive field;
        # trains from scratch on small datasets. Architecturally
        # complementary to MaxViT for ensembles.
        from models.coatnet_temporal import CoAtNetTemporal

        return CoAtNetTemporal(
            num_classes=num_classes,
            pretrained=pretrained,
            variant=str(cfg.model.get("variant", "coatnet_0_rw_224")),
            dropout=float(cfg.model.get("dropout", 0.3)),
            use_gradient_checkpointing=bool(
                cfg.model.get("use_gradient_checkpointing", False)
            ),
        )

    raise ValueError(f"Unknown model.name: {name}")


def _sample_beta_lambda(alpha: float) -> float:
    """Sample λ ~ Beta(alpha, alpha). Returns 1.0 if alpha <= 0 (mixing disabled)."""
    if alpha <= 0.0:
        return 1.0
    return float(torch.distributions.Beta(alpha, alpha).sample().item())


def _rand_cutmix_bbox(H: int, W: int, lam: float) -> Tuple[int, int, int, int]:
    """Random rectangle (y1, y2, x1, x2) with target area ≈ (1 - lam) * H * W."""
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    ch = int(H * cut_ratio)
    cw = int(W * cut_ratio)
    cy = int(torch.randint(0, H, (1,)).item())
    cx = int(torch.randint(0, W, (1,)).item())
    y1 = max(0, cy - ch // 2)
    y2 = min(H, cy + ch // 2)
    x1 = max(0, cx - cw // 2)
    x2 = min(W, cx + cw // 2)
    return y1, y2, x1, x2


def _apply_mixup_cutmix(
    video: torch.Tensor,
    labels: torch.Tensor,
    use_mixup: bool,
    use_cutmix: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup or CutMix to a (B, T, C, H, W) batch of clips.

    Returns (mixed_video, labels_a, labels_b, lambda). If no mixing happens,
    labels_a == labels_b == labels and lambda == 1.0 (downstream loss reduces
    to plain CE on labels_a).

    When both flags are on, each batch flips a coin and picks one of the two
    (standard timm-style policy).
    """
    if not (use_mixup or use_cutmix):
        return video, labels, labels, 1.0
    if torch.rand(()).item() >= mix_prob:
        return video, labels, labels, 1.0

    if use_mixup and use_cutmix:
        mode = "mixup" if torch.rand(()).item() < 0.5 else "cutmix"
    elif use_mixup:
        mode = "mixup"
    else:
        mode = "cutmix"

    B = video.size(0)
    perm = torch.randperm(B, device=video.device)
    labels_b = labels[perm]

    if mode == "mixup":
        lam = _sample_beta_lambda(mixup_alpha)
        # Linear interpolation in pixel space (post-normalization). Identical
        # crop/flip/etc was already applied per-clip, so no temporal break.
        video = lam * video + (1.0 - lam) * video[perm]
    else:  # cutmix
        lam = _sample_beta_lambda(cutmix_alpha)
        _, _, _, H, W = video.shape
        y1, y2, x1, x2 = _rand_cutmix_bbox(H, W, lam)
        if (y2 - y1) > 0 and (x2 - x1) > 0:
            # Same rectangle pasted on all T frames of every clip → temporal
            # consistency preserved.
            video[:, :, :, y1:y2, x1:x2] = video[perm][:, :, :, y1:y2, x1:x2]
            # Adjust λ to the actual cut area (handles edge clipping).
            lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(H * W)
        # else: cut had zero area, leave video and lam as-is

    return video, labels, labels_b, lam


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    log_prior_for_train: Optional[torch.Tensor] = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 1.0,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch.

    When use_mixup/use_cutmix are on, training mixes pairs of clips per batch.
    The reported "accuracy" then counts matches against the primary label
    (labels_a) — it under-reports a bit but stays consistent for monitoring.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        video_batch, labels_a, labels_b, lam = _apply_mixup_cutmix(
            video_batch,
            labels,
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(video_batch)  # (B, num_classes)
            # Logit Adjustment (Menon et al. 2020): add tau*log_prior to logits
            # at TRAINING time so the network learns to produce log P(x|y)
            # natively. At eval, the raw logits are used (no adjustment), which
            # gives the correctly de-biased predictions.
            logits_for_loss = logits
            if log_prior_for_train is not None:
                logits_for_loss = logits + log_prior_for_train
            if lam < 1.0:
                loss = lam * loss_fn(logits_for_loss, labels_a) + (1.0 - lam) * loss_fn(
                    logits_for_loss, labels_b
                )
            else:
                loss = loss_fn(logits_for_loss, labels_a)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        # Accuracy is computed on RAW logits to mirror inference behavior.
        predictions = logits.argmax(dim=1)
        # Match against the primary (un-permuted) label — biased but consistent.
        correct += int((predictions == labels_a).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    # K-fold mode (opt-in). When `dataset.kfold_num` is set to an integer >= 2
    # AND `dataset.kfold_index` is in [0, kfold_num), use a stratified K-fold
    # split instead of the default random val split. The held-out fold becomes
    # the internal validation set; the remaining folds become train. This is
    # what `oof_predict.py` then exploits to build out-of-fold softmaxes.
    kfold_num = cfg.dataset.get("kfold_num", None)
    kfold_index = cfg.dataset.get("kfold_index", None)
    kfold_info: Optional[Dict[str, Any]] = None
    if kfold_num is not None and kfold_index is not None:
        kfold_num_i = int(kfold_num)
        kfold_index_i = int(kfold_index)
        train_samples, val_samples, fold_id_by_pos = kfold_split(
            all_samples,
            num_folds=kfold_num_i,
            fold_index=kfold_index_i,
            seed=int(cfg.dataset.seed),
        )
        print(
            f"K-fold split: num_folds={kfold_num_i}  fold_index={kfold_index_i}  "
            f"train={len(train_samples)}  val={len(val_samples)}",
            flush=True,
        )
        # Persist enough info that the OOF script can reconstruct which clips
        # were held out *without* re-running the split.
        kfold_info = {
            "num_folds": kfold_num_i,
            "fold_index": kfold_index_i,
            "seed": int(cfg.dataset.seed),
            # The (path, label) pairs that the model NEVER saw — these are the
            # clips for which this checkpoint produces honest OOF predictions.
            "val_sample_paths": [str(p) for p, _ in val_samples],
            "val_sample_labels": [int(lbl) for _, lbl in val_samples],
        }
    else:
        train_samples, val_samples = split_train_val(
            all_samples,
            val_ratio=float(cfg.dataset.val_ratio),
            seed=int(cfg.dataset.seed),
        )

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
    use_imagenet_norm = bool(cfg.model.pretrained)
    image_size = int(cfg.dataset.get("image_size", 224))
    train_transform = build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        use_horizontal_flip=bool(cfg.training.get("use_horizontal_flip", False)),
        use_random_crop=bool(cfg.training.get("use_random_crop", False)),
        random_crop_scale=tuple(cfg.training.get("random_crop_scale", (0.7, 1.0))),
        random_crop_ratio=tuple(cfg.training.get("random_crop_ratio", (0.85, 1.15))),
        use_color_jitter=bool(cfg.training.get("use_color_jitter", False)),
        color_jitter_strength=float(cfg.training.get("color_jitter_strength", 0.2)),
        use_random_erasing=bool(cfg.training.get("use_random_erasing", False)),
        random_erasing_p=float(cfg.training.get("random_erasing_p", 0.25)),
        random_erasing_scale=tuple(
            cfg.training.get("random_erasing_scale", (0.02, 0.2))
        ),
        random_erasing_ratio=tuple(
            cfg.training.get("random_erasing_ratio", (0.3, 3.3))
        ),
        use_rotation=bool(cfg.training.get("use_rotation", False)),
        rotation_degrees=float(cfg.training.get("rotation_degrees", 5.0)),
        use_sharpness=bool(cfg.training.get("use_sharpness", False)),
        sharpness_strength=float(cfg.training.get("sharpness_strength", 0.5)),
        use_blur=bool(cfg.training.get("use_blur", False)),
        blur_p=float(cfg.training.get("blur_p", 0.2)),
        blur_kernel=int(cfg.training.get("blur_kernel", 5)),
        blur_sigma=tuple(cfg.training.get("blur_sigma", (0.1, 1.5))),
    )
    eval_transform = build_transforms(
        image_size=image_size,
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    num_workers = int(cfg.training.num_workers)
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.training.get("persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(
            cfg.training.get("prefetch_factor", 4)
        )

    use_balanced_sampler = bool(
        cfg.training.get("use_class_balanced_sampler", False)
    )
    if use_balanced_sampler:
        balance_method = str(cfg.training.get("class_balance_method", "sqrt"))
        sample_weights = compute_sample_weights(train_samples, method=balance_method)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )
        print(
            f"Class-balanced sampler enabled (method={balance_method!r}, "
            f"{len(train_samples)} samples/epoch with oversampling of rare classes)."
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(cfg).to(device)

    best_val_accuracy = 0.0
    resume_path = cfg.training.get("resume_checkpoint")
    if resume_path:
        resume_path = Path(resume_path).resolve()
        print(f"Resuming weights from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_accuracy = float(ckpt.get("val_accuracy", 0.0))
        print(f"Resuming best val accuracy from checkpoint: {best_val_accuracy:.4f}")

    # ---- Loss function: opt-in Focal Loss (default = CE with smoothing) ----
    # Knobs (all default to legacy behavior):
    #   training.use_focal_loss      (bool, default False)
    #   training.focal_gamma         (float, default 2.0)
    #   training.use_class_weights   (bool, default False)
    #   training.class_weights_method  ("inv_freq" | "inv_sqrt_freq" |
    #                                   "effective_num" | "none", default "inv_sqrt_freq")
    #   training.class_weights_beta  (float, default 0.9999, only for "effective_num")
    #   training.label_smoothing     (float, default 0.1)
    use_focal_loss = bool(cfg.training.get("use_focal_loss", False))
    use_class_weights = bool(cfg.training.get("use_class_weights", False))
    label_smoothing = float(cfg.training.get("label_smoothing", 0.1))
    class_weights_tensor: Optional[torch.Tensor] = None
    if use_class_weights:
        class_weights_method = str(cfg.training.get("class_weights_method", "inv_sqrt_freq"))
        class_weights_beta = float(cfg.training.get("class_weights_beta", 0.9999))
        class_weights_tensor = compute_class_weights(
            train_samples,
            num_classes=int(cfg.model.num_classes),
            method=class_weights_method,
            beta=class_weights_beta,
        ).to(device)
        print(
            f"Class weights: ENABLED  (method={class_weights_method}, "
            f"mean={float(class_weights_tensor.mean()):.3f})"
        )

    if use_focal_loss:
        focal_gamma = float(cfg.training.get("focal_gamma", 2.0))
        loss_fn: nn.Module = FocalLoss(
            gamma=focal_gamma, alpha_weight=class_weights_tensor
        ).to(device)
        # Focal loss + label smoothing is a non-standard combination — warn if both.
        if label_smoothing > 0.0:
            print(
                f"NOTE: focal_loss=True ignores label_smoothing (was {label_smoothing}). "
                "The focusing factor already provides implicit regularization on easy samples."
            )
        print(f"Loss: FocalLoss(gamma={focal_gamma}, alpha_weight={'yes' if class_weights_tensor is not None else 'no'})")
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights_tensor, label_smoothing=label_smoothing
        ).to(device)
        print(
            f"Loss: CrossEntropyLoss(label_smoothing={label_smoothing}, "
            f"weight={'yes' if class_weights_tensor is not None else 'no'})"
        )

    # ---- Logit Adjustment (Menon et al. 2020): opt-in, default OFF ---------
    #   training.use_logit_adjustment  (bool, default False)
    #   training.logit_adjustment_tau  (float, default 1.0)
    log_prior_for_train: Optional[torch.Tensor] = None
    if bool(cfg.training.get("use_logit_adjustment", False)):
        la_tau = float(cfg.training.get("logit_adjustment_tau", 1.0))
        la_train_dir = cfg.training.get("logit_adjustment_train_dir", None)
        if la_train_dir is None:
            la_train_dir = cfg.dataset.train_dir
        log_prior_for_train = compute_prior_logits(
            train_dir=la_train_dir,
            num_classes=int(cfg.model.num_classes),
            alpha=la_tau,
            missing_class_logit=100.0,
        ).to(device)
        print(
            f"Logit adjustment: ENABLED  (tau={la_tau}, source={la_train_dir})  "
            "— inference uses raw logits (no post-hoc calibration needed)."
        )

    optimizer_name = str(cfg.training.get("optimizer", "adam")).lower()
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    lr = float(cfg.training.lr)

    # Layer-wise LR decay: head gets full LR, earlier transformer blocks get
    # progressively smaller LRs. Standard for ViT fine-tuning.
    use_llrd = bool(cfg.training.get("lwlrdecay", False))
    if use_llrd:
        if not hasattr(model, "get_param_groups_llrd"):
            raise ValueError(
                f"training.lwlrdecay=true but model {cfg.model.name!r} does "
                "not implement get_param_groups_llrd(). Only V-JEPA 2 supports "
                "LLRD currently — set training.lwlrdecay=false."
            )
        llrd_rate = float(cfg.training.get("llrd_rate", 0.65))
        param_groups = model.get_param_groups_llrd(
            base_lr=lr, weight_decay=weight_decay, decay_rate=llrd_rate
        )
        lrs = sorted({float(g["lr"]) for g in param_groups})
        print(
            f"Layer-wise LR decay: rate={llrd_rate}, {len(param_groups)} groups, "
            f"LR range [{lrs[0]:.2e}, {lrs[-1]:.2e}]"
        )
    elif hasattr(model, "no_weight_decay") and weight_decay > 0:
        # ViT convention: exclude CLS tokens, position embeddings, biases and
        # LayerNorm params from weight decay. The model declares which named
        # params via no_weight_decay(); 1-D params (biases / norm gammas) are
        # caught automatically.
        no_decay_names = set(model.no_weight_decay())
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = (
                name in no_decay_names
                or param.ndim <= 1
                or name.endswith(".bias")
            )
            (no_decay_params if is_no_decay else decay_params).append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        print(
            f"Weight decay split: {sum(p.numel() for p in decay_params):,} "
            f"params w/ wd={weight_decay}, "
            f"{sum(p.numel() for p in no_decay_params):,} params w/o "
            f"(named: {sorted(no_decay_names)})"
        )
    else:
        param_groups = model.parameters()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = float(cfg.training.get("momentum", 0.9))
        nesterov = bool(cfg.training.get("nesterov", False))
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        raise ValueError(f"Unknown training.optimizer: {optimizer_name}")
    extra = ""
    if optimizer_name == "sgd":
        extra = f", momentum={cfg.training.get('momentum', 0.9)}"
    print(
        f"Optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay}{extra})"
    )

    # ---- LR schedule (opt-in, default OFF = flat LR / legacy behavior) ------
    # "cosine": linear warmup for `warmup_epochs`, then cosine decay to
    # `min_lr_ratio * lr` over the remaining epochs. LambdaLR multiplies each
    # param group's base LR by the same factor, so per-group LRs (LLRD /
    # weight-decay split) keep their relative scale.
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scheduler_name = str(cfg.training.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
        total_epochs = int(cfg.training.epochs)
        min_lr_ratio = float(cfg.training.get("min_lr_ratio", 0.0))

        def _lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            denom = max(1, total_epochs - warmup_epochs)
            progress = min(max(float(epoch - warmup_epochs) / float(denom), 0.0), 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(
            f"LR schedule: cosine (warmup_epochs={warmup_epochs}, "
            f"total_epochs={total_epochs}, min_lr_ratio={min_lr_ratio})"
        )
    elif scheduler_name == "none":
        print("LR schedule: none (constant LR)")
    else:
        raise ValueError(
            f"Unknown training.scheduler: {scheduler_name!r} (use 'none' or 'cosine')"
        )

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    use_amp = bool(cfg.training.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    mixup_kwargs: Dict[str, Any] = {
        "use_mixup": bool(cfg.training.get("use_mixup", False)),
        "mixup_alpha": float(cfg.training.get("mixup_alpha", 0.2)),
        "use_cutmix": bool(cfg.training.get("use_cutmix", False)),
        "cutmix_alpha": float(cfg.training.get("cutmix_alpha", 1.0)),
        "mix_prob": float(cfg.training.get("mix_prob", 1.0)),
    }
    if mixup_kwargs["use_mixup"] or mixup_kwargs["use_cutmix"]:
        modes = []
        if mixup_kwargs["use_mixup"]:
            modes.append(f"Mixup(α={mixup_kwargs['mixup_alpha']})")
        if mixup_kwargs["use_cutmix"]:
            modes.append(f"CutMix(α={mixup_kwargs['cutmix_alpha']})")
        print(
            f"Batch mixing enabled: {' + '.join(modes)} "
            f"(applied with p={mixup_kwargs['mix_prob']})"
        )

    # ---- Weights & Biases (opt-in, default OFF) ---------------------------
    # Enable with `wandb.enabled=true`. The full resolved Hydra config is sent
    # as the run's `config`, so every hyperparameter (sampler, scheduler,
    # temporal_pool, dropout, optimizer, lr, ...) is filterable/groupable in
    # the dashboard. Run name defaults to the checkpoint file stem so the 8
    # ablation runs are individually identifiable.
    wandb_cfg = cfg.get("wandb", None)
    wandb_run = None
    if wandb_cfg is not None and bool(wandb_cfg.get("enabled", False)):
        import wandb

        run_name = wandb_cfg.get("name") or checkpoint_path.stem
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "video-anticipation")),
            entity=wandb_cfg.get("entity", None),
            name=str(run_name),
            tags=list(wandb_cfg.get("tags", []) or []),
            mode=str(wandb_cfg.get("mode", "online")),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(
            f"W&B logging: ENABLED (project={wandb_cfg.get('project', 'video-anticipation')}, "
            f"run={run_name}, mode={wandb_cfg.get('mode', 'online')})"
        )

    for epoch in range(int(cfg.training.epochs)):
        # LR used during this epoch (captured before the post-epoch step).
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, use_amp,
            log_prior_for_train=log_prior_for_train,
            **mixup_kwargs,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, use_amp
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | lr {current_lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "lr": current_lr,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                },
                step=epoch + 1,
            )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )
            if kfold_info is not None:
                payload["kfold_info"] = kfold_info

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )
            if wandb_run is not None:
                wandb_run.summary["best_val_accuracy"] = best_val_accuracy

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")

    if wandb_run is not None:
        wandb_run.summary["best_val_accuracy"] = best_val_accuracy
        wandb_run.finish()


if __name__ == "__main__":
    main()
