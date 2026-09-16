"""
Evaluate a saved checkpoint on the **full** validation split: reports top-1
and top-5 accuracy, plus **per-class accuracy** and the **top confusions** for
the worst classes.

Uses ``dataset.val_dir`` (entire folder; no ``split_train_val``).

Example (from ``src/``)::

    python evaluate.py training.checkpoint_path=best_model.pt

The per-class report shows, for each class:
  - support (samples in val)
  - top-1 accuracy on that class
  - top-3 confusions (most frequent wrong predictions for this true class)

This pinpoints which actions the model struggles with so you can target
augmentations / training tweaks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, compute_prior_logits, set_seed


def _build_class_names(val_dir: Path, num_classes: int) -> List[str]:
    """Read class folder names under val_dir and return a list[idx] -> human name.

    Folders are named like ``017_Closing_something``. Strips the numeric prefix
    and replaces underscores with spaces. Falls back to "class_<i>" if not found.
    """
    names: Dict[int, str] = {}
    if val_dir.is_dir():
        for entry in sorted(val_dir.iterdir()):
            if not entry.is_dir():
                continue
            m = re.match(r"^(\d+)_(.*)$", entry.name)
            if m is None:
                continue
            idx = int(m.group(1))
            label = m.group(2).replace("_", " ").strip()
            names[idx] = label
    return [names.get(i, f"class_{i}") for i in range(num_classes)]


def _print_per_class_report(
    per_class_correct: torch.Tensor,
    per_class_total: torch.Tensor,
    confusion: torch.Tensor,
    class_names: List[str],
    worst_n: int = 10,
    top_confusions: int = 3,
) -> None:
    """Per-class recall + precision table + confusion drill-down for worst classes.

    - recall[k]    = TP_k / (TP_k + FN_k) = "of clips truly k, how many caught"
    - precision[k] = TP_k / (TP_k + FP_k) = "when model says k, how often right"

    A class with high recall but low precision is being OVER-predicted (typical
    after aggressive class-balanced sampling). Low recall + low precision = the
    model genuinely can't represent that action.
    """
    num_classes = per_class_total.numel()
    # Column sums of the confusion matrix = times each class was predicted.
    per_class_predicted = confusion.sum(dim=0)

    recall = torch.where(
        per_class_total > 0,
        per_class_correct.float() / per_class_total.clamp(min=1).float(),
        torch.full_like(per_class_correct, -1.0, dtype=torch.float),
    )
    precision = torch.where(
        per_class_predicted > 0,
        per_class_correct.float() / per_class_predicted.clamp(min=1).float(),
        torch.full_like(per_class_correct, -1.0, dtype=torch.float),
    )

    valid_mask = per_class_total > 0
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
    # Sort by recall ascending → worst classes first.
    sorted_by_acc = sorted(valid_indices, key=lambda i: float(recall[i].item()))

    print("\n=== Per-class recall + precision (sorted worst recall → best) ===")
    print(
        f"  {'idx':>3}  {'class':30s}  {'support':>7}  "
        f"{'pred':>5}  {'rec':>6}  {'prec':>6}"
    )
    print(
        f"  {'---':>3}  {'-'*30}  {'-------':>7}  "
        f"{'-----':>5}  {'-----':>6}  {'-----':>6}"
    )
    for idx in sorted_by_acc:
        n = int(per_class_total[idx].item())
        pred_count = int(per_class_predicted[idx].item())
        r = float(recall[idx].item())
        p = float(precision[idx].item())
        prec_str = f"{p:.3f}" if p >= 0.0 else "  n/a"
        print(
            f"  {idx:>3}  {class_names[idx][:30]:30s}  {n:>7}  "
            f"{pred_count:>5}  {r:>6.3f}  {prec_str:>6}"
        )

    if not sorted_by_acc:
        return

    print(f"\n=== Top-{top_confusions} confusions for the {worst_n} worst classes ===")
    for idx in sorted_by_acc[:worst_n]:
        row = confusion[idx].clone()           # predicted-class counts when true=idx
        row[idx] = 0                            # zero out correct predictions
        if row.sum().item() == 0:
            continue
        # Take top-k wrong predictions.
        vals, conf_idx = row.topk(min(top_confusions, num_classes), largest=True)
        total_true = int(per_class_total[idx].item())
        r = float(recall[idx].item())
        print(
            f"\n  [{idx}] {class_names[idx]}  (support={total_true}, recall={r:.3f})"
        )
        for v, c in zip(vals.tolist(), conf_idx.tolist()):
            if v == 0:
                continue
            pct = 100.0 * v / max(total_true, 1)
            print(
                f"      → confused with [{c}] {class_names[c]:30s}  "
                f"{v} times  ({pct:.1f}%)"
            )


def load_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Rebuild the model from the Hydra config stored in the checkpoint (same as training).

    Checkpoints must include ``config`` (saved by ``train.py``). No duplicate
    architecture list here—``build_model`` is the single construction site.
    """
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError(
            "Checkpoint has no 'config' entry. Train with the current train.py so the "
            "full Hydra config is saved with the weights."
        )
    cfg = OmegaConf.create(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(raw, device)

    # Normalization must match how the checkpoint was trained (ImageNet stats if pretrained).
    pretrained_used = bool(raw.get("pretrained", cfg.model.pretrained))
    # Use the same image_size as training: stored in the checkpoint's config,
    # falling back to the current cfg / 224.
    ckpt_cfg = OmegaConf.create(raw["config"]) if "config" in raw else cfg
    image_size = int(
        ckpt_cfg.dataset.get("image_size", cfg.dataset.get("image_size", 224))
    )
    eval_transform = build_transforms(
        image_size=image_size,
        is_training=False,
        use_imagenet_norm=pretrained_used,
    )

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(cfg.model.num_classes)

    # ---- Optional prior calibration (opt-in, default OFF) ----------------
    # Enable with:  +calibration.use_prior=true
    # Override source with: +calibration.train_dir=/path/to/train  (defaults
    # to dataset.train_dir from the loaded config). Subtracts log P(y) from
    # the raw logits at inference time, undoing the class-frequency bias the
    # model picked up at training. This is purely post-hoc — your existing
    # checkpoint is unchanged.
    prior_logits: torch.Tensor | None = None
    calibration_cfg = cfg.get("calibration", None)
    if calibration_cfg is not None and bool(calibration_cfg.get("use_prior", False)):
        prior_train_dir = calibration_cfg.get("train_dir", cfg.dataset.train_dir)
        prior_alpha = float(calibration_cfg.get("alpha", 1.0))
        prior_logits = compute_prior_logits(
            train_dir=prior_train_dir,
            num_classes=num_classes,
            alpha=prior_alpha,
        ).to(device)
        print(
            f"\nPrior calibration: ENABLED  (alpha={prior_alpha}, from {prior_train_dir})",
            flush=True,
        )
    else:
        print("\nPrior calibration: disabled (set +calibration.use_prior=true to enable)")

    # ---- EXPERIMENTAL: per-class logit boost (opt-in, default OFF) --------
    # Multiplies the softmax probability of chosen classes by a factor — which
    # is mathematically identical to adding log(factor) to their logits — to
    # push up the recall of under-predicted classes. Enable with e.g.:
    #   +logit_boost.enabled=true +logit_boost.factor=3 '+logit_boost.classes=[26,16,11,17]'
    # Per-class factors (override the global one, can also be <1 to suppress):
    #   '+logit_boost.per_class={26:5.0,22:0.5}'
    # WARNING: for top-1 this typically trades accuracy away on LOW-PRECISION
    # classes — watch the `prec` column below. Delete this whole block (and the
    # `boost_offset` line in the loop) to revert.
    boost_offset: torch.Tensor | None = None
    boost_cfg = cfg.get("logit_boost", None)
    if boost_cfg is not None and bool(boost_cfg.get("enabled", False)):
        factors = torch.ones(num_classes, dtype=torch.float32)
        default_factor = float(boost_cfg.get("factor", 1.0))
        for c in (boost_cfg.get("classes", []) or []):
            ci = int(c)
            if 0 <= ci < num_classes:
                factors[ci] = default_factor
        for k, v in dict(boost_cfg.get("per_class", {}) or {}).items():
            ci = int(k)
            if 0 <= ci < num_classes:
                factors[ci] = float(v)
        if bool((factors <= 0).any()):
            raise ValueError(
                "logit_boost factors must be > 0 (they multiply probabilities)."
            )
        boost_offset = torch.log(factors).to(device)  # +log(f) on logit == ×f on prob
        boosted = {
            i: round(float(factors[i]), 3)
            for i in range(num_classes)
            if abs(float(factors[i]) - 1.0) > 1e-9
        }
        print(f"Logit boost: ENABLED  (class -> factor: {boosted})", flush=True)
    else:
        print("Logit boost: disabled (set +logit_boost.enabled=true to enable)")

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    # Per-class buckets (CPU, accumulated across batches).
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    # confusion[t, p] = number of clips whose true class is t and pred is p.
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for video_batch, labels in val_loader:
            video_batch = video_batch.to(device)
            labels = labels.to(device)
            logits = model(video_batch)  # (B, num_classes)

            # Prior calibration (no-op if prior_logits is None).
            if prior_logits is not None:
                logits = logits - prior_logits

            # EXPERIMENTAL per-class logit boost (no-op if boost_offset is None).
            if boost_offset is not None:
                logits = logits + boost_offset

            # Top-1: argmax class matches label
            predictions_top1 = logits.argmax(dim=1)
            correct_top1 += int((predictions_top1 == labels).sum().item())

            # Top-5: label appears in the five largest logits per row
            _, predictions_top5 = logits.topk(5, dim=1, largest=True, sorted=True)
            # (B, 5) compared with (B, 1) -> (B, 5) boolean, True if label in top-5
            matches_top5 = predictions_top5.eq(labels.view(-1, 1)).any(dim=1)
            correct_top5 += int(matches_top5.sum().item())

            total += labels.size(0)

            # Per-class bookkeeping (move to CPU once per batch).
            labels_cpu = labels.cpu()
            preds_cpu = predictions_top1.cpu()
            per_class_total.scatter_add_(
                0, labels_cpu, torch.ones_like(labels_cpu)
            )
            hits = (preds_cpu == labels_cpu).long()
            per_class_correct.scatter_add_(0, labels_cpu, hits)
            # confusion[true, pred] += 1, batched.
            flat = labels_cpu * num_classes + preds_cpu
            confusion.view(-1).scatter_add_(
                0, flat, torch.ones_like(flat)
            )

    top1_accuracy = correct_top1 / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)

    print(f"\nValidation samples: {len(val_dataset)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")

    class_names = _build_class_names(val_dir, num_classes)
    _print_per_class_report(
        per_class_correct,
        per_class_total,
        confusion,
        class_names,
        worst_n=int(cfg.get("eval_worst_n", 10)),
        top_confusions=int(cfg.get("eval_top_confusions", 3)),
    )


if __name__ == "__main__":
    main()
