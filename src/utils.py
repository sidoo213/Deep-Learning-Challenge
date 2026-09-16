"""
Small helpers: reproducibility, clip-level transforms, and metrics.

The transforms returned by ``build_transforms`` operate on a **list of PIL
frames** rather than a single image. All random parameters (crop, flip, color
jitter, random erasing) are sampled ONCE per clip and applied identically to
every frame. This preserves temporal consistency, which matters a lot for
Something-Something where most of the signal is in inter-frame motion.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# A clip transform maps a list of PIL frames to a list of (C, H, W) tensors.
ClipTransform = Callable[[List[Image.Image]], List[torch.Tensor]]


def _make_normalize(use_imagenet_norm: bool) -> transforms.Normalize:
    if use_imagenet_norm:
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _sample_erase_rect(
    tensor: torch.Tensor,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    max_attempts: int = 10,
) -> Optional[Tuple[int, int, int, int]]:
    """Sample (i, j, h, w) for a random-erasing rectangle, or None if no valid
    rectangle was found within max_attempts. Same shape returned per clip,
    applied identically to every frame.
    """
    _, H, W = tensor.shape
    area = H * W
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(max_attempts):
        target_area = area * float(torch.empty(()).uniform_(scale[0], scale[1]))
        aspect = math.exp(float(torch.empty(()).uniform_(log_lo, log_hi)))
        h = int(round(math.sqrt(target_area * aspect)))
        w = int(round(math.sqrt(target_area / aspect)))
        if 0 < h < H and 0 < w < W:
            i = int(torch.randint(0, H - h + 1, (1,)).item())
            j = int(torch.randint(0, W - w + 1, (1,)).item())
            return i, j, h, w
    return None


class _TrainClipTransform:
    """Train-time augmentation. All random params sampled once per clip and
    applied identically to every frame.
    """

    def __init__(
        self,
        image_size: int,
        use_imagenet_norm: bool,
        use_horizontal_flip: bool,
        use_random_crop: bool,
        random_crop_scale: Tuple[float, float],
        random_crop_ratio: Tuple[float, float],
        use_color_jitter: bool,
        color_jitter_strength: float,
        use_random_erasing: bool,
        random_erasing_p: float,
        random_erasing_scale: Tuple[float, float],
        random_erasing_ratio: Tuple[float, float],
        use_rotation: bool,
        rotation_degrees: float,
        use_sharpness: bool,
        sharpness_strength: float,
        use_blur: bool,
        blur_p: float,
        blur_kernel: int,
        blur_sigma: Tuple[float, float],
    ) -> None:
        self.image_size = image_size
        self.use_horizontal_flip = use_horizontal_flip
        self.use_random_crop = use_random_crop
        self.random_crop_scale = tuple(random_crop_scale)
        self.random_crop_ratio = tuple(random_crop_ratio)
        self.use_color_jitter = use_color_jitter
        self.color_jitter_strength = float(color_jitter_strength)
        self.use_random_erasing = use_random_erasing
        self.random_erasing_p = float(random_erasing_p)
        self.random_erasing_scale = tuple(random_erasing_scale)
        self.random_erasing_ratio = tuple(random_erasing_ratio)
        self.use_rotation = use_rotation
        self.rotation_degrees = float(rotation_degrees)
        self.use_sharpness = use_sharpness
        self.sharpness_strength = float(sharpness_strength)
        self.use_blur = use_blur
        self.blur_p = float(blur_p)
        self.blur_kernel = int(blur_kernel) if int(blur_kernel) % 2 == 1 else int(blur_kernel) + 1
        self.blur_sigma = tuple(blur_sigma)

        self.normalize = _make_normalize(use_imagenet_norm)

        if use_color_jitter:
            s = self.color_jitter_strength
            self._color_jitter = transforms.ColorJitter(
                brightness=s, contrast=s, saturation=s, hue=s * 0.5
            )
        else:
            self._color_jitter = None

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        size = self.image_size
        first = frames[0]

        # ---- Sample shared parameters once per clip ----
        if self.use_random_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                first,
                scale=self.random_crop_scale,
                ratio=self.random_crop_ratio,
            )
        else:
            i = j = 0
            w_px, h_px = first.size  # PIL: (W, H)
            h, w = h_px, w_px

        if self.use_rotation and self.rotation_degrees > 0:
            angle = float(
                torch.empty(()).uniform_(
                    -self.rotation_degrees, self.rotation_degrees
                ).item()
            )
        else:
            angle = 0.0

        do_flip = self.use_horizontal_flip and (torch.rand(()).item() < 0.5)

        if self._color_jitter is not None:
            fn_idx, b_f, c_f, s_f, h_f = self._color_jitter.get_params(
                self._color_jitter.brightness,
                self._color_jitter.contrast,
                self._color_jitter.saturation,
                self._color_jitter.hue,
            )
        else:
            fn_idx = ()
            b_f = c_f = s_f = h_f = None

        if self.use_sharpness and self.sharpness_strength > 0:
            sharpness_factor = float(
                torch.empty(()).uniform_(
                    max(0.0, 1.0 - self.sharpness_strength),
                    1.0 + self.sharpness_strength,
                ).item()
            )
        else:
            sharpness_factor = 1.0

        do_blur = self.use_blur and (torch.rand(()).item() < self.blur_p)
        if do_blur:
            blur_sigma = float(
                torch.empty(()).uniform_(self.blur_sigma[0], self.blur_sigma[1]).item()
            )
        else:
            blur_sigma = 0.0

        # ---- Apply per frame with the SHARED params ----
        out: List[torch.Tensor] = []
        for img in frames:
            if self.use_random_crop:
                img = TF.resized_crop(img, i, j, h, w, [size, size])
            else:
                img = TF.resize(img, [size, size])

            if self.use_rotation and abs(angle) > 1e-3:
                img = TF.rotate(img, angle, fill=0)

            if do_flip:
                img = TF.hflip(img)

            if self._color_jitter is not None:
                for fn_id in fn_idx:
                    fn_id_int = int(fn_id)
                    if fn_id_int == 0 and b_f is not None:
                        img = TF.adjust_brightness(img, b_f)
                    elif fn_id_int == 1 and c_f is not None:
                        img = TF.adjust_contrast(img, c_f)
                    elif fn_id_int == 2 and s_f is not None:
                        img = TF.adjust_saturation(img, s_f)
                    elif fn_id_int == 3 and h_f is not None:
                        img = TF.adjust_hue(img, h_f)

            if self.use_sharpness and abs(sharpness_factor - 1.0) > 1e-3:
                img = TF.adjust_sharpness(img, sharpness_factor)

            if do_blur:
                img = TF.gaussian_blur(
                    img,
                    kernel_size=[self.blur_kernel, self.blur_kernel],
                    sigma=[blur_sigma, blur_sigma],
                )

            tensor = TF.to_tensor(img)
            tensor = self.normalize(tensor)
            out.append(tensor)

        # ---- Random erasing: shared rectangle across the whole clip ----
        if (
            self.use_random_erasing
            and torch.rand(()).item() < self.random_erasing_p
        ):
            rect = _sample_erase_rect(
                out[0], self.random_erasing_scale, self.random_erasing_ratio
            )
            if rect is not None:
                ei, ej, eh, ew = rect
                for t in out:
                    t[:, ei : ei + eh, ej : ej + ew] = 0.0

        return out


class _EvalClipTransform:
    """Eval-time transform: deterministic resize + ToTensor + Normalize, same
    on every frame.
    """

    def __init__(self, image_size: int, use_imagenet_norm: bool) -> None:
        self.image_size = image_size
        self.normalize = _make_normalize(use_imagenet_norm)

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        size = self.image_size
        out: List[torch.Tensor] = []
        for img in frames:
            img = TF.resize(img, [size, size])
            tensor = TF.to_tensor(img)
            tensor = self.normalize(tensor)
            out.append(tensor)
        return out


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    use_horizontal_flip: bool = True,
    use_random_crop: bool = False,
    random_crop_scale: Tuple[float, float] = (0.7, 1.0),
    random_crop_ratio: Tuple[float, float] = (0.85, 1.15),
    use_color_jitter: bool = False,
    color_jitter_strength: float = 0.2,
    use_random_erasing: bool = False,
    random_erasing_p: float = 0.25,
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2),
    random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
    use_rotation: bool = False,
    rotation_degrees: float = 5.0,
    use_sharpness: bool = False,
    sharpness_strength: float = 0.5,
    use_blur: bool = False,
    blur_p: float = 0.2,
    blur_kernel: int = 5,
    blur_sigma: Tuple[float, float] = (0.1, 1.5),
) -> ClipTransform:
    """Build a clip-level augmentation pipeline.

    Returns a callable mapping ``List[PIL.Image] -> List[torch.Tensor]``. All
    random parameters are sampled once per clip and applied identically to
    every frame, so the temporal motion within a clip is preserved.
    """
    if is_training:
        return _TrainClipTransform(
            image_size=image_size,
            use_imagenet_norm=use_imagenet_norm,
            use_horizontal_flip=use_horizontal_flip,
            use_random_crop=use_random_crop,
            random_crop_scale=random_crop_scale,
            random_crop_ratio=random_crop_ratio,
            use_color_jitter=use_color_jitter,
            color_jitter_strength=color_jitter_strength,
            use_random_erasing=use_random_erasing,
            random_erasing_p=random_erasing_p,
            random_erasing_scale=random_erasing_scale,
            random_erasing_ratio=random_erasing_ratio,
            use_rotation=use_rotation,
            rotation_degrees=rotation_degrees,
            use_sharpness=use_sharpness,
            sharpness_strength=sharpness_strength,
            use_blur=use_blur,
            blur_p=blur_p,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
        )
    return _EvalClipTransform(
        image_size=image_size, use_imagenet_norm=use_imagenet_norm
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def compute_sample_weights(
    samples: List[Tuple[Path, int]],
    method: str = "sqrt",
) -> List[float]:
    """Per-sample weights for ``WeightedRandomSampler`` based on class frequency.

    method:
        "inv"  -> weight = 1 / count(class)   (full inverse frequency)
        "sqrt" -> weight = 1 / sqrt(count)    (softer; recommended default)
        "none" -> weight = 1 for all samples  (no rebalancing)
    """
    if method not in {"inv", "sqrt", "none"}:
        raise ValueError(f"Unknown class_balance_method: {method!r}")

    counts: dict[int, int] = {}
    for _path, label in samples:
        counts[label] = counts.get(label, 0) + 1

    if method == "none":
        return [1.0] * len(samples)

    import math as _math

    per_class: dict[int, float] = {}
    for c, n in counts.items():
        if method == "inv":
            per_class[c] = 1.0 / float(n)
        else:  # "sqrt"
            per_class[c] = 1.0 / _math.sqrt(float(n))

    return [per_class[label] for _path, label in samples]


def compute_class_weights(
    samples: List[Tuple[Path, int]],
    num_classes: int,
    method: str = "inv_freq",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Per-class scalar weights tensor for losses like ``FocalLoss(alpha_weight=...)``.

    Mirrors ``compute_sample_weights`` but yields a length-``num_classes``
    tensor instead of per-sample weights. Weights are normalized so their
    mean equals 1.0 (so the absolute loss magnitude stays comparable to
    plain cross-entropy).

    Methods:
        ``"none"``           -> all 1s (no rebalancing).
        ``"inv_freq"``       -> w_k = 1 / count_k (full inverse frequency).
        ``"inv_sqrt_freq"``  -> w_k = 1 / sqrt(count_k) (softer; safer).
        ``"effective_num"``  -> Class-Balanced Loss (Cui et al. 2019):
                                w_k = (1 - beta) / (1 - beta**count_k).
                                Smoothly interpolates between "none" (beta=0)
                                and "inv_freq" (beta -> 1). beta=0.9999 is
                                a common default for fine-tuning regimes.

    Missing classes (count_k == 0) get weight 0 — neutralized in the loss.
    """
    if method not in {"none", "inv_freq", "inv_sqrt_freq", "effective_num"}:
        raise ValueError(f"Unknown class-weight method: {method!r}")

    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _path, label in samples:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1.0

    if method == "none":
        return torch.ones(num_classes, dtype=torch.float32)

    if method == "inv_freq":
        weights = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
    elif method == "inv_sqrt_freq":
        weights = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0).sqrt(), torch.zeros_like(counts))
    else:  # "effective_num"
        b = float(beta)
        eff = 1.0 - torch.pow(torch.tensor(b, dtype=torch.float64), counts)
        weights = torch.where(counts > 0, (1.0 - b) / eff.clamp(min=1e-12), torch.zeros_like(counts))

    present = (weights > 0).sum().clamp(min=1)
    weights = weights / (weights.sum() / present.double())  # mean over present classes = 1
    return weights.float()


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) for multi-class classification.

    .. math::
        \\text{FL}(p_t) = -\\alpha_t \\cdot (1 - p_t)^{\\gamma} \\cdot \\log(p_t)

    where ``p_t`` is the softmax probability assigned to the true class.
    Compared to vanilla cross-entropy, the ``(1 - p_t)^gamma`` factor
    **down-weights easy examples** (high ``p_t``) and **focuses gradient on
    hard ones** (low ``p_t``, often the confidently-wrong predictions where
    the model is fooled by a semantic twin).

    ``gamma = 0`` recovers weighted cross-entropy. ``gamma = 2`` is the
    typical setting from the paper and works well on class-imbalanced
    classification.

    Note: focal loss is **incompatible with label smoothing** in a clean
    formulation (the focusing factor is per-true-class). If you previously
    relied on smoothing for regularization, consider keeping it OFF when
    using focal loss — the focusing effect provides similar implicit
    regularization on easy examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha_weight is not None:
            self.register_buffer("alpha_weight", alpha_weight.float())
        else:
            self.alpha_weight = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        target = target.long()
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        focal_factor = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(self.gamma)
        loss = -focal_factor * log_p_t

        if self.alpha_weight is not None:
            alpha_t = self.alpha_weight.to(logits.device).gather(0, target)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_prior_logits(
    train_dir: str | Path,
    num_classes: int,
    alpha: float = 1.0,
    missing_class_logit: float = 100.0,
) -> torch.Tensor:
    """Compute α · log P(y) from class-folder counts in ``train_dir``.

    Used at inference time for **prior calibration**: subtracting this tensor
    from the raw model logits removes the bias inherited from class-imbalanced
    training data. Mathematically, the trained network approximates::

        log P(y | x) ≈ log P(x | y) + log P(y_train)

    Subtracting ``α · log P(y_train)`` from the logits yields a quantity
    proportional to ``P(x | y)·P(y_train)^(1-α)``. With α=1.0 this is the
    pure Bayes posterior P(x | y); smaller α dampens the correction.

    Args:
        train_dir: Directory whose immediate subfolders are named
            ``"NNN_ClassName"`` with NNN the class index used in training.
        num_classes: Number of model output classes.
        alpha: Strength of the calibration in [0, 1].
            * 0.0  -> no calibration (returns zeros).
            * 1.0  -> full Bayes prior removal (default).
            * 0.5  -> half-strength; safer when the model has only partially
              memorized the prior (often the case when training with
              class-balanced sampling or mixup).
        missing_class_logit: Value assigned to classes that have **zero**
            training samples (e.g. the 33-class subset where index 27 has no
            folder). A large positive value ensures the calibrated logit
            ``raw - missing_class_logit`` is effectively -inf, so the
            calibrated model never predicts that index.

    Returns:
        Tensor of shape ``(num_classes,)``, dtype ``float32``.

    .. note::
        Earlier versions of this function used a tiny smoothing constant
        (``log(1e-9) ≈ -20.7``) for missing classes. That backfired: the
        calibration step ``logit - (-20.7) = logit + 20.7`` **boosted**
        empty classes instead of suppressing them, causing the calibrated
        model to predict only the missing index. The fix below assigns a
        large positive log-prior to missing classes so subtraction crushes
        them.
    """
    path = Path(train_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Prior train dir not found: {path}")

    counts = torch.zeros(num_classes, dtype=torch.float64)
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        match = re.match(r"^(\d+)_", entry.name)
        if match is None:
            continue
        idx = int(match.group(1))
        if idx < 0 or idx >= num_classes:
            continue
        counts[idx] = float(sum(1 for v in entry.iterdir() if v.is_dir()))

    total = counts.sum().item()
    if total <= 0:
        raise RuntimeError(
            f"compute_prior_logits: no class folders / videos found under {path}"
        )

    present = counts > 0
    # Log-prior for classes that have at least one sample.
    safe_counts = torch.where(present, counts, torch.ones_like(counts))
    log_prior = torch.log(safe_counts / total)

    # Missing classes: large positive value → calibrated logit becomes -∞.
    log_prior = torch.where(
        present,
        log_prior,
        torch.full_like(log_prior, float(missing_class_logit)),
    )

    return (float(alpha) * log_prior).float()


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Shuffle then split (video_path, label) pairs into train/val portions."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples


def kfold_split(
    samples: List[Tuple[Path, int]],
    num_folds: int,
    fold_index: int,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[int]]:
    """Stratified K-fold split of (video_path, label) pairs.

    Each clip is assigned to exactly one fold by a deterministic per-class
    round-robin after shuffling. This guarantees:
      * Every clip appears in val of exactly one fold (essential for OOF).
      * Class distribution is approximately preserved across folds.
      * The assignment is reproducible from (seed, num_folds) alone.

    Returns:
        train_samples: clips in folds != fold_index.
        val_samples:   clips in fold == fold_index (held out).
        clip_fold_ids: for the FULL input list (in input order, NOT shuffled),
                       the fold id each clip was assigned to. Use this to
                       map per-clip OOF predictions back to the original
                       ordering of ``samples`` later.
    """
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")
    if not (0 <= fold_index < num_folds):
        raise ValueError(
            f"fold_index must be in [0, {num_folds-1}], got {fold_index}"
        )

    rng = random.Random(seed)

    # Group sample positions by label so we can round-robin per class.
    per_class_positions: dict[int, List[int]] = {}
    for pos, (_path, label) in enumerate(samples):
        per_class_positions.setdefault(int(label), []).append(pos)
    # Shuffle within each class so the per-fold assignment is randomized
    # but still deterministic for a given seed.
    for label in per_class_positions:
        rng.shuffle(per_class_positions[label])

    fold_id_by_pos = [0] * len(samples)
    for label, positions in per_class_positions.items():
        # Stagger the starting fold per class so small classes don't all
        # land in fold 0 first.
        start = label % num_folds
        for k, pos in enumerate(positions):
            fold_id_by_pos[pos] = (start + k) % num_folds

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    for pos, sample in enumerate(samples):
        if fold_id_by_pos[pos] == fold_index:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(
            f"kfold_split produced an empty side: train={len(train_samples)} "
            f"val={len(val_samples)} (num_folds={num_folds}, fold_index={fold_index})."
        )

    return train_samples, val_samples, fold_id_by_pos
