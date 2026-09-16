#!/usr/bin/env python3
"""
Build out-of-fold (OOF) softmax predictions for a K-fold-trained model family.

Workflow recap:

    1. Train K models with the same config but distinct `dataset.kfold_index`:
           uv run python src/train.py experiment=tsm \\
               +dataset.kfold_num=5 +dataset.kfold_index=0 \\
               training.checkpoint_path=tsm_fold0.pt
           ... (k=1..4) ...
       Each .pt stores a `kfold_info` dict listing the clips it never saw.

    2. Run this script to glue the K checkpoints into one (N_train, C)
       softmax tensor where row i holds the prediction for train clip i
       produced by the fold model that did NOT see it:

           uv run python src/oof_predict.py \\
               --checkpoints tsm_fold0.pt tsm_fold1.pt tsm_fold2.pt \\
                             tsm_fold3.pt tsm_fold4.pt \\
               --output oof_tsm.pt

    3. Pass that .pt to ensemble_predict.py via --oof-softmaxes (one file per
       model family). Ensemble weights fit on OOF predictions are leakage-free
       because every model produced its row without ever having seen that clip.

Backward-compatibility / safety:
  * Existing checkpoints without `kfold_info` are rejected with a clear error.
  * The script verifies that the union of held-out clips covers exactly the
    train set (no gaps, no duplicates). The covering check is on (path, label)
    tuples, so re-extracting frames doesn't break it.
  * Output ordering is sorted by sample-path string so different model
    families produce row-aligned tensors automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="The K fold checkpoints of one model family. Each must contain a "
        "'kfold_info' entry written by train.py with K-fold mode enabled.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path of the output .pt file (a dict with 'softmax', 'labels', "
        "'sample_paths').",
    )
    p.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Override dataset.train_dir. Defaults to the value stored in the "
        "first checkpoint's config (which is what the K-fold split was run on).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_model_and_meta(
    checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in ckpt or ckpt["config"] is None:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has no 'config' entry — re-train it "
            "with the current train.py."
        )
    if "kfold_info" not in ckpt:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} has no 'kfold_info'. Re-train it "
            "with +dataset.kfold_num=K +dataset.kfold_index=I so the held-out "
            "fold is recorded inside the .pt."
        )
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    meta: Dict[str, Any] = {
        "num_frames": int(ckpt.get("num_frames", cfg.dataset.num_frames)),
        "pretrained": bool(ckpt.get("pretrained", cfg.model.pretrained)),
        "image_size": int(cfg.dataset.get("image_size", 224)),
        "config": cfg,
        "model_name": str(cfg.model.name),
        "kfold_info": dict(ckpt["kfold_info"]),
    }
    return model, meta


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tag: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    for batch_idx, (video_batch, labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        logits = model(video_batch)
        probs = F.softmax(logits, dim=-1).cpu()
        probs_chunks.append(probs)
        label_chunks.append(labels.cpu())
        if batch_idx % log_every == 0 or batch_idx == n_batches:
            print(f"    [{tag}] inference batch {batch_idx}/{n_batches}", flush=True)
    return torch.cat(probs_chunks, dim=0), torch.cat(label_chunks, dim=0)


def main() -> None:
    args = parse_args()

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # ---- Pre-pass: cheap header peek to discover the train_dir + train list -
    first_ckpt = torch.load(args.checkpoints[0], map_location="cpu")
    first_cfg = OmegaConf.create(first_ckpt["config"])
    train_dir = Path(
        args.train_dir if args.train_dir is not None else first_cfg.dataset.train_dir
    ).resolve()
    print(f"Train dir: {train_dir}", flush=True)

    # The reference list of (path, label) — its length is the OOF tensor row count.
    # We sort by path string so the output is deterministic and row-aligned across
    # different model families.
    all_samples_unsorted = collect_video_samples(train_dir)
    all_samples = sorted(all_samples_unsorted, key=lambda pl: str(pl[0]))
    path_to_position: Dict[str, int] = {
        str(p): i for i, (p, _) in enumerate(all_samples)
    }
    n_train = len(all_samples)
    print(f"Train clips: {n_train}", flush=True)

    # Pre-fill the OOF storage with NaN so we can detect any uncovered row.
    num_classes_storage: Optional[int] = None
    oof_softmax: Optional[torch.Tensor] = None
    oof_labels = torch.full((n_train,), -1, dtype=torch.long)
    covered = torch.zeros(n_train, dtype=torch.bool)

    # ---- Loop over the K fold checkpoints --------------------------------
    seen_num_folds: Optional[int] = None
    seen_seed: Optional[int] = None
    seen_fold_indices: List[int] = []

    for idx, ckpt_path_str in enumerate(args.checkpoints, start=1):
        ckpt_path = Path(ckpt_path_str).resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")

        print(f"\n[{idx}/{len(args.checkpoints)}] Loading {ckpt_path.name}", flush=True)
        model, meta = _load_model_and_meta(ckpt_path, device)
        info = meta["kfold_info"]
        nf = int(info["num_folds"])
        fi = int(info["fold_index"])
        seed = int(info["seed"])

        # Sanity: all K checkpoints must come from the SAME K-fold split.
        if seen_num_folds is None:
            seen_num_folds = nf
            seen_seed = seed
        else:
            if seen_num_folds != nf or seen_seed != seed:
                raise SystemExit(
                    f"Checkpoint {ckpt_path.name} has num_folds={nf} seed={seed} "
                    f"but previous ones had num_folds={seen_num_folds} "
                    f"seed={seen_seed}. All K checkpoints must share the split."
                )
        if fi in seen_fold_indices:
            raise SystemExit(
                f"fold_index={fi} appears twice across the supplied checkpoints."
            )
        seen_fold_indices.append(fi)

        # The held-out clips were stored at training time. Rebuild the
        # sample_list pointing at the (possibly absolute) paths and the labels
        # that were saved alongside.
        val_paths = [Path(p) for p in info["val_sample_paths"]]
        val_labels = [int(lb) for lb in info["val_sample_labels"]]
        sample_list: List[Tuple[Path, int]] = list(zip(val_paths, val_labels))

        # Make sure those clips still exist (paths can move across machines).
        for vp in val_paths:
            if str(vp) not in path_to_position:
                raise SystemExit(
                    f"Held-out clip {vp} is not in the current train_dir "
                    f"({train_dir}). Either restore it or override --train-dir."
                )

        transform = build_transforms(
            image_size=meta["image_size"],
            is_training=False,
            use_imagenet_norm=meta["pretrained"],
        )
        dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=meta["num_frames"],
            transform=transform,
            sample_list=sample_list,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

        print(
            f"    fold={fi}/{nf}  num_frames={meta['num_frames']}  "
            f"image_size={meta['image_size']}  pretrained={meta['pretrained']}  "
            f"clips={len(sample_list)}",
            flush=True,
        )
        probs, labels = _run_inference(model, loader, device, tag=f"fold{fi}")

        # Materialize OOF storage once we know num_classes.
        if num_classes_storage is None:
            num_classes_storage = int(probs.size(1))
            oof_softmax = torch.full(
                (n_train, num_classes_storage), float("nan"), dtype=torch.float32
            )
        else:
            if int(probs.size(1)) != num_classes_storage:
                raise SystemExit(
                    f"Checkpoint {ckpt_path.name} produces "
                    f"{int(probs.size(1))} classes but earlier ones had "
                    f"{num_classes_storage}."
                )

        # Scatter the fold's predictions into the global OOF tensor.
        assert oof_softmax is not None
        for k, vp in enumerate(val_paths):
            row = path_to_position[str(vp)]
            if covered[row]:
                raise SystemExit(
                    f"Clip {vp} is covered by more than one fold — the K-fold "
                    f"split is inconsistent across checkpoints."
                )
            oof_softmax[row] = probs[k]
            oof_labels[row] = int(labels[k].item())
            covered[row] = True

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Sanity check: every clip must be covered exactly once -----------
    if seen_num_folds is None:
        raise SystemExit("No checkpoints loaded.")
    if len(seen_fold_indices) != seen_num_folds:
        raise SystemExit(
            f"Got {len(seen_fold_indices)} fold checkpoints "
            f"({sorted(seen_fold_indices)}) but the split declared "
            f"num_folds={seen_num_folds}. Pass all K checkpoints."
        )
    uncovered = (~covered).sum().item()
    if uncovered > 0:
        raise SystemExit(
            f"{uncovered}/{n_train} train clips were not predicted by any "
            "fold. Did the K-fold split match the current train_dir?"
        )

    assert oof_softmax is not None
    assert num_classes_storage is not None

    # Quick diagnostic: in-sample OOF top-1 (an honest accuracy estimate).
    preds = oof_softmax.argmax(dim=-1)
    top1 = float((preds == oof_labels).float().mean().item())
    print(
        f"\nOOF top-1 on the full train set ({n_train} clips): {top1:.4f}",
        flush=True,
    )

    # Write the blob expected by ensemble_predict.py --oof-softmaxes.
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "softmax": oof_softmax,
            "labels": oof_labels,
            "sample_paths": [str(p) for p, _ in all_samples],
            "num_folds": seen_num_folds,
            "seed": seen_seed,
            "train_dir": str(train_dir),
        },
        output_path,
    )
    print(f"\nWrote OOF softmax tensor to {output_path}", flush=True)


if __name__ == "__main__":
    main()
