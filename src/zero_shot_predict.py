#!/usr/bin/env python3
"""
Zero-shot submission using V-JEPA 2 finetuned on full SSv2 (174 classes).

No training. Loads the model with its original SSv2 head intact, maps the
challenge's 33 classes to their SSv2 counterparts by name, runs inference on
the test split, writes a submission CSV in the same format as
``create_submission.py``::

    video_name,predicted_class

Usage (from ``src/``)::

    python zero_shot_predict.py
    python zero_shot_predict.py +zeroshot.model_id=facebook/vjepa2-vitl-fpc16-256-ssv2 \\
                                +zeroshot.image_size=256
    python zero_shot_predict.py dataset.test_dir=/path/to/test \\
                                dataset.submission_output=zeroshot.csv

The list of challenge classes is read from ``dataset.train_dir`` by walking
its top-level ``NNN_Class_name`` folders. The integer index is the leading
``NNN`` so it matches what ``VideoFrameDataset`` will use during evaluation.

Crash-safe streaming writes
---------------------------
Predictions are written incrementally to ``dataset.submission_output`` after
every batch (single write + flush + fsync). If the process is killed or the
machine loses power, just relaunch the same command: the script heals any
partial trailing line in the CSV, reads which videos have already been
predicted, and resumes on the remaining ones. Toggles::

    +zeroshot.resume=false    # delete the existing CSV and start from scratch
    +zeroshot.fsync=false     # skip the per-batch fsync (faster but loses
                              # the last few batches on power loss)
"""

from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)
from dataset.video_dataset import VideoFrameDataset
from models.b_vjepa2 import VJEPA2ZeroShotClassifier
from utils import build_transforms, set_seed


# --------------------------------------------------------------------------- #
# Name normalization                                                          #
# --------------------------------------------------------------------------- #

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: str) -> str:
    """Lower-case, strip leading 'NNN_' prefix, collapse non-alphanumeric to
    single spaces. Robust to "[something]" vs "something", underscores vs
    spaces, punctuation, etc.

    Examples:
        '000_Pouring_something_into_something' -> 'pouring something into something'
        'Pouring [something] into [something]' -> 'pouring something into something'
        "Pretending to throw something"        -> 'pretending to throw something'
    """
    n = re.sub(r"^\d+_", "", name)
    n = _NORMALIZE_RE.sub(" ", n.lower()).strip()
    return re.sub(r"\s+", " ", n)


# --------------------------------------------------------------------------- #
# Crash-safe submission writing (resume after power loss / SIGKILL)           #
# --------------------------------------------------------------------------- #


def _heal_csv(path: Path) -> None:
    """Truncate any orphan bytes after the last newline. Idempotent.

    If the previous run crashed mid-write (power loss, SIGKILL), the file may
    end with a partial row that has no trailing ``\\n``. We scan from the end,
    find the last newline, and truncate everything after it. After this, every
    line the parser sees is complete.
    """
    if not path.is_file():
        return
    with open(path, "rb+") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return
        chunk = 4096
        pos = size
        while pos > 0:
            read_at = max(0, pos - chunk)
            f.seek(read_at)
            data = f.read(pos - read_at)
            nl = data.rfind(b"\n")
            if nl >= 0:
                f.truncate(read_at + nl + 1)
                return
            pos = read_at
        f.truncate(0)


def _load_done_video_names(path: Path, num_classes: int) -> Set[str]:
    """Return the set of video names already present in the submission CSV.

    Strict validation per row: exactly 2 columns, non-empty name, second column
    parseable as int in ``[0, num_classes)``. The header row naturally fails
    (``"predicted_class"`` is not an int) and is silently dropped, as are any
    malformed rows. Those videos will simply be re-predicted on resume.
    """
    if not path.is_file():
        return set()
    done: Set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 2:
                continue
            name, label = row[0], row[1]
            if not name:
                continue
            try:
                lbl = int(label)
            except (TypeError, ValueError):
                continue
            if 0 <= lbl < num_classes:
                done.add(name)
    return done


def _fsync_dir(dir_path: Path) -> None:
    """fsync the directory entry so a freshly created file's existence is durable."""
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# Class discovery + mapping                                                   #
# --------------------------------------------------------------------------- #


def _read_challenge_classes(train_dir: Path) -> Dict[int, Tuple[str, str]]:
    """Walk ``train_dir`` and return ``{class_idx: (raw_folder_name, normalized_name)}``.

    Expects folder names like ``000_Pouring_something_into_something``.
    """
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    out: Dict[int, Tuple[str, str]] = {}
    for p in sorted(train_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)_", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        out[idx] = (p.name, _normalize_name(p.name))
    if not out:
        raise RuntimeError(
            f"No 'NNN_class_name' folders under {train_dir}. Did you point "
            "dataset.train_dir at the right split?"
        )
    return out


def _build_class_mapping(
    challenge_classes: Dict[int, Tuple[str, str]],
    ssv2_id2label: Dict[int, str],
    allow_missing: bool = False,
    missing_fallback_ssv2_idx: int = 0,
) -> List[int]:
    """Build a list ``mapping[challenge_idx] = ssv2_idx`` covering 0..N-1.

    Matching strategy (per challenge class):
      1. Exact match on the normalized name.
      2. **Prefix match**: if the challenge name is a prefix of exactly one
         SSv2 label, use that. Handles cases where the challenge folder name
         was truncated by the dataset packager (a real failure mode here —
         e.g. ``015_Pretending_to_pour_something_out_of_something_but_something_``).
      3. Otherwise: collected into ``missing``.

    If ``allow_missing`` is True, unmatched classes are mapped to
    ``missing_fallback_ssv2_idx`` (default 0) and a warning is printed.
    Otherwise raises a useful diagnostic.
    """
    ssv2_pairs: List[Tuple[int, str]] = [
        (int(idx), _normalize_name(label)) for idx, label in ssv2_id2label.items()
    ]
    ssv2_exact: Dict[str, int] = {norm: idx for idx, norm in ssv2_pairs}

    num_classes = max(challenge_classes.keys()) + 1
    mapping: List[int] = [-1] * num_classes
    missing: List[Tuple[int, str]] = []
    prefix_resolved: List[Tuple[int, str, int, str]] = []  # ch_idx, ch_raw, s_idx, s_raw

    for idx in range(num_classes):
        info = challenge_classes.get(idx)
        if info is None:
            missing.append((idx, "<no folder for this index>"))
            continue
        raw_name, norm_name = info

        # 1) Exact match
        s_idx = ssv2_exact.get(norm_name)
        if s_idx is not None:
            mapping[idx] = s_idx
            continue

        # 2) Prefix match (challenge name is a prefix of an SSv2 label)
        candidates = [
            (i, n) for i, n in ssv2_pairs
            if n.startswith(norm_name)
        ]
        if len(candidates) == 1:
            s_idx, _ = candidates[0]
            mapping[idx] = s_idx
            prefix_resolved.append((idx, raw_name, s_idx, ssv2_id2label[s_idx]))
            continue

        if len(candidates) > 1:
            # Show up to 3 ambiguous candidates in the error
            sample = ", ".join(
                f"[{i}] {ssv2_id2label[i]!r}" for i, _ in candidates[:3]
            )
            extra = f" (+{len(candidates)-3} more)" if len(candidates) > 3 else ""
            missing.append(
                (idx, f"{raw_name} (ambiguous prefix; matches: {sample}{extra})")
            )
        else:
            missing.append((idx, raw_name))

    if prefix_resolved:
        print(f"Resolved {len(prefix_resolved)} class(es) via prefix match "
              f"(challenge name was a prefix of the SSv2 label):", flush=True)
        for ch_idx, ch_raw, s_idx, s_raw in prefix_resolved:
            print(f"  challenge[{ch_idx}] {ch_raw!r}", flush=True)
            print(f"     -> SSv2[{s_idx}] {s_raw!r}", flush=True)

    if missing:
        details = "\n".join(
            f"  challenge[{i}] = {n!r}" for i, n in missing[:15]
        )
        if allow_missing:
            print(f"WARNING: {len(missing)} class(es) unresolved, mapped to "
                  f"SSv2 index {missing_fallback_ssv2_idx} as a fallback:\n"
                  f"{details}", flush=True)
            for ch_idx, _ in missing:
                if mapping[ch_idx] == -1:
                    mapping[ch_idx] = missing_fallback_ssv2_idx
            return mapping

        sample_ssv2 = "\n".join(
            f"  ssv2[{i}] = {l!r}"
            for i, l in list(ssv2_id2label.items())[:10]
        )
        raise RuntimeError(
            f"{len(missing)} challenge class(es) could not be matched to "
            f"SSv2 id2label:\n{details}\n\n"
            f"First few SSv2 labels for comparison:\n{sample_ssv2}\n\n"
            "Fixes:\n"
            "  - If a folder is missing entirely (e.g. challenge[27]), inspect\n"
            "    your train_dir to see what's actually there.\n"
            "  - To proceed despite missing classes (predictions for those\n"
            "    will be wrong but the CSV will be complete), pass\n"
            "    `+zeroshot.allow_missing=true`.\n"
            "  - If names look almost identical but differ in punctuation,\n"
            "    tweak `_normalize_name`."
        )

    return mapping


def _peek_id2label(model_id: str) -> Dict[int, str]:
    """Load just the config (cheap) to grab id2label without paying for weights."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    raw = getattr(cfg, "id2label", None)
    if raw is None:
        raise RuntimeError(f"{model_id} has no id2label in its HF config.")
    return {int(k): str(v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Zero-shot knobs (override at CLI with the `+zeroshot.foo=bar` syntax).
    zs = cfg.get("zeroshot", {}) or {}
    model_id = str(zs.get("model_id", "facebook/vjepa2-vitg-fpc64-384-ssv2"))
    image_size = int(zs.get("image_size", 384))
    allow_missing = bool(zs.get("allow_missing", False))
    missing_fallback = int(zs.get("missing_fallback_ssv2_idx", 0))
    # Crash-safety knobs:
    #   resume=true (default) -> if the output CSV already exists, heal any
    #     partial trailing line, skip already-predicted videos, and append.
    #   resume=false -> delete any existing CSV and start fresh.
    #   fsync=true (default) -> after each batch, call fsync() to force the
    #     OS to write to physical media. Adds a few ms per batch; survives
    #     power loss. Disable only if you're sure the run won't be interrupted.
    resume = bool(zs.get("resume", True))
    use_fsync = bool(zs.get("fsync", True))

    num_frames = int(cfg.dataset.num_frames)
    train_dir = Path(cfg.dataset.train_dir).resolve()
    test_dir = Path(cfg.dataset.test_dir).resolve()
    output_path = Path(cfg.dataset.submission_output).resolve()

    # ---- Step 1: discover challenge class names from disk ------------------ #
    print(f"Discovering challenge classes from: {train_dir}", flush=True)
    challenge_classes = _read_challenge_classes(train_dir)
    expected = max(challenge_classes.keys()) + 1
    gaps = [i for i in range(expected) if i not in challenge_classes]
    print(f"  Found {len(challenge_classes)} classes "
          f"(indices {min(challenge_classes)}..{max(challenge_classes)}).",
          flush=True)
    if gaps:
        print(f"  Missing index(es) in train_dir: {gaps}", flush=True)
    print("  Discovered folders (all):", flush=True)
    for ch_idx in sorted(challenge_classes.keys()):
        raw, _ = challenge_classes[ch_idx]
        print(f"    [{ch_idx:3d}] {raw}", flush=True)

    # ---- Step 2: read SSv2 id2label from the HF config --------------------- #
    print(f"Reading SSv2 labels from HF config: {model_id}", flush=True)
    ssv2_id2label = _peek_id2label(model_id)
    print(f"  Model native classes: {len(ssv2_id2label)}.", flush=True)

    # ---- Step 3: build challenge -> SSv2 index mapping --------------------- #
    mapping = _build_class_mapping(
        challenge_classes,
        ssv2_id2label,
        allow_missing=allow_missing,
        missing_fallback_ssv2_idx=missing_fallback,
    )
    print("Class mapping (challenge_idx -> ssv2_idx | name):", flush=True)
    for ch_idx in sorted(challenge_classes.keys()):
        raw, _ = challenge_classes[ch_idx]
        print(f"  {ch_idx:3d} -> {mapping[ch_idx]:3d} | {raw}", flush=True)

    # ---- Step 4: build the zero-shot model --------------------------------- #
    print(f"Loading model weights: {model_id} "
          f"(num_frames={num_frames}, image_size={image_size})", flush=True)
    model = VJEPA2ZeroShotClassifier(
        class_indices=mapping,
        model_id=model_id,
        num_frames=num_frames,
        image_size=image_size,
    ).to(device).eval()
    print(f"Model on device: {device}", flush=True)

    # ---- Step 5: build the test loader (mirrors create_submission.py) ------ #
    # V-JEPA 2 was trained with ImageNet normalization stats.
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)

    manifest_cfg = cfg.dataset.get("test_manifest")
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        print(f"Reading manifest: {manifest_path}", flush=True)
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_dir, video_names)
    else:
        print(f"No manifest; discovering test videos under: {test_dir}", flush=True)
        video_names, video_dirs = discover_all_test_videos(test_dir)
    print(f"  {len(video_dirs)} test clip folders.", flush=True)

    # ---- Resume support: heal partial CSV + filter already-done videos ---- #
    if not resume and output_path.is_file():
        print(f"  resume=false: deleting existing {output_path}", flush=True)
        output_path.unlink()

    done_set: Set[str] = set()
    if resume:
        _heal_csv(output_path)
        done_set = _load_done_video_names(output_path, int(cfg.num_classes))

    if done_set:
        unknown = done_set - set(video_names)
        if unknown:
            print(f"  WARNING: {len(unknown)} name(s) in existing CSV are not "
                  f"in the current test list — ignored.", flush=True)
        before = len(video_names)
        kept = [
            (n, d) for n, d in zip(video_names, video_dirs) if n not in done_set
        ]
        video_names = [n for n, _ in kept]
        video_dirs = [d for _, d in kept]
        print(f"  Resume: {before - len(video_names)}/{before} clips already "
              f"done; {len(video_names)} remaining.", flush=True)
        if not video_names:
            print(f"Nothing to do: every clip is already in {output_path}.",
                  flush=True)
            return

    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]
    dataset = VideoFrameDataset(
        root_dir=test_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=sample_list,
    )
    batch_size = int(cfg.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # ---- Step 6: streaming inference with per-batch durable writes -------- #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not output_path.is_file() or output_path.stat().st_size == 0

    print(f"Inference: {len(dataset)} clips, batch_size={batch_size}, "
          f"{len(loader)} batches", flush=True)
    print(f"  {'Writing fresh CSV' if need_header else 'Appending to existing CSV'}"
          f": {output_path}", flush=True)

    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    rows_this_run = 0
    cursor = 0

    # Open in append mode. _heal_csv ran earlier, so the file (if it exists)
    # ends cleanly on a newline. We serialise each batch into an in-memory
    # buffer and issue a single write() per batch; on Linux ext4 with
    # O_APPEND, writes < PIPE_BUF (4KB) are atomic, so a crash mid-write
    # either keeps the whole batch or loses it whole. After every batch we
    # flush() + fsync() so completed batches are durable on disk even if the
    # machine loses power immediately afterwards.
    csv_file = output_path.open("a", newline="", encoding="utf-8")
    try:
        if need_header:
            csv_file.write("video_name,predicted_class\n")
            csv_file.flush()
            if use_fsync:
                os.fsync(csv_file.fileno())
                _fsync_dir(output_path.parent)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                video_batch = batch[0].to(device)
                logits = model(video_batch)
                preds = [int(p) for p in logits.argmax(dim=1).cpu().tolist()]
                actual_bs = len(preds)
                batch_names = video_names[cursor:cursor + actual_bs]
                cursor += actual_bs

                buf = io.StringIO()
                bw = csv.writer(buf)
                for name, pred in zip(batch_names, preds):
                    bw.writerow([name, pred])
                csv_file.write(buf.getvalue())
                csv_file.flush()
                if use_fsync:
                    os.fsync(csv_file.fileno())

                rows_this_run += actual_bs
                if batch_idx % log_interval == 0 or batch_idx == n_batches:
                    print(f"  batch {batch_idx}/{n_batches} "
                          f"(+{rows_this_run} rows persisted this run)",
                          flush=True)
    finally:
        csv_file.close()

    if cursor != len(video_names):
        print(f"WARNING: wrote {cursor} rows but expected {len(video_names)} — "
              "CSV is still valid for what was written.", flush=True)

    print(f"Done. CSV at {output_path} (+{rows_this_run} rows this run).",
          flush=True)


if __name__ == "__main__":
    main()
