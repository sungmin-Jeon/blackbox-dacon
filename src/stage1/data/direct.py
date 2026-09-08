"""Direct-capture MP4 dataset for Stage 1 replay classification."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from inference import _clip_ids, _decode_stage1_clip


LABEL_TO_INDEX = {
    "ORIGINAL": 0,
    "RERECORDED": 1,
}


def _usable_mask(frame: pd.DataFrame) -> pd.Series:
    """Interpret the optional CSV usable column without treating 'False' as true."""
    if "usable" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)

    values = frame["usable"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    normalized = values.fillna("").astype(str).str.strip().str.lower()
    accepted = {"1", "true", "t", "yes", "y"}
    return normalized.isin(accepted)


class DirectStage1Dataset(Dataset):
    """Read labeled MP4 files listed in ``stage1_split.csv``.

    The dataset returns the same pair as ``BaiduMoireDataset``:

        clip: ``[3, frames, size, size]`` normalized float tensor
        label: scalar long tensor, ORIGINAL=0 and RERECORDED=1

    ``video_root`` can replace the Drive paths stored in the CSV with a fast
    local copy that contains ``source/`` and ``recaptured/`` directories.
    """

    def __init__(
        self,
        split_csv: str | Path,
        *,
        split: str,
        video_root: str | Path | None = None,
        frames: int = 16,
        size: int = 224,
        cache_dir: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if frames <= 0:
            raise ValueError("frames must be greater than zero")
        if size <= 0:
            raise ValueError("size must be greater than zero")

        self.split_csv = Path(split_csv).expanduser().resolve()
        self.split = split
        self.video_root = (
            Path(video_root).expanduser().resolve()
            if video_root is not None
            else None
        )
        self.frames = frames
        self.size = size
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else None
        )

        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split CSV does not exist: {self.split_csv}")
        if self.video_root is not None and not self.video_root.is_dir():
            raise FileNotFoundError(f"Video root does not exist: {self.video_root}")

        table = pd.read_csv(self.split_csv)
        required = {"source_id", "video_id", "kind", "label", "split", "path"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Split CSV is missing columns: {missing}")

        table = table[(table["split"] == split) & _usable_mask(table)].copy()
        if table.empty:
            raise ValueError(f"No usable {split!r} rows in: {self.split_csv}")

        unknown_labels = sorted(set(table["label"].astype(str)) - set(LABEL_TO_INDEX))
        if unknown_labels:
            raise ValueError(f"Unknown Stage 1 labels: {unknown_labels}")

        table["label_index"] = table["label"].map(LABEL_TO_INDEX).astype(int)
        table = table.sort_values(
            ["label_index", "source_id", "video_id"],
            kind="stable",
        ).reset_index(drop=True)

        self.samples: list[dict] = []
        for row in table.to_dict(orient="records"):
            path = self._resolve_video_path(row)
            if not path.is_file():
                raise FileNotFoundError(f"Video does not exist: {path}")

            self.samples.append(
                {
                    "source_id": str(row["source_id"]),
                    "video_id": str(row["video_id"]),
                    "kind": str(row["kind"]),
                    "label": int(row["label_index"]),
                    "path": path,
                }
            )

        # Used by the balanced sampler in build.py.
        self.labels = [sample["label"] for sample in self.samples]

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_video_path(self, row: dict) -> Path:
        if self.video_root is None:
            return Path(str(row["path"])).expanduser().resolve()

        directory_by_kind = {
            "source": "source",
            "recaptured": "recaptured",
        }
        kind = str(row["kind"])
        if kind not in directory_by_kind:
            raise ValueError(f"Unknown video kind: {kind!r}")

        filename = Path(str(row["path"])).name
        return (self.video_root / directory_by_kind[kind] / filename).resolve()

    def _cache_path(self, sample: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        return (
            self.cache_dir
            / f"frames_{self.frames}_size_{self.size}"
            / sample["kind"]
            / f"{sample['video_id']}.pt"
        )

    def _decode(self, path: Path) -> torch.Tensor:
        frame_ids = _clip_ids(path, self.frames, slot=0, slots=1)
        return _decode_stage1_clip(path, self.size, frame_ids)

    def _load_clip(self, sample: dict) -> torch.Tensor:
        cache_path = self._cache_path(sample)
        source_path: Path = sample["path"]
        source_stat = source_path.stat()

        if cache_path is not None and cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                cached.get("source_size") == source_stat.st_size
                and cached.get("source_mtime_ns") == source_stat.st_mtime_ns
            ):
                return cached["clip"].float()

        clip = self._decode(source_path)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
            torch.save(
                {
                    # Half precision keeps the optional local cache reasonably small.
                    "clip": clip.half(),
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                },
                temporary,
            )
            temporary.replace(cache_path)

        return clip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        clip = self._load_clip(sample)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return clip, label

    def sample_info(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "source_id": sample["source_id"],
            "video_id": sample["video_id"],
            "kind": sample["kind"],
            "label": sample["label"],
            "path": str(sample["path"]),
            "split": self.split,
        }
