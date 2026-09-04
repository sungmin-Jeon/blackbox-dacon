"""Baidu moire dataset loader for Stage 1 replay classification."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.video import crop_tensor


# 클래스 번호는 제출 추론 코드와 동일하게 유지한다.
CLASS_TO_LABEL = {
    "gt_rgb": 0,       # ORIGINAL
    "moire_rgb": 1,    # RERECORDED
}

LABEL_TO_CLASS = {
    0: "ORIGINAL",
    1: "RERECORDED",
}

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

MEAN = torch.tensor(
    [0.45, 0.45, 0.45],
    dtype=torch.float32,
)[:, None, None, None]

STD = torch.tensor(
    [0.225, 0.225, 0.225],
    dtype=torch.float32,
)[:, None, None, None]


def _parse_frame_name(path: Path) -> tuple[str, int]:
    """
    v001_01.png을 다음과 같이 분리한다.

    sequence_id: v001
    frame_number: 1
    """
    try:
        sequence_id, frame_number = path.stem.rsplit("_", 1)
        return sequence_id, int(frame_number)
    except ValueError as error:
        raise ValueError(
            f"Baidu 파일명 형식이 올바르지 않습니다: {path.name}"
        ) from error


def _uniform_indices(
    total: int,
    frames: int,
) -> np.ndarray:
    """전체 프레임에 걸쳐 지정한 개수의 인덱스를 균일하게 선택한다."""
    if total <= 0:
        raise ValueError("total은 1 이상이어야 합니다.")

    if frames <= 0:
        raise ValueError("frames는 1 이상이어야 합니다.")

    return np.linspace(
        0,
        total - 1,
        frames,
    ).round().astype(np.int64)


class BaiduMoireDataset(Dataset):
    """
    Baidu gt_rgb/moire_rgb 이미지 시퀀스를 읽는 Dataset.

    예상 폴더 구조:

        split_dir/
        ├── gt_rgb/
        │   ├── v001_01.png
        │   ├── v001_02.png
        │   └── ...
        └── moire_rgb/
            ├── v001_01.png
            ├── v001_02.png
            └── ...

    반환:

        clip: [3, frames, size, size]
        label: 0 또는 1
    """

    def __init__(
        self,
        split_dir: str | Path,
        frames: int = 16,
        size: int = 224,
        expected_source_frames: int | None = 60,
    ) -> None:
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.frames = frames
        self.size = size
        self.expected_source_frames = expected_source_frames

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"데이터 폴더가 없습니다: {self.split_dir}"
            )

        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(
                f"학습 샘플을 찾지 못했습니다: {self.split_dir}"
            )

    def _build_samples(self) -> list[dict]:
        samples = []

        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = self.split_dir / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"클래스 폴더가 없습니다: {class_dir}"
                )

            grouped_paths: dict[str, list[tuple[int, Path]]] = defaultdict(list)

            for path in class_dir.iterdir():
                if not path.is_file():
                    continue

                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                sequence_id, frame_number = _parse_frame_name(path)

                grouped_paths[sequence_id].append(
                    (frame_number, path)
                )

            for sequence_id, numbered_paths in grouped_paths.items():
                # v001_01, v001_02, ... 순서로 정렬
                numbered_paths.sort(key=lambda item: item[0])

                frame_numbers = [
                    frame_number
                    for frame_number, _ in numbered_paths
                ]

                image_paths = [
                    path
                    for _, path in numbered_paths
                ]

                if self.expected_source_frames is not None:
                    if len(image_paths) != self.expected_source_frames:
                        raise ValueError(
                            f"{class_dir / sequence_id}: "
                            f"예상 프레임={self.expected_source_frames}, "
                            f"실제 프레임={len(image_paths)}"
                        )

                if len(image_paths) < self.frames:
                    raise ValueError(
                        f"{class_dir / sequence_id}: "
                        f"{self.frames}장을 선택해야 하지만 "
                        f"{len(image_paths)}장만 존재합니다."
                    )

                if len(frame_numbers) != len(set(frame_numbers)):
                    raise ValueError(
                        f"중복된 프레임 번호가 있습니다: "
                        f"{class_dir / sequence_id}"
                    )

                samples.append({
                    "sequence_id": sequence_id,
                    "class_name": class_name,
                    "label": label,
                    "image_paths": image_paths,
                })

        # 실행할 때마다 동일한 순서를 유지한다.
        samples.sort(
            key=lambda sample: (
                sample["label"],
                sample["sequence_id"],
            )
        )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_paths = sample["image_paths"]

        selected_indices = _uniform_indices(
            total=len(image_paths),
            frames=self.frames,
        )

        frame_tensors = []

        for frame_index in selected_indices:
            image_path = image_paths[int(frame_index)]

            bgr = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if bgr is None:
                raise ValueError(
                    f"이미지를 읽을 수 없습니다: {image_path}"
                )

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB,
            )

            frame_tensor = crop_tensor(
                rgb,
                size=self.size,
            )

            frame_tensors.append(frame_tensor)

        # [3, frames, size, size]
        clip = torch.stack(
            frame_tensors,
            dim=1,
        )

        # Stage 1 baseline과 동일한 정규화
        clip = (clip - MEAN) / STD

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return clip, label

    def sample_info(self, index: int) -> dict:
        """검사와 디버깅에 사용할 샘플 정보를 반환한다."""
        sample = self.samples[index]

        return {
            "sequence_id": sample["sequence_id"],
            "class_name": sample["class_name"],
            "label": sample["label"],
            "total_frames": len(sample["image_paths"]),
            "first_frame": str(sample["image_paths"][0]),
            "last_frame": str(sample["image_paths"][-1]),
        }