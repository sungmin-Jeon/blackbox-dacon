"""Colab widget for annotating Stage 2 accident videos.

The widget keeps human annotation uncertainty (min/best/max) separate from
the competition's +/-0.3 second scoring tolerance.  It uses one-based frame
numbers for the AIHub clips while OpenCV decoding remains zero-based.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "ID",
    "video_path",
    "fps",
    "total_frames",
    "entry_frame",
    "entry_min",
    "entry_max",
    "entry_time",
    "entry_min_time",
    "entry_max_time",
    "collision_frame",
    "collision_min",
    "collision_max",
    "collision_time",
    "collision_min_time",
    "collision_max_time",
    "entry_side",
    "evasion_space",
    "entry_confidence",
    "collision_confidence",
    "lane_basis",
    "status",
    "notes",
    "updated_at",
]


def frame_number_to_time(frame_number: int | None, fps: float, base: int = 1) -> float | None:
    """Convert an external frame number to seconds."""
    if frame_number is None:
        return None
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_number < base:
        raise ValueError(f"frame_number must be >= {base}")
    return (frame_number - base) / fps


def tolerance_radius_frames(fps: float, seconds: float = 0.3) -> int:
    """Largest integer frame offset that stays within the time tolerance."""
    if fps <= 0 or seconds < 0:
        raise ValueError("fps must be positive and seconds must be non-negative")
    return math.floor(fps * seconds + 1e-9)


class Stage2Labeler:
    """Interactive one-video-at-a-time Stage 2 labeler for Colab/Jupyter."""

    def __init__(
        self,
        video_root: str | Path,
        csv_path: str | Path,
        evidence_dir: str | Path | None = None,
        *,
        frame_number_base: int = 1,
        neighborhood_radius: int = 2,
        initial_video_id: str | None = None,
        include_collision: bool = True,
    ) -> None:
        try:
            import cv2
            import ipywidgets as widgets
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
        except ImportError as exc:  # pragma: no cover - exercised in Colab
            raise RuntimeError(
                "Stage2Labeler requires opencv-python, matplotlib, pandas, "
                "numpy, ipywidgets and IPython. These are preinstalled in Colab."
            ) from exc

        self.cv2 = cv2
        self.widgets = widgets
        self.plt = plt
        self.np = np
        self.pd = pd
        self._display = display

        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)
        self.evidence_dir = Path(evidence_dir) if evidence_dir else self.csv_path.parent / "evidence"
        self.frame_number_base = frame_number_base
        self.neighborhood_radius = max(1, int(neighborhood_radius))
        self.include_collision = bool(include_collision)

        self.videos = sorted(
            path
            for path in self.video_root.rglob("*")
            if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if not self.videos:
            raise FileNotFoundError(f"No videos found under {self.video_root}")

        self._video_by_id = {path.stem: path for path in self.videos}
        if len(self._video_by_id) != len(self.videos):
            raise ValueError("Video stems must be unique under video_root")

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._saved = self._load_saved_rows()
        self._marks: dict[str, int | None] = {}
        self.current_video: Path | None = None
        self.fps = 0.0
        self.total_frames = 0

        self._build_widgets(initial_video_id)
        self._load_video(self.video_select.value)

    def _load_saved_rows(self) -> dict[str, dict[str, Any]]:
        if not self.csv_path.is_file():
            return {}
        frame = self.pd.read_csv(self.csv_path, keep_default_na=False)
        return {str(row["ID"]): row.to_dict() for _, row in frame.iterrows()}

    def _build_widgets(self, initial_video_id: str | None) -> None:
        w = self.widgets
        video_ids = list(self._video_by_id)
        selected = initial_video_id if initial_video_id in self._video_by_id else video_ids[0]

        self.video_select = w.Dropdown(options=video_ids, value=selected, description="영상")
        self.overview_button = w.Button(description="전체 흐름 30장", button_style="info")
        self.prev5 = w.Button(description="-5")
        self.prev1 = w.Button(description="-1")
        self.next1 = w.Button(description="+1")
        self.next5 = w.Button(description="+5")
        self.slider = w.IntSlider(
            min=self.frame_number_base,
            max=self.frame_number_base,
            step=1,
            value=self.frame_number_base,
            description="프레임",
            continuous_update=False,
            layout=w.Layout(width="70%"),
        )

        self.mark_labels = {
            name: w.HTML(value=f"<b>{title}:</b> -")
            for name, title in (
                ("entry_min", "진입 min"),
                ("entry_frame", "진입 best"),
                ("entry_max", "진입 max"),
                ("collision_min", "충돌 min"),
                ("collision_frame", "충돌 best"),
                ("collision_max", "충돌 max"),
            )
        }
        self.mark_buttons = {}
        for name, title in (
            ("entry_min", "진입 min 지정"),
            ("entry_frame", "진입 best 지정"),
            ("entry_max", "진입 max 지정"),
            ("collision_min", "충돌 min 지정"),
            ("collision_frame", "충돌 best 지정"),
            ("collision_max", "충돌 max 지정"),
        ):
            button = w.Button(description=title)
            button.on_click(lambda _, key=name: self._set_mark(key))
            self.mark_buttons[name] = button

        self.entry_side = w.ToggleButtons(
            options=[("미선택", ""), ("LEFT", "LEFT"), ("RIGHT", "RIGHT")],
            description="진입 방향",
        )
        self.evasion_space = w.ToggleButtons(
            options=[("미선택", ""), ("없음 0", 0), ("있음 1", 1)],
            description="회피 공간",
        )
        confidence_options = [("미선택", ""), ("HIGH", "HIGH"), ("MEDIUM", "MEDIUM"), ("LOW", "LOW")]
        self.entry_confidence = w.ToggleButtons(options=confidence_options, description="진입 확신")
        self.collision_confidence = w.ToggleButtons(options=confidence_options, description="충돌 확신")
        self.lane_basis = w.ToggleButtons(
            options=[("미선택", ""), ("VISIBLE", "VISIBLE"), ("EXTENDED", "EXTENDED"), ("UNCLEAR", "UNCLEAR")],
            description="차선 기준",
        )
        self.status = w.ToggleButtons(
            options=[("KEEP", "KEEP"), ("REVIEW", "REVIEW"), ("SKIP", "SKIP")],
            value="KEEP",
            description="상태",
        )
        self.notes = w.Textarea(description="메모", layout=w.Layout(width="90%", height="70px"))
        self.save_button = w.Button(description="CSV 저장", button_style="success")
        self.next_video_button = w.Button(description="다음 영상")
        self.clear_button = w.Button(description="현재 라벨 초기화", button_style="warning")
        self.message = w.HTML()
        self.frame_output = w.Output()
        self.overview_output = w.Output()

        self.video_select.observe(lambda change: self._load_video(change["new"]), names="value")
        self.slider.observe(lambda _: self._render_neighborhood(), names="value")
        self.prev5.on_click(lambda _: self._step(-5))
        self.prev1.on_click(lambda _: self._step(-1))
        self.next1.on_click(lambda _: self._step(1))
        self.next5.on_click(lambda _: self._step(5))
        self.overview_button.on_click(lambda _: self._render_overview())
        self.save_button.on_click(lambda _: self._save())
        self.next_video_button.on_click(lambda _: self._next_video())
        self.clear_button.on_click(lambda _: self._clear_form())

        entry_box = w.VBox([
            w.HBox([self.mark_buttons["entry_min"], self.mark_buttons["entry_frame"], self.mark_buttons["entry_max"]]),
            w.HBox([self.mark_labels["entry_min"], self.mark_labels["entry_frame"], self.mark_labels["entry_max"]]),
        ])
        collision_box = w.VBox([
            w.HBox([self.mark_buttons["collision_min"], self.mark_buttons["collision_frame"], self.mark_buttons["collision_max"]]),
            w.HBox([self.mark_labels["collision_min"], self.mark_labels["collision_frame"], self.mark_labels["collision_max"]]),
        ])
        annotation_widgets = [
            w.HBox([self.video_select, self.overview_button]),
            w.HBox([self.prev5, self.prev1, self.slider, self.next1, self.next5]),
            self.frame_output,
            entry_box,
        ]
        if self.include_collision:
            annotation_widgets.append(collision_box)
        annotation_widgets.extend([
            self.entry_side,
            self.evasion_space,
            self.entry_confidence,
        ])
        if self.include_collision:
            annotation_widgets.append(self.collision_confidence)
        annotation_widgets.extend([
            self.lane_basis,
            self.status,
            self.notes,
            w.HBox([self.save_button, self.next_video_button, self.clear_button]),
            self.message,
            self.overview_output,
        ])
        self.ui = w.VBox(annotation_widgets)

    def show(self) -> None:
        self._display(self.ui)

    def _read_video_info(self, path: Path) -> tuple[float, int]:
        cap = self.cv2.VideoCapture(str(path))
        fps = float(cap.get(self.cv2.CAP_PROP_FPS))
        total = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or total <= 0:
            raise ValueError(f"Could not read video metadata: {path}")
        return fps, total

    def _load_video(self, video_id: str) -> None:
        self.current_video = self._video_by_id[video_id]
        self.fps, self.total_frames = self._read_video_info(self.current_video)
        self.slider.min = self.frame_number_base
        self.slider.max = self.frame_number_base + self.total_frames - 1
        self._clear_form(render=False)
        self._restore_saved(video_id)
        self.slider.value = self.frame_number_base
        self.message.value = (
            f"<b>{video_id}</b> · FPS {self.fps:.3f} · {self.total_frames} frames · "
            f"±0.3초 = ±{tolerance_radius_frames(self.fps)} frames"
        )
        self._render_neighborhood()

    def _restore_saved(self, video_id: str) -> None:
        row = self._saved.get(video_id)
        if not row:
            return
        for key in self._marks:
            value = row.get(key, "")
            self._marks[key] = int(float(value)) if value not in ("", None) else None
        self.entry_side.value = row.get("entry_side", "")
        evasion = row.get("evasion_space", "")
        self.evasion_space.value = int(float(evasion)) if evasion not in ("", None) else ""
        self.entry_confidence.value = row.get("entry_confidence", "")
        self.collision_confidence.value = row.get("collision_confidence", "")
        self.lane_basis.value = row.get("lane_basis", "")
        self.status.value = row.get("status", "KEEP") or "KEEP"
        self.notes.value = row.get("notes", "")
        self._refresh_mark_labels()

    def _clear_form(self, *, render: bool = True) -> None:
        self._marks = {
            "entry_min": None,
            "entry_frame": None,
            "entry_max": None,
            "collision_min": None,
            "collision_frame": None,
            "collision_max": None,
        }
        self.entry_side.value = ""
        self.evasion_space.value = ""
        self.entry_confidence.value = ""
        self.collision_confidence.value = ""
        self.lane_basis.value = ""
        self.status.value = "KEEP"
        self.notes.value = ""
        self._refresh_mark_labels()
        if render:
            self.message.value = "현재 폼을 초기화했습니다. CSV의 기존 행은 저장 전까지 유지됩니다."

    def _set_mark(self, key: str) -> None:
        self._marks[key] = int(self.slider.value)
        self._refresh_mark_labels()

    def _refresh_mark_labels(self) -> None:
        titles = {
            "entry_min": "진입 min",
            "entry_frame": "진입 best",
            "entry_max": "진입 max",
            "collision_min": "충돌 min",
            "collision_frame": "충돌 best",
            "collision_max": "충돌 max",
        }
        for key, label in self.mark_labels.items():
            value = self._marks.get(key)
            label.value = f"<b>{titles[key]}:</b> {value if value is not None else '-'}"

    def _step(self, amount: int) -> None:
        self.slider.value = min(self.slider.max, max(self.slider.min, self.slider.value + amount))

    def _read_frame(self, frame_number: int):
        assert self.current_video is not None
        index = frame_number - self.frame_number_base
        cap = self.cv2.VideoCapture(str(self.current_video))
        cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = cap.read()
        cap.release()
        if not success:
            return None
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def _render_neighborhood(self) -> None:
        center = int(self.slider.value)
        start = max(self.slider.min, center - self.neighborhood_radius)
        end = min(self.slider.max, center + self.neighborhood_radius)
        items = [(number, self._read_frame(number)) for number in range(start, end + 1)]
        items = [(number, frame) for number, frame in items if frame is not None]
        with self.frame_output:
            self.frame_output.clear_output(wait=True)
            if not items:
                print("프레임을 읽지 못했습니다.")
                return
            fig, axes = self.plt.subplots(1, len(items), figsize=(4.8 * len(items), 4.5), dpi=120)
            axes = self.np.atleast_1d(axes).ravel()
            for ax, (number, frame) in zip(axes, items):
                ax.imshow(frame)
                color = "red" if number == center else "black"
                marker = "★ " if number == center else ""
                time_value = frame_number_to_time(number, self.fps, self.frame_number_base)
                ax.set_title(f"{marker}frame {number:03d}\n{time_value:.3f}s", color=color)
                ax.axis("off")
            self.plt.tight_layout()
            self.plt.show()

    def _render_overview(self) -> None:
        numbers = self.np.linspace(self.slider.min, self.slider.max, 30).round().astype(int)
        with self.overview_output:
            self.overview_output.clear_output(wait=True)
            for start in range(0, len(numbers), 6):
                page = numbers[start : start + 6]
                fig, axes = self.plt.subplots(2, 3, figsize=(21, 10), dpi=120)
                for ax in axes.flat:
                    ax.axis("off")
                for ax, number in zip(axes.flat, page):
                    frame = self._read_frame(int(number))
                    if frame is None:
                        continue
                    time_value = frame_number_to_time(int(number), self.fps, self.frame_number_base)
                    ax.imshow(frame)
                    ax.set_title(f"frame {int(number):03d} / {time_value:.2f}s")
                    ax.axis("off")
                self.plt.tight_layout()
                self.plt.show()

    @staticmethod
    def _ordered(values: list[int | None]) -> bool:
        present = [value for value in values if value is not None]
        return present == sorted(present)

    def _validation_error(self) -> str | None:
        if self.status.value == "SKIP":
            return None
        entry = [self._marks["entry_min"], self._marks["entry_frame"], self._marks["entry_max"]]
        if any(value is None for value in entry):
            return "진입의 min/best/max를 모두 지정하세요. 판단 불가 영상은 SKIP으로 저장하세요."
        if not self._ordered(entry):
            return "진입 프레임은 min ≤ best ≤ max여야 합니다."
        if self.include_collision:
            collision = [
                self._marks["collision_min"],
                self._marks["collision_frame"],
                self._marks["collision_max"],
            ]
            if any(value is None for value in collision):
                return "충돌의 min/best/max를 모두 지정하세요. 판단 불가 영상은 SKIP으로 저장하세요."
            if not self._ordered(collision):
                return "충돌 프레임은 min ≤ best ≤ max여야 합니다."
            if self._marks["entry_frame"] > self._marks["collision_frame"]:  # type: ignore[operator]
                return "진입 대표 프레임이 충돌 대표 프레임보다 늦습니다. 영상을 재확인하세요."
        if self.entry_side.value not in {"LEFT", "RIGHT"}:
            return "진입 방향을 선택하세요."
        if self.evasion_space.value not in {0, 1}:
            return "회피 공간 0/1을 선택하세요."
        if not self.entry_confidence.value:
            return "진입 확신도를 선택하세요."
        if self.include_collision and not self.collision_confidence.value:
            return "충돌 확신도를 선택하세요."
        if not self.lane_basis.value:
            return "차선 기준을 선택하세요."
        return None

    def _time(self, key: str) -> float | str:
        value = self._marks[key]
        converted = frame_number_to_time(value, self.fps, self.frame_number_base)
        return "" if converted is None else round(converted, 6)

    def _row(self) -> dict[str, Any]:
        assert self.current_video is not None
        skipped = self.status.value == "SKIP"
        row: dict[str, Any] = {
            "ID": self.current_video.stem,
            "video_path": str(self.current_video),
            "fps": round(self.fps, 6),
            "total_frames": self.total_frames,
            "entry_side": "" if skipped else self.entry_side.value,
            "evasion_space": "" if skipped else self.evasion_space.value,
            "entry_confidence": "" if skipped else self.entry_confidence.value,
            "collision_confidence": (
                "" if skipped or not self.include_collision else self.collision_confidence.value
            ),
            "lane_basis": "" if skipped else self.lane_basis.value,
            "status": self.status.value,
            "notes": self.notes.value.strip(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for key in self._marks:
            excluded = key.startswith("collision_") and not self.include_collision
            row[key] = "" if skipped or excluded or self._marks[key] is None else self._marks[key]
        for prefix in ("entry", "collision"):
            excluded = prefix == "collision" and not self.include_collision
            row[f"{prefix}_time"] = "" if skipped or excluded else self._time(f"{prefix}_frame")
            row[f"{prefix}_min_time"] = "" if skipped or excluded else self._time(f"{prefix}_min")
            row[f"{prefix}_max_time"] = "" if skipped or excluded else self._time(f"{prefix}_max")
        return {column: row.get(column, "") for column in CSV_COLUMNS}

    def _write_csv(self, row: dict[str, Any]) -> None:
        self._saved[row["ID"]] = row
        temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for video_id in sorted(self._saved):
                writer.writerow({column: self._saved[video_id].get(column, "") for column in CSV_COLUMNS})
        temporary.replace(self.csv_path)

    def _save_evidence(self, event: str, frame_number: int | None) -> None:
        if frame_number is None or self.current_video is None:
            return
        frame = self._read_frame(frame_number)
        if frame is None:
            return
        target = self.evidence_dir / f"{self.current_video.stem}_{event}_{frame_number:06d}.jpg"
        bgr = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
        self.cv2.imwrite(str(target), bgr)

    def _save(self) -> None:
        error = self._validation_error()
        if error:
            self.message.value = f"<span style='color:#b00020'><b>저장 실패:</b> {error}</span>"
            return
        row = self._row()
        self._write_csv(row)
        if self.status.value != "SKIP":
            self._save_evidence("entry", self._marks["entry_frame"])
            if self.include_collision:
                self._save_evidence("collision", self._marks["collision_frame"])
        self.message.value = (
            f"<span style='color:green'><b>저장 완료:</b> {self.csv_path}<br>"
            f"평가 허용 반경은 이 영상에서 ±{tolerance_radius_frames(self.fps)}프레임이며, "
            "CSV의 min/max는 사람의 라벨 불확실성입니다.</span>"
        )

    def _next_video(self) -> None:
        options = list(self._video_by_id)
        index = options.index(self.video_select.value)
        self.video_select.value = options[(index + 1) % len(options)]
