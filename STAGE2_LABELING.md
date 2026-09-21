# Stage 2 Colab labeling pilot

The first-pass labeler records one-based AIHub frame numbers. Human uncertainty
(`min`, `best`, `max`) is stored separately from Dacon's `+/-0.3 second`
evaluation tolerance.

## Colab usage

Mount Drive and unzip the three archives into `/content/stage2_pilot` as in the
project notes. Clone or update this repository, then run:

```python
%cd /content/blackbox-dacon

from src.stage2.labeler_widget import Stage2Labeler

labeler = Stage2Labeler(
    video_root="/content/stage2_pilot/videos",
    csv_path=(
        "/content/drive/MyDrive/2026_Dacon/data/stage2/annotations/"
        "stage2_labels_v1.csv"
    ),
    initial_video_id="bb_1_000104_vehicle_40_156",
)
labeler.show()
```

Use `전체 흐름 30장` for coarse navigation, then move by five or one frame.
For each event, set `min`, `best`, and `max`. Choose direction, evasion space,
confidence, and the lane basis before saving. `SKIP` may be saved without event
labels. Saving updates the video's row instead of adding a duplicate and writes
the selected entry/collision frames under the adjacent `evidence` directory.

The CSV is the durable output. Files extracted under `/content` may be discarded
when the Colab runtime ends.
