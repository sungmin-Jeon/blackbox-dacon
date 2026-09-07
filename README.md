# Blackbox Dacon baseline scripts

The original notebooks remain under `notebooks/` as references. Training code
is separated by Stage, while submission inference stays in one server-facing
file:

- `train_stage1.py`, `train_stage2.py`, `train_stage3.py`: independent Stage trainers.
- `train_baseline.py`: optional convenience runner for any combination of Stages.
- `inference.py`: exposes the three functions called by the evaluation server.
- `build_submit.py`: validates the files and creates `submit.zip`.

```text
src/
├── common/
├── stage1/
│   ├── data/          # Baidu Dataset and DataLoader builders
│   ├── checkpoint.py  # best.pt, last.pt, config, metrics
│   ├── engine.py      # training and validation loops
│   ├── experiment.py  # Baidu-only experiment orchestration
│   ├── metrics.py     # Macro-F1 and class prediction ratios
│   ├── model.py       # MViTv2-S classifier
│   ├── optim.py       # optimizer and scheduler builders
│   └── train.py       # original Stage 1 baseline trainer
├── stage2/
│   ├── model.py
│   └── train.py
└── stage3/
    ├── model.py
    └── train.py
```

## Colab

Mount Google Drive in a small Colab launcher notebook, update this repository,
and run the scripts. Replace the Drive paths below with the actual locations.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd "/content"
!git clone https://github.com/sungmin-Jeon/blackbox-dacon.git
%cd "/content/blackbox-dacon"
!pip install -r requirements.txt
```

Train the Baidu-only Stage 1 experiment. Read data from Colab's local disk and
write checkpoints and metrics to Google Drive:

```python
!python train_stage1.py \
  --data-dir "/content/datasets/baidu_moire" \
  --model-dir "/content/drive/MyDrive/2026_Dacon/sungmin/stage1/baidu_baseline" \
  --epochs 10 \
  --batch-size 2 \
  --frames 16 \
  --size 224 \
  --lr 1e-4
```

The run saves `best.pt` by validation Macro-F1, `last.pt`, `metrics.csv`, and
`config.json`. Use a new `--model-dir` for every experiment.

Evaluate a Stage 1 checkpoint on labeled source and recaptured MP4 files using
the exact preprocessing implemented by submission inference:

```python
!python eval_stage1.py \
  --data-dir "/content/drive/MyDrive/2026_Dacon/data/stage1" \
  --checkpoint "/path/to/stage1/baidu_baseline_v1/best.pt" \
  --output "/path/to/stage1/baidu_baseline_v1/custom_eval.csv"
```

Train the remaining baseline Stages independently:

```python
!python train_stage2.py \
  --data-dir "/content/drive/MyDrive/blackbox-dacon/data/stage2" \
  --model-dir "/content/drive/MyDrive/blackbox-dacon/model/stage2" \
  --epochs 1
```

```python
!python train_stage3.py \
  --data-dir "/content/drive/MyDrive/blackbox-dacon/data/stage3" \
  --model-dir "/content/drive/MyDrive/blackbox-dacon/model/stage3" \
  --epochs 1
```

Or train all baseline Stages:

```python
!python train_baseline.py \
  --data-dir "/content/drive/MyDrive/blackbox-dacon/data" \
  --model-dir "/content/drive/MyDrive/blackbox-dacon/model" \
  --epochs 1
```

Build the submission archive:

```python
!python build_submit.py \
  --inference-file "/content/drive/MyDrive/blackbox-dacon/inference.py" \
  --requirements-file "/content/drive/MyDrive/blackbox-dacon/requirements.txt" \
  --model-dir "/content/drive/MyDrive/blackbox-dacon/model" \
  --output "/content/drive/MyDrive/blackbox-dacon/submit.zip"
```

The builder requires the four baseline checkpoint files and checks that the ZIP
contains `predict_stage1`, `predict_stage2`, and `predict_stage3`.

When Stage checkpoints live in separate experiment folders, assemble them
without manually copying files:

```python
!python prepare_submit.py \
  --stage1-checkpoint "/path/to/stage1/baidu_baseline_v1/best.pt" \
  --stage2-checkpoint "/path/to/stage2/best.pt" \
  --stage2-backbone "/path/to/stage2/resnet18-f37072fd.pth" \
  --stage3-checkpoint "/path/to/stage3/best.pt" \
  --output "/path/to/submissions/baidu_v1/submit_baidu_v1.zip"
```

This creates the ZIP directly and writes a neighboring `.manifest.json` file
recording the exact source path, size, and SHA-256 hash of every model.
