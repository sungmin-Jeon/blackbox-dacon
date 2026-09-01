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
│   ├── model.py
│   └── train.py
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
%cd "/content/drive/MyDrive/blackbox-dacon"
!git pull origin main
!pip install -r requirements.txt
```

Train one Stage independently:

```python
!python train_stage1.py \
  --data-dir "/content/drive/MyDrive/blackbox-dacon/data/stage1" \
  --model-dir "/content/drive/MyDrive/blackbox-dacon/model/stage1" \
  --epochs 1
```

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
