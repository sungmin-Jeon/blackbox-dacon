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

When initializing from an existing Stage 1 checkpoint, choose how much of the
MViT to update with `--fine-tune-scope`: `full` (default), `head`, or
`last-block` (the final MViT block, final norm, and classification head).

## Direct temporal-view experiments (Colab)

Temporal sampling is versioned in the checkpoint and shared by Direct
training, validation, `eval_stage1.py`, and submission inference. Old
checkpoints have no temporal configuration and retain the original single
uniform view exactly.

| Training option | Behavior |
| --- | --- |
| `--temporal-mode uniform` | Select 16 frames uniformly across the full video (legacy default). |
| `--temporal-mode multi-burst` | Split the video into four segments and select four consecutive frames per segment. Training jitters each burst start; validation is centered and deterministic. |
| `--temporal-mode mixed` | Randomly use uniform or jittered multi-burst for each training sample. |

`--temporal-eval-mode auto` follows the training mode: uniform evaluates one
uniform view, multi-burst evaluates one centered multi-burst view, and mixed
evaluates both views and averages their probabilities. Override it with
`uniform`, `multi-burst`, or `both`. The `both` setting runs MViT twice per
video. `--frames` must be divisible by `--temporal-bursts` whenever a
multi-burst view is used. The recommended first ablation keeps the center crop
and all other settings fixed:

```python
%cd /content/blackbox-dacon
!python train_stage1.py \
  --dataset direct \
  --split-csv "/content/drive/MyDrive/2026_Dacon/sungmin/stage1/data/stage1_split.csv" \
  --video-root "/content/direct_stage1" \
  --model-dir "/content/drive/MyDrive/2026_Dacon/sungmin/stage1/direct_v0_multiburst_v1" \
  --cache-dir "/content/stage1_temporal_cache" \
  --temporal-mode multi-burst --temporal-bursts 4 \
  --spatial-mode center \
  --frames 16 --size 224 --epochs 30 --batch-size 2 \
  --lr 1e-5 --seed 42 --early-stopping-patience 5
```

After the single-view ablation, train a model that accepts both views with
`--temporal-mode mixed --temporal-eval-mode both`. Randomized training clips
are intentionally not cached. Deterministic validation clips can be cached,
and their cache keys include the full temporal configuration. Offline
evaluation CSVs include each view's probability as well as their mean.

## Direct v0 spatial-selection experiments (Colab)

The model and uniform 16-frame temporal sampling stay unchanged. One crop
location is used for the whole clip; the output remains `[3, 16, 224, 224]`.
The following options apply to `--dataset direct`:

| `--spatial-mode` | Behavior |
| --- | --- |
| `center` (default) | Original short-side resize, then center crop. Old checkpoints keep this behavior. |
| `native-center` | Center crop before resizing: a control for native crop scale. |
| `random` | Uniformly select one native crop from the candidate grid per training draw. Evaluation uses a fixed filename/seed-based selection. |
| `fft` | Select one native crop with the largest clip-averaged high-frequency power. |

`--crop-size 224` is the native-pixel crop edge, independent of model input
`--size 224`. `--crop-grid 5` puts up to 5 positions on each axis, spanning the
frame including its edges. Random and FFT use the **same candidate grid and
crop size**. Frames smaller than the crop are rejected rather than silently
changing the scale. FFT ties use the first candidate in row-major order.

The FFT score is an explicit **adaptation**, not an exact implementation of
[DGOAS (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Lee_Domain-Generalized_Object_Anti-Spoofing_Bridging_Gaps_and_Patch_Selection_for_Robust_WACV_2025_paper.pdf):
convert a patch to luminance in `[0,1]`, subtract its mean, apply a 2D Hann
window, calculate orthonormal FFT power, and average bins whose radial
frequency is at least `--fft-min-freq 0.25` cycles/pixel. Sum over decoded
sampled frames (equivalent ranking to the mean), then select the maximum.
This is a high-frequency-energy score, **not a replay probability**. It may
also select textured original scenes. It uses one crop, not the paper's five
64-pixel patches. RGB crops, not spectra, are passed to MViT.

After transferring the updated repository to Colab, restart the runtime if
the old functions have already been imported. Mount Drive again as needed.
Adjust these paths to your existing **v0 split CSV** (with a `split` column),
not the unsplit `stage1_manifest.csv`:

```python
%cd /content/blackbox-dacon
!python train_stage1.py \
  --dataset direct \
  --split-csv "/content/drive/MyDrive/2026_Dacon/sungmin/stage1/data/stage1_split.csv" \
  --video-root "/content/direct_stage1" \
  --model-dir "/content/drive/MyDrive/2026_Dacon/sungmin/stage1/direct_v0_fft_v1" \
  --cache-dir "/content/stage1_spatial_cache" \
  --spatial-mode fft --crop-size 224 --crop-grid 5 \
  --frames 16 --size 224 --epochs 30 --batch-size 2 \
  --lr 1e-5 --seed 42 --early-stopping-patience 5
```

For a random-selection comparison change `--spatial-mode random` and use a
new `--model-dir`. Keep the source split, initialization and training budget
fixed. Both commands initialize from torchvision pretrained weights by default.
FFT decoding makes two passes, so the first cache-building pass is slower.
Final deterministic crops are cached separately by the full spatial settings
and preprocessing version. **Random training ignores the clip cache** so that
each draw can select a new position; decoding can therefore be slower every
epoch. Validation random crops are deterministic and cacheable.

Settings are stored in `config.json` and both checkpoints. `eval_stage1.py`
and submission `predict_stage1` read them automatically. Use the updated
`inference.py` when packaging a new spatial model. Existing three-argument
calls to `_decode_stage1_clip` still mean **legacy center crop**; when using
the custom R021–R056 notebook, update its imports and decode call explicitly:

```python
from inference import _clip_ids, _decode_stage1_clip, _stage1_spatial_config

# After loading checkpoint; also works with old checkpoints.
spatial = _stage1_spatial_config(checkpoint)
print("Evaluation spatial settings:", spatial)

# Inside the existing video loop:
frame_ids = _clip_ids(path, frames=frames, slot=0, slots=1)
clip = _decode_stage1_clip(path, size, frame_ids, spatial=spatial)
```

Local CPU checks: `python -m unittest discover -s tests -v`.

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
