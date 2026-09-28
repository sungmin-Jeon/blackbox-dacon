# Stage 2 Direct Spatial V1

이 실험은 최종 Stage 2를 두 체크포인트로 구성하기 위한 두 번째 분기다.

- 기존 `collision_model.pt`: MM-AU로 학습한 `collision_frame`
- 새 `direct_model.pt`: 직접 라벨링 자료로 학습한 `entry_frame`, `entry_side`, `evasion_space`

Direct 모델은 전체 화면을 하나의 512차원 벡터로 평균 내지 않는다. 고정된 ImageNet
ResNet18의 공간 특징맵을 저장하고, 학습 가능한 공간 attention과 BiGRU를 학습한다.
기본값은 해상도가 더 높은 `layer3` 특징 `[T, 256, H/16, W/16]`이다.

## 1. 코랩에서 코드 갱신

```bash
%cd /content/blackbox-dacon
!git pull
```

필요하면 비디오 ZIP을 먼저 `/content/stage2` 아래에 해제한다. 다음 경로는 실제
환경에 맞게 한 번만 확인한다.

```python
from pathlib import Path

LABELS = Path(
    "/content/drive/MyDrive/2026_Dacon/data/stage2/annotations/"
    "stage2_labels_unique_keep_split_seed42_v1.csv"
)
VIDEO_ROOT = Path(
    "/content/stage2/stage2_labeled_unique_keep_v1/videos"
)
BACKBONE = Path(
    "/content/drive/MyDrive/2026_Dacon/sungmin/stage2/"
    "resnet18-f37072fd.pth"
)
FEATURE_DIR = Path("/content/stage2_direct_layer3_v1")
OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/2026_Dacon/data/stage2/experiments/direct_spatial"
)

for path in (LABELS, VIDEO_ROOT, BACKBONE):
    print(path, path.exists())
```

## 2. 공간 특징 추출

Drive에 수 GB의 작은 파일을 반복해서 읽는 것보다 `/content`에 특징을 만든 뒤 바로
학습하는 편이 빠르다. 기존 파일은 자동으로 건너뛰므로 중단 후 같은 명령을 다시
실행할 수 있다.

```bash
!python extract_stage2_spatial.py \
  --labels "/content/drive/MyDrive/2026_Dacon/data/stage2/annotations/stage2_labels_unique_keep_split_seed42_v1.csv" \
  --video-root "/content/stage2/stage2_labeled_unique_keep_v1/videos" \
  --backbone "/content/drive/MyDrive/2026_Dacon/sungmin/stage2/resnet18-f37072fd.pth" \
  --output-dir "/content/stage2_direct_layer3_v1" \
  --layer layer3 \
  --short-edge 256 \
  --batch-size 64
```

`layer3` 캐시는 영상 172개 기준으로 수 GB가 될 수 있다. 로컬 저장 공간이 부족하면
`--layer layer4`와 다른 출력 폴더를 사용한다. 학습 스크립트는 채널 수를 자동으로
읽는다. 서로 다른 layer의 캐시를 같은 폴더에 섞지 않는다.

## 3. 세 과업 공동 학습

```bash
!python train_stage2_direct.py \
  --labels "/content/drive/MyDrive/2026_Dacon/data/stage2/annotations/stage2_labels_unique_keep_split_seed42_v1.csv" \
  --feature-dir "/content/stage2_direct_layer3_v1" \
  --output-root "/content/drive/MyDrive/2026_Dacon/data/stage2/experiments/direct_spatial" \
  --name "direct_layer3_attention_bigru_v1" \
  --entry-loss gaussian_ce \
  --sigma-sec 0.1 \
  --hidden-size 128 \
  --num-layers 1 \
  --epochs 50 \
  --early-patience 7
```

학습 목표는 다음과 같다.

```text
1.0 × entry Gaussian CE
+ 0.5 × entry-side CrossEntropy
+ 0.5 × evasion-space CrossEntropy
```

각 영상에서 값이 비어 있는 과업은 그 loss만 제외한다. 학습 중 evasion은 라벨링한
충돌 위치에 최대 ±4프레임 jitter를 주어 사용한다. `evasion_space`는 있지만
`collision_frame`이 비어 있는 영상은 전체 영상 문맥으로 학습한다. 검증 evasion도
충돌 라벨이 있으면 해당 위치를, 없으면 전체 영상 문맥을 사용한다. 실제 제출에서는
MM-AU Collision 모델이 예측한 위치를 넣어야 하며 성능이 더 낮아질 수 있다.

최고 모델은 다음 기준으로 저장한다.

```text
0.35 × entry Accuracy@0.3초
+ 0.15 × side Macro-F1
+ 0.15 × evasion Macro-F1
```

출력 폴더에는 아래 파일이 생긴다.

```text
best_model.pt
config.json
history.csv
labels_split.csv
val_predictions.csv
```

`inference.py`는 체크포인트의 `feature_config`를 읽어 학습과 추론의 resize, ResNet
layer, 정규화를 동일하게 적용한다. 제출 ZIP을 만들 때 `prepare_submit.py`의
`--stage2-direct-checkpoint`에 이 실험의 `best_model.pt`를 전달한다.
