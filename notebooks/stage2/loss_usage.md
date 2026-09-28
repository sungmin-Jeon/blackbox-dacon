# Stage 2 loss 선택 방법

현재처럼 **한 영상의 전체 프레임을 한 번에 학습**하는 노트북용이다.
모델 출력은 padding 없는 `[1, T]` 또는 `[T]` raw logits이다.

## 1. 공통 함수 셀

`src/stage2/losses.py`의 내용을 노트북 공통 셀 하나에 복사해 실행한다.
해당 파일이 코랩의 저장소에도 있다면 저장소 루트에서 다음 import로 대신한다.

```python
from src.stage2.losses import make_temporal_loss
```

## 2. 실험 설정 셀

```python
LOSS_CONFIG = {"name": "gaussian_ce", "sigma_sec": 0.1}
train_loss_fn = make_temporal_loss(**LOSS_CONFIG)

# 기존 evaluate_entry / check_entry_fit에서 사용하는 공통 비교 기준 유지
criterion = torch.nn.CrossEntropyLoss()
print("학습 loss 설정:", train_loss_fn.config)
```

새 실험마다 같은 seed로 모델과 optimizer, scheduler를 **새로 생성**한다.
동일한 데이터 분할, 특징 파일, 모델 크기, 나머지 학습 설정을 유지한다.

## 3. 학습 루프에서 loss 한 줄 교체

```python
# s는 train_samples의 영상 하나
logits = model(s["x"].unsqueeze(0).to(device))
loss = train_loss_fn(logits, s)
loss.backward()
```

기존 optimizer.zero_grad(), gradient clipping, optimizer.step()은 그대로 둔다.
검증 함수의 loss는 기존 hard CE를 유지한다. 최고 모델 선정도 동일하게
±0.3초 정확도를 우선하고 동률이면 hard CE를 비교한다.
서로 다른 학습 loss의 숫자 크기 자체를 비교하지 않는다.

체크포인트에 설정을 추가한다.

```python
# 기존 torch.save({...}, checkpoint_path)의 딕셔너리에 추가
"loss_config": train_loss_fn.config,
```

현재 sample의 `frames`, `target_index`, `target_frame`, `fps`를 그대로 사용한다.
`target_index`는 배열의 0부터 시작하는 위치이고 `frames`는 실제 프레임 번호다.
가변 FPS 영상에는 `frame_times`에 실제 초 단위 시각을 제공한다.
이 값이 있으면 FPS 환산 대신 사용한다.

## 선택 가능한 temporal loss

| 이름 | 주요 설정 | 학습 목표 |
| --- | --- | --- |
| `hard_ce` | 없음 | 정답 한 프레임에 확률 집중 |
| `gaussian_ce` | `sigma_sec=0.1` | 정답 근처에 가우시안 분포로 확률 배분 |
| `triangular_ce` | `radius_sec=0.3` | 정답에서 멀어질수록 선형으로 줄어드는 목표 |
| `window_ce` | `radius_sec=0.3` | 범위 안 프레임 모두에 균등한 목표 |
| `annotation_ce` | sample에 min/max 필요 | 사람이 표시한 범위 안에 균등한 목표 |
| `label_smoothing` | `smoothing=0.1` | 정답 이외의 전체 프레임에도 확률 일부 배분 |
| `window_nll` | `radius_sec=0.3` | 범위 안 확률의 합을 높임 |
| `annotation_nll` | sample에 min/max 필요 | 사람의 허용 구간 안 확률의 합을 높임 |
| `gaussian_kl` | `sigma_sec=0.1` | 같은 목표의 Gaussian CE와 gradient가 같음 |
| `wasserstein1` | 없음 | 확률로 가중한 절대 시간 오차 최소화 |
| `cdf_l2` | 없음 | 누적분포 차이의 제곱을 시간 간격으로 적분 |
| `bce` | `radius_sec=0.0`, `pos_weight=1.0` 또는 `"balanced"` | 각 프레임을 독립적으로 양성/음성 분류 |
| `focal_ce` | `gamma=2.0` | 정답 확률이 낮은 영상에 더 큰 비중 |
| `focal_bce` | `gamma=2.0`, `alpha=None`, `radius_sec=0.0` | 어려운 프레임에 더 큰 비중 |
| `heatmap_mse` | `sigma_sec=0.1` | sigmoid 출력과 최대값 1인 가우시안 비교 |
| `ranking` | `margin=1.0`, `radius_sec=0.0` | 정답 점수가 범위 밖 점수보다 높도록 학습 |
| `ordinal_bce` | 없음 | 각 시점까지 사건이 발생했는지 누적확률로 학습 |

`annotation_*`를 쓰려면 sample 생성 단계에서 원래 라벨의 `entry_min`,
`entry_max`를 `target_min_frame`, `target_max_frame`으로 넣는다.
구간이 없으면 오류가 발생한다. 결측 정답을 임의로 0으로 채우지 않는다.

이 목록은 실험 후보이며 모두 시도해야 한다는 뜻은 아니다.
먼저 **hard CE 기준점 → Gaussian CE**를 같은 조건에서 비교한다.
Gaussian KL은 같은 gradient이므로 별도의 개선 실험으로 셀 필요가 없다.
`window_nll`은 구간 확률 합을 높이는 대리 목표이고, 최종 argmax의
±0.3초 정확도를 직접 계산한 loss는 아니다. 모델·라벨 문제를 자동으로 해결하지 않는다.

## 분류와 회귀는 별도

```python
from src.stage2.losses import make_classification_loss, make_regression_loss

side_loss_fn = make_classification_loss("ce")
evasion_loss_fn = make_classification_loss("ce")
# raw logits [B, 2], target [B] 정수 0/1
loss_side = side_loss_fn(side_logits, side_targets)

# 시간 자체를 출력하는 모델을 별도로 만들 때만 사용
time_loss_fn = make_regression_loss("huber", delta=0.3)
loss_time = time_loss_fn(predicted_seconds, target_seconds)
```

분류: `ce`, `bce`, `focal`, `soft_macro_f1`.
CE는 `weight`, `label_smoothing`을, BCE는 `pos_weight`를 받을 수 있다.
BCE는 같은 shape의 실수 0/1 정답이 필요하다.
`soft_macro_f1`은 공식 Macro-F1과 다른 대리 목표이며 영상 하나씩 학습할 때는 권하지 않는다.
결측 라벨은 해당 과업의 loss 계산에서 제외한다.

회귀: `l1`, `mse`, `huber`, `smooth_l1`.
프레임별 logits와 정답 index를 그대로 회귀 loss에 넣으면 안 된다.
예측 시각을 미분 가능하게 출력해야 하며, argmax 결과에는 역전파되지 않는다.
