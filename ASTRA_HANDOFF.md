# Astra Project Handoff

> 작성 기준일: 2026-09-14 (Asia/Seoul)  
> 근거 범위: 이 프로젝트와 관련된 대화 기록 및 현재 저장소에서 확인한 내용  
> 표기 규칙: **[수행]** 실제 실행·생성·제출·측정이 확인됨 / **[계획]** 하기로 했으나 완료 근거 없음 / **[논의]** 후보로만 검토 / **[미확인]** 대화나 저장소에서 확인할 수 없음

새로운 연구 아이디어는 이 문서에 추가하지 않았다. 동일 항목의 값이 변한 경우 최신 상태를 본문에 쓰고, 이전 값은 변경 이력 또는 실험별 기록에 남겼다.

## 1. Project Overview

- **대회/프로젝트 목적:** 블랙박스 영상을 입력으로 받아 3개 Stage의 예측을 수행하는 DACON 프로젝트이다. 대회 ID는 저장소 원격 주소 기준 `236753`이지만, 대회의 정식 명칭은 대화에서 **미확인**이다.
- **Stage 1:** 영상이 원본(`ORIGINAL`)인지 화면을 다시 촬영한 영상(`RERECORDED`)인지 이진 분류.
- **Stage 1 label 해석:** 코드와 평가 로그에서 `ORIGINAL`은 class 0/negative, `RERECORDED`는 class 1/positive(`P(R)`)로 다뤘다.
- **Stage 2:** 사고의 주요 시점과 상황 예측. 대화·코드에서 확인된 출력은 충돌 시점, 진입 시점, 회피 공간, 진입 방향이다.
- **Stage 3:** 10 Hz 단위 차량 행동 예측. 가감속 4개 클래스와 조향 3개 클래스를 예측한다.
- **현재 집중 중인 Stage:** **Stage 1**. Stage 2·3은 초기 베이스라인을 제출 ZIP에 유지했으며, 이후 대화와 실험은 거의 모두 Stage 1에 집중했다.
- **최종 목표:** 세 Stage를 모두 포함하는 유효한 제출물을 만들고, 특히 Stage 1에서 숨은 평가 데이터에 일반화되는 원본/재촬영 분류기를 얻는 것. 목표 순위나 목표 점수는 **미확인**이다.

## 2. Competition Constraints

| 항목 | 확인 내용 | 상태 |
|---|---|---|
| 평가 metric | Stage 1은 `ORIGINAL`, `RERECORDED` 두 클래스의 F1을 동일 가중 평균한 **Macro-F1**. Stage 2·3의 정확한 metric 정의는 대화에서 미확인 | Stage 1 확인 / 나머지 미확인 |
| 제출 방식 | CSV만 제출하는 방식이 아니라, `inference.py`, `requirements.txt`, `model/stage1/best.pt`, `model/stage2/best.pt`, `model/stage2/resnet18-f37072fd.pth`, `model/stage3/best.pt`를 담은 ZIP. `predict_stage1/2/3` 함수가 각 Stage 결과 DataFrame을 반환 | 확인 |
| inference 시간 제한 | 공식 제한값은 **미확인**. 실제 제출 관측 시간은 약 `13분 56초~16분 41초` | 제한 미확인 / 실측 있음 |
| 패키지 설치 제한 | 별도 설치 허용 범위는 **미확인**. 제출용 `requirements.txt`에는 고정 버전이 들어감 | 미확인 |
| 모델/ZIP 용량 제한 | **미확인** | 미확인 |
| 평가 GPU/CPU/RAM | 공식 사양은 **미확인**. 현재 제출 코드는 CUDA가 없으면 오류를 내도록 구현되어 있음 | 사양 미확인 |
| 인터넷 사용 | 평가 환경에서 인터넷 사용 불가 | 확인 |
| 외부 데이터 규정 | Baidu RawVDemoiré의 라이선스가 CC BY-NC-SA 4.0이라는 논의가 있었고, 상금 대회 사용 가능 여부와 대회 외부 데이터 허용 규칙을 별도로 확인해야 한다고 정리됨. 실제 확인 완료 근거는 없음 | 미확인 |
| 입력/출력 단위 | Stage 1은 비디오 파일별 1개 `answer`를 출력 | 확인 |
| 구현상 제약 | 현재 Stage 1 추론은 영상 전체에서 균등 추출한 16프레임, 224×224 한 clip을 사용. 제출 환경에서는 모든 checkpoint가 ZIP 내부에 있어야 함 | 확인 |
| Stage 1 decision threshold | 현재 추론은 `P(RERECORDED) >= 0.5`이면 `RERECORDED`, 아니면 `ORIGINAL` | 구현 확인 |

## 3. Current Data Assets

### 3.1 대회 원본·프로젝트 원본 데이터

| 이름 | 출처·규모 | positive/negative 의미 | 확보/사용 | 사용 목적·경로 |
|---|---|---|---|---|
| 대회 제공 Stage 1 예시/원본 | DACON 제공. 정확한 파일 수는 미확인 | `ORIGINAL`=원본, `RERECORDED`=재촬영 | [수행] 초기 베이스라인·smoke 제출에 사용 | 초기 제출 구조 및 파이프라인 검증. 상세 경로 미확인 |
| AIHub 블랙박스 원본 | AIHub. 인벤토리에서 원본/소스 영상 **4,724개** 확인 | AIHub 원본은 `ORIGINAL`; 여기서 물리 재촬영 또는 synthetic replay를 만들면 `RERECORDED` | [수행] 확보·사용 | 직접 재촬영의 원본 R001~R056 선정, FAS-Aug paired 데이터 생성. 예시 경로 `/content/data/stage1/_source_extracted/batch_0...`; 인벤토리 `stage1_mainfast.csv` |

대회 숨은 test 분포, 실제 `ORIGINAL:RERECORDED` 비율, 촬영 장치·환경 구성은 모두 **미확인**이다.

### 3.2 직접 재촬영 데이터

| 버전/이름 | 규모 | 의미 | 확보 여부 | 실제 사용 여부 | 목적·경로 |
|---|---:|---|---|---|---|
| Direct v0 | 원본 source 20개(R001~R020), 원본 20 + 재촬영 60 = **80 videos** | 원본 20=`ORIGINAL`; source당 3개 재촬영=`RERECORDED` | [수행] 확보 | [수행] train/val, zero-shot 평가, fine-tuning, 제출에 사용 | Drive `/content/drive/MyDrive/2026_Dacon/data/stage1/{source,recaptured}`; 로컬 `/content/direct_stage1`; split `/content/drive/MyDrive/2026_Dacon/sungmin/stage1/data/stage1_split.csv`; cache `/content/stage1_clip_cache` |
| Direct v1 확장 | 원본 source 56개(R001~R056), 원본 56 + 재촬영 132 = **188 videos** | `ORIGINAL` 56, `RERECORDED` 132 | [수행] 확보 | [수행] train/val 및 여러 fine-tuning 실험에 사용 | split `/content/drive/MyDrive/2026_Dacon/sungmin/stage1/data_v1/stage1_split_v1_seed42.csv`; cache `/content/stage1_clip_cache_v1` |
| LG/IP13P E03 추가 촬영분 | R021~R056에 대응하는 iPhone MOV **36개** | 모두 `RERECORDED` 예정 | [수행] 파일 존재 및 dry-run mapping 확인 | 실제 rename/apply, manifest 반영, 학습 사용은 **미확인** | 원본 위치 `/content/drive/MyDrive/recaptured`; 계획 파일 `rename_plan_LG_IP13P_E03.csv` |
| LG/IP13P 표준화 촬영 세트 | R001~R056 × E01/E03/E05를 동일 장비 조합으로 채우는 계획 | 원본=`ORIGINAL`, 각 재촬영=`RERECORDED` | [계획] 일부 E03(R021~R056)만 존재 확인 | 미사용 | E01 R001~R056 촬영, E03 R001~R020 보충, E05 R001~R056 촬영 계획 |

Direct v1의 최신 확인 분포는 train **44 source / 149 videos**(`O=44`, `R=105`), val **12 source / 39 videos**(`O=12`, `R=27`)이며 source 중복은 없다.

### 3.3 외부 데이터

| 이름 | 출처·규모 | positive/negative 의미 | 확보 여부 | 실제 사용 여부 | 사용 목적·경로 |
|---|---|---|---|---|---|
| Baidu RawVDemoiré RGB | Baidu. train `gt_rgb` 15,000 frames + `moire_rgb` 15,000 frames, 1:1 매칭. 파일명 `v001_01.png`~`v250_60.png`; train 250 paired sequence×60 frames → 분류 clip 500개. val은 학습 로그상 50 pair → 100 clip | `gt_rgb`=`ORIGINAL`, `moire_rgb`=`RERECORDED` proxy | [수행] 다운로드·압축 해제·pair audit 완료 | [수행] pretraining, 단독 제출, Direct fine-tuning 초기화에 사용 | Drive `/content/drive/MyDrive/2026_Dacon/data/외부데이터/baidu_moire/{train,val}`; Colab `/content/datasets/baidu_moire` |
| Baidu `gt_raw` / `moire_raw` | RawVDemoiré의 NPZ raw 데이터 | `gt_raw`=원본 센서 raw, `moire_raw`=모아레 raw | [논의] 존재만 검토 | 미확보·미사용 | RGB만으로 Stage 1을 먼저 진행하기로 함 |
| DLC-2021 | 문서 재촬영 영상 **1,424개**, 전체 약 **106 GB**로 논의됨 | 문서 원본/재촬영 계열 | [논의] | 미확보·미사용 | 후보 외부 데이터였으나 용량·도메인 차이 때문에 우선순위 하락 |
| VDMoiré | 약 **290 video pairs × 60 frames**, 720p로 논의됨 | 원본/모아레 pair | [논의] | 미확보·미사용 | 비디오 모아레 후보 |
| GOSet | **2,849 videos**, 신청 필요로 논의됨 | 원본/재촬영 관련 | [논의] | 미확보·미사용 | 비디오 재촬영 후보 |
| 기타 공개 후보 | mToFNet 12,529 images, DoFNet 2,752 pairs, NTU-ROSE 2k+2.7k, ICL-COMMSP 약 8 GB, BJTU IIS 636+706, MTurk rebroadcast 14.5k+14.5k, FHDMi 12k pairs, UHDM 5k, TMM22 948, TIP2018 135k patches, OULU-NPU 4,950, SiW 4,478, ROSE-Youtu 3,350, Replay-Attack 1,300, Replay-Mobile 1,190, CASIA-FASD 600, MSU-MFSD 280, WMCA 1,941/replay 348, CelebA-Spoof 625,537, CeFA 5,346 및 AMNet·UniDemoiré | 각 데이터별 정의가 달라 대회 O/R로의 정확한 대응은 미확인 | [논의] | 모두 미확보·미사용 | 외부 데이터 후보 조사에만 포함 |

### 3.4 Synthetic / generated data

| 이름 | 구성·규모 | 의미 | 확보/사용 | 목적·경로 |
|---|---|---|---|---|
| AIHub + FAS-Aug paired split | AIHub 원본과 FAS-Aug replay 출력의 paired 데이터. train **836**, val **210**, 총 **1,046 samples** | AIHub 원본=`ORIGINAL`, FAS-Aug replay=`RERECORDED` | [수행] 생성 산출물 확보·학습 사용 | pretraining 및 Direct fine-tuning 초기화. split `/content/aihub_fas_paired_split.csv`; cache `/content/aihub_fas_clip_cache`; run `/content/drive/MyDrive/2026_Dacon/sungmin/stage1/aihub_fas_paired_v1` |
| 대회 베이스라인 synthetic rerecord 예시 | 대회 예시 또는 초기 코드에서 만든 synthetic/dependent 샘플 | 물리 재촬영이 아니라 제출 구조용 `RERECORDED` 예시 | [수행] 초기 smoke에만 사용 | 제출 파이프라인 검증. 정확한 생성 조건·규모 미확인 |

FAS-Aug의 정확한 변환 파라미터와 각 효과 조합은 대화에서 완전히 확인되지 않았다. 또한 Direct 원본과 FAS-Aug 원본이 모두 AIHub에서 왔기 때문에 source overlap 가능성이 지적됐지만, 실제 중복 제거 여부는 **미확인**이다.

## 4. Real Recapture Data Collection

### 4.1 실제 확보된 촬영 구성

- **v0 원본 수:** R001~R020, 총 20개.
- **v0 재촬영 수:** 원본당 3개, 총 60개.
- **v1 원본 수:** R001~R056, 총 56개.
- **v1 재촬영 수:** 총 132개. 원본별 재촬영 개수가 완전히 동일하지는 않다.
- **장비 코드:**
  - `MCBP2_IP17_E05`: 20개. HEVC/hvc1. R001~R010은 주로 1920×1080 약 30 fps, R011~R020은 주로 3840×2160 약 60 fps. 샘플 길이 약 7.4~22.4초.
  - `MCBP_IP13_E01`: 20개. H264/avc1, 1920×1080, 약 30 fps, 약 7~8초.
  - `MCBP_IP13_E03`: 20개. H264/avc1, 1920×1080, 약 30 fps, 약 7~8초.
  - `LG_IP13P_E03`: R021~R056에 대응하는 MOV 36개가 확인됨.
- 코드의 정확한 제품명은 대화에서 확정되지 않았다. `MCBP`, `MCBP2`, `IP13`, `IP17`, `LG`, `IP13P`를 특정 MacBook/iPhone/LG 모델명으로 확장하는 것은 **불확실**하므로 그대로 장비 코드로 보존한다.
- **환경 코드:** `E01`, `E03`, `E05`. 각 코드의 정확한 조명·거리·각도·실내외 정의는 **미확인**이다.

### 4.2 파일명 규칙과 mapping

- 원본 source ID는 `R001`, `R002`, … 형식으로 정규화했다.
- 재촬영 파일명 규칙은 `R001_<display>_<camera>_<environment>.<ext>`이다. 예: `R001_MCBP_IP13_E01.mov`.
- 일부 과거 파일의 `R01` 표기도 코드에서 `R001`로 정규화했다.
- LG/IP13P E03의 원본 MOV는 `IMG_7497...` 같은 숫자 파일명이었다.
- 처음에는 `creation_time` 기반 mapping도 검토했으나, 사용자가 **숫자 파일명 순서가 촬영 순서와 일치하고 번호 중간 누락은 문제없다**고 확인했다. 최신 mapping 원칙은 숫자 파일명 오름차순 ↔ R021~R056 순서이다.
- 실제 파일 변경 전 `rename_plan_LG_IP13P_E03.csv`를 만드는 dry-run을 수행했다. rename/move를 실제 적용했는지는 **미확인**이다.

### 4.3 train/val/test 사용

- Direct v0: source 단위 split으로 train 16 source/64 videos, val 4 source/16 videos(`O=4`, `R=12`). 실제 학습·검증·제출 모델 생성에 사용했다.
- Direct v1: seed 42 source-group split으로 train 44 source/149 videos, val 12 source/39 videos. 실제 학습·검증에 사용했다.
- 동일 source의 원본과 재촬영은 같은 split에 둬서 source leakage를 막았다.
- 현재 val은 **source holdout**일 뿐, 촬영 환경·장치 조합을 완전히 분리한 environment holdout은 아니다.
- R001~R056 전체를 train으로 쓰고 고정 3 epoch 학습하는 최종 방식이 논의됐으나, 실행 근거는 없다.
- 숨은 대회 test와 직접 촬영 환경의 대응 관계는 미확인이다.

### 4.4 메타데이터 편향

- Direct v1에서 원본 56개 중 55개가 MPEG4/mp4v 15 fps 계열이고, 재촬영은 모두 H264/HEVC 30/60 fps 계열이었다.
- R050 원본은 예외적으로 H264 30 fps였으며 val에 포함됐다.
- 따라서 source split이더라도 codec/fps shortcut을 학습할 가능성이 있다는 문제가 대화에서 명시적으로 제기됐다.

## 5. Preprocessing / Augmentation

| 항목 | 내용 | 상태 |
|---|---|---|
| 균등 16프레임 추출 | 영상 전체에서 16개 프레임을 균등 선택 | **구현 완료·실험 완료** |
| resize + center crop | 짧은 변을 224에 맞춘 뒤 224×224 center crop | **구현 완료·실험 완료** |
| normalization | mean `(0.45,0.45,0.45)`, std `(0.225,0.225,0.225)` | **구현 완료·실험 완료** |
| one-clip inference | 현재 Stage 1은 `slots=1` 한 clip 사용 | **구현 완료·실험 완료** |
| 과거 3-slot 연속 clip | 초기 baseline 추론에는 영상을 3구간으로 나눠 각 구간 연속 16프레임을 쓰는 구성이 있었음 | **과거 구현**, 이후 현재 uniform one-clip로 변경 |
| class-balanced sampling | Direct 학습에서 O/R row 비율 불균형을 완화하는 sampler | **구현 완료·실험 완료** |
| FAS-Aug replay | AIHub 원본으로 synthetic `RERECORDED`를 생성한 산출물을 paired 학습에 사용 | **실험 완료**. 생성 세부 파라미터는 미확인 |
| horizontal flip 0.5 | clip 전체에 동일한 flip을 적용하는 약한 augmentation | **논의만 함** |
| random crop/resize | scale 0.90~1.00 수준의 약한 공간 변형 | **논의만 함** |
| brightness/contrast | ±10% 수준, clip-consistent color 변형 | **논의만 함** |
| temporal jitter | 시작 위치 변화 또는 프레임 선택 변화 | **논의만 함**. 현재 cache는 고정 clip이라 미구현 |
| common compression/resolution augmentation | O/R 양쪽 모두에 재압축, 해상도 변경, gamma, blur/noise를 적용해 codec shortcut을 줄이는 방식 | **논의만 함** |
| display-camera simulation | 재촬영 class에 화면 픽셀 구조, perspective, reflection/glare, 밝기 banding, color shift 등을 합성 | **논의만 함**. FAS-Aug 외 별도 구현 근거 없음 |
| moiré simulation | R 전용 약한 모아레 합성 | **논의만 함** |
| blur/noise/compression | 공통 또는 R 전용으로 조절하는 후보 | **논의만 함** |
| perspective | 화면 촬영 기하 변형 후보 | **논의만 함** |
| rolling shutter/banding | 이동 band 또는 밝기 줄무늬 후보 | **논의만 함** |
| temporal flicker / frame-rate mismatch | clip 차원 깜빡임, 주기성, display-camera FPS 차이 모사 | **논의만 함** |
| global + local clip | global uniform 16프레임과 local consecutive 16프레임을 함께 사용 | **논의만 함** |
| multi-crop/full-frame/local high-res | 공간적 모아레가 국소적일 수 있어 여러 crop을 쓰는 구성 | **논의만 함** |
| multi-clip inference | 영상당 3~5 clip 확률 평균 | **논의만 함**. 현재 코드는 1 clip |

현재 저장소의 Direct/Baidu 주력 학습에는 **랜덤 augmentation이 없다**. `resize → center crop → normalize`는 고정 전처리이며, Direct cache 또한 매 epoch 같은 프레임을 내놓는다.

## 6. Models Tried or Discussed

| 모델 | frame/video | pretrained | loss | temporal modeling | 실제 사용 | 결과 |
|---|---|---|---|---|---|---|
| `mvit_v2_s` | video-based | torchvision Kinetics pretrained 또는 이전 checkpoint | CrossEntropy | 16-frame MViT spatiotemporal attention | **실제 사용**: Baidu, Direct, FAS-Aug, Stage 1 제출 | 현재 Stage 1의 유일한 주력 모델. 최고 대회 Stage 1 점수 0.8308436777 |
| ResNet18 + 2-layer BiGRU | video-based(frame encoder+sequence model) | ResNet18 pretrained weight 사용 | 정확한 Stage 2 loss는 이 대화에서 미확인 | ResNet18 frame feature 후 양방향 GRU, hidden 192×2 | **실제 사용**: Stage 2 baseline·제출 | 모든 기록된 제출에서 Stage 2 점수 0.159801139. 충돌 시점 외 일부 출력은 초기값/미학습 상태라는 설명이 있었음 |
| MViTv2-S dual head | video-based | pretrained 여부는 현재 Stage 3 baseline 코드상 사용되지만 대화에서 세부 확인 불충분 | 두 task loss의 구체식 미확인 | MViT video feature | **실제 사용**: Stage 3 baseline·제출 | 모든 기록된 제출에서 Stage 3 점수 0.1324991733 |
| head-only fine-tuning | MViTv2-S | pretrained/checkpoint | CE | MViT 동일 | 코드 옵션 **구현**, 실제 실험은 미확인 | 결과 없음 |
| last-block fine-tuning | MViTv2-S | Baidu checkpoint | CE | MViT 동일 | **실제 사용** 1회 | Direct v1 val Macro-F1 0.4040으로 약함 |
| full fine-tuning | MViTv2-S | torchvision/Baidu/FAS checkpoint | CE | MViT 동일 | **반복 사용** | Direct v0 torchvision init이 최고 대회 점수 |
| LoRA | MViTv2-S attention/conv에 adapter | pretrained 기반 예정 | 미정 | backbone 일부만 저랭크 적응 | **논의만 함** | 결과 없음 |
| FFT/frequency 또는 chromaticity branch | 2-branch 후보 | 미정 | 미정 | spatial/frequency 결합 | **논의만 함** | 결과 없음 |
| global/local two-branch | MViT 기반 후보 | 미정 | 미정 | global uniform + local consecutive clip 결합 | **논의만 함** | 결과 없음 |

실제 Stage 1 loss는 모든 확인된 학습에서 CrossEntropy였다. focal loss와 label smoothing은 후보/누락 항목으로만 언급됐고 사용 근거가 없다.

## 7. Training Pipeline

### 7.1 실제 수행한 순서

1. **초기 3-Stage baseline smoke 제출**
   - Stage 1은 대회 예시 기반 baseline, Stage 2·3은 제공/구현된 baseline checkpoint로 ZIP 제출 파이프라인을 검증했다.
2. **Baidu 단독 Stage 1 학습**
   - torchvision Kinetics pretrained MViTv2-S → Baidu RGB O/R proxy 학습.
   - 먼저 notebook prototype과 1-epoch smoke를 돌린 뒤 full run을 수행했다.
3. **Baidu 모델의 Direct zero-shot 평가**
   - 물리 재촬영 80개에 그대로 평가해 심한 domain gap을 확인했다.
4. **Direct v0 단독 fine-tuning**
   - torchvision pretrained MViTv2-S를 Direct v0에서 full fine-tuning.
   - 이 checkpoint가 현재 최고 대회 점수를 냈다.
5. **Baidu → Direct v0 staged fine-tuning**
   - Baidu best checkpoint를 초기값으로 Direct v0 전체 layer fine-tuning.
6. **AIHub + FAS-Aug synthetic pretraining**
   - paired synthetic split에서 MViTv2-S를 학습.
7. **FAS-Aug → Direct v0 staged fine-tuning**
   - FAS-Aug best checkpoint에서 Direct v0 full fine-tuning.
8. **Direct v1 확장 데이터 학습**
   - 56 source/188 video 데이터로 torchvision init full fine-tuning.
9. **Baidu → Direct v1 fine-tuning scope 비교**
   - last-block과 full fine-tuning을 각각 실행. last-block은 약했고 full이 훨씬 좋았다.

### 7.2 공통 실제 학습 조건

- optimizer: AdamW
- scheduler: cosine decay
- weight decay: 0.01
- loss: CrossEntropy
- AMP 사용
- checkpoint 선택 metric: validation Macro-F1
- early stopping 사용
- Direct는 class-balanced sampler 사용
- random augmentation 없음
- Direct/Baidu clip은 고정 uniform 16 frames, 224×224

### 7.3 계획·논의만 된 staged/curriculum

- Direct source-balanced sampling: epoch마다 source별 original 1개 + 재촬영 1개를 뽑는 1:1 구성.
- Baidu와 Direct를 domain-balanced batch로 섞은 공동 학습.
- 동일한 validation split을 고정한 뒤 `no augmentation → weak augmentation → full vs last-block → common compression/resolution → 제한적 recapture synthesis` 순서로 한 변수씩 비교.
- R001~R056 전체 train, 고정 3 epoch 최종 학습.
- L2-SP 또는 LoRA를 통한 보수적 fine-tuning.
- 위 항목은 모두 실행 완료 근거가 없다.

## 8. Domain Adaptation

| 방법 | 검토 이유 | 적용 여부 | 결과 |
|---|---|---|---|
| 단순 staged fine-tuning | 외부 proxy domain(Baidu/FAS-Aug)의 표현을 실제 Direct domain에 맞추기 위해 | **실제 적용** | Baidu→Direct v0 val 0.7949, leaderboard 0.7929746204. FAS→Direct v0 val 0.9086, leaderboard 0.7252208598. Baidu→Direct v1 full val 0.9023, leaderboard 0.7758072612 |
| DANN | Baidu/FAS/Direct 간 domain gap 완화 후보 | **논의만 함** | 미실험. 카메라·화면 단서 자체가 label 신호일 수 있어 이를 지울 위험이 있다고 논의됨 |
| CORAL | feature covariance 정렬의 비교적 단순한 domain adaptation 후보 | **논의만 함** | 미실험 |
| MMD | source/target feature distribution 정렬 후보 | **논의만 함** | 미실험 |
| domain-balanced mixed training | Baidu의 규모와 Direct의 타깃 근접성을 함께 사용 | **계획/논의** | 미실험 |
| environment holdout validation | 동일 촬영장비/환경 shortcut을 검증하기 위해 | **계획** | 미구현·미실험 |
| codec/fps 공통 augmentation | O/R 간 메타데이터 shortcut 완화 | **계획/논의** | 미실험 |

대화에서 DANN/CORAL/MMD의 실제 코드, checkpoint, score는 나오지 않았다.

## 9. Experiments and Results

`대회 점수` 열은 DACON 제출 대시보드에서 본 Stage 1 점수이다. 이것이 public leaderboard인지 별도 test score인지 대화에서 명시적으로 구분되지 않았으므로 **public/private 구분 미확인**으로 둔다. private leaderboard 점수는 전부 미확인이다.

| ID / 설명 | 날짜·순서 | 데이터 | 모델 | augmentation | 주요 hyperparameter | validation metric / score | 대회 Stage 1 score | private | inference time | 해석 | 판정 |
|---|---|---|---|---|---|---|---:|---|---|---|---|
| E00 초기 3-Stage baseline smoke | 2026-09-01 | 대회 예시/초기 baseline | Stage1 baseline, Stage2 ResNet18+BiGRU, Stage3 MViT dual-head | 상세 미확인 | 상세 미확인 | 미측정 | **0.4045598914** | 미확인 | 13m56s | 제출 ZIP과 세 Stage 호출이 동작함을 확인했으나 Stage1은 약함 | 기술적 성공 / 성능 약함 |
| E01 Baidu notebook prototype | 초기 Baidu 실험 | Baidu train 250 pair, val 50 pair로 표시 | MViTv2-S로 추정 | 없음으로 추정, 세부 미확인 | fixed 16/224; 나머지 미확인 | Macro-F1 **0.9200**, train loss 0.5483, val loss 0.2886 | 미제출 | 미확인 | 초기 notebook 수치. 이후 modular smoke와 설정 대응관계는 완전히 확정되지 않음 | 성공(프로토타입) |
| E02 Baidu modular smoke 1 epoch | 2026-09-04 이후 | train 500 clips, val 100 clips | pretrained MViTv2-S, full | 없음 | CE, AdamW, lr 1e-4, batch 2, frames 16, size 224, AMP | Macro-F1 **0.8368**, O 0.8596, R 0.8140; train loss 0.6257, val loss 0.3636, pred O/R 0.640/0.360 | 미제출 | 미확인 | 코드·loader 검증 | 성공(smoke) |
| E03 Baidu full `baidu_baseline_v1` | 2026-09-04~07 | Baidu 500/100 clips | torchvision pretrained MViTv2-S, full | 없음 | CE, AdamW+cosine, lr 1e-4, batch 2, wd 0.01, max 30, patience 7, fixed 16/224, AMP | 최고 확인 Macro-F1 **0.9900** at epoch 6, O 0.9899, R 0.9901. 중간 e1 0.8368, e2 0.7895, e3 0.9599, e5 0.9800 | **0.5093434343** | 미확인 | 15m04s | Baidu 내부 val은 거의 완벽하지만 대회 일반화는 낮음 | 내부 성공 / 일반화 실패 |
| E04 Baidu best → Direct v0 zero-shot | 2026-09-07 | Direct v0 80 videos | E03 checkpoint | 없음 | threshold 0.5, current fixed 16/224 | Accuracy 0.3500, Macro-F1 **0.3463**, O 0.3953, R 0.2973; confusion `[[17,3],[49,11]]`; mean P(R) source 0.161626, recapture 0.192101; ROC-AUC 0.50875 | 미제출 | 미확인 | 해당 없음 | 실제 재촬영으로 전이가 거의 안 됨 | 실패 |
| E04a threshold scan on E04 | E04 직후 | Direct v0 80 | E03 checkpoint | 없음 | threshold 0.01 | 최고 탐색 Macro-F1 **0.44792168**, confusion `[[12,8],[35,25]]` | 미제출 | 미확인 | 해당 없음 | threshold 보정만으로 해결 불가 | 실패 |
| E04b 환경별 zero-shot 분석 | E04 직후 | 환경별 original+recapture pair. 별도 pair 분석은 환경당 40 videos(원본 20+재촬영 20)로 보고됨 | E03 checkpoint | 없음 | threshold 0.5로 추정 | E05: mean P(R) 0.228408, replay recall 0.25; E01: 0.102230/0.05; E03: 0.245665/0.25. 별도 pair 분석은 E05 AUC 0.5525/Macro-F1 0.5055, E01 0.4388/0.3452, E03 0.5350/0.5055 | 미제출 | 미확인 | 해당 없음 | 장비·환경별 차이는 있으나 어느 환경도 안정적이지 않음. 두 분석의 sample 구성은 일부 불확실 | 실패/진단 |
| E05 Direct v0 smoke 1 epoch | 2026-09-08 | train 64, val 16 | torchvision pretrained MViTv2-S, full | 없음 | lr 1e-5, batch 2, fixed 16/224 | Macro-F1 **0.4170**, O 0.3077, R 0.5263; train loss 0.6735, val loss 0.7013; pred O/R 0.562/0.438 | 미제출 | 미확인 | 해당 없음 | 작은 실제 데이터 파이프라인 동작 확인 | 성공(smoke), 성능 약함 |
| E06 Direct v0 full `direct_raw_v1` | 2026-09-08 | train 64, val 16; source split | torchvision pretrained MViTv2-S, full 34M | 없음 | CE, AdamW+cosine, lr 1e-5, batch 2, wd 0.01, max 30, patience 5, AMP, balanced sampler, fixed 16/224 | best e8 Macro-F1 **0.6135**, accuracy 0.6875, O 0.4444, R 0.7826, val loss 0.5513, pred O/R 0.312/0.688; early stop e13 | **0.8308436777** | 미확인 | 16m10s | val은 높지 않지만 숨은 평가 분포에 가장 잘 맞은 현재 최고 제출 | **성공 / 현재 best** |
| E07 Baidu → Direct v0 full | 2026-09-08 | Direct v0 64/16, init E03 | MViTv2-S full | 없음 | 새 AdamW, lr 1e-5, batch 2, max 30, patience 5, fixed 16/224 | best e5 Macro-F1 **0.7949**, accuracy 0.875, O 0.6667, R 0.9231, pred O/R 0.125/0.875; training log val loss 0.4534. 후속 checkpoint 보고는 0.4537로 0.0003 충돌; early stop e10 | **0.7929746204** | 미확인 | 15m19s | Direct val은 E06보다 높지만 대회 점수는 낮음 | 부분 성공 / E06 대비 약함 |
| E08 AIHub+FAS-Aug pretraining | 2026-09-09 | synthetic paired train 836, val 210 | torchvision pretrained MViTv2-S, full | FAS-Aug generated replay | log상 lr 3e-5, fixed 16/224, max 30, patience 7. 이전 명령 텍스트에 lr 1e-5가 있어 충돌하나 실제 결과 로그는 3e-5 | best e10 Macro-F1 **0.9952**, O 0.9953, R 0.9952, pred O/R 0.505/0.495, val loss 0.0344; e11 loss 0.0325이나 F1 동일; early stop e17 | 미제출 | 미확인 | 해당 없음 | synthetic 내부 분류는 거의 완벽 | 내부 성공 / 일반화 미확인 |
| E09 FAS-Aug best → Direct v0 zero-shot | E08 후 | Direct v0 val 16 | E08 checkpoint | FAS-Aug는 pretraining에만 사용 | threshold 0.5 | loss 1.8351, accuracy 0.375, Macro-F1 **0.3651**, O 0.4444, R 0.2857, pred O/R 0.875/0.125 | 미제출 | 미확인 | 해당 없음 | synthetic signature와 실제 재촬영 사이 domain gap | 실패 |
| E10 FAS-Aug → Direct v0 full | 2026-09-09 | Direct v0 64/16, init E08 | MViTv2-S full | Direct 단계 없음 | lr 1e-5, batch 2, fixed 16/224, patience 5 | best e1 Macro-F1 **0.9086**, O 0.8571, R 0.9600, pred O/R 0.188/0.812, val loss 0.2251. e2 0.7949로 하락, 후반 train loss 0.0162까지 감소. early-stop epoch는 대화 기록의 e7/e8 대응이 불명확 | **0.7252208598** | 미확인 | 15m37s | 작은 Direct val에는 매우 높지만 대회 일반화가 약하고 빠르게 과적합 | 실패/약함 |
| E11 Direct v1 full `direct_full` | 2026-09-10 전후 | train 149, val 39; 56 source | torchvision pretrained MViTv2-S, full 34,231,682 | 없음 | lr 2e-5, batch 4로 정리됨, fixed 16/224, balanced sampler | best e11 Macro-F1 **0.9117**, O 0.8800, R 0.9434, pred O/R 0.333/0.667, val loss 0.1990; early stop e18 | **0.7539450278** (`direct_v1.zip`) | 미확인 | 16m41s | 데이터 증가로 val은 크게 상승했지만 대회 점수는 E06보다 하락 | 일반화 실패 |
| E12 Direct v1 대안 `ch_direct_v1` | 2026-09-10 전후 | Direct v1 | torchvision pretrained MViTv2-S full로 추정 | 없음 | 표기상 batch 2, lr 1e-5. E11과 동일 checkpoint인지 별도 run인지 일부 불확실 | Macro-F1 **0.9117**로 보고됨 | **0.7557530189** | 미확인 | 15m19s | E11보다 0.0018 높지만 여전히 E06보다 낮음 | 약함 / 대응관계 일부 불확실 |
| E13 Baidu → Direct v1 last-block | 2026-09-10 전후 | Direct v1 149/39, init E03 | MViTv2-S last-block; trainable 7,103,234 / total 34,231,682 (20.75%) | 없음 | lr 1e-5, batch 2로 정리됨, fixed 16/224 | best e9 Macro-F1 **0.4040**, O 0.4651, R 0.3429, pred O/R 0.795/0.205, val loss 0.8371; early stop e16 | 미제출 | 미확인 | 해당 없음 | 제한된 업데이트 범위로 Direct 적응 실패 | 실패 |
| E14 Baidu → Direct v1 full | 2026-09-10 | Direct v1 149/39, init E03 | MViTv2-S full | 없음 | lr 1e-5, batch 2, fixed 16/224 | best e3 Macro-F1 **0.9023**, O 0.8571, R 0.9474, pred O/R 0.231/0.769, val loss 0.4453; early stop e10 | **0.7758072612** | 미확인 | 15m23s | 같은 확장 데이터의 Direct-only보다 대회 점수는 약간 높지만 E06보다 낮음 | 부분 성공 / best 미달 |

참고: 모든 위 제출에서 Stage 2 점수는 **0.159801139**, Stage 3 점수는 **0.1324991733**으로 동일했다. 즉, 제출 간 변화는 Stage 1 모델에서 발생한 것으로 해석했다.

## 10. Best Known Result

- **모델:** torchvision Kinetics pretrained `mvit_v2_s`, 모든 layer full fine-tuning.
- **실험:** E06 `direct_raw_v1`.
- **데이터:** Direct v0, 원본 source 20개에서 생성된 80 videos. train 16 source/64 videos, val 4 source/16 videos.
- **augmentation:** random augmentation 없음. 전체 영상 uniform 16 frames, short side resize 224, center crop 224×224, 고정 normalization.
- **학습 조건:** CrossEntropy, AdamW, cosine scheduler, lr `1e-5`, batch `2`, weight decay `0.01`, AMP, balanced sampler, early stopping patience 5. best epoch 8, early stop epoch 13.
- **validation:** Macro-F1 **0.6135**.
- **대회 Stage 1 점수:** **0.8308436777**.
- **점수 성격:** DACON 제출 대시보드/leaderboard score. public인지 private인지 명시적으로 확인되지 않음. private score 없음.
- **현재 best인 이유:** 대화에서 확인 가능한 모든 Stage 1 제출 중 가장 높은 점수이다. 두 번째는 Baidu→Direct v0의 0.7929746204, 세 번째는 Baidu→Direct v1 full의 0.7758072612이다.
- **중요한 해석:** validation 최고 모델이 대회 최고 모델은 아니다. Synthetic val 0.9952, Baidu val 0.9900, Direct v1 val 0.9117이 모두 대회 점수로는 E06보다 낮았다.

## 11. Failed / Weak Approaches

| 접근 | 실제 한 일 | 결과 | 약하다고 판단한 근거 |
|---|---|---|---|
| Baidu-only | Baidu RGB pair로 MViTv2-S full 학습 후 제출 | val 0.9900, 대회 0.5093434343 | 내부 val과 대회 점수의 큰 괴리 |
| Baidu zero-shot on Direct | Baidu best를 물리 재촬영 80개에 직접 평가 | Macro-F1 0.3463, AUC 0.50875 | 거의 무작위 수준이고 R을 O로 많이 분류 |
| threshold만 조정 | Baidu→Direct zero-shot threshold scan | 최고 0.4479 at 0.01 | 분류 표현 자체가 맞지 않아 threshold로 해결되지 않음 |
| FAS-Aug zero-shot | synthetic best를 Direct val에 직접 평가 | Macro-F1 0.3651 | synthetic와 real recapture 간 domain gap |
| FAS-Aug→Direct v0 | synthetic pretrain 후 Direct full fine-tuning | val 0.9086, 대회 0.7252208598 | val은 높지만 E06보다 대회 점수가 크게 낮고, epoch 1 이후 하락·과적합 |
| Direct v1 데이터 단순 확대 | 20→56 source, 80→188 video로 늘려 full fine-tuning | val 0.9117, 대회 0.753945/0.755753 | 데이터 수 증가가 숨은 분포 일반화로 이어지지 않음 |
| Baidu→Direct v1 last-block | 마지막 block+head만 학습 | val 0.4040 | full fine-tuning 0.9023보다 크게 낮음 |
| Baidu→Direct v1 full | Baidu init 후 확장 Direct에 full fine-tuning | val 0.9023, 대회 0.7758072612 | 확장 Direct-only보다 조금 낫지만 E06 best보다 낮음 |
| 고해상도 영상 순차 decode | 4K/60fps Direct 영상을 프레임 0부터 순차 decode | 준비 시간이 지나치게 길어짐 | 필요한 frame index로 seek하는 방식으로 코드 최적화 완료 |

다음은 실패로 확정할 수 없는 위험 요소이다.

- FAS-Aug와 Direct가 같은 AIHub source를 공유했을 가능성: 실제 overlap 검증·제거 여부가 미확인이라 val 누수 가능성이 남아 있다.
- Direct v1의 높은 val/낮은 대회 점수: codec/fps나 촬영 환경 shortcut 때문이라는 해석이 유력했지만 인과 검증은 아직 없다.
- `last-block`은 E13 설정에서는 실패했으나, 모든 초기화·augmentation 조건에서 폐기됐다고 결정한 것은 아니다.

## 12. Current Hypotheses

아래는 대화에서 실제로 제기되거나 현재 유력하다고 정리된 가설이며, 아직 증명된 사실은 아니다.

1. **실제 물리 재촬영 데이터가 가장 중요하다.** Direct v0가 내부 val은 낮아도 대회에서 최고였고, Baidu/FAS synthetic proxy는 단독 또는 초기화로 충분하지 않았다.
2. **현재 Direct validation은 숨은 평가 분포를 대표하지 못한다.** source는 분리했지만 display/camera/environment는 겹쳐 있어 같은 재촬영 파이프라인의 fingerprint를 학습할 수 있다.
3. **codec/fps shortcut 가능성이 크다.** Direct v1의 O는 거의 모두 MPEG4 15fps, R은 H264/HEVC 30/60fps여서 content보다 container/codec/FPS 단서를 사용할 수 있다.
4. **같은 장비·환경의 데이터를 더 추가하면 오히려 편향이 강화될 수 있다.** 20→56 source 확장에서 val은 상승했지만 대회 점수는 하락했다.
5. **FAS-Aug가 synthetic 생성 규칙을 학습했을 가능성이 있다.** synthetic val 0.9952에 비해 Direct zero-shot 0.3651, 대회 0.7252였다.
6. **Baidu pretraining은 일부 도움이 될 수 있으나 최종 해법은 아니었다.** Direct v1에서는 Direct-only보다 약간 높은 대회 점수를 냈지만, 전체 최고 Direct v0에는 못 미쳤다.
7. **고정 center crop·고정 16프레임 한 clip이 recapture 단서를 놓칠 수 있다.** 모아레·rolling band·flicker가 국소적 또는 시간적으로 불규칙할 수 있다는 논의가 있었다.
8. **일부 val 영상은 장면 자체의 shortcut을 유발할 수 있다.** `direct_full` 진단에서 R014 원본을 R로 강하게 예측(P(R)≈0.837), R017 재촬영 일부를 O로 예측(P(R)≈0.065/0.170)한 사례가 언급됐다. 이 진단의 정확한 checkpoint 대응은 `direct_full`로 보이나 완전 확정은 아니다.

## 13. Open Problems

- **데이터 다양성:** 절대 수보다 display/camera/environment 조합 다양성이 부족하다.
- **validation 설계:** source holdout은 했지만 environment holdout 또는 장비 조합 holdout이 없다. val과 대회 순위가 반복적으로 역전된다.
- **domain gap:** Baidu/FAS synthetic ↔ 실제 재촬영 ↔ 대회 숨은 분포 간 차이가 크다.
- **temporal modeling:** MViT를 사용하지만 입력은 고정 uniform 16프레임 한 clip뿐이다. flicker, rolling band, frame-rate mismatch를 충분히 관찰하는지는 미확인이다.
- **overfitting:** Direct가 작고 FAS→Direct는 best가 epoch 1에서 나온 뒤 급격히 하락했다.
- **test distribution uncertainty:** 대회 test 장비, 화면, codec, fps, 조명, class ratio가 모두 미확인이다.
- **metadata shortcut:** O/R 간 codec·fps 차이가 label과 거의 결합돼 있다.
- **FAS/Direct source overlap:** 중복 제거 여부가 미확인이다.
- **augmentation 부재:** 현재 주력 코드는 랜덤 공간·색·시간 augmentation이 없다.
- **Direct cache 고정성:** 같은 clip을 매 epoch 사용해 temporal diversity가 없다.
- **새 LG/IP13P 데이터 통합:** dry-run mapping 이후 실제 rename, manifest 반영, split, 학습 결과가 미확인이다.
- **외부 데이터 사용 규정·라이선스:** 확인 완료 근거가 없다.
- **R050 codec 진단:** 예외적인 H264 30fps original을 이용해 shortcut을 점검하자는 논의는 있었으나 결과는 미확인이다.
- **Stage 2·3:** 초기 baseline에서 거의 진전이 없고, 정확한 metric·개선 실험도 미확인이다.

## 14. Next Planned Experiments

새 제안이 아니라, 대화에서 실제 다음 단계로 잡았던 항목만 적는다.

### 최신 실험 순서로 정리된 항목

1. 동일 validation 기준에서 현재 **no-augmentation baseline**을 기준점 A로 고정.
2. clip-consistent weak spatial/color/time augmentation을 더한 B 실험.
3. 같은 조건에서 **full vs last-block** fine-tuning 비교 C.
4. O/R 양쪽에 compression/resolution 변형을 적용하는 D.
5. 그 뒤에만 제한적으로 recapture synthesis를 넣는 E.

위 A의 과거 baseline 결과는 있으나, A~E를 동일 split·동일 조건으로 다시 비교 완료했다는 근거는 없다. B~E는 미실험이다.

### 데이터 수집·검증 계획

- LG display + IP13P camera로 R001~R056의 E01 촬영.
- 기존 E03 R021~R056을 유지하고 R001~R020을 보충.
- E05 R001~R056 촬영.
- 숫자 파일명 순서와 R ID를 mapping CSV로 보존.
- R021~R056 추가분을 기존 0.8308436777 모델로 별도 평가.
- source holdout과 별도로 environment holdout validation을 구성.

### 이전부터 남아 있는 계획

- source-balanced Direct sampling.
- Baidu+Direct domain-balanced mixed training.
- common codec/fps augmentation으로 shortcut 점검.
- EDA, 오류 샘플 확인, 필요 시 Grad-CAM/FFT 통계. 실제 완료 기록은 없음.
- R001~R056 전체 train, 고정 3 epoch 최종 학습. 실행 여부 미확인.

## 15. Repository / Environment

- **repo name/path:** `blackbox-dacon`; 로컬 `/home/sungmin/blackbox-dacon`.
- **remote:** `https://github.com/sungmin-Jeon/blackbox-dacon`.
- **branch/workflow:** `main` branch에서 WSL/Ubuntu 로컬 개발 → commit/push → Colab에서 pull하여 실행하는 흐름. 현재 확인 시 working tree는 clean이었다.
- **주요 Git 이력:**
  - 2026-08-31: initial baseline notebooks.
  - 2026-09-01: Stage별 baseline 분리, Python 3.13 dependency 수정.
  - 2026-09-04: Baidu loader·DataLoader·modular Stage 1 pipeline.
  - 2026-09-07: 제출 assembler, Stage 1 video eval, source ID 정규화, video decoding 최적화.
  - 2026-09-08: Direct dataset 및 Baidu/Direct unified training.
  - 2026-09-09: fine-tuning scope 옵션.
- **Colab 환경:** 실제 학습에 사용. 확인된 런타임은 `PyTorch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, `NumPy 2.1.2`, `OpenCV 4.10.0`, CUDA 사용 가능, GPU `Tesla T4`.
- **requirements 주요 pin:** torch 2.8.0, torchvision 0.23.0, numpy 2.1.2, pandas 2.2.3, opencv-python-headless 4.10.0, tqdm 4.66.5, Pillow 11.
- **Google Drive:** 데이터와 checkpoint의 영구 저장소로 사용. `/content`는 압축 해제·cache용 임시 로컬 공간.
- **주요 모델 경로:** `/content/drive/MyDrive/2026_Dacon/sungmin/stage1/...`; Stage 2·3 baseline은 `/content/drive/MyDrive/blackbox-dacon/model/...` 경로가 사용됨.
- **repo 내 현재 주요 파일:** `train_stage1.py`, `train_stage2.py`, `train_stage3.py`, `inference.py`, `eval_stage1.py`, `prepare_submit.py`, `build_submit.py`, `src/stage1/*`, `src/stage2/*`, `src/stage3/*`.
- **현재 구현:** `--data-mode baidu|direct`, `--init-ckpt`, `--fine-tune-scope full|head|last-block`, Direct clip cache, source split, Macro-F1 checkpointing, early stopping, 제출 ZIP 조립.
- **repo에 데이터/학습 weight가 실제 포함돼 있는지:** 현재 workspace에서는 확인되지 않으며, 대화상 데이터·checkpoint는 주로 Drive에 있다.

## 16. Important Decisions Already Made

### 이미 결정·적용된 것

- 연구 우선순위는 Stage 1.
- Stage 1 주력 backbone은 pretrained MViTv2-S.
- Direct split은 source ID 단위로 묶어 원본과 그 재촬영이 train/val에 갈라지지 않게 함.
- Baidu는 RGB `gt_rgb/moire_rgb`만 사용하고 raw NPZ는 받지 않음.
- 실험별 output directory를 분리하고 기존 checkpoint를 덮어쓰지 않음.
- 큰 ZIP은 Drive에 보관하고 Colab 로컬로 압축 해제·cache하여 학습.
- 현재 최고 기준점은 Direct v0 full fine-tuning 제출 0.8308436777.
- 새 직접 촬영 파일은 장비·환경 코드와 source ID가 드러나는 규칙으로 보존.
- 변화 원인을 분리하기 위해 한 번에 하나의 요소를 비교해야 한다는 원칙.

### 보류·미실험

- DANN, CORAL, MMD.
- LoRA, L2-SP.
- FFT/frequency branch, global/local two-branch.
- multi-clip/multi-crop 추론.
- weak spatial/color/temporal augmentation과 common codec/resolution augmentation.
- source-balanced mixed-domain training.
- environment holdout, 5-fold/group CV.
- 전체 56 source final 3-epoch fit.

### 폐기 또는 우선순위 하락

- Baidu raw NPZ 사용: 현재 Stage 1에는 불필요하다고 보고 제외.
- DLC-2021을 첫 외부 데이터로 사용하는 안: 106 GB 규모와 문서 도메인 차이로 우선순위 하락.
- Baidu-only를 최종 Stage 1 해법으로 보는 안: 대회 0.5093 및 Direct zero-shot 실패로 약함이 확인됨.
- FAS-Aug-only/zero-shot을 실제 재촬영 대체물로 보는 안: Direct zero-shot 0.3651로 부적합.
- Baidu→Direct v1 last-block의 해당 설정: val 0.4040으로 실패. 다만 last-block 전체를 영구 폐기한 결정은 아님.
- Direct v1을 단순히 더 많은 source라는 이유로 best로 채택하는 안: 대회 점수가 Direct v0보다 낮아 채택하지 않음.

## 17. Timeline

| 시점 | 진행 사항 |
|---|---|
| 2026-08-31 | 3-Stage baseline notebooks를 저장소에 추가 |
| 2026-09-01 | Stage별 baseline 분리, Python 3.13 의존성 수정. 첫 제출: Stage1 0.4045598914 / Stage2 0.159801139 / Stage3 0.1324991733 |
| 2026-09-04 | Baidu RGB loader와 modular Stage 1 pipeline 구현. notebook prototype 및 1-epoch smoke 수행 |
| 2026-09-04~06 | Baidu full 학습, 내부 val Macro-F1 0.9900 도달 |
| 2026-09-07 | Baidu-only 제출 Stage1 0.5093434343. Direct v0 80개에 zero-shot 평가하여 Macro-F1 0.3463, AUC 0.50875 확인. video seek decode 최적화와 제출 assembler 정리 |
| 2026-09-08 | Direct v0 smoke 및 full 학습. `direct_raw_v1` 제출 Stage1 **0.8308436777**로 현재 best. Baidu→Direct v0 제출 0.7929746204 |
| 2026-09-09 | AIHub+FAS-Aug paired pretraining val 0.9952. Direct zero-shot 0.3651. FAS→Direct v0 제출 0.7252208598. fine-tuning scope 옵션 추가 |
| 2026-09-10 전후 | Direct v1을 56 source/188 video로 확장. Direct-only val 0.9117이나 제출 0.753945/0.755753. Baidu→Direct v1 last-block val 0.4040, full val 0.9023 및 제출 0.7758072612 |
| 2026-09-10 이후 | validation-leaderboard 불일치, codec/fps shortcut, 환경 다양성 부족을 핵심 문제로 정리. R014/R017 오류 사례와 R050 codec 예외를 진단 대상으로 논의 |
| 2026-09-14까지 | LG/IP13P E03 R021~R056 36개 파일의 순서 mapping dry-run 확인. E01/E03/E05 표준화 재촬영 계획 수립. 최신 연구 순서를 no-aug 기준점 → weak aug → fine-tune scope → common compression/resolution → 제한적 synthesis로 정리했으나 후속 실험 결과는 없음 |

## 18. Astra Handoff Summary

이 프로젝트는 블랙박스 비디오를 다루는 3-Stage DACON 과제이며, 지금까지의 실질적인 연구는 Stage 1 원본(`ORIGINAL`) 대 재촬영(`RERECORDED`) 분류에 집중됐다. Stage 2는 ResNet18+2-layer BiGRU baseline, Stage 3는 MViTv2-S dual-head baseline을 유지하고 있다. 확인된 모든 제출에서 Stage 2는 0.159801139, Stage 3는 0.1324991733으로 고정돼 있었다. 현재 Stage 1 주력 모델은 Kinetics pretrained `mvit_v2_s`이고, 입력은 영상 전체에서 균등 추출한 16프레임 한 clip, 224×224 center crop이다. 실제 학습은 CrossEntropy, AdamW, cosine scheduler, AMP, Macro-F1 checkpoint 선택, early stopping을 사용했다. 현재 주력 코드에는 랜덤 augmentation이 없다.

사용한 핵심 데이터는 세 계열이다. 첫째, Baidu RawVDemoiré RGB는 train 250 pair×60 frames로 `gt_rgb`와 `moire_rgb`가 정확히 1:1 대응한다. 이를 O/R proxy로 학습한 모델은 내부 val Macro-F1 0.9900이었지만 대회 Stage 1 점수는 0.5093434343이었다. Direct v0 80개에 zero-shot 평가했을 때 Macro-F1 0.3463, ROC-AUC 0.50875에 그쳤다. 따라서 Baidu 내부 성능은 실제 재촬영이나 대회 일반화를 대표하지 않는다.

둘째, AIHub 원본과 FAS-Aug replay로 만든 synthetic paired 데이터는 train 836, val 210이다. 여기서는 val Macro-F1 0.9952까지 올라갔지만 Direct zero-shot은 0.3651이었다. FAS checkpoint를 Direct v0에 full fine-tuning하면 작은 Direct val에서는 0.9086이었으나 대회는 0.7252208598이었다. FAS-Aug가 synthetic 생성 규칙을 학습했다는 가설이 제기됐다. Direct와 FAS 원본이 같은 AIHub pool에서 왔기 때문에 source overlap 가능성도 있으나, 중복 제거 여부는 확인되지 않았다.

셋째이자 가장 중요한 데이터는 실제 물리 재촬영 Direct 데이터다. v0는 R001~R020 원본 20개와 source당 3개 재촬영 60개, 총 80개다. source 단위로 train 16 source/64 videos, val 4 source/16 videos로 나눴다. torchvision pretrained MViTv2-S를 augmentation 없이 full fine-tuning한 `direct_raw_v1`은 val Macro-F1이 0.6135에 불과했지만 대회에서 **0.8308436777**을 기록해 현재 최고다. lr 1e-5, batch 2, balanced sampler를 썼고 best epoch는 8, early stop은 13이었다. 내부 val이 가장 낮은 편인 모델이 대회에서는 가장 좋았다는 점이 전체 프로젝트의 핵심 사실이다.

Direct v1은 R001~R056 원본 56개와 재촬영 132개, 총 188개로 늘어났다. seed 42 source split은 train 44 source/149 videos, val 12 source/39 videos다. Direct-only full fine-tuning은 val 0.9117까지 올랐지만 대회는 0.7539450278 또는 설정 라벨이 일부 다른 제출에서 0.7557530189였다. Baidu→Direct v1 full은 val 0.9023, 대회 0.7758072612였고, last-block만 학습하면 val 0.4040으로 실패했다. 따라서 단순 데이터 증가와 높은 in-domain val은 숨은 분포 성능을 보장하지 않았다.

현재 가장 강한 해석은 validation 설계와 촬영 metadata 편향 문제다. source ID leakage는 막았지만 train/val이 같은 display-camera-environment 계열을 공유한다. 또한 Direct v1 원본 56개 중 55개가 MPEG4 15fps이고, 재촬영은 모두 H264/HEVC 30/60fps라 label과 codec/fps가 강하게 결합돼 있다. 모델이 실제 재촬영 artifact보다 이 shortcut이나 고정 촬영 파이프라인 fingerprint를 학습했을 가능성이 제기됐다. 이것은 유력 가설이지 아직 인과 검증된 사실은 아니다. 숨은 test의 장비·환경·codec·fps·class ratio는 전부 미확인이다.

직접 촬영은 계속 확대하는 방향으로 논의됐다. 최신 확인으로 LG/IP13P E03의 R021~R056 MOV 36개가 있고, 숫자 파일명 순서가 촬영 순서와 일치한다는 사용자 확인에 따라 R021~R056으로 매핑하는 dry-run CSV를 만들었다. 실제 rename/apply와 학습 반영은 미확인이다. 이후 동일 LG/IP13P 조합으로 E01 R001~R056, E03의 R001~R020 보충, E05 R001~R056을 촬영하는 계획이 세워졌다. 환경 코드 E01/E03/E05의 정확한 물리 조건과 장비 코드의 정식 제품명은 확인되지 않았으므로 임의 확장하면 안 된다.

최신 실험 계획은 동일 validation을 고정하고 한 변수씩 비교하는 순서였다. A는 현재 no-augmentation baseline, B는 clip-consistent weak spatial/color/time augmentation, C는 full 대 last-block, D는 O/R 양쪽 common compression/resolution augmentation, E는 그 이후 제한적 recapture synthesis다. source-balanced sampling, Baidu+Direct domain-balanced mixed training, environment holdout validation, multi-clip/global-local 입력, DANN/CORAL/MMD, LoRA, FFT branch도 논의됐지만 실행 결과는 없다. Astra는 이 항목들을 완료된 연구로 오인하면 안 된다.

현재 상태를 한 문장으로 요약하면: **실제 Direct v0에서 단순 full fine-tuning한 MViTv2-S가 대회 Stage 1 0.8308436777로 가장 잘 됐고, 더 큰 Direct v1·Baidu·FAS-Aug는 validation을 크게 높였지만 숨은 평가 성능을 낮췄으며, 핵심 미해결 문제는 촬영 환경/codec/fps shortcut과 validation-test 분포 불일치다.**
