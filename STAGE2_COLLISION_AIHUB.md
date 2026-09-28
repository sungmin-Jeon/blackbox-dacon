# Stage 2 AIHub-only collision experiment

This experiment keeps ImageNet ResNet18 frozen, caches one 512-dimensional
vector per video frame, and trains a new Collision BiGRU from scratch using only
AIHub rows that have `collision_frame`. It does not initialize the temporal model
from MM-AU.

```bash
python train_stage2_collision_aihub.py \
  --labels "/content/drive/MyDrive/2026_Dacon/data/stage2/annotations/stage2_labels_unique_keep_split_seed42_v1.csv" \
  --video-root "/content/stage2/stage2_labeled_unique_keep_v1/videos" \
  --backbone "/content/drive/MyDrive/2026_Dacon/sungmin/stage2/resnet18-f37072fd.pth" \
  --feature-dir "/content/stage2_collision_aihub_features_v1" \
  --baseline-predictions "/content/drive/MyDrive/2026_Dacon/data/stage2/annotations/stage2_collision_predictions_v1.csv" \
  --output-root "/content/drive/MyDrive/2026_Dacon/data/stage2/experiments/collision_aihub_only" \
  --name "collision_aihub_only_gaussian_v1" \
  --loss gaussian_ce \
  --sigma-sec 0.1 \
  --hidden-size 64 \
  --num-layers 1 \
  --epochs 50 \
  --early-patience 7
```

The first run builds the local feature cache. Later runs reuse it. The output
prints the existing MM-AU model and the new AIHub-only model on exactly the same
validation IDs. Model selection uses collision Accuracy@0.3 seconds, with
validation loss only as a tie-breaker.
