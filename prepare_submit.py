"""Assemble checkpoints from separate experiments and build a submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from build_submit import build_archive, validate_archive, validate_inputs, validate_source
from src.stage2.collision_predict import collision_model_from_checkpoint
from src.stage2.collision_feature_compare import CollisionFeatureModel
from src.stage2.direct_model import direct_model_from_checkpoint
from src.stage2.entry_feature_compare import entry_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-direct-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-collision-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-entry-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-side-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-evasion-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-backbone", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path, required=True)
    parser.add_argument("--inference-file", type=Path, default=Path("inference.py"))
    parser.add_argument("--requirements-file", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage2_checkpoints(
    collision_path: Path,
    direct_path: Path,
) -> tuple[str, dict, dict]:
    """Fail before packaging when Stage 2 paths point at incompatible files."""
    collision_checkpoint = torch.load(
        collision_path, map_location="cpu", weights_only=False
    )
    _, collision_format = collision_model_from_checkpoint(collision_checkpoint)

    direct_checkpoint = torch.load(
        direct_path, map_location="cpu", weights_only=False
    )
    required = {"model_config", "model_state_dict"}
    missing = required - set(direct_checkpoint)
    if missing:
        raise ValueError(
            f"Stage 2 Direct checkpoint is incompatible; missing {sorted(missing)}. "
            f"Selected file: {direct_path}"
        )
    direct_model_from_checkpoint(direct_checkpoint)
    feature_config = direct_checkpoint.get("feature_config", {})
    layer = feature_config.get("layer", "layer3")
    if layer not in {"layer3", "layer4"}:
        raise ValueError(f"Unsupported Stage 2 Direct feature layer: {layer!r}")
    expected_channels = 256 if layer == "layer3" else 512
    actual_channels = int(direct_checkpoint["model_config"]["input_channels"])
    if actual_channels != expected_channels:
        raise ValueError(
            "Stage 2 Direct checkpoint input channels do not match its feature layer: "
            f"{actual_channels} vs {layer} ({expected_channels})"
        )
    return collision_format, collision_checkpoint, direct_checkpoint


def _load_checkpoint(path: Path, label: str) -> tuple[Path, dict]:
    resolved = _existing_file(path, label)
    return resolved, torch.load(resolved, map_location="cpu", weights_only=False)


def _validate_stage2_final(args: argparse.Namespace) -> dict[str, tuple[Path, dict]]:
    paths = {
        "collision": args.stage2_collision_checkpoint,
        "entry": args.stage2_entry_checkpoint,
        "side": args.stage2_side_checkpoint,
        "evasion": args.stage2_evasion_checkpoint,
    }
    if not all(paths.values()):
        missing = [name for name, path in paths.items() if path is None]
        raise ValueError(f"Missing final Stage 2 checkpoints: {missing}")
    loaded = {
        name: _load_checkpoint(path, f"Stage 2 {name} checkpoint")
        for name, path in paths.items()
    }
    collision = loaded["collision"][1]
    collision_model = CollisionFeatureModel(**collision["model_config"])
    collision_model.load_state_dict(collision["model_state_dict"])
    for name in ("entry", "side"):
        checkpoint = loaded[name][1]
        model = entry_model_from_checkpoint(checkpoint)
        if checkpoint.get("model_class") != "EntrySideFeatureModel":
            raise ValueError(f"Stage 2 {name} checkpoint has no Side-capable model")
        del model
    direct_model_from_checkpoint(loaded["evasion"][1])

    signatures = []
    for _, checkpoint in loaded.values():
        config = checkpoint.get("feature_config", {})
        signatures.append(
            (
                config.get("layer", "layer3"),
                int(config.get("short_edge", 256)),
                tuple(config.get("mean", (0.485, 0.456, 0.406))),
                tuple(config.get("std", (0.229, 0.224, 0.225))),
                config.get("crop"),
            )
        )
    if len(set(signatures)) != 1 or signatures[0][4] is not None:
        raise ValueError("Final Stage 2 checkpoints use incompatible feature settings")
    return loaded


def main() -> None:
    args = parse_args()
    inference_file = _existing_file(args.inference_file, "inference file")
    requirements_file = _existing_file(args.requirements_file, "requirements file")
    output = args.output.expanduser().resolve()

    final_requested = any(
        path is not None
        for path in (
            args.stage2_collision_checkpoint,
            args.stage2_entry_checkpoint,
            args.stage2_side_checkpoint,
            args.stage2_evasion_checkpoint,
        )
    )
    model_sources = {
        "stage1/best.pt": _existing_file(args.stage1_checkpoint, "Stage 1 checkpoint"),
        "stage2/resnet18-f37072fd.pth": _existing_file(
            args.stage2_backbone,
            "Stage 2 backbone",
        ),
        "stage3/best.pt": _existing_file(args.stage3_checkpoint, "Stage 3 checkpoint"),
    }
    if final_requested:
        loaded = _validate_stage2_final(args)
        for name, (path, checkpoint) in loaded.items():
            model_sources[f"stage2/{name}.pt"] = path
            print(
                f"Stage 2 {name}: epoch={checkpoint.get('epoch')} | "
                f"task={checkpoint.get('task', 'unspecified')}"
            )
    else:
        if args.stage2_checkpoint is None or args.stage2_direct_checkpoint is None:
            raise ValueError(
                "Provide either all four final Stage 2 checkpoints or both legacy "
                "--stage2-checkpoint and --stage2-direct-checkpoint"
            )
        stage2_checkpoint = _existing_file(
            args.stage2_checkpoint, "Stage 2 checkpoint"
        )
        stage2_direct_checkpoint = _existing_file(
            args.stage2_direct_checkpoint, "Stage 2 Direct checkpoint"
        )
        collision_format, collision_checkpoint, direct_checkpoint = (
            _validate_stage2_checkpoints(stage2_checkpoint, stage2_direct_checkpoint)
        )
        print(f"Stage 2 collision checkpoint format: {collision_format}")
        print(
            f"Stage 2 collision task: "
            f"{collision_checkpoint.get('task', 'unspecified')}"
        )
        print(f"Stage 2 Direct task: {direct_checkpoint.get('task', 'unspecified')}")
        model_sources["stage2/best.pt"] = stage2_checkpoint
        model_sources["stage2/direct.pt"] = stage2_direct_checkpoint

    source = validate_source(inference_file)
    with tempfile.TemporaryDirectory(prefix="blackbox-submit-") as temporary:
        model_root = Path(temporary) / "model"
        for relative, model_source in model_sources.items():
            target = model_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_source, target)

        validate_inputs(requirements_file, model_root)
        build_archive(source, requirements_file, model_root, output)
        archive_names = validate_archive(output)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_zip": str(output),
        "submission_size_bytes": output.stat().st_size,
        "inference_file": str(inference_file),
        "requirements_file": str(requirements_file),
        "models": {
            relative: {
                "source": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for relative, path in model_sources.items()
        },
        "archive_contents": archive_names,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Created: {output}")
    print(f"Size: {output.stat().st_size / 1024**2:.2f} MB")
    print(f"Manifest: {manifest_path}")
    print("Models:")
    for relative, path in model_sources.items():
        print(f" - {relative} <- {path}")


if __name__ == "__main__":
    main()
