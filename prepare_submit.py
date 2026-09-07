"""Assemble checkpoints from separate experiments and build a submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from build_submit import build_archive, validate_archive, validate_inputs, validate_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    inference_file = _existing_file(args.inference_file, "inference file")
    requirements_file = _existing_file(args.requirements_file, "requirements file")
    output = args.output.expanduser().resolve()

    model_sources = {
        "stage1/best.pt": _existing_file(args.stage1_checkpoint, "Stage 1 checkpoint"),
        "stage2/best.pt": _existing_file(args.stage2_checkpoint, "Stage 2 checkpoint"),
        "stage2/resnet18-f37072fd.pth": _existing_file(
            args.stage2_backbone,
            "Stage 2 backbone",
        ),
        "stage3/best.pt": _existing_file(args.stage3_checkpoint, "Stage 3 checkpoint"),
    }

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
