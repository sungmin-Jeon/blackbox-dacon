"""Build and validate the baseline-compatible submit.zip archive."""

from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path


REQUIRED_FUNCTIONS = {"predict_stage1", "predict_stage2", "predict_stage3"}
REQUIRED_MODELS = {
    "stage1/best.pt",
    "stage2/best.pt",
    "stage2/resnet18-f37072fd.pth",
    "stage3/best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-file", type=Path, default=Path("inference.py"))
    parser.add_argument("--requirements-file", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def top_level_functions(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate_source(inference_file: Path) -> str:
    if not inference_file.is_file():
        raise FileNotFoundError(f"Inference file does not exist: {inference_file}")
    source = inference_file.read_text(encoding="utf-8")
    missing = sorted(REQUIRED_FUNCTIONS - top_level_functions(source, str(inference_file)))
    if missing:
        raise RuntimeError(f"Missing inference functions: {missing}")
    return source


def validate_inputs(requirements_file: Path, model_dir: Path) -> None:
    if not requirements_file.is_file():
        raise FileNotFoundError(f"Requirements file does not exist: {requirements_file}")
    missing = sorted(relative for relative in REQUIRED_MODELS if not (model_dir / relative).is_file())
    if missing:
        raise FileNotFoundError(f"Missing model files under {model_dir}: {missing}")


def build_archive(
    source: str,
    requirements_file: Path,
    model_dir: Path,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("inference.py", source)
        archive.write(requirements_file, "requirements.txt")
        for model_file in sorted(path for path in model_dir.rglob("*") if path.is_file()):
            if model_file.resolve() == output:
                continue
            archive_name = (Path("model") / model_file.relative_to(model_dir)).as_posix()
            archive.write(model_file, archive_name)


def validate_archive(output: Path) -> list[str]:
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        zipped_source = archive.read("inference.py").decode("utf-8")

    required_paths = {"inference.py", "requirements.txt"} | {
        f"model/{relative}" for relative in REQUIRED_MODELS
    }
    missing_paths = sorted(required_paths - set(names))
    if missing_paths:
        raise RuntimeError(f"Missing files in submit.zip: {missing_paths}")

    missing_functions = sorted(
        REQUIRED_FUNCTIONS - top_level_functions(zipped_source, f"{output}/inference.py")
    )
    if missing_functions:
        raise RuntimeError(f"Missing functions in zipped inference.py: {missing_functions}")
    return names


def main() -> None:
    args = parse_args()
    inference_file = args.inference_file.expanduser().resolve()
    requirements_file = args.requirements_file.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    source = validate_source(inference_file)
    validate_inputs(requirements_file, model_dir)
    build_archive(source, requirements_file, model_dir, output)
    names = validate_archive(output)

    print(f"Created: {output}")
    print(f"Size: {output.stat().st_size / 1024**2:.2f} MB")
    print("Archive contents:")
    for name in names:
        print(f" - {name}")


if __name__ == "__main__":
    main()
