"""Check model files needed by the MangaTransfer GUI pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BALLONSTRANSLATOR_ROOT = ROOT / "BallonsTranslator"
MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def main() -> int:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ocr-engine",
        choices=sorted(manifest["ocr"]),
        default="PaddleOCRVLManga",
        help="OCR engine selected in the GUI",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device selected in the GUI; CPU detection requires the ONNX model",
    )
    args = parser.parse_args()

    detector_device = "cpu" if args.device == "cpu" else "other"
    resources = [
        *manifest["detector"][detector_device],
        *manifest["ocr"][args.ocr_engine],
        *manifest["inpainter"]["lama_large_512px"],
    ]

    missing = []
    for resource in resources:
        path = BALLONSTRANSLATOR_ROOT.joinpath(*resource["path"].split("/"))
        if path.is_file():
            print(f"[OK]      {path}")
        else:
            print(f"[MISSING] {path}")
            print(f"          {resource['url']}")
            missing.append(path)

    if missing:
        print(f"\nMissing {len(missing)} required model/resource file(s).")
        return 1

    print("\nAll checked model/resource files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
