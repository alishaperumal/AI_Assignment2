"""Check that the locally added dataset and Person B package are complete and unchanged.

Each teammate adds the Kaggle data and the ``person_b_final`` package locally
(they are too large for Git). This compares every file with the SHA-256 hashes
recorded by Person B:

    python verify_inputs.py                       # data/ and runs/person_b_final/
    python verify_inputs.py --data-root D:/filament/MAGFiLO_1.0_Kaggle_2026

It only reads files; nothing is modified or downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(root: Path, expected: dict) -> dict:
    """``expected`` maps relative paths to (bytes, sha256); returns problems found."""
    def one(item):
        relative, (size, digest) = item
        path = root / relative
        if not path.is_file():
            return relative, "missing"
        if path.stat().st_size != size or _sha256(path) != digest:
            return relative, "changed"
        return relative, "ok"

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, expected.items()))
    problems = {name: status for name, status in results if status != "ok"}
    return dict(root=str(root), checked=len(results), ok=len(results) - len(problems), problems=problems)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--package", default=str(ROOT / "runs" / "person_b_final"))
    parser.add_argument("--skip-package", action="store_true")
    args = parser.parse_args(argv)
    raw = json.loads((ROOT / "results/person_b_study/raw_dataset_manifest.json").read_text(encoding="utf8"))
    reports = {"dataset": check(Path(args.data_root),
                                {f["path"]: (f["size"], f["sha256"]) for f in raw["files"]})}
    if not args.skip_package:
        package = Path(args.package)
        manifest = package / "manifest.json"
        if manifest.is_file():
            files = json.loads(manifest.read_text(encoding="utf8"))["files"]
            reports["person_b_package"] = check(package, {k: (v["bytes"], v["sha256"]) for k, v in files.items()})
        else:
            reports["person_b_package"] = dict(root=str(package), checked=0, ok=0,
                                               problems={"manifest.json": "missing"})
    print(json.dumps(reports, indent=2))
    failed = [name for name, report in reports.items() if report["problems"]]
    if failed:
        raise SystemExit(f"Problems found in: {', '.join(failed)}")
    print("All files match their recorded hashes.")


if __name__ == "__main__":
    main()
