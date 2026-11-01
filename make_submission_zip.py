"""Assemble the assignment ZIP: code, submission.csv, slides and video (Person C).

The assignment requires one archive with (1) all code, (2) the exact
submission.csv uploaded to Kaggle, (3) the video presentation (at most 50 MB
and 10 minutes) and (4) the PowerPoint slides. Datasets, caches, checkpoints and
run outputs are excluded. Example, from the repository root:

    python make_submission_zip.py --submission runs/person_c/final/submission.csv \
        --slides delivery/COS711_A2_team_presentation.pptx --video presentation.mp4 \
        --output COS711_A2_submission.zip

Use ``--allow-missing-video`` only for a rehearsal archive. The command never
uploads anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_PATTERNS = ["*.py", "eda.ipynb", "requirements.txt", "README.md", "HANDOFF.md", "PERSON_C_START_HERE.md",
                 "AGENTS.md", ".gitignore", "tests/*.py", "configs/*.yaml", "configs/person_c/*.yaml",
                 "splits/*.json", "delivery/*.py", "delivery/*.mjs", "delivery/*.ps1", "delivery/*.md",
                 "jobs/person_c/*.ps1", "jobs/person_c/*.md",
                 "delivery/*.json", "results/*.json", "results/person_b_study/*.json", "results/person_b_study/*.csv",
                 "results/person_b_study/*.md", "results/person_b_study/figures/*.png", "results/person_c/*",
                 "figures/*.png", "figures/person_c/*.png"]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".npz", ".zip"}
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_VIDEO_SECONDS = 600
MAX_OTHER_FILE_BYTES = 20 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_files() -> list[Path]:
    files = set()
    for pattern in CODE_PATTERNS:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(p for p in files if p.suffix.lower() not in FORBIDDEN_SUFFIXES)


def check_submission(path: Path) -> dict:
    with path.open(newline="", encoding="utf8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    if header != ["filament_id", "segmentation_rle"]:
        raise ValueError(f"unexpected submission header: {header}")
    return dict(rows=rows, sha256=sha256(path))


def video_seconds(path: Path):
    """Duration via ffprobe, or the ffmpeg bundled with imageio-ffmpeg; None if neither exists."""
    import re

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                                capture_output=True, text=True, check=False)
        try:
            return float(json.loads(result.stdout)["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                            capture_output=True, text=True, check=False)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    return None if not match else int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", required=True, help="the exact CSV uploaded to Kaggle")
    parser.add_argument("--slides", required=True, help="PowerPoint used in the video")
    parser.add_argument("--video", help="final video (mp4), at most 50 MB and 10 minutes")
    parser.add_argument("--screenshot", help="optional leaderboard screenshot for the top-100 bonus")
    parser.add_argument("--output", default="COS711_A2_submission.zip")
    parser.add_argument("--allow-missing-video", action="store_true")
    args = parser.parse_args(argv)

    submission, slides = Path(args.submission), Path(args.slides)
    facts = dict(submission=check_submission(submission))
    if slides.suffix.lower() != ".pptx" or not slides.is_file():
        raise ValueError("--slides must be an existing .pptx file")
    extras = [(submission, "submission.csv"), (slides, slides.name)]
    if args.video:
        video = Path(args.video)
        size, seconds = video.stat().st_size, video_seconds(video)
        if size > MAX_VIDEO_BYTES:
            raise ValueError(f"video is {size / 2**20:.1f} MB; the limit is 50 MB")
        if seconds is not None and seconds > MAX_VIDEO_SECONDS:
            raise ValueError(f"video lasts {seconds / 60:.1f} minutes; the limit is 10")
        facts["video"] = dict(name=video.name, megabytes=round(size / 2**20, 2),
                              minutes=None if seconds is None else round(seconds / 60, 2),
                              duration_checked=seconds is not None)
        extras.append((video, video.name))
    elif not args.allow_missing_video:
        raise ValueError("--video is required for the final archive (use --allow-missing-video for a rehearsal)")
    if args.screenshot:
        extras.append((Path(args.screenshot), "leaderboard_" + Path(args.screenshot).name))

    output = Path(args.output).resolve()
    files = code_files()
    manifest = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            if path.resolve() == output or path.stat().st_size > MAX_OTHER_FILE_BYTES:
                continue
            name = "code/" + path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            manifest.append(f"{sha256(path)}  {name}")
        for path, name in extras:
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix.lower() == ".mp4" else None)
            manifest.append(f"{sha256(path)}  {name}")
        facts["n_code_files"] = len(files)
        archive.writestr("SUBMISSION_MANIFEST.txt", "\n".join(manifest) + "\n")
        archive.writestr("SUBMISSION_FACTS.json", json.dumps(facts, indent=2) + "\n")
    facts["zip_megabytes"] = round(output.stat().st_size / 2**20, 2)
    print(json.dumps(facts, indent=2))
    print(f"wrote {output}")
    if not args.video:
        print("WARNING: no video included; this archive is a rehearsal, not the final submission.")


if __name__ == "__main__":
    main()
