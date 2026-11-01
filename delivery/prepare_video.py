"""Merge the team's recordings and compress the video below the assignment limits.

The assignment allows at most 10 minutes and 50 MB, with each member visible
(head and shoulders) in the upper-right corner. Record each section with your
screen recorder's webcam overlay in the upper right, then run, from the
repository root:

    python delivery/prepare_video.py part_A.mp4 part_B.mp4 part_C.mp4 --output presentation.mp4

Parts are joined in the given order, scaled to 1280 px wide and encoded with
H.264/AAC at a bitrate computed from the total duration so that the file stays
under ~48 MB. It uses the ffmpeg binary bundled with ``imageio-ffmpeg``
(``pip install imageio-ffmpeg``), so no system installation is needed.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

LIMIT_MB, LIMIT_SECONDS, TARGET_MB, AUDIO_KBPS = 50, 600, 47.5, 64


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration_seconds(path: Path) -> float:
    """Parse the container duration from ffmpeg's banner (ffprobe is not bundled)."""
    result = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise ValueError(f"could not read the duration of {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", help="recordings in presentation order")
    parser.add_argument("--output", default="presentation.mp4")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args(argv)
    parts = [Path(p) for p in args.parts]
    total = sum(duration_seconds(p) for p in parts)
    if total > LIMIT_SECONDS:
        raise SystemExit(f"Total duration {total / 60:.2f} min exceeds the 10-minute limit; trim before encoding.")
    video_kbps = int(min(1500, max(150, TARGET_MB * 8 * 1024 / total - AUDIO_KBPS)))
    output = Path(args.output)
    with tempfile.TemporaryDirectory() as tmp:
        inputs = []
        for p in parts:
            inputs += ["-i", str(p)]
        streams = "".join(f"[{i}:v:0]scale={args.width}:-2,setsar=1,fps=30[v{i}];" for i in range(len(parts)))
        joined = "".join(f"[v{i}][{i}:a:0]" for i in range(len(parts)))
        graph = f"{streams}{joined}concat=n={len(parts)}:v=1:a=1[v][a]"
        command = [ffmpeg(), "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                   "-c:v", "libx264", "-preset", "medium", "-b:v", f"{video_kbps}k", "-maxrate", f"{int(video_kbps * 1.5)}k",
                   "-bufsize", f"{video_kbps * 2}k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k",
                   "-movflags", "+faststart", str(output)]
        subprocess.run(command, check=True, cwd=tmp)
    size_mb = output.stat().st_size / 2**20
    final_seconds = duration_seconds(output)
    print(f"{output}: {size_mb:.1f} MB, {final_seconds / 60:.2f} min, video {video_kbps} kbit/s")
    if size_mb > LIMIT_MB or final_seconds > LIMIT_SECONDS:
        raise SystemExit("Output exceeds the assignment limits; lower --width or shorten the recording.")


if __name__ == "__main__":
    main()
