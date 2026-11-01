"""Run one local GPU job with a log, an exclusive lock, and sleep prevention.

Example: python local_job.py --log runs/logs/baseline.log -- train.py --config ...
The child uses this interpreter. Closing the process releases the Windows sleep
request and file lock; no permanent power settings are changed.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a Python script and its arguments after --")
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB free: archive completed runs before continuing")
    (ROOT / "runs").mkdir(exist_ok=True)
    lock = (ROOT / "runs" / ".gpu-job.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        awake = None
    args.log.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.log.with_suffix(".job.json")
    state = dict(command=[sys.executable, "-u", *command], pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 sleep_prevention=awake, status="running", free_bytes_at_start=shutil.disk_usage(ROOT).free)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf8")
    start = time.perf_counter()
    child = None
    code = 1
    try:
        with args.log.open("a", encoding="utf8") as log:
            log.write(f"\nInvocation {state['started_utc']}\n")
            child = subprocess.Popen(state["command"], cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace",
                bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state["child_pid"] = child.pid
            state_path.write_text(json.dumps(state, indent=2), encoding="utf8")
            print(f"Job PID {child.pid}; complete log: {args.log}", flush=True)
            for line in child.stdout:
                log.write(line)
                log.flush()
                if line.strip().startswith(("Run ", "Epoch ", "validation:", "Starting ",
                        "Finished ", "Skipping ", "Recorded ", "test prediction:", "Traceback", "Error")):
                    print(line.strip(), flush=True)
            code = child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(status="completed" if code == 0 else "failed", exit_code=code,
                     elapsed_seconds=time.perf_counter() - start,
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     free_bytes_at_end=shutil.disk_usage(ROOT).free)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf8")
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()
    print(f"Job {state['status']}; exit={code}; wall time={state['elapsed_seconds']:.1f}s", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
