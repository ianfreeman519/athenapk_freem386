#!/usr/bin/env python3
"""Run a command on the first available advisory-locked GPU."""

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", required=True)
parser.add_argument("--lock-dir", type=Path, required=True)
parser.add_argument("--poll-seconds", type=float, default=0.2)
args, command = parser.parse_known_args()
if command and command[0] == "--":
    command = command[1:]
gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
if not gpu_ids or not command or args.poll_seconds <= 0.0:
    parser.error("positive poll interval, at least one GPU, and a command are required")

args.lock_dir.mkdir(parents=True, exist_ok=True)
while True:
    for gpu_id in gpu_ids:
        lock_fd = os.open(args.lock_dir / f"gpu-{gpu_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_fd)
            continue
        # Preserve this descriptor across exec so AthenaPK holds the lock until exit.
        os.set_inheritable(lock_fd, True)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["ATHENAPK_ASSIGNED_GPU"] = gpu_id
        print(f"gpu-lock: CUDA device {gpu_id}: {' '.join(command)}", file=sys.stderr, flush=True)
        os.execvp(command[0], command)
    time.sleep(args.poll_seconds)
