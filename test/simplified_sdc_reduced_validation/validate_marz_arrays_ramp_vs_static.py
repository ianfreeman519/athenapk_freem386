#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)",
    re.IGNORECASE,
)
SHAPE_RE = re.compile(r"DATASPACE\s+SIMPLE\s+\{\s*\(\s*([^)]*?)\s*\)")


def h5dump_text(path: Path, dataset: str) -> str:
    return subprocess.check_output(
        ["h5dump", "-m", "%.17g", "-w", "0", "-d", dataset, str(path)], text=True
    )


def parse_dataset(path: Path, dataset: str) -> np.ndarray:
    text = h5dump_text(path, dataset)
    shape = None
    values: list[float] = []
    in_data = False
    for line in text.splitlines():
        if shape is None:
            match = SHAPE_RE.search(line)
            if match:
                shape = tuple(int(x.strip()) for x in match.group(1).split(","))
        stripped = line.strip()
        if stripped == "DATA {":
            in_data = True
            continue
        if in_data:
            if stripped == "}":
                break
            payload = stripped.split(":", 1)[1] if ":" in stripped else stripped
            values.extend(float(tok) for tok in TOKEN_RE.findall(payload))
    if shape is None:
        raise RuntimeError(f"Could not find shape for {dataset} in {path}")
    arr = np.array(values, dtype=float)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise RuntimeError(f"{dataset} expected {expected} values, got {arr.size}")
    return arr.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ramped and static MARZ array-source early-time response."
    )
    parser.add_argument("--ramp", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--peak-ratio-max", type=float, default=0.5)
    args = parser.parse_args()

    ramp = parse_dataset(Path(args.ramp), "/curlBz")
    static = parse_dataset(Path(args.static), "/curlBz")

    if not np.all(np.isfinite(ramp)) or not np.all(np.isfinite(static)):
        raise RuntimeError("Comparison outputs contain non-finite current diagnostics.")

    ramp_peak = float(np.max(np.abs(ramp)))
    static_peak = float(np.max(np.abs(static)))
    ratio = ramp_peak / max(static_peak, 1.0e-30)

    print(f"ramp_curlBz_peak={ramp_peak:.16e}")
    print(f"static_curlBz_peak={static_peak:.16e}")
    print(f"ramp_to_static_peak_ratio={ratio:.16e}")

    if static_peak <= 0.0:
        raise RuntimeError("Static comparison case did not produce wire current.")
    if ratio >= args.peak_ratio_max:
        raise RuntimeError("Ramped source did not stay meaningfully below static early-time drive.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
