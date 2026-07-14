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
        description="Validate reduced MARZ-sheet initial-state setup."
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--x1min", type=float, required=True)
    parser.add_argument("--x1max", type=float, required=True)
    parser.add_argument("--x2min", type=float, required=True)
    parser.add_argument("--x2max", type=float, required=True)
    parser.add_argument("--b0", type=float, required=False)
    parser.add_argument("--divb-tol", type=float, default=1.0e-4)
    parser.add_argument("--pressure-rel-tol", type=float, default=5.0e-3)
    args = parser.parse_args()

    path = Path(args.file)
    cons = parse_dataset(path, "/cons")
    prim = parse_dataset(path, "/prim")

    rho = cons[:, 0, :, :, :]
    b1 = cons[:, 5, :, :, :]
    b2 = cons[:, 6, :, :, :]
    pressure = prim[:, 4, :, :, :]

    finite_ok = np.all(np.isfinite(cons)) and np.all(np.isfinite(prim))
    if not finite_ok:
        raise RuntimeError("Initial-state dump contains non-finite values.")
    if np.min(rho) <= 0.0:
        raise RuntimeError(f"Density is non-positive: {np.min(rho):.6e}")
    if np.min(pressure) <= 0.0:
        raise RuntimeError(f"Pressure is non-positive: {np.min(pressure):.6e}")

    ni = cons.shape[-1]
    nj = cons.shape[-2]
    dx = (args.x1max - args.x1min) / ni
    dy = (args.x2max - args.x2min) / nj
    divb = (b1[..., 1:-1, 2:] - b1[..., 1:-1, :-2]) / (2.0 * dx) + (
        b2[..., 2:, 1:-1] - b2[..., :-2, 1:-1]
    ) / (2.0 * dy)
    max_divb = float(np.max(np.abs(divb)))
    max_b = float(max(np.max(np.abs(b1)), np.max(np.abs(b2)), 1.0e-30))
    normalized_divb = max_divb * max(dx, dy) / max_b

    center_i = ni // 2
    center_j = nj // 2
    center_pressure = float(np.mean(pressure[..., center_j, center_i]))
    upstream_pressure = float(
        0.5
        * (
            np.mean(pressure[..., center_j, 0])
            + np.mean(pressure[..., center_j, -1])
        )
    )
    asymptotic_b = float(
        0.5
        * (
            abs(np.mean(b2[..., center_j, 0]))
            + abs(np.mean(b2[..., center_j, -1]))
        )
    )
    enhancement = center_pressure - upstream_pressure
    expected = 0.5 * asymptotic_b * asymptotic_b
    rel_err = abs(enhancement - expected) / expected

    print(f"rho_min={np.min(rho):.16e}")
    print(f"pressure_min={np.min(pressure):.16e}")
    print(f"temperature_proxy_min={np.min(pressure / rho):.16e}")
    print(f"divb_max_abs={max_divb:.16e}")
    print(f"divb_normalized={normalized_divb:.16e}")
    print(f"asymptotic_B2={asymptotic_b:.16e}")
    print(f"center_pressure={center_pressure:.16e}")
    print(f"upstream_pressure={upstream_pressure:.16e}")
    print(f"pressure_enhancement={enhancement:.16e}")
    print(f"pressure_enhancement_expected={expected:.16e}")
    print(f"pressure_enhancement_rel_err={rel_err:.16e}")

    if normalized_divb > args.divb_tol:
        raise RuntimeError(
            "Initial-state normalized divB exceeded tolerance: "
            f"{normalized_divb:.3e} > {args.divb_tol:.3e}"
        )
    if rel_err > args.pressure_rel_tol:
        raise RuntimeError(
            "Pressure enhancement does not match total-pressure balance: "
            f"{rel_err:.3e} > {args.pressure_rel_tol:.3e}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
