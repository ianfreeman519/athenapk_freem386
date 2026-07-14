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


def squeeze_field(arr: np.ndarray) -> np.ndarray:
    return np.squeeze(arr)


def current_profile(curl_bz: np.ndarray) -> np.ndarray:
    field = squeeze_field(curl_bz)
    if field.ndim == 1:
        return np.abs(field)
    if field.ndim == 2:
        return np.mean(np.abs(field), axis=0)
    raise RuntimeError(f"Unexpected curlBz rank after squeeze: {field.ndim}")


def half_width(profile: np.ndarray, x1min: float, x1max: float) -> float:
    ni = profile.size
    dx = (x1max - x1min) / ni
    x = x1min + (np.arange(ni) + 0.5) * dx
    peak_idx = int(np.argmax(profile))
    peak_val = profile[peak_idx]
    if peak_val <= 0.0:
        return float("inf")
    mask = profile >= 0.5 * peak_val
    return float(np.max(np.abs(x[mask] - x[peak_idx])))


def boundary_ratio(curl_bz: np.ndarray) -> float:
    field = squeeze_field(curl_bz)
    profile_2d = np.abs(field) if field.ndim == 2 else np.abs(field[None, :])
    ni = profile_2d.shape[-1]
    edge = max(1, ni // 8)
    center = profile_2d[:, edge:-edge]
    boundary = np.concatenate((profile_2d[:, :edge].ravel(), profile_2d[:, -edge:].ravel()))
    return float(np.max(center) / max(np.max(boundary), 1.0e-30))


def peak_location(curl_bz: np.ndarray, x1min: float, x1max: float) -> float:
    profile = current_profile(curl_bz)
    ni = profile.size
    dx = (x1max - x1min) / ni
    return float(x1min + (np.argmax(profile) + 0.5) * dx)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate reduced MARZ-sheet smoke evolution."
    )
    parser.add_argument("--first", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--x1min", type=float, required=True)
    parser.add_argument("--x1max", type=float, required=True)
    parser.add_argument("--expected-iters", type=int, default=2)
    parser.add_argument("--temperature-floor", type=float, default=1.0e4)
    parser.add_argument("--boundary-ratio-min", type=float, default=2.0)
    args = parser.parse_args()

    first = Path(args.first)
    final = Path(args.final)

    cons_final = parse_dataset(final, "/cons")
    temp_first = parse_dataset(first, "/T")
    temp_final = parse_dataset(final, "/T")
    curl_first = parse_dataset(first, "/curlBz")
    curl_final = parse_dataset(final, "/curlBz")
    eta_final = parse_dataset(final, "/eta")
    iters_final = parse_dataset(final, "/thermal_sdc_iter_count")

    finite_ok = (
        np.all(np.isfinite(cons_final))
        and np.all(np.isfinite(temp_final))
        and np.all(np.isfinite(curl_final))
        and np.all(np.isfinite(eta_final))
        and np.all(np.isfinite(iters_final))
    )
    if not finite_ok:
        raise RuntimeError("Smoke-test output contains non-finite values.")

    rho_final = cons_final[:, 0, :, :, :]
    if np.min(rho_final) <= 0.0:
        raise RuntimeError(f"Density became non-positive: {np.min(rho_final):.3e}")
    if np.min(temp_final) <= 0.0:
        raise RuntimeError(f"Temperature became non-positive: {np.min(temp_final):.3e}")

    peak_first = float(np.max(np.abs(curl_first)))
    peak_final = float(np.max(np.abs(curl_final)))
    width_first = half_width(current_profile(curl_first), args.x1min, args.x1max)
    width_final = half_width(current_profile(curl_final), args.x1min, args.x1max)
    ratio = boundary_ratio(curl_final)
    x_peak_final = peak_location(curl_final, args.x1min, args.x1max)
    x_peak_tol = 0.1 * (args.x1max - args.x1min)

    print(f"curlBz_peak_first={peak_first:.16e}")
    print(f"curlBz_peak_final={peak_final:.16e}")
    print(f"sheet_half_width_first={width_first:.16e}")
    print(f"sheet_half_width_final={width_final:.16e}")
    print(f"temperature_min_first={np.min(temp_first):.16e}")
    print(f"temperature_min_final={np.min(temp_final):.16e}")
    print(f"density_max_final={np.max(rho_final):.16e}")
    print(f"eta_max_final={np.max(eta_final):.16e}")
    print(f"boundary_ratio={ratio:.16e}")
    print(f"peak_location_final={x_peak_final:.16e}")
    print(f"iter_count_min={np.min(iters_final):.16e}")
    print(f"iter_count_max={np.max(iters_final):.16e}")

    if peak_first <= 0.0:
        raise RuntimeError("No current sheet is present in the first dump.")
    if not (peak_final >= 0.95 * peak_first or width_final <= 1.05 * width_first):
        raise RuntimeError(
            "The early sheet neither strengthened nor thinned in the smoke run."
        )
    if np.min(temp_final) <= 1.05 * args.temperature_floor:
        raise RuntimeError(
            "The early smoke run collapsed too close to the temperature floor."
        )
    if ratio < args.boundary_ratio_min:
        raise RuntimeError(
            "Boundary contamination dominates the central layer in the smoke run."
        )
    if abs(x_peak_final) > x_peak_tol:
        raise RuntimeError(
            "The current layer drifted too far from the domain center in the smoke run."
        )
    if np.min(iters_final) != args.expected_iters or np.max(iters_final) != args.expected_iters:
        raise RuntimeError(
            "Unexpected simplified-SDC iteration count range: "
            f"[{np.min(iters_final):.3e}, {np.max(iters_final):.3e}]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
