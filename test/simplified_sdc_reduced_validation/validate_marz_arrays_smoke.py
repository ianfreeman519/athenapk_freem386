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


def x2_profile(field: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(np.abs(field))
    if squeezed.ndim == 1:
        return squeezed
    if squeezed.ndim != 2:
        raise RuntimeError(f"Unexpected squeezed field rank: {squeezed.ndim}")
    return np.mean(squeezed, axis=1)


def x2_coords(nj: int, x2min: float, x2max: float) -> np.ndarray:
    return x2min + (np.arange(nj) + 0.5) * (x2max - x2min) / nj


def wire_peak_locations(profile: np.ndarray, coords: np.ndarray) -> tuple[float, float]:
    nj = profile.size
    lower_idx = int(np.argmax(profile[: nj // 2]))
    upper_idx = nj // 2 + int(np.argmax(profile[nj // 2 :]))
    return float(coords[lower_idx]), float(coords[upper_idx])


def center_to_edge_ratio(field: np.ndarray) -> float:
    squeezed = np.squeeze(np.abs(field))
    if squeezed.ndim == 1:
        squeezed = squeezed[:, None]
    ni = squeezed.shape[-1]
    edge = max(1, ni // 8)
    center = squeezed[:, edge:-edge] if edge < ni // 2 else squeezed
    boundary = np.concatenate((squeezed[:, :edge].ravel(), squeezed[:, -edge:].ravel()))
    return float(np.max(center) / max(np.max(boundary), 1.0e-30))


def min_location(field: np.ndarray, x2min: float, x2max: float) -> float:
    squeezed = np.squeeze(field)
    if squeezed.ndim == 1:
        profile = squeezed
    else:
        finite = np.where(np.isfinite(squeezed), squeezed, np.inf)
        profile = np.min(finite, axis=-1)
    coords = x2_coords(profile.size, x2min, x2max)
    return float(coords[int(np.argmin(profile))])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MARZ-like array-source smoke evolution."
    )
    parser.add_argument("--first", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--x2min", type=float, required=True)
    parser.add_argument("--x2max", type=float, required=True)
    parser.add_argument("--array-separation", type=float, required=True)
    parser.add_argument("--expected-iters", type=int, default=2)
    parser.add_argument("--temperature-floor", type=float, default=1.0e4)
    parser.add_argument("--boundary-ratio-min", type=float, default=1.5)
    parser.add_argument("--position-tol", type=float, default=2.5e-2)
    parser.add_argument("--midplane-current-growth", type=float, default=1.1)
    args = parser.parse_args()

    first = Path(args.first)
    final = Path(args.final)

    cons_final = parse_dataset(final, "/cons")
    temp_first = parse_dataset(first, "/T")
    temp_final = parse_dataset(final, "/T")
    curl_first = parse_dataset(first, "/curlBz")
    curl_final = parse_dataset(final, "/curlBz")
    eta_final = parse_dataset(final, "/eta")
    dt_heat_final = parse_dataset(final, "/dt_heat_local")
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
    if not np.isfinite(dt_heat_final).any():
        raise RuntimeError("No finite local heating timestep was recorded in the smoke run.")

    profile_first = x2_profile(curl_first)
    profile_final = x2_profile(curl_final)
    coords = x2_coords(profile_final.size, args.x2min, args.x2max)
    peak_first = float(np.max(np.abs(curl_first)))
    peak_final = float(np.max(np.abs(curl_final)))
    lower_first, upper_first = wire_peak_locations(profile_first, coords)
    lower_final, upper_final = wire_peak_locations(profile_final, coords)
    ratio = center_to_edge_ratio(curl_final)
    half_sep = 0.5 * args.array_separation
    dt_heat_min_loc = min_location(dt_heat_final, args.x2min, args.x2max)
    mid_idx = profile_final.size // 2
    mid_first = float(profile_first[mid_idx])
    mid_final = float(profile_final[mid_idx])

    print(f"curlBz_peak_first={peak_first:.16e}")
    print(f"curlBz_peak_final={peak_final:.16e}")
    print(f"lower_wire_current_first={lower_first:.16e}")
    print(f"upper_wire_current_first={upper_first:.16e}")
    print(f"lower_wire_current_final={lower_final:.16e}")
    print(f"upper_wire_current_final={upper_final:.16e}")
    print(f"midplane_current_first={mid_first:.16e}")
    print(f"midplane_current_final={mid_final:.16e}")
    print(f"temperature_min_first={np.min(temp_first):.16e}")
    print(f"temperature_min_final={np.min(temp_final):.16e}")
    print(f"density_max_final={np.max(rho_final):.16e}")
    print(f"eta_max_final={np.max(eta_final):.16e}")
    print(f"dt_heat_min_x2_final={dt_heat_min_loc:.16e}")
    print(f"boundary_to_center_current_ratio={ratio:.16e}")
    print(f"iter_count_min={np.min(iters_final):.16e}")
    print(f"iter_count_max={np.max(iters_final):.16e}")

    if peak_final <= 0.0:
        raise RuntimeError("No current developed in the smoke run.")
    if peak_final < 0.95 * peak_first:
        raise RuntimeError("The localized source failed to sustain its early current.")
    if np.min(temp_final) <= 1.05 * args.temperature_floor:
        raise RuntimeError("The smoke run collapsed too close to the temperature floor.")
    if ratio < args.boundary_ratio_min:
        raise RuntimeError("Boundary contamination dominates the evolved current profile.")
    if abs(lower_first + half_sep) > args.position_tol or abs(upper_first - half_sep) > args.position_tol:
        raise RuntimeError("The initial current is not wire-localized.")
    if mid_final < args.midplane_current_growth * mid_first:
        raise RuntimeError("The midplane current did not grow beyond the wire-seeded state.")
    if np.min(iters_final) != args.expected_iters or np.max(iters_final) != args.expected_iters:
        raise RuntimeError(
            "Unexpected simplified-SDC iteration count range: "
            f"[{np.min(iters_final):.3e}, {np.max(iters_final):.3e}]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
