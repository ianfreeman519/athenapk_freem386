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


def profile_along_x2(field: np.ndarray, center_i: int) -> np.ndarray:
    squeezed = np.squeeze(np.abs(field))
    if squeezed.ndim == 1:
        return squeezed
    if squeezed.ndim != 2:
        raise RuntimeError(f"Unexpected squeezed field rank: {squeezed.ndim}")
    return squeezed[:, center_i]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MARZ-like array-source initial-state setup."
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--x2min", type=float, required=True)
    parser.add_argument("--x2max", type=float, required=True)
    parser.add_argument("--array-separation", type=float, required=True)
    parser.add_argument("--expected-drive-amplitude", type=float, required=True)
    parser.add_argument("--symmetry-tol", type=float, default=5.0e-2)
    parser.add_argument("--position-tol", type=float, default=2.0e-2)
    parser.add_argument("--midplane-current-max-fraction", type=float, default=0.35)
    args = parser.parse_args()

    path = Path(args.file)
    cons = parse_dataset(path, "/cons")
    prim = parse_dataset(path, "/prim")
    temp = parse_dataset(path, "/T")
    curl_bz = parse_dataset(path, "/curlBz")

    rho = cons[:, 0, :, :, :]
    pressure = prim[:, 4, :, :, :]

    finite_ok = (
        np.all(np.isfinite(cons))
        and np.all(np.isfinite(prim))
        and np.all(np.isfinite(temp))
        and np.all(np.isfinite(curl_bz))
    )
    if not finite_ok:
        raise RuntimeError("Initial-state dump contains non-finite values.")
    if np.min(rho) <= 0.0:
        raise RuntimeError(f"Density is non-positive: {np.min(rho):.6e}")
    if np.min(pressure) <= 0.0:
        raise RuntimeError(f"Pressure is non-positive: {np.min(pressure):.6e}")
    if np.min(temp) <= 0.0:
        raise RuntimeError(f"Temperature is non-positive: {np.min(temp):.6e}")

    ni = cons.shape[-1]
    nj = cons.shape[-2]
    x2 = args.x2min + (np.arange(nj) + 0.5) * (args.x2max - args.x2min) / nj
    center_i = ni // 2

    rho_line = np.mean(rho[..., center_i], axis=tuple(range(rho[..., center_i].ndim - 1)))
    curl_line = profile_along_x2(curl_bz, center_i)

    lower_idx = int(np.argmax(rho_line[: nj // 2]))
    upper_idx = nj // 2 + int(np.argmax(rho_line[nj // 2 :]))
    lower_pos = float(x2[lower_idx])
    upper_pos = float(x2[upper_idx])
    expected_half_sep = 0.5 * args.array_separation
    mirrored = rho_line[::-1]
    symmetry_err = float(
        np.linalg.norm(rho_line - mirrored) / max(np.linalg.norm(rho_line), 1.0e-30)
    )

    peak_curl = float(np.max(np.abs(curl_bz)))
    lower_curl_idx = int(np.argmax(curl_line[: nj // 2]))
    upper_curl_idx = nj // 2 + int(np.argmax(curl_line[nj // 2 :]))
    lower_curl_pos = float(x2[lower_curl_idx])
    upper_curl_pos = float(x2[upper_curl_idx])
    mid_idx = nj // 2
    midplane_curl = float(curl_line[mid_idx])
    wire_curl_peak = float(max(curl_line[lower_curl_idx], curl_line[upper_curl_idx]))

    print(f"rho_min={np.min(rho):.16e}")
    print(f"pressure_min={np.min(pressure):.16e}")
    print(f"temperature_min={np.min(temp):.16e}")
    print(f"lower_array_x2={lower_pos:.16e}")
    print(f"upper_array_x2={upper_pos:.16e}")
    print(f"array_symmetry_rel_l2={symmetry_err:.16e}")
    print(f"curlBz_peak_abs={peak_curl:.16e}")
    print(f"lower_current_x2={lower_curl_pos:.16e}")
    print(f"upper_current_x2={upper_curl_pos:.16e}")
    print(f"midplane_current_abs={midplane_curl:.16e}")
    print(f"wire_current_peak_abs={wire_curl_peak:.16e}")
    print(f"drive_amplitude_expected={args.expected_drive_amplitude:.16e}")

    if abs(lower_pos + expected_half_sep) > args.position_tol:
        raise RuntimeError(
            f"Lower array is misplaced: x2={lower_pos:.3e}, expected {-expected_half_sep:.3e}"
        )
    if abs(upper_pos - expected_half_sep) > args.position_tol:
        raise RuntimeError(
            f"Upper array is misplaced: x2={upper_pos:.3e}, expected {expected_half_sep:.3e}"
        )
    if symmetry_err > args.symmetry_tol:
        raise RuntimeError(
            f"Array symmetry exceeded tolerance: {symmetry_err:.3e} > {args.symmetry_tol:.3e}"
        )

    if args.expected_drive_amplitude > 0.0:
        if peak_curl <= 0.0:
            raise RuntimeError("Driven initial state should contain non-zero wire current.")
        if abs(lower_curl_pos + expected_half_sep) > args.position_tol:
            raise RuntimeError(
                "Lower current peak is not wire-localized in the initial state."
            )
        if abs(upper_curl_pos - expected_half_sep) > args.position_tol:
            raise RuntimeError(
                "Upper current peak is not wire-localized in the initial state."
            )
        if midplane_curl > args.midplane_current_max_fraction * max(wire_curl_peak, 1.0e-30):
            raise RuntimeError("Initial midplane current is too strong for a localized source.")
    else:
        if peak_curl > 1.0e-10:
            raise RuntimeError("Zero-drive initial state should not contain seeded current.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
