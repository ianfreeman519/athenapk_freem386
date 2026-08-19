#!/usr/bin/env python3
"""Analyze the analytic Ohmic-diffusion matrix used by the resistive CT workflow."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt


CASES = {
    "uniform_16_16_16": ("uniform", 0, 16, 16, 16),
    "sin1d_16_4_1": ("sin1d", 1, 16, 4, 1),
    "sin1d_32_4_1": ("sin1d", 1, 32, 4, 1),
    "sin1d_64_4_1": ("sin1d", 1, 64, 4, 1),
    "rot_jy_64_4_4": ("rot_jy", 2, 64, 4, 4),
    "rot_jx_4_64_4": ("rot_jx", 3, 4, 64, 4),
    "fourier2d_16_16_1": ("fourier2d", 10, 16, 16, 1),
    "fourier2d_32_32_1": ("fourier2d", 10, 32, 32, 1),
    "fourier2d_64_64_1": ("fourier2d", 10, 64, 64, 1),
    "abc3d_8_8_8": ("abc3d", 20, 8, 8, 8),
    "abc3d_16_16_16": ("abc3d", 20, 16, 16, 16),
}

MINIMUM_RATES = {"sin1d": 1.8, "fourier2d": 1.8, "abc3d": 1.6}
TWO_PI = 2.0 * math.pi


def parse_input(value: str) -> tuple[str, Path]:
    try:
        name, filename = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("inputs must have the form CASE=FILE") from exc
    if name not in CASES:
        raise argparse.ArgumentTypeError(f"unknown resistive CT case: {name}")
    return name, Path(filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--parthenon-tools", type=Path, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--tlim", type=float, required=True)
    parser.add_argument("--amplitude", type=float, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--pass-marker", type=Path, required=True)
    return parser.parse_args()


def cell_volumes(data_file) -> np.ndarray:
    return np.einsum(
        "ai,aj,ak->aijk",
        np.diff(data_file.zf),
        np.diff(data_file.yf),
        np.diff(data_file.xf),
    )


def weighted_l1(error: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.abs(error) * weights) / np.sum(weights))


def convergence_rate(entries: list[tuple[int, float]]) -> float:
    entries = sorted(entries)
    resolutions, errors = np.asarray(entries, dtype=float).T
    if np.any(errors <= 0.0) or np.any(~np.isfinite(errors)):
        return float("nan")
    return float(-np.polyfit(np.log(resolutions), np.log(errors), 1)[0])


def main() -> None:
    args = parse_args()
    if args.eta <= 0.0 or args.tlim <= 0.0 or args.amplitude <= 0.0:
        raise SystemExit("eta, tlim, and amplitude must all be positive")

    supplied = dict(args.input)
    missing = set(CASES) - set(supplied)
    extra = set(supplied) - set(CASES)
    if missing or extra:
        raise SystemExit(f"input matrix mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    missing_files = [str(path) for path in supplied.values() if not path.is_file()]
    if missing_files:
        raise SystemExit("missing PHDF inputs: " + ", ".join(missing_files))

    sys.path.insert(0, str(args.parthenon_tools))
    try:
        import phdf
    except ModuleNotFoundError as exc:
        raise SystemExit(f"could not import phdf from {args.parthenon_tools}") from exc

    decay_1d = math.exp(-args.eta * TWO_PI**2 * args.tlim)
    decay_2d = math.exp(-args.eta * 2.0 * TWO_PI**2 * args.tlim)
    rows: list[dict[str, object]] = []
    convergence: dict[str, list[tuple[int, float]]] = {
        "sin1d": [],
        "fourier2d": [],
        "abc3d": [],
    }
    failures: list[str] = []

    for case, (family, iprob, nx1, nx2, nx3) in CASES.items():
        data = phdf.phdf(str(supplied[case]))
        components = data.GetComponents(
            [
                "prim_magnetic_field_1",
                "prim_magnetic_field_2",
                "prim_magnetic_field_3",
            ],
            flatten=False,
        )
        bx = components["prim_magnetic_field_1"]
        by = components["prim_magnetic_field_2"]
        bz = components["prim_magnetic_field_3"]
        z, y, x = data.GetVolumeLocations(flatten=False)
        volume = cell_volumes(data)
        normalized_l1 = float("nan")
        amplitude_error = float("nan")
        phase_error = float("nan")
        passed = True

        if iprob == 0:
            normalized_l1 = max(
                float(np.max(np.abs(bx - 0.25))),
                float(np.max(np.abs(by + 0.125))),
                float(np.max(np.abs(bz - 0.0625))),
            )
            passed = normalized_l1 < 5.0e-13
        elif iprob in (1, 2, 3):
            coordinate = x if iprob in (1, 2) else y
            phase = np.sin(TWO_PI * coordinate)
            quadrature = np.cos(TWO_PI * coordinate)
            field = by if iprob == 1 else bz
            expected = args.amplitude * decay_1d * phase
            normalized_l1 = weighted_l1(field - expected, volume) / args.amplitude
            sine_amplitude = np.sum(field * phase * volume) / np.sum(phase**2 * volume)
            cosine_amplitude = np.sum(field * quadrature * volume) / np.sum(
                quadrature**2 * volume
            )
            amplitude_error = abs(sine_amplitude / (args.amplitude * decay_1d) - 1.0)
            phase_error = abs(cosine_amplitude) / args.amplitude
            if family == "sin1d":
                convergence[family].append((nx1, normalized_l1))
            else:
                passed = amplitude_error < 2.0e-2 and phase_error < 2.0e-3
        elif iprob == 10:
            expected_bx = (
                args.amplitude * TWO_PI * np.sin(TWO_PI * x) * np.cos(TWO_PI * y)
            )
            expected_by = (
                -args.amplitude * TWO_PI * np.cos(TWO_PI * x) * np.sin(TWO_PI * y)
            )
            scale = args.amplitude * TWO_PI
            normalized_l1 = (
                weighted_l1(bx - decay_2d * expected_bx, volume)
                + weighted_l1(by - decay_2d * expected_by, volume)
            ) / (2.0 * scale)
            convergence[family].append((nx1, normalized_l1))
        else:
            expected_bx = args.amplitude * (
                np.sin(TWO_PI * z) + np.cos(TWO_PI * y)
            )
            expected_by = args.amplitude * (
                np.sin(TWO_PI * x) + np.cos(TWO_PI * z)
            )
            expected_bz = args.amplitude * (
                np.sin(TWO_PI * y) + np.cos(TWO_PI * x)
            )
            normalized_l1 = (
                weighted_l1(bx - decay_1d * expected_bx, volume)
                + weighted_l1(by - decay_1d * expected_by, volume)
                + weighted_l1(bz - decay_1d * expected_bz, volume)
            ) / (3.0 * args.amplitude)
            convergence[family].append((nx1, normalized_l1))

        if not passed:
            failures.append(f"{case}: per-case tolerance failed")
        rows.append(
            {
                "case": case,
                "family": family,
                "iprob": iprob,
                "nx1": nx1,
                "nx2": nx2,
                "nx3": nx3,
                "normalized_l1": normalized_l1,
                "amplitude_error": amplitude_error,
                "phase_error": phase_error,
                "passed": passed,
            }
        )

    rates = {family: convergence_rate(entries) for family, entries in convergence.items()}
    for family, minimum in MINIMUM_RATES.items():
        rate = rates[family]
        if not math.isfinite(rate) or rate < minimum:
            failures.append(
                f"{family}: convergence rate {rate:.6g} is below required {minimum}"
            )

    for output in (args.output_csv, args.output_summary, args.output_plot, args.pass_marker):
        output.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        "Resistive CT analytic test summary",
        f"eta={args.eta:g}, tlim={args.tlim:g}, amplitude={args.amplitude:g}",
        "",
    ]
    for row in rows:
        summary_lines.append(
            f"{row['case']}: L1={row['normalized_l1']:.8e}, "
            f"amplitude_error={row['amplitude_error']:.8e}, "
            f"phase_error={row['phase_error']:.8e}, passed={row['passed']}"
        )
    summary_lines.append("")
    for family, rate in rates.items():
        summary_lines.append(
            f"{family}: convergence_rate={rate:.6f}, minimum={MINIMUM_RATES[family]:.2f}"
        )
    summary_lines.append("")
    summary_lines.extend(["FAIL: " + failure for failure in failures] or ["PASS"])
    summary = "\n".join(summary_lines) + "\n"
    args.output_summary.write_text(summary, encoding="utf-8")
    print(summary, end="")

    figure, axis = plt.subplots()
    for family, entries in convergence.items():
        resolutions, errors = np.asarray(sorted(entries), dtype=float).T
        axis.loglog(resolutions, errors, "o-", label=family)
    axis.set_xlabel("linear resolution")
    axis.set_ylabel("normalized volume-weighted L1 error")
    axis.grid(True, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_plot)
    plt.close(figure)

    if failures:
        raise SystemExit("; ".join(failures))
    args.pass_marker.write_text("PASS\n", encoding="utf-8")


if __name__ == "__main__":
    main()
