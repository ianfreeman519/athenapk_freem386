#!/usr/bin/env python3
"""Validate Ohmic heating, current-sheet scaling, CT divergence, and UCT parity."""

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

GAMMA = 5.0 / 3.0
TWO_PI = 2.0 * math.pi


def keyed_path(value: str) -> tuple[str, Path]:
    key, path = value.split("=", 1)
    return key, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heating", action="append", type=keyed_path, required=True)
    parser.add_argument("--reconnection", action="append", type=keyed_path, required=True)
    parser.add_argument("--parthenon-tools", type=Path, required=True)
    parser.add_argument("--heating-eta", type=float, required=True)
    parser.add_argument("--heating-tlim", type=float, required=True)
    parser.add_argument("--reconnection-tlim", type=float, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--pass-marker", type=Path, required=True)
    return parser.parse_args()


def volumes(data) -> np.ndarray:
    return np.einsum("ai,aj,ak->aijk", np.diff(data.zf), np.diff(data.yf), np.diff(data.xf))


def primitive(data) -> dict[str, np.ndarray]:
    names = [
        "prim_density", "prim_pressure", "prim_velocity_1", "prim_velocity_2",
        "prim_velocity_3", "prim_magnetic_field_1", "prim_magnetic_field_2",
        "prim_magnetic_field_3",
    ]
    return data.GetComponents(names, flatten=False)


def energies(data) -> tuple[float, float, float, float]:
    q = primitive(data)
    vol = volumes(data)
    rho = q["prim_density"]
    thermal = float(np.sum(q["prim_pressure"] / (GAMMA - 1.0) * vol))
    kinetic = float(np.sum(0.5 * rho * sum(q[f"prim_velocity_{d}"] ** 2 for d in (1, 2, 3)) * vol))
    magnetic = float(np.sum(0.5 * sum(q[f"prim_magnetic_field_{d}"] ** 2 for d in (1, 2, 3)) * vol))
    return thermal, kinetic, magnetic, thermal + kinetic + magnetic


def max_face_divb(history: Path) -> float:
    # maxFaceDivB is the last registered CT history variable.
    values = np.genfromtxt(history)
    values = np.atleast_2d(values)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.max(np.abs(values[:, -1])))


def final_file(directory: Path, prefix: str) -> Path:
    path = directory / f"parthenon.{prefix}.final.phdf"
    if not path.is_file():
        raise SystemExit(f"missing output: {path}")
    return path


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.parthenon_tools))
    try:
        import phdf
    except ModuleNotFoundError as exc:
        raise SystemExit(f"could not import phdf from {args.parthenon_tools}") from exc

    heating_dirs = dict(args.heating)
    recon_dirs = dict(args.reconnection)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    heating_metrics: dict[str, dict[str, float]] = {}
    rates: dict[str, list[tuple[float, float]]] = {}

    for fluid, directory in heating_dirs.items():
        prefix = f"heating_{fluid}"
        initial = phdf.phdf(str(directory / f"parthenon.{prefix}.00000.phdf"))
        final = phdf.phdf(str(final_file(directory, prefix)))
        if not all(np.all(np.isfinite(value)) for value in primitive(final).values()):
            failures.append(f"{fluid} heating contains non-finite primitives")
        th0, ke0, me0, et0 = energies(initial)
        th1, ke1, me1, et1 = energies(final)
        magnetic_loss = me0 - me1
        thermal_gain = th1 - th0
        resolved_gain = thermal_gain + (ke1 - ke0)
        closure = abs(resolved_gain - magnetic_loss) / max(abs(magnetic_loss), 1.0e-300)
        thermal_fraction = thermal_gain / max(abs(magnetic_loss), 1.0e-300)
        total_error = abs(et1 - et0) / max(abs(et0), 1.0e-300)
        expected_ratio = math.exp(-2.0 * args.heating_eta * TWO_PI**2 * args.heating_tlim)
        decay_error = abs(me1 / me0 - expected_ratio) / expected_ratio
        divb = max_face_divb(directory / "parthenon.out1.hst")
        heating_metrics[fluid] = {"thermal_gain": thermal_gain, "magnetic_loss": magnetic_loss}
        if magnetic_loss <= 0.0 or thermal_gain <= 0.0:
            failures.append(f"{fluid} heating did not convert magnetic energy to thermal energy")
        if closure > 0.05 or thermal_fraction < 0.90:
            failures.append(f"{fluid} heating closure/fraction failed: {closure:g}, {thermal_fraction:g}")
        if total_error > 5.0e-5 or decay_error > 0.08:
            failures.append(f"{fluid} heating conservation/decay failed: {total_error:g}, {decay_error:g}")
        if divb > 1.0e-10:
            failures.append(f"{fluid} heating maxFaceDivB={divb:g}")
        rows.append({"test": "heating", "fluid": fluid, "eta": args.heating_eta,
                     "metric": thermal_fraction, "analytic_error": decay_error,
                     "energy_error": total_error, "max_face_divb": divb})

    for key, directory in recon_dirs.items():
        fluid, eta_text = key.split(",", 1)
        eta = float(eta_text)
        label = f"eta{eta:g}".replace(".", "p")
        prefix = f"reconnection_{fluid}_{label}"
        initial = phdf.phdf(str(directory / f"parthenon.{prefix}.00000.phdf"))
        final = phdf.phdf(str(final_file(directory, prefix)))
        q0, q1 = primitive(initial), primitive(final)
        if not all(np.all(np.isfinite(value)) for value in q1.values()):
            failures.append(f"{fluid}, eta={eta:g} reconnection contains non-finite primitives")
        vol0, vol1 = volumes(initial), volumes(final)
        unsigned0 = float(np.sum(np.abs(q0["prim_magnetic_field_2"]) * vol0))
        unsigned1 = float(np.sum(np.abs(q1["prim_magnetic_field_2"]) * vol1))
        # Two periodic sheets each remove 4 B sqrt(eta*t/pi) of unsigned flux.
        rate = (unsigned0 - unsigned1) / (8.0 * args.reconnection_tlim)
        expected = math.sqrt(eta / (math.pi * args.reconnection_tlim))
        analytic_error = abs(rate / expected - 1.0)
        divb = max_face_divb(directory / "parthenon.out1.hst")
        rates.setdefault(fluid, []).append((eta, rate))
        if rate <= 0.0 or analytic_error > 0.25:
            failures.append(f"{fluid}, eta={eta:g} current-sheet rate error={analytic_error:g}")
        if divb > 1.0e-10:
            failures.append(f"{fluid}, eta={eta:g} maxFaceDivB={divb:g}")
        rows.append({"test": "reconnection", "fluid": fluid, "eta": eta,
                     "metric": rate, "analytic_error": analytic_error,
                     "energy_error": float("nan"), "max_face_divb": divb})

    slopes: dict[str, float] = {}
    for fluid, entries in rates.items():
        eta, rate = np.asarray(sorted(entries), dtype=float).T
        slope = float(np.polyfit(np.log(eta), np.log(rate), 1)[0])
        slopes[fluid] = slope
        if not 0.40 <= slope <= 0.60:
            failures.append(f"{fluid} reconnection exponent={slope:g}, expected 0.5")

    fluids = sorted(heating_metrics)
    if len(fluids) == 2:
        left, right = fluids
        for metric in ("thermal_gain", "magnetic_loss"):
            a, b = heating_metrics[left][metric], heating_metrics[right][metric]
            if abs(a - b) / max(abs(a), abs(b), 1.0e-300) > 0.05:
                failures.append(f"cross-fluid heating mismatch in {metric}")
        for (eta_a, rate_a), (eta_b, rate_b) in zip(sorted(rates[left]), sorted(rates[right])):
            if eta_a != eta_b or abs(rate_a - rate_b) / max(abs(rate_a), abs(rate_b)) > 0.10:
                failures.append(f"cross-fluid reconnection mismatch at eta={eta_a:g}")

    for path in (args.output_csv, args.output_summary, args.output_plot, args.pass_marker):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["Extended resistive CT validation", ""]
    lines += [f"{fluid}: reconnection exponent={slope:.6f} (required 0.40--0.60)" for fluid, slope in slopes.items()]
    lines += ["", *("FAIL: " + item for item in failures)] if failures else ["", "PASS"]
    summary = "\n".join(lines) + "\n"
    args.output_summary.write_text(summary, encoding="utf-8")
    print(summary, end="")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for fluid, entries in sorted(rates.items()):
        eta, rate = np.asarray(sorted(entries), dtype=float).T
        axes[0].loglog(eta, rate, "o-", label=f"{fluid} ({slopes[fluid]:.3f})")
    eta_ref = np.asarray(sorted({eta for entries in rates.values() for eta, _ in entries}))
    axes[0].loglog(eta_ref, np.sqrt(eta_ref / (math.pi * args.reconnection_tlim)), "k--", label="analytic")
    axes[0].set(xlabel="eta", ylabel="effective reconnection rate")
    axes[0].legend()
    axes[0].grid(True, which="both")
    heating_fluids = list(heating_metrics)
    axes[1].bar(
        heating_fluids,
        [heating_metrics[fluid]["thermal_gain"] for fluid in heating_fluids],
    )
    axes[1].set(ylabel="thermal-energy gain", title="Ohmic heating")
    fig.tight_layout()
    fig.savefig(args.output_plot)
    plt.close(fig)

    if failures:
        raise SystemExit("; ".join(failures))
    args.pass_marker.write_text("PASS\n", encoding="utf-8")


if __name__ == "__main__":
    main()
