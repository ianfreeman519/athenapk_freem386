#!/usr/bin/env python3
"""Compute observed convergence orders for the CPAW test."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


COLUMNS = {
    "||dq||": 4,
    "density": 5,
    "M1": 6,
    "M2": 7,
    "M3": 8,
    "E": 9,
    "B1c": 10,
    "B2c": 11,
    "B3c": 12,
}


def default_error_file() -> Path:
    here = Path(__file__).resolve().parent
    return here / "cpaw-errors.dat"


def read_error_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            if len(values) >= max(COLUMNS.values()) + 1:
                rows.append(values)
    rows.sort(key=lambda row: row[0])
    return rows


def observed_order(n0: float, e0: float, n1: float, e1: float) -> float:
    if e0 <= 0.0 or e1 <= 0.0 or n0 <= 0.0 or n1 <= n0:
        return float("nan")
    return math.log(e0 / e1) / math.log(n1 / n0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute observed orders from cpaw-errors.dat."
    )
    parser.add_argument(
        "error_file",
        nargs="?",
        type=Path,
        default=default_error_file(),
        help="Path to the CPAW error .dat file.",
    )
    args = parser.parse_args()

    rows = read_error_rows(args.error_file)
    if len(rows) < 2:
        raise SystemExit(f"Need at least two data rows in {args.error_file}")

    lines: list[str] = []
    lines.append("Circularly polarized Alfven wave")
    lines.append(f"Reading: {args.error_file}")
    lines.append("")
    header = f"{'N0->N1':>12}  " + "  ".join(f"{name:>10}" for name in COLUMNS)
    lines.append(header)
    lines.append("-" * len(header))
    for coarse, fine in zip(rows[:-1], rows[1:]):
        # The standard CPAW setup uses nx1 = 2*N and nx2 = nx3 = N.
        # Report and compute orders using that base resolution N.
        n0 = coarse[1]
        n1 = fine[1]
        orders = [
            observed_order(n0, coarse[col], n1, fine[col]) for col in COLUMNS.values()
        ]
        line = f"{int(n0):5d}->{int(n1):<5d}  "
        line += "  ".join(f"{p:10.4f}" if math.isfinite(p) else f"{'nan':>10}" for p in orders)
        lines.append(line)

    lines.append("")
    lines.append("Errors used:")
    err_header = f"{'N':>8}  " + "  ".join(f"{name:>12}" for name in COLUMNS)
    lines.append(err_header)
    lines.append("-" * len(err_header))
    for row in rows:
        line = f"{int(row[1]):8d}  "
        line += "  ".join(f"{row[col]:12.6e}" for col in COLUMNS.values())
        lines.append(line)

    report = "\n".join(lines)
    print(report)


if __name__ == "__main__":
    main()
