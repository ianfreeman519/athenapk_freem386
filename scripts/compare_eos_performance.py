#!/usr/bin/env python3
"""Paired, alternating upstream/candidate timestep benchmark (CPU or GPU).

Run on an exclusive allocation, using identically configured Release binaries.
Exit 0: all upper confidence bounds <1%; 1: regression; 2: inconclusive/noisy.
Raw stdout, inputs, metadata, paired measurements, and statistics are retained.
"""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Confirm exclusive hardware and controlled runtime settings",
    )
    parser.add_argument(
        "--fluids", nargs="+", choices=["euler", "glmmhd"], default=["euler", "glmmhd"]
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        type=int,
        help="Meshblock sizes (default: size and size/2)",
    )
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.pairs < 10 or args.size <= 0 or args.size % 2 or args.steps < 10:
        parser.error("Require >=10 pairs, an even size, and >=10 steps")
    blocks = args.blocks or [args.size, args.size // 2]
    if any(b <= 0 or args.size % b for b in blocks):
        parser.error("Every block size must be a positive divisor of size")
    binaries = {k: str(getattr(args, k).resolve()) for k in ["baseline", "candidate"]}
    report = {
        "backend": args.backend,
        "host": platform.node(),
        "platform": platform.platform(),
        "binaries": binaries,
        "input": str(args.input.resolve()),
        "steps": args.steps,
        "affinity": sorted(os.sched_getaffinity(0)),
        "exclusive": args.exclusive,
        "sha256": {
            k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
            for k, v in binaries.items()
        },
        "cases": [],
    }
    report["build_cache"] = {}
    for kind, binary in binaries.items():
        cache = Path(binary).parent.parent / "CMakeCache.txt"
        if cache.exists():
            report["build_cache"][kind] = [
                line
                for line in cache.read_text().splitlines()
                if re.match(
                    r"(?:CMAKE_(?:CXX_COMPILER|CXX_FLAGS|BUILD_TYPE)|Kokkos_|PARTHENON_)",
                    line,
                )
            ]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logdir = args.output.with_suffix(".logs")
    logdir.mkdir(exist_ok=True)
    status = 0
    for fluid in args.fluids:
        for block in blocks:
            case = {"fluid": fluid, "block": block, "runs": []}
            overrides = [
                f"hydro/fluid={fluid}",
                "hydro/reconstruction=plm",
                f"hydro/riemann={'hllc' if fluid == 'euler' else 'hlld'}",
                "parthenon/time/integrator=vl2",
                "parthenon/time/ncycle_out=0",
                "parthenon/time/perf_cycle_offset=5",
                f"parthenon/time/nlim={args.steps}",
                "parthenon/time/tlim=1e20",
                "parthenon/mesh/refinement=none",
                "problem/linear_wave/compute_error=false",
            ]
            overrides += [
                f"{section}/dt=-1"
                for section in re.findall(
                    r"<([^>]*output[^>]*)>", args.input.read_text()
                )
            ]
            for axis in [1, 2, 3]:
                overrides += [
                    f"parthenon/mesh/nx{axis}={args.size}",
                    f"parthenon/meshblock/nx{axis}={block}",
                ]
            case["overrides"] = overrides
            # One full warm-up of each executable; alternating order thereafter.
            for pair in range(-1, args.pairs):
                order = (
                    ["baseline", "candidate"]
                    if pair % 2 == 0
                    else ["candidate", "baseline"]
                )
                result = {}
                for kind in order:
                    with tempfile.TemporaryDirectory(
                        prefix="athenapk-eos-perf-"
                    ) as tmp:
                        proc = subprocess.run(
                            [
                                binaries[kind],
                                "-i",
                                str(args.input.resolve()),
                                *overrides,
                            ],
                            cwd=tmp,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    label = f"{fluid}-b{block}-p{pair}-{kind}"
                    (logdir / f"{label}.log").write_text(proc.stdout)
                    proc.check_returncode()
                    rate = re.search(
                        r"zone-cycles/wallsecond\s*=\s*([\d.eE+-]+)", proc.stdout
                    )
                    elapsed = re.search(
                        r"walltime used\s*=\s*([\d.eE+-]+)", proc.stdout
                    )
                    if not rate:
                        raise RuntimeError(f"No throughput in {label}; inspect log")
                    result[kind] = {
                        "rate": float(rate[1]),
                        "seconds": float(elapsed[1]) if elapsed else None,
                    }
                if pair >= 0:
                    case["runs"].append(result)
            logs = [
                math.log(r["baseline"]["rate"] / r["candidate"]["rate"])
                for r in case["runs"]
            ]
            # Conservative two-sided 95% Student-t critical value for >=10 pairs.
            margin = 2.262 * statistics.stdev(logs) / math.sqrt(len(logs))
            mean = statistics.mean(logs)
            case.update(
                slowdown=math.expm1(mean),
                lower95=math.expm1(mean - margin),
                upper95=math.expm1(mean + margin),
            )
            short = any(
                r[k]["seconds"] is None or r[k]["seconds"] < args.min_seconds
                for r in case["runs"]
                for k in binaries
            )
            case["verdict"] = (
                "inconclusive"
                if short or not args.exclusive
                else (
                    "pass"
                    if case["upper95"] < 0.01
                    else "regression" if case["lower95"] >= 0.01 else "inconclusive"
                )
            )
            if case["verdict"] != "pass":
                status = max(status, 1 if case["verdict"] == "regression" else 2)
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                fluid,
                block,
                case["verdict"],
                f"slowdown={case['slowdown']:.2%} 95%CI=[{case['lower95']:.2%}, {case['upper95']:.2%}]",
                flush=True,
            )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
