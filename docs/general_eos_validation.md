# General EOS validation

The ideal-gas integration is implemented on `general-eos-support`. It is not yet
accepted against the requested performance and regression gates: CPU overhead
exceeds 1%, and the existing turbulence reference does not match the generalized
wave estimates. No tabulated material is exposed in runtime inputs.

## Provenance

The baseline is official AthenaPK upstream commit
`99dd6f4bd675f02412813fd487a91c9f357f259a`, rather than the user fork.
Both builds use Parthenon `e299e2c1cdfac588ebc46b805af4b67e2370d773`
and Kokkos `8e15454876298832f013ebcaf08d53825b5bda40`.
Singularity-EOS is pinned to `6bbe9cd1f2da13760181813a16038e9a67f11a42`.

Measurements used Release builds, GCC 12.3, CUDA 12.4 for GPU builds,
CMake 3.26.3, OpenMPI 4.1.5, and HDF5 1.14 on `dev-amd20-v100`.
The CPU was an Intel Xeon Platinum 8260; the GPU was a V100S PCIe 32 GB.
The node was not reserved. Consequently, all performance-runner acceptance
verdicts are inconclusive, even where the measured confidence interval is small.

## Correctness

* Focused EOS tests passed in CPU double precision, CPU single precision, and
  on the V100. These include device execution of the stiffened-gas fixture,
  thermodynamic derivatives, and conserved-to-primitive safeguards.
* The new continuous-versus-restarted MHD test passed on serial CPU, two MPI
  ranks, and the V100. It compares all conserved fields after 20 cycles.
* The GPU serial regression matrix passed 17 of 18 tests. The remaining test
  is the legacy turbulence reference comparison.
* The CPU matrix passed 12 of 18 tests; the other six reached their 600-second
  timeouts. These were turbulence, hydro convergence, MHD convergence, conduction
  ring convergence, multidimensional conduction ring, and field loop. Timed-out
  tests are unresolved, not passes. The two-rank hydro Riemann suite passed.
* Invalid gamma and invalid unprotected pressure were rejected; an enabled
  pressure floor recovered the tested invalid internal-energy state.

The general EOS solver uses Davis wave bounds in hydro HLLC and hydro/MHD HLLE.
These alter turbulence trajectories. Relative to unchanged upstream, the tested
candidate's final sonic Mach number changed by about -0.0028%, Alfven Mach number
by -0.143%, and total energy by -4.1e-9; total mass agreed to roundoff. Both runs
retained 512 distinct tracers. These aggregate comparisons do not establish
equivalence of the tracer trajectories.

Unchanged upstream also missed the strict two-ULP tracer reference on this
toolchain, with much smaller differences. That does not explain away the larger
candidate differences. The legacy reference and tolerances remain unchanged.

The full GPU matrix preceded the final constructor and disabled-velocity-ceiling
cleanup. The final focused EOS and restart tests passed, and the final turbulence
conserved fields were bit-for-bit identical to the pre-cleanup candidate.

## Whole-step performance

The final comparison alternated baseline/candidate order over ten paired runs,
with warmups and output disabled. Positive values below mean slowdown. The
intervals are paired 95% confidence intervals reported by the runner.

| Backend | Fluid / solver | Mesh block | Slowdown | 95% interval |
| --- | --- | --- | --- | --- |
| CPU | Euler / HLLC | 32 cubed | 1.47% | 1.25% to 1.69% |
| CPU | Euler / HLLC | 16 cubed | 1.58% | 0.92% to 2.25% |
| CPU | GLMMHD / HLLD | 32 cubed | 2.09% | 1.67% to 2.51% |
| CPU | GLMMHD / HLLD | 16 cubed | 2.19% | 1.86% to 2.52% |
| GPU | Euler / HLLC | 64 cubed | -3.53% | -3.59% to -3.46% |
| GPU | Euler / HLLC | 32 cubed | -2.97% | -3.77% to -2.16% |
| GPU | GLMMHD / HLLD | 64 cubed | 0.86% | 0.77% to 0.95% |
| GPU | GLMMHD / HLLD | 32 cubed | 0.64% | 0.07% to 1.22% |

CPU runs used one pinned core, a 32-cubed grid, and 100 steps. GPU runs used a
64-cubed grid and 2,000 steps. Both used PLM/VL2. The GPU hydro speedup includes
the changed wave estimates; it cannot be attributed solely to EOS call cost.
The CPU results do not meet the requested 1% target. Neither backend establishes
the full acceptance gate on this unreserved hardware and limited workload matrix.

The warmed fused thermodynamics microbenchmark reported approximately 2.59%
CPU and 1.34% GPU throughput overhead, with matching checksums. This diagnostic
does not measure whole-timestep overhead. Optimized CPU disassembly contained no
separate calls to the inspected thermodynamic methods. Static GPU resource
inspection matched 82 core conversion, flux, and timestep kernels; MHD conversion
register use stayed at 80 and hydro conversion fell from 71 to 69, with no
increase in local memory for those matched kernels. This is not runtime profiling.

## Evidence and remaining work

Local raw results are retained in the ignored `build-eos-results/` directory:
`cpu-final.json`, `gpu-final.json`, their logs, `regression-gpu.log`,
`regression-serial.log`, `regression-mpi-smoke.log`, `cpu-final-checks.log`,
`gpu-final-checks.log`, `turbulence-comparison.json`,
`final-cleanup-state-comparison.json`, `invalid-states.json`,
`cpu-microbenchmark-final.txt`, `gpu-microbenchmark-final.txt`, and
`gpu-kernel-resources.json`. These artifacts are local evidence, not tracked
repository fixtures; performance JSON includes binary hashes and build settings.

Next steps are to profile and reduce CPU overhead while preserving generic EOS
calls, repeat timing on reserved hardware, complete the timed-out CPU regressions,
and decide how to validate turbulence under the changed solver. Shock, source,
AMR, and multi-rank performance cases also remain. See
[general EOS support](general_eos.md) for reproduction commands and the additional
requirements before enabling tabulated materials.

The November 25 continuation added this report and performed static checks only;
it did not compile or rerun simulations or benchmarks.
