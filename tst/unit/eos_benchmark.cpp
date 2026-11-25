// Copyright (c) 2026, AthenaPK Collaboration. All rights reserved.
// Licensed under the 3-Clause BSD License (the LICENSE file).
#include <cmath>
#include <iostream>

#include "eos/thermodynamics.hpp"

// A diagnostic kernel benchmark, not a substitute for whole-step measurements.
// Input arrays prevent constant folding; output is reduced and printed.
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    using parthenon::Real;
    constexpr int n = 1 << 20;
    const Real gamma = 5.0 / 3.0;
    const GasThermodynamics gas(singularity::IdealGas(gamma - 1.0, 1.0));
    Kokkos::View<Real *> rho("rho", n), energy("energy", n), output("output", n);
    Kokkos::parallel_for(
        "EOS benchmark inputs", n, KOKKOS_LAMBDA(int i) {
          rho(i) = 0.1 + (i % 101) * 0.01;
          energy(i) = 0.2 + (i % 97) * 0.01;
        });
    auto run = [&](bool singularity) {
      Kokkos::parallel_for(
          "EOS benchmark", n, KOKKOS_LAMBDA(int i) {
            const Real r = rho(i), e = energy(i);
            if (singularity) {
              const Real p = gas.Pressure(r, e);
              output(i) = gas.EnergyDensity(r, p) + sqrt(gas.SoundSpeedSquared(r, p));
            } else {
              const Real p = (gamma - 1.0) * r * e;
              output(i) = p / (gamma - 1.0) + sqrt(gamma * p / r);
            }
          });
      Kokkos::fence();
    };
    for (bool singularity : {false, true}) {
      Kokkos::Timer warmup;
      do {
        run(singularity);
      } while (warmup.seconds() < 0.25);
      int repeats = 0;
      Kokkos::Timer timer;
      do {
        run(singularity);
        ++repeats;
      } while (timer.seconds() < 0.5);
      const double seconds = timer.seconds();
      Real checksum = 0;
      Kokkos::parallel_reduce(
          "EOS benchmark checksum", n, KOKKOS_LAMBDA(int i, Real &s) { s += output(i); },
          checksum);
      std::cout << (singularity ? "singularity" : "legacy") << " seconds=" << seconds
                << " calls/s=" << double(n) * repeats / seconds
                << " checksum=" << checksum << '\n';
    }
  }
  Kokkos::finalize();
}
