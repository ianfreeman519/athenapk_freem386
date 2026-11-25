// Copyright (c) 2026, AthenaPK Collaboration. All rights reserved.
// Licensed under the 3-Clause BSD License (the LICENSE file).
#include <cmath>
#include <iostream>
#include <limits>

#include "eos/adiabatic_glmmhd.hpp"
#include "eos/adiabatic_hydro.hpp"

using parthenon::Real;

// Stiffened-gas fixture: gamma=2, p_infinity=1, Cv=3.
// It catches accidental ideal-gas conversions in the adapter.
struct NonIdeal {
  KOKKOS_INLINE_FUNCTION Real PressureFromDensityInternalEnergy(Real r, Real e) const {
    return r * e - 2;
  }
  KOKKOS_INLINE_FUNCTION void InternalEnergyFromDensityPressure(Real r, Real p,
                                                                Real &e) const {
    e = (p + 2) / r;
  }
  KOKKOS_INLINE_FUNCTION Real TemperatureFromDensityInternalEnergy(Real r, Real e) const {
    return (e - 1 / r) / 3;
  }
  KOKKOS_INLINE_FUNCTION Real InternalEnergyFromDensityTemperature(Real r, Real t) const {
    return 3 * t + 1 / r;
  }
  KOKKOS_INLINE_FUNCTION Real BulkModulusFromDensityInternalEnergy(Real r, Real e) const {
    return 2 * r * e - 2;
  }
  KOKKOS_INLINE_FUNCTION Real GruneisenParamFromDensityInternalEnergy(Real, Real) const {
    return 1;
  }
  KOKKOS_INLINE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(Real, Real) const {
    return 3;
  }
};

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    for (Real gamma : {Real(1.4), Real(5.0 / 3.0), Real(2.0)}) {
      auto owner = GasThermodynamics(singularity::IdealGas(gamma - 1, 2.5)).GetOnDevice();
      const auto gas = owner;
      int bad = 0;
      Kokkos::parallel_reduce(
          "EOS round trips", 81,
          KOKKOS_LAMBDA(int i, int &n) {
            const Real rho = pow(10.0, (i % 9) - 4);
            const Real e = pow(10.0, (i / 9) - 4);
            const Real p = gas.Pressure(rho, e);
            const Real tol = 64 * std::numeric_limits<Real>::epsilon();
            if (fabs(gas.InternalEnergy(rho, p) / e - 1) > tol ||
                fabs(gas.EnergyFromTemperature(rho, gas.Temperature(rho, e)) / e - 1) >
                    tol ||
                fabs(gas.SoundSpeedSquared(rho, p) / (gamma * p / rho) - 1) > tol ||
                fabs(gas.PressureDerivativeEnergy(rho, e) / ((gamma - 1) * rho) - 1) >
                    tol ||
                fabs(gas.PressureDerivativeDensity(rho, e) / ((gamma - 1) * e) - 1) > tol)
              ++n;
          },
          bad);
      failures += bad;
      owner.Finalize();
    }
    const Thermodynamics<NonIdeal> nonideal(NonIdeal{});
    int nonideal_failures = 0;
    Kokkos::parallel_reduce(
        "Non-ideal EOS fixture", 1,
        KOKKOS_LAMBDA(int, int &n) {
          if (fabs(nonideal.InternalEnergy(2, nonideal.Pressure(2, 5)) - 5) > 1e-6 ||
              fabs(nonideal.SoundSpeedSquared(2, 8) - 9) > 1e-6 ||
              fabs(nonideal.PressureDerivativeEnergy(2, 5) - 2) > 1e-6 ||
              fabs(nonideal.PressureDerivativeDensity(2, 5) - 5) > 1e-6 ||
              fabs(nonideal.TemperatureFromPressure(2, 8) - 1.5) > 1e-6)
            ++n;
        },
        nonideal_failures);
    failures += nonideal_failures;

    const GasThermodynamics gas(singularity::IdealGas(0.4, 1.0));

    // Both fluids: exercise floors, ceilings, momentum/energy corrections,
    // scalar conversion, and magnetic energy subtraction on the selected device.
    for (bool magnetic : {false, true}) {
      Kokkos::View<Real ****> u("u", 10, 1, 1, 1), w("w", 10, 1, 1, 1);
      const Real inf = std::numeric_limits<Real>::infinity();
      const AdiabaticHydroEOS hydro(0.1, 1.0, -1, 2.0, 3.0, gas);
      const AdiabaticGLMMHDEOS mhd(0.1, 1.0, -1, 2.0, 3.0, gas);
      Kokkos::parallel_for(
          "EOS limits", 1, KOKKOS_LAMBDA(int) {
            u(IDN, 0, 0, 0) = 0.5;
            u(IM1, 0, 0, 0) = 4;
            u(IEN, 0, 0, 0) = 30;
            if (magnetic) u(IB1, 0, 0, 0) = 2;
            u(9, 0, 0, 0) = 0.25;
            if (magnetic)
              mhd.ConsToPrim(u, w, 9, 1, 0, 0, 0);
            else
              hydro.ConsToPrim(u, w, 9, 1, 0, 0, 0);
          });
      auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
      auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
      const Real tol = 64 * std::numeric_limits<Real>::epsilon();
      if (fabs(hw(IDN, 0, 0, 0) - 1) > tol || fabs(hw(IV1, 0, 0, 0) - 2) > tol ||
          fabs(hw(IPR, 0, 0, 0) - 1.2) > tol || fabs(hw(9, 0, 0, 0) - 0.25) > tol ||
          fabs(hu(IEN, 0, 0, 0) - (magnetic ? 7 : 5)) > tol)
        ++failures;
      const AdiabaticHydroEOS floor(0.4, -1, -1, inf, inf, gas);
      Kokkos::parallel_for(
          "EOS pressure floor", 1, KOKKOS_LAMBDA(int) {
            u(IDN, 0, 0, 0) = 2;
            u(IM1, 0, 0, 0) = 0;
            u(IEN, 0, 0, 0) = -1;
            floor.ConsToPrim(u, w, 9, 1, 0, 0, 0);
          });
      hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
      hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
      if (fabs(hw(IPR, 0, 0, 0) - 0.4) > tol || fabs(hu(IEN, 0, 0, 0) - 1) > tol)
        ++failures;
    }
  }
  Kokkos::finalize();
  std::cout << "EOS failures: " << failures << '\n';
  return failures ? 1 : 0;
}
