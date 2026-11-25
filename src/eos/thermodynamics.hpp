// Copyright (c) 2026, AthenaPK Collaboration. All rights reserved.
// Licensed under the 3-Clause BSD License (the LICENSE file).
#ifndef EOS_THERMODYNAMICS_HPP_
#define EOS_THERMODYNAMICS_HPP_

#include <cmath>
#include <singularity-eos/eos/eos_ideal.hpp>
#include <type_traits>

#include "basic_types.hpp"

// All quantities are in AthenaPK code units, except temperature (Kelvin when
// physical units/composition are configured). Energy arguments are specific.
// Keep the model concrete: generic API calls inline without per-cell dispatch.
template <typename Model>
class Thermodynamics {
 public:
  static constexpr bool constant_specific_heat =
      std::is_same_v<Model, singularity::IdealGas>;

  Thermodynamics() = default;

  explicit Thermodynamics(Model model) : model_(model) {}

  KOKKOS_INLINE_FUNCTION
  parthenon::Real Pressure(parthenon::Real rho, parthenon::Real sie) const {
    return model_.PressureFromDensityInternalEnergy(rho, sie);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real InternalEnergy(parthenon::Real rho, parthenon::Real pressure) const {
    using ModelReal = decltype(model_.PressureFromDensityInternalEnergy(rho, pressure));
    ModelReal sie = 0;
    model_.InternalEnergyFromDensityPressure(rho, pressure, sie);
    return sie;
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real EnergyDensity(parthenon::Real rho, parthenon::Real pressure) const {
    return rho * InternalEnergy(rho, pressure);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real Temperature(parthenon::Real rho, parthenon::Real sie) const {
    return model_.TemperatureFromDensityInternalEnergy(rho, sie);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real TemperatureFromPressure(parthenon::Real rho,
                                          parthenon::Real pressure) const {
    return Temperature(rho, InternalEnergy(rho, pressure));
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real EnergyFromTemperature(parthenon::Real rho,
                                        parthenon::Real temperature) const {
    return model_.InternalEnergyFromDensityTemperature(rho, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SpecificHeat(parthenon::Real rho, parthenon::Real sie) const {
    return model_.SpecificHeatFromDensityInternalEnergy(rho, sie);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real BulkModulus(parthenon::Real rho, parthenon::Real pressure) const {
    return model_.BulkModulusFromDensityInternalEnergy(rho,
                                                       InternalEnergy(rho, pressure));
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SoundSpeedSquared(parthenon::Real rho, parthenon::Real pressure) const {
    return BulkModulus(rho, pressure) / rho;
  }

  // Explicit host-side lifetime hooks for future device-resident tables.
  // Copies borrow model resources; the owning adapter must finalize exactly once.
  Thermodynamics GetOnDevice() { return Thermodynamics(model_.GetOnDevice()); }
  void Finalize() { model_.Finalize(); }

  // Derivatives for source integrators, at fixed density or fixed energy.
  KOKKOS_INLINE_FUNCTION
  parthenon::Real PressureDerivativeEnergy(parthenon::Real rho,
                                           parthenon::Real sie) const {
    return rho * model_.GruneisenParamFromDensityInternalEnergy(rho, sie);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real PressureDerivativeDensity(parthenon::Real rho,
                                            parthenon::Real sie) const {
    const auto pressure = Pressure(rho, sie);
    return (model_.BulkModulusFromDensityInternalEnergy(rho, sie) -
            pressure * PressureDerivativeEnergy(rho, sie) / rho) /
           rho;
  }

 private:
  Model model_{};
};

using GasThermodynamics = Thermodynamics<singularity::IdealGas>;

#endif // EOS_THERMODYNAMICS_HPP_
