//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusion.hpp
//! \brief

#ifndef HYDRO_DIFFUSION_DIFFUSION_HPP_
#define HYDRO_DIFFUSION_DIFFUSION_HPP_

// C++ headers
#include <algorithm>
#include <cmath>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../../eos/tabulated_eos.hpp"

using namespace parthenon::package::prelude;

namespace limiters {
/*----------------------------------------------------------------------------*/
/* vanleer: van Leer slope limiter
 */

KOKKOS_INLINE_FUNCTION Real vanleer(const Real A, const Real B) {
  if (A * B > 0) {
    return 2.0 * A * B / (A + B);
  } else {
    return 0.0;
  }
}

/*----------------------------------------------------------------------------*/
/* minmod: minmod slope limiter
 */

KOKKOS_INLINE_FUNCTION Real minmod(const Real A, const Real B) {
  if (A * B > 0) {
    if (A > 0) {
      return std::min(A, B);
    } else {
      return std::max(A, B);
    }
  } else {
    return 0.0;
  }
}

/*----------------------------------------------------------------------------*/
/* mc: monotonized central slope limiter
 */

KOKKOS_INLINE_FUNCTION Real mc(const Real A, const Real B) {
  return minmod(2.0 * minmod(A, B), (A + B) / 2.0);
}
/*----------------------------------------------------------------------------*/
/* limiter2 and limiter4: call slope limiters to preserve monotonicity
 */

KOKKOS_INLINE_FUNCTION Real lim2(const Real A, const Real B) {
  /* slope limiter */
  return mc(A, B);
}

KOKKOS_INLINE_FUNCTION Real lim4(const Real A, const Real B, const Real C, const Real D) {
  return lim2(lim2(A, B), lim2(C, D));
}
} // namespace limiters

struct ThermalDiffusivity {
 private:
  Real mbar_, me_, kb_;
  Conduction conduction_;
  ConductionCoeff conduction_coeff_type_;
  // "free" coefficient/prefactor. Value depends on conduction is set in the constructor.
  Real coeff_;
  TabulatedEOS eos_;

 public:
  KOKKOS_INLINE_FUNCTION
  ThermalDiffusivity(Conduction conduction, ConductionCoeff conduction_coeff_type,
                     Real coeff, Real mbar, Real me, Real kb,
                     TabulatedEOS eos = TabulatedEOS())
      : mbar_(mbar), me_(me), kb_(kb), conduction_(conduction),
        conduction_coeff_type_(conduction_coeff_type), coeff_(coeff), eos_(eos) {}

  KOKKOS_INLINE_FUNCTION
  Real Get(const Real pres, const Real rho, const Real temperature = NAN,
           const Real zbar = NAN) const {
    if (conduction_coeff_type_ == ConductionCoeff::fixed) {
      if (!eos_.IsInitialized()) return coeff_;
      const Real local_temperature = Kokkos::isfinite(temperature)
                                         ? temperature
                                         : Temperature(pres, rho);
      return coeff_ * HeatCapacity(rho, local_temperature);
    } else if (conduction_coeff_type_ == ConductionCoeff::spitzer) {
      auto local_temperature = temperature;
      auto local_zbar = zbar;
      if (eos_.IsInitialized()) {
        const auto eps = eos_.InternalEnergyFromDensityPressure(rho, pres);
        const auto state = eos_.StateFromDensityInternalEnergy(rho, eps);
        local_temperature = state.temperature;
        local_zbar = state.zbar;
      }
      const Real T_cgs = Kokkos::isfinite(local_temperature)
                             ? local_temperature
                             : mbar_ / kb_ * pres / rho;
      local_zbar = Kokkos::isfinite(local_zbar) ? local_zbar : 1.0;
      if (eos_.IsInitialized() && local_zbar <= 0.0) return 0.0;
      PARTHENON_REQUIRE(T_cgs > 0.0 && local_zbar > 0.0,
                        "Spitzer conduction requires positive temperature and Zbar.");
      const Real kappa = coeff_ * std::pow(T_cgs, 2.5) / local_zbar;
      return kappa * (eos_.IsInitialized() || Kokkos::isfinite(temperature)
                          ? 1.0
                          : mbar_ / kb_) /
             rho;
    }
    return 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  Real Temperature(const Real pres, const Real rho) const {
    if (!eos_.IsInitialized()) return pres / rho;
    const auto eps = eos_.InternalEnergyFromDensityPressure(rho, pres);
    return eos_.StateFromDensityInternalEnergy(rho, eps).temperature;
  }

  KOKKOS_INLINE_FUNCTION
  Real HeatCapacity(const Real rho, const Real temperature) const {
    if (!eos_.IsInitialized()) return 1.0;
    constexpr Real delta = 1.e-4;
    const Real tlo = temperature * (1.0 - delta);
    const Real thi = temperature * (1.0 + delta);
    return (eos_.InternalEnergyFromDensityTemperature(rho, thi) -
            eos_.InternalEnergyFromDensityTemperature(rho, tlo)) /
           (thi - tlo);
  }

  KOKKOS_INLINE_FUNCTION
  Real StableDiffusivity(const Real pres, const Real rho) const {
    if (!eos_.IsInitialized()) return Get(pres, rho);
    const Real temperature = Temperature(pres, rho);
    return Get(pres, rho, temperature) / HeatCapacity(rho, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  bool UsesGeneralEOS() const { return eos_.IsInitialized(); }

  KOKKOS_INLINE_FUNCTION
  Conduction GetType() const { return conduction_; }

  KOKKOS_INLINE_FUNCTION
  ConductionCoeff GetCoeffType() const { return conduction_coeff_type_; }
};

Real EstimateConductionTimestep(MeshData<Real> *md);

//! Calculate isotropic thermal conduction with fixed coefficient
void ThermalFluxIsoFixed(MeshData<Real> *md);
//! Calculate thermal conduction (general case incl. anisotropic and saturated)
void ThermalFluxGeneral(MeshData<Real> *md);

struct MomentumDiffusivity {
 private:
  Real mbar_, me_, kb_;
  Viscosity viscosity_;
  ViscosityCoeff viscosity_coeff_type_;
  // "free" coefficient/prefactor. Value depends on viscosity set in the constructor.
  Real coeff_;

 public:
  KOKKOS_INLINE_FUNCTION
  MomentumDiffusivity(Viscosity viscosity, ViscosityCoeff viscosity_coeff_type,
                      Real coeff, Real mbar, Real me, Real kb)
      : viscosity_(viscosity), viscosity_coeff_type_(viscosity_coeff_type), coeff_(coeff),
        mbar_(mbar), me_(me), kb_(kb) {}

  KOKKOS_INLINE_FUNCTION
  Real Get(const Real pres, const Real rho) const;

  KOKKOS_INLINE_FUNCTION
  Viscosity GetType() const { return viscosity_; }

  KOKKOS_INLINE_FUNCTION
  ViscosityCoeff GetCoeffType() const { return viscosity_coeff_type_; }
};

Real EstimateViscosityTimestep(MeshData<Real> *md);

//! Calculate isotropic viscosity with fixed coefficient
void MomentumDiffFluxIsoFixed(MeshData<Real> *md);
//! Calculate viscosity (general case incl. anisotropic)
void MomentumDiffFluxGeneral(MeshData<Real> *md);

struct OhmicDiffusivity {
 private:
  Resistivity resistivity_;
  ResistivityCoeff resistivity_coeff_type_;
  // "free" coefficient/prefactor. Value depends on resistivity set in the constructor.
  Real coeff_;
  Real temperature_from_p_over_rho_;
  Real zbar_;
  Real eta_cgs_to_code_;
  Real eta_max_;
  TabulatedEOS eos_;

 public:
  KOKKOS_INLINE_FUNCTION
  OhmicDiffusivity(Resistivity resistivity, ResistivityCoeff resistivity_coeff_type,
                   Real coeff, Real temperature_from_p_over_rho, Real zbar,
                   Real eta_cgs_to_code, Real eta_max,
                   TabulatedEOS eos = TabulatedEOS())
      : resistivity_(resistivity), resistivity_coeff_type_(resistivity_coeff_type),
        coeff_(coeff), temperature_from_p_over_rho_(temperature_from_p_over_rho),
        zbar_(zbar), eta_cgs_to_code_(eta_cgs_to_code), eta_max_(eta_max), eos_(eos) {}

  KOKKOS_INLINE_FUNCTION
  Real Get(const Real pres, const Real rho, const Real temperature = NAN,
           const Real zbar = NAN) const {
    if (resistivity_coeff_type_ == ResistivityCoeff::fixed) {
      return coeff_;
    } else if (resistivity_coeff_type_ == ResistivityCoeff::spitzer) {
      // Convert p/rho from code units to temperature in kelvin.
      auto local_temperature = temperature;
      auto local_zbar = zbar;
      if (eos_.IsInitialized()) {
        const auto eps = eos_.InternalEnergyFromDensityPressure(rho, pres);
        const auto state = eos_.StateFromDensityInternalEnergy(rho, eps);
        local_temperature = state.temperature;
        local_zbar = state.zbar;
      }
      const Real temperature_kelvin = Kokkos::isfinite(local_temperature)
                                          ? local_temperature
                                          : temperature_from_p_over_rho_ * pres / rho;
      PARTHENON_REQUIRE(temperature_kelvin > 0.0,
                        "Spitzer resistivity requires positive temperature.");

      // Magnetic diffusivity in Heaviside-Lorentz CGS [cm^2/s], then code units.
      local_zbar = Kokkos::isfinite(local_zbar) ? local_zbar : zbar_;
      if (eos_.IsInitialized() && local_zbar <= 0.0) return 0.0;
      PARTHENON_REQUIRE(local_zbar > 0.0,
                        "Spitzer resistivity requires positive Zbar.");
      Real eta = 1.02688e12 * local_zbar * coeff_ /
                 std::pow(temperature_kelvin, 1.5) *
                 eta_cgs_to_code_;
      if (eta_max_ > 0.0) {
        eta = std::min(eta, eta_max_);
      }
      return eta;
    }
    PARTHENON_FAIL("Unknown Resistivity coeff");
  }

  KOKKOS_INLINE_FUNCTION
  Resistivity GetType() const { return resistivity_; }

  KOKKOS_INLINE_FUNCTION
  ResistivityCoeff GetCoeffType() const { return resistivity_coeff_type_; }

  KOKKOS_INLINE_FUNCTION
  Real InternalEnergy(const Real rho, const Real pres, const Real gm1) const {
    return eos_.IsInitialized() ? rho * eos_.InternalEnergyFromDensityPressure(rho, pres)
                                : pres / gm1;
  }
};

Real EstimateResistivityTimestep(MeshData<Real> *md);

//! Calculate Ohmic face fluxes, evaluating the configured coefficient at each face.
void OhmicDiffFlux(MeshData<Real> *md);

//! Add Ohmic electric fields to the edge EMFs used by constrained transport.
TaskStatus AddOhmicEdgeEMF(MeshData<Real> *md);

// Calculate all diffusion fluxes, i.e., update the .flux views in md
TaskStatus CalcDiffFluxes(StateDescriptor *hydro_pkg, MeshData<Real> *md);

#endif //  HYDRO_DIFFUSION_DIFFUSION_HPP_
