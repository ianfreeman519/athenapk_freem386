//========================================================================================
//// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
///// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
///// Licensed under the 3-clause BSD License, see LICENSE file for details
/////========================================================================================
//! \file cluster_utils.hpp
//  \brief Utilities for galaxy cluster functions
#ifndef CLUSTER_CLUSTER_UTILS_HPP_
#define CLUSTER_CLUSTER_UTILS_HPP_

// parthenon headers
#include <basic_types.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "utils/error_checking.hpp"

namespace cluster {

// Add a density to the conserved variables while keeping velocity fixed
template <typename View4D>
KOKKOS_INLINE_FUNCTION void
AddDensityToConsAtFixedVel(const parthenon::Real density, View4D &cons,
                           const View4D &prim, const int &k, const int &j, const int &i) {
  // Add density such that velocity is fixed
  cons(IDN, k, j, i) += density;
  cons(IM1, k, j, i) += density * prim(IV1, k, j, i);
  cons(IM2, k, j, i) += density * prim(IV2, k, j, i);
  cons(IM3, k, j, i) += density * prim(IV3, k, j, i);
  cons(IEN, k, j, i) +=
      density * (0.5 * (SQR(prim(IV1, k, j, i)) + SQR(prim(IV2, k, j, i)) +
                        SQR(prim(IV3, k, j, i))));
}

// Add a density to the conserved variables while keeping velocity and
// temperature ( propto pressure/density) fixed
template <typename View4D, typename EOS>
KOKKOS_INLINE_FUNCTION void
AddDensityToConsAtFixedVelTemp(const parthenon::Real density, View4D &cons,
                               const View4D &prim, const EOS &eos, const int &k,
                               const int &j, const int &i) {
  const Real old_rho = cons(IDN, k, j, i);
  const Real old_e = eos.InternalEnergy(prim(IDN, k, j, i), prim(IPR, k, j, i));
  const Real temperature = eos.Temperature(prim(IDN, k, j, i), old_e);
  const Real new_rho = old_rho + density;
  PARTHENON_REQUIRE(new_rho > 0.0, "Mass update produced nonpositive density");
  const Real thermal_delta = new_rho * eos.EnergyFromTemperature(new_rho, temperature) -
                             old_rho * eos.EnergyFromTemperature(old_rho, temperature);
  cons(IDN, k, j, i) += density;
  cons(IM1, k, j, i) += density * prim(IV1, k, j, i);
  cons(IM2, k, j, i) += density * prim(IV2, k, j, i);
  cons(IM3, k, j, i) += density * prim(IV3, k, j, i);
  cons(IEN, k, j, i) +=
      density * (0.5 * (SQR(prim(IV1, k, j, i)) + SQR(prim(IV2, k, j, i)) +
                        SQR(prim(IV3, k, j, i)))) +
      thermal_delta;
}

} // namespace cluster

#endif // CLUSTER_CLUSTER_UTILS_HPP_
