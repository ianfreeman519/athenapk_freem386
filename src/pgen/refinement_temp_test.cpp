//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file refinement_temp_test.cpp
//! \brief Testing ways to trigger anomalous heating across refinement with dot(v,b)!=0
//!
//! REFERENCE: Trust Me Bro I made it all up
//========================================================================================

// Parthenon headers
#include "Kokkos_MathematicalFunctions.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

namespace refinement_temp_test {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  Real gm1  = pin->GetReal("hydro", "gamma") - 1.0;
  Real B0   = pin->GetOrAddReal("problem/refinement_temp_test", "B0", 1e6);
  Real B_phi_degrees = pin->GetOrAddReal("problem/refinement_temp_test", "field_angle_degrees", 0.0);
  Real v0 = pin->GetOrAddReal("problem/refinement_temp_test", "v0", 1e7);
  Real v_phi_degrees = pin->GetOrAddReal("problem/refinement_temp_test", "velocity_angle_degrees", 0.0);
  Real T0 = pin->GetOrAddReal("problem/refinement_temp_test", "T0", 1e4);
  Real rho0 = pin->GetOrAddReal("problem/refinement_temp_test", "rho0", 2.4);

  auto &coords = pmb->coords;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  Real k_b = units.k_boltzmann();
  Real atomic_mass_unit = units.atomic_mass_unit();
  Real m_bar = pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;


  pmb->par_for(
      "ProblemGenerator: refinement_temp_test", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        (void)coords;

        u(IDN, k, j, i) = rho0; // Density

        u(IM1, k, j, i) = rho0 * v0 * cos(v_phi_degrees * M_PI / 180.0);
        u(IM2, k, j, i) = rho0 * v0 * sin(v_phi_degrees * M_PI / 180.0);
        u(IM3, k, j, i) = 0.0;

        // Set background pressure and magnetic fields
        u(IB1, k, j, i) = B0*cos(B_phi_degrees * M_PI / 180.0);
        u(IB2, k, j, i) = B0*sin(B_phi_degrees * M_PI / 180.0);
        u(IB3, k, j, i) = 0.0;
        // Pressure
        Real P = T0 * k_b * rho0 / m_bar;

        // Internal energy profile
        u(IEN, k, j, i) =
            P / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
    });
  }
} // namespace refinement_temp_test
