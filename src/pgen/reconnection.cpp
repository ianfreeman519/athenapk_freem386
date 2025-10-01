//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file reconnection.cpp
//! \brief Problem generator for magnetic reconnection in 2d.
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

namespace reconnection {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Setting up derived fields:
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  // Defining m to pass to the field definition
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  // Field definitions
  hydro_pkg->AddField("curlBx", m);
  hydro_pkg->AddField("curlBy", m);
  hydro_pkg->AddField("curlBz", m);
}

// storing the curls just before output
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm) {
  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &data = pmb->meshblock_data.Get(); // This is for grabbing the meshblocks defined above

  // Get derived fields
  auto &curlBx = data->Get("curlBx").data;
  auto &curlBy = data->Get("curlBy").data;
  auto &curlBz = data->Get("curlBz").data;

  // Getting indices???
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Actually computing and storing curl data?
  pmb->par_for(
      "reconnection::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real term1, term2;
        // curlBx = dBzdy - dBydz
        term1 = (u(IB3,k,j+1,i) - u(IB3,k,j-1,i))/(coords.Xc<2>(j+1)-coords.Xc<2>(j-1));
        term2 = (u(IB2,k+1,j,i) - u(IB2,k-1,j,i))/(coords.Xc<3>(k+1)-coords.Xc<3>(k-1));
        curlBx(k, j, i) = term1 - term2;
        // curlBy = dBxdz - dBzdx
        term1 = (u(IB1,k+1,j,i) - u(IB1,k-1,j,i))/(coords.Xc<3>(k+1)-coords.Xc<3>(k-1));
        term2 = (u(IB3,k,j,i+1) - u(IB3,k,j,i-1))/(coords.Xc<1>(i+1)-coords.Xc<1>(i-1));
        curlBy(k, j, i) = term1 - term2;
        // curlBz = dBydx - dBxdy
        term1 = (u(IB2,k,j,i+1) - u(IB2,k,j,i-1))/(coords.Xc<1>(i+1)-coords.Xc<1>(i-1));
        term2 = (u(IB1,k,j+1,i) - u(IB1,k,j-1,i))/(coords.Xc<2>(j+1)-coords.Xc<2>(j-1));
        curlBz(k, j, i) = term1 - term2;
      }
  );
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  Real gm1  = pin->GetReal("hydro", "gamma") - 1.0;
  Real B0   = pin->GetOrAddReal("problem/reconnection", "B0", 1.0) / std::sqrt(4*M_PI);
  Real rho0 = pin->GetOrAddReal("problem/reconnection", "rho0", 1.0);
  Real T0   =  pin->GetOrAddReal("problem/reconnection", "T0", 2.0);
  Real w    = pin->GetOrAddReal("problem/reconnection", "w", 0.5);
  Real delta = pin->GetOrAddReal("problem/reconnection", "delta", 0.5);
  Real powP = pin->GetOrAddReal("problem/reconnection", "powP", 2.0);
  
  // Checking if spitzer or fixed ohmic resistivity is turned on:
  Real k_b, atomic_mass_unit, m_bar;
  auto detected_resistivity_type = pin->GetString("diffusion", "resistivity_coeff");
if (detected_resistivity_type == "spitzer") {
    // if spitzer is defined, grab the hydro package and units now...
    auto hydro_pkg = pmb->packages.Get("Hydro");
    const auto units = hydro_pkg->Param<Units>("units");
    k_b = units.k_boltzmann();
    atomic_mass_unit = units.atomic_mass_unit();
    m_bar = pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;
  } else {
    PARTHENON_FAIL("Unknown resitivity type given in input file");
  }

  // Printing out input values for slurm records
  std::cout << "========================================" << std::endl;
  std::cout << "Input parameters:" << std::endl;
  std::cout << "gamma ..... " << pin->GetReal("hydro", "gamma") << std::endl;
  std::cout << "B0 [Gauss]  " << B0 * std::sqrt(4*M_PI) << std::endl;
  std::cout << "rho0 [g] .. " << rho0 << std::endl;
  std::cout << "T0 [K] .... " << T0 << std::endl;
  std::cout << "w [cm] .... " << w << std::endl;
  std::cout << "delta [cm]  " << delta << std::endl;
  std::cout << "powP [-] .. " << powP << std::endl;
  std::cout << "resistivity: " << detected_resistivity_type << std::endl;
  std::cout << "========================================" << std::endl;


  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: reconnection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x, y;
        x = coords.Xc<1>(i);
        y = coords.Xc<2>(j);
        u(IDN, k, j, i) = rho0; // Density

        u(IM1, k, j, i) = 0.0;  // Initial Momentum is zero
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        Real b1x, b2x, b1y, b2y;  // Helper variables for clarity
        // Gaussian Loop in x (clamping exponential terms to prevent infinitely small and large values)
        Real exp1 = std::clamp(-std::pow(4*(x+1.0)/w, 2) - std::pow(4*y/delta, 2), -300.0, 300.0);
        Real exp2 = std::clamp(-std::pow(4*(x-1.0)/w, 2) - std::pow(4*y/delta, 2), -300.0, 300.0);
        b1x = 2*M_PI*w*y/delta * std::exp(exp1);
        b2x = 2*M_PI*w*y/delta * std::exp(exp2);
        // Gaussian Loop in y
        b1y = 2*M_PI*delta/w * (-1.0 - x) * std::exp(exp1);
        b2y = 2*M_PI*delta/w * ( 1.0 - x) * std::exp(exp2);
        u(IB1, k, j, i) = B0*(b1x + b2x + std::tanh(y/delta));
        u(IB2, k, j, i) = B0*(b1y + b2y);
        u(IB3, k, j, i) = 0.0;

        // if spitzer turned on, use units, otherwise treat T0 as an initial P_thermal
        Real P_thermal;
        if (detected_resistivity_type == "spitzer") {
          P_thermal = T0 * k_b / (m_bar * u(IDN, k, j, i));
        } else {
          P_thermal = T0;
        }
        Real P = P_thermal * (1.0 + std::pow(std::abs(y), powP));

        u(IEN, k, j, i) =
            P / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      });
}
} // namespace reconnection
