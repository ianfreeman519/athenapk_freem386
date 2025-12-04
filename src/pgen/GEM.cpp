//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file GEM.cpp
//! \brief Problem generator for magnetic GEM reconnection in 2d.
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
#include "../hydro/diffusion/diffusion.hpp" // For storing eta later

namespace GEM {
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
  hydro_pkg->AddField("beta", m);
  hydro_pkg->AddField("eta", m);
  hydro_pkg->AddField("T", m);
}

// storing the curls just before output
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm) {
  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &w = mbd->Get("prim").data;
  auto &data = pmb->meshblock_data.Get(); // This is for grabbing the meshblocks defined above
  auto hydro_pkg = pmb->packages.Get("Hydro"); // This is for grabbing the calculated diffusivity
  const bool has_ohm_diff = hydro_pkg->AllParams().hasKey("ohm_diff");
  OhmicDiffusivity ohm_diff_dev(Resistivity::none, ResistivityCoeff::none, 0.0, 0.0, 0.0,
                                0.0, 0.0); // Dummy init
  if (has_ohm_diff) {
    ohm_diff_dev = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  }

  // Get derived fields
  auto &curlBx = data->Get("curlBx").data;
  auto &curlBy = data->Get("curlBy").data;
  auto &curlBz = data->Get("curlBz").data;
  auto &eta_field    = data->Get("eta").data;
  auto &beta_field   = data->Get("beta").data;
  auto &T_field      = data->Get("T").data;
  const auto units = hydro_pkg->Param<Units>("units");

  // Getting indices
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Actually computing and storing curl data
  pmb->par_for(
      "GEM::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
        // Calculating 
        // Use primitive pressure for temperature; conserved array doesn't store p.
        Real rho = w(IDN, k, j, i);
        Real p = w(IPR, k, j, i);
        Real mbar = hydro_pkg->Param<Real>("mbar");
        Real kb = units.k_boltzmann();
        T_field(k, j, i) = mbar / kb * p / rho;

        // beta = p / (B^2 / 2) - in Heaviside Lorentz units, this is p / (0.5 * 4pi * B^2)
        beta_field(k, j, i) = p / (0.5 * 4 * M_PI * (SQR(u(IB1,k,j,i)) + SQR(u(IB2,k,j,i)) + SQR(u(IB3,k,j,i))));
        Real eta_val = 0.0;
        if (has_ohm_diff) {
          eta_val = ohm_diff_dev.Get(p, rho);
        }
        eta_field(k, j, i) = eta_val;
      }
  );
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto detected_resistivity_type = pin->GetOrAddString("diffusion", "resistivity_coeff", "none");

  Real gm1  = pin->GetReal("hydro", "gamma") - 1.0;
  Real B0   = pin->GetOrAddReal("problem/GEM", "B0", 1.0) / std::sqrt(4 * M_PI);  // Assuming input is in Gauss
  Real Z    = pin->GetOrAddReal("hydro", "Z", 1.0);
  Real rho0   = pin->GetOrAddReal("problem/GEM", "rho0", 1.0);
  Real rhoinf = 0.2 * rho0;
  Real T0   = pin->GetOrAddReal("problem/GEM", "T0", 1.0);
  // psi0 in the input is interpreted as a vector-potential amplitude (fraction of B0*lambda)
  // so the resulting perturbation field is O(psi0 * B0).
  Real psi0 = pin->GetOrAddReal("problem/GEM", "psi0", 0.1);
  Real lambda=pin->GetOrAddReal("problem/GEM", "lambda", 0.5);
  psi0 = psi0 * B0 * lambda; // convert to B*L units for A_z
  Real Lx   = pin->GetReal("problem/GEM", "Lx");
  Real Ly   = pin->GetReal("problem/GEM", "Ly");

  // Checking if spitzer or fixed ohmic resistivity is turned on:
  Real k_b, atomic_mass_unit, m_bar, P_thermal_central;
  // Grabbing hydro pkg and units objects now...
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  k_b = units.k_boltzmann();
  atomic_mass_unit = units.atomic_mass_unit();
  m_bar = pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;
  P_thermal_central = T0 * k_b * rho0 / m_bar;

  // Printing out input values for slurm records
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "gamma ...... " << pin->GetReal("hydro", "gamma") << std::endl;
    std::cout << "B0 [Gauss] . " << B0 * std::sqrt(4*M_PI) << std::endl;
    std::cout << "rho0 [g] ... " << rho0 << std::endl;
    std::cout << "T0 [K] ..... " << T0 << std::endl;
    std::cout << "P_th [erg/cc]" << P_thermal_central << std::endl;
    std::cout << "psi0 [G*cm]. " << psi0 << std::endl;
    std::cout << "lambda ..... " << lambda << std::endl;
    std::cout << "Lx [cm] .... " << Lx << std::endl;
    std::cout << "Ly [cm] .... " << Ly << std::endl;
  }

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: GEM", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x, y;
        x = coords.Xc<1>(i);
        y = coords.Xc<2>(j);
        u(IDN, k, j, i) = rho0 / std::pow(std::cosh(y/lambda), 2) + rhoinf; // Density

        u(IM1, k, j, i) = 0.0;  // Initial Momentum is zero
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        // Perturbation from A_z = psi0 * cos(2pi x/Lx) * cos(pi y/Ly)
        // (psi0 already has B*L units)
        Real bx, by;  // storing curl(A_z)
        bx = M_PI*psi0*std::cos(2*M_PI*x/Lx)*std::sin(M_PI*y/Ly)/(Ly);
        by = 2*M_PI*psi0*std::sin(2*M_PI*x/Lx)*std::cos(M_PI*y/Ly)/(Lx);
  
        u(IB1, k, j, i) = B0*std::tanh(y/lambda) + bx;
        u(IB2, k, j, i) = by;
        u(IB3, k, j, i) = 0.0;

        // if spitzer turned on, use units, otherwise treat T0 as an initial P_thermal
        Real P_thermal = T0 * k_b * u(IDN, k, j, i) / m_bar;

        Real P = P_thermal;

        u(IEN, k, j, i) =
            P / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));

      });
}
} // namespace GEM
