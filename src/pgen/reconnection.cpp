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
#include "../hydro/diffusion/diffusion.hpp" // For storing eta later

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
        // Calculating 
        Real rho = u(IDN, k, j, i);
        Real p = u(IPR, k, j, i);
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
  Real B0   = pin->GetOrAddReal("problem/reconnection", "B0", 1.0) / std::sqrt(4*M_PI);
  Real rho0 = pin->GetOrAddReal("problem/reconnection", "rho0", 1.0);
  Real T0   =  pin->GetOrAddReal("problem/reconnection", "T0", 2.0);
  Real w    = pin->GetOrAddReal("problem/reconnection", "w", 0.5);
  Real delta = pin->GetOrAddReal("problem/reconnection", "delta", 0.5);
  Real powP = pin->GetOrAddReal("problem/reconnection", "powP", 2.0);
  Real powV = pin->GetOrAddReal("problem/reconnection", "powV", 0.0);
  Real v0   = pin->GetOrAddReal("problem/reconnection", "v0", 0.0);
  Real powRho = pin->GetOrAddReal("problem/reconnection", "powRho", 0.0);
  Real v_offset = pin->GetOrAddReal("problem/reconnection", "v_offset", 1.0); 
  Real loop_x1 = pin->GetOrAddReal("problem/reconnection", "loop_x1", -1.0);
  Real loop_x2 = pin->GetOrAddReal("problem/reconnection", "loop_x2",  1.0);
  
  // Checking if spitzer or fixed ohmic resistivity is turned on:
  Real k_b, atomic_mass_unit, m_bar, P_thermal_central;
  // Grabbing hydro pkg and units objects now...
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  k_b = units.k_boltzmann();
  atomic_mass_unit = units.atomic_mass_unit();
  m_bar = pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;
  // std::cout << "P_thermal_central = " << T0 << " * " << k_b << " * " << rho0 << " / " << m_bar << std::endl;
  P_thermal_central = T0 * k_b * rho0 / m_bar;

  // Printing out input values for slurm records
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "gamma ..... " << pin->GetReal("hydro", "gamma") << std::endl;
    std::cout << "B0 [Gauss]  " << B0 * std::sqrt(4*M_PI) << std::endl;
    std::cout << "rho0 [g] .. " << rho0 << std::endl;
    std::cout << "v0 [cm/s] . " << v0 << std::endl;
    std::cout << "T0 [K] .... " << T0 << std::endl;
    std::cout << "w [cm] .... " << w << std::endl;
    std::cout << "v_offset .. " << v_offset << std::endl;
    std::cout << "delta [cm]  " << delta << std::endl;
    std::cout << "powP [-] .. " << powP << std::endl;
    std::cout << "powV [-] .. " << powV << std::endl;
    std::cout << "powRho [-]  " << powRho << std::endl;
    std::cout << "resistivity:  " << detected_resistivity_type << std::endl;
    std::cout << "mbar: ..... " << m_bar << std::endl;
    // Displaying a sample beta value near center of domain
    std::cout << P_thermal_central << " / (0.5 * 4 * 3.14 * SQR(" << B0 << "))" << std::endl;
    Real beta_sample = P_thermal_central / (0.5 * 4 * M_PI * SQR(B0));
    std::cout << "Initializing with central plasma beta ~ " << beta_sample << std::endl;
    std::cout << "========================================" << std::endl;
  }

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: reconnection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x, y;
        x = coords.Xc<1>(i);
        y = coords.Xc<2>(j);
        u(IDN, k, j, i) = rho0 * (1.0 + std::pow(std::abs(y), powRho)); // Density

        u(IM1, k, j, i) = 0.0;  // Initial Momentum is zero
        u(IM2, k, j, i) = -1.0 * SIGN(y) * u(IDN, k, j, i) * v0 * (v_offset + std::pow(std::abs(y), powV));
        u(IM3, k, j, i) = 0.0;

        Real b1x, b2x, b1y, b2y;  // Helper variables for clarity
        // Gaussian Loop in x (clamping exponential terms to prevent infinitely small and large values)
        Real exp1 = std::clamp(-std::pow(4*(x-loop_x1)/w, 2) - std::pow(4*y/delta, 2), -300.0, 300.0);
        Real exp2 = std::clamp(-std::pow(4*(x-loop_x2)/w, 2) - std::pow(4*y/delta, 2), -300.0, 300.0);
        b1x = 2*M_PI*w*y/delta * std::exp(exp1);
        b2x = 2*M_PI*w*y/delta * std::exp(exp2);
        // Gaussian Loop in y
        b1y = 2*M_PI*delta/w * (loop_x1 - x) * std::exp(exp1);
        b2y = 2*M_PI*delta/w * (loop_x2 - x) * std::exp(exp2);
        u(IB1, k, j, i) = B0*(b1x + b2x + std::tanh(y/delta));
        u(IB2, k, j, i) = B0*(b1y + b2y);
        u(IB3, k, j, i) = 0.0;

        // if spitzer turned on, use units, otherwise treat T0 as an initial P_thermal
        Real P_thermal = T0 * k_b * u(IDN, k, j, i)/ m_bar;
        Real P = P_thermal * (1.0 + std::pow(std::abs(y), powP));

        u(IEN, k, j, i) =
            P / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      });
}
} // namespace reconnection
