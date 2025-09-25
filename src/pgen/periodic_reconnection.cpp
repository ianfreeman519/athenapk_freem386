//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file periodic_reconnection.cpp
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

namespace periodic_reconnection {
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
      "periodic_reconnection::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
  Real B0   = pin->GetOrAddReal("problem/periodic_reconnection", "B0", 1.0);
  Real rho0 = pin->GetOrAddReal("problem/periodic_reconnection", "rho0", 1.0);
  Real P0   = pin->GetOrAddReal("problem/periodic_reconnection", "P0", 2.0);
  Real delta= pin->GetOrAddReal("problem/periodic_reconnection", "delta", 0.5);
  Real P1 = pin->GetOrAddReal("problem/periodic_reconnection", "P1", 1.0);
  Real Nloop= pin->GetOrAddReal("problem/periodic_reconnection", "Nloop", 2);
  Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  Real Lx   = x1max - x1min;
  Real Ly   = x2max - x2min;

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

        // Helper variables
        Real bx = 0.0;
        Real by = 0.0;
        Real w = 0.5 * Lx / Nloop;
        // For loop over the field loops
        for (Real xl = -Lx/2; xl <= Lx/2; xl += Lx/(Nloop-1)){
          // Clamping exponential terms such that 1e-86 < exp(expterm) < 1e86
          // Without this extra step, exp(-very_large_number) leads to underflow nans
          Real expterm = -std::pow(4*(x-xl)/w, 2) - std::pow(4*y/delta, 2);
          expterm = std::clamp(expterm, -300.0, 300.0); 
          bx += 2*M_PI*w*y/delta * std::exp(expterm);
          by += -2*M_PI*delta/w * (x-xl) * std::exp(expterm);
        };

        // Set background pressure and magnetic fields
        u(IB1, k, j, i) = B0*bx + B0*std::tanh(y/delta);
        u(IB2, k, j, i) = B0*by;
        u(IB3, k, j, i) = 0.0;
        // Pressure
        Real P = P0 + P1*std::pow(std::sin((M_PI*y)/(2*Ly)), 2);

        // Internal energy profile
        u(IEN, k, j, i) =
            P / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
    });
  } 
} // namespace periodic_reconnection
