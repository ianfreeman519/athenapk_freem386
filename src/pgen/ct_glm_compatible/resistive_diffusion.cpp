//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file resistive_diffusion.cpp
//! \brief Analytic magnetic-diffusion problems for face-centered constrained transport.

#include <cmath>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "utils/error_checking.hpp"

namespace resistive_diffusion {
using namespace parthenon::driver::prelude;

namespace {

KOKKOS_INLINE_FUNCTION
Real FourierAz(const Real x, const Real y, const Real amp, const Real kx,
               const Real ky) {
  return amp * sin(kx * x) * sin(ky * y);
}

} // namespace

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  PARTHENON_REQUIRE_THROWS(hydro_pkg->Param<Fluid>("fluid") == Fluid::ucthlldmhd,
                           "resistive_diffusion requires hydro/fluid=ucthlldmhd");
  const int ndim = pmb->pmy_mesh->ndim;

  const int iprob = pin->GetOrAddInteger("problem/resistive_diffusion", "iprob", 0);
  PARTHENON_REQUIRE_THROWS(iprob == 0 || iprob == 1 || iprob == 2 || iprob == 3 ||
                               iprob == 10 || iprob == 20,
                           "Unknown problem/resistive_diffusion/iprob");

  const Real rho0 = pin->GetOrAddReal("problem/resistive_diffusion", "rho", 1.0);
  const Real p0 = pin->GetOrAddReal("problem/resistive_diffusion", "pressure", 1.0);
  const Real amp = pin->GetOrAddReal("problem/resistive_diffusion", "amp", 1.e-6);
  const Real bx0 = pin->GetOrAddReal("problem/resistive_diffusion", "Bx", 0.25);
  const Real by0 = pin->GetOrAddReal("problem/resistive_diffusion", "By", -0.125);
  const Real bz0 = pin->GetOrAddReal("problem/resistive_diffusion", "Bz", 0.0625);
  const int mode_x = pin->GetOrAddInteger("problem/resistive_diffusion", "mode_x", 1);
  const int mode_y = pin->GetOrAddInteger("problem/resistive_diffusion", "mode_y", 1);
  const int mode_z = pin->GetOrAddInteger("problem/resistive_diffusion", "mode_z", 1);
  PARTHENON_REQUIRE_THROWS(rho0 > 0.0 && p0 > 0.0,
                           "resistive_diffusion requires positive density and pressure");
  PARTHENON_REQUIRE_THROWS(mode_x > 0 && mode_y > 0 && mode_z > 0,
                           "resistive_diffusion mode numbers must be positive");

  const auto &mesh_size = pmb->pmy_mesh->mesh_size;
  const Real lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real ly = mesh_size.xmax(parthenon::X2DIR) - mesh_size.xmin(parthenon::X2DIR);
  const Real lz = mesh_size.xmax(parthenon::X3DIR) - mesh_size.xmin(parthenon::X3DIR);
  const Real kx = 2.0 * M_PI * mode_x / lx;
  const Real ky = 2.0 * M_PI * mode_y / ly;
  const Real kz = 2.0 * M_PI * mode_z / lz;
  const Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  auto &bface_dev = mbd->Get("Bface").data;
  auto u = u_dev.GetHostMirrorAndCopy();
  auto bface = bface_dev.GetHostMirrorAndCopy();
  auto b1f = bface.Get(IBF1, 0, 0, 0);
  auto b2f = bface.Get(IBF2, 0, 0, 0);
  auto b3f = bface.Get(IBF3, 0, 0, 0);

  // F1 values live at x faces and at cell centers in the transverse directions.
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        Real value = 0.0;
        if (iprob == 0) {
          value = bx0;
        } else if (iprob == 10) {
          const Real x = coords.Xf<1>(i);
          const Real y0 = coords.Xf<2>(j);
          const Real y1 = coords.Xf<2>(j + 1);
          value = (FourierAz(x, y1, amp, kx, ky) -
                   FourierAz(x, y0, amp, kx, ky)) /
                  (y1 - y0);
        } else if (iprob == 20) {
          value = amp * (sin(kz * coords.Xc<3>(k)) +
                         cos(ky * coords.Xc<2>(j)));
        }
        b1f(k, j, i) = value;
      }
    }
  }

  // F2 values live at y faces and at cell centers in x and z.
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        Real value = 0.0;
        if (iprob == 0) {
          value = by0;
        } else if (iprob == 1) {
          value = amp * sin(kx * coords.Xc<1>(i));
        } else if (iprob == 10) {
          const Real y = coords.Xf<2>(j);
          const Real x0 = coords.Xf<1>(i);
          const Real x1 = coords.Xf<1>(i + 1);
          value = -(FourierAz(x1, y, amp, kx, ky) -
                    FourierAz(x0, y, amp, kx, ky)) /
                  (x1 - x0);
        } else if (iprob == 20) {
          value = amp * (sin(kx * coords.Xc<1>(i)) +
                         cos(kz * coords.Xc<3>(k)));
        }
        b2f(k, j, i) = value;
      }
    }
  }

  // F3 values live at z faces and at cell centers in x and y.
  for (int k = kb.s; k <= kb.e + (ndim >= 3 ? 1 : 0); ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        Real value = 0.0;
        if (iprob == 0) {
          value = bz0;
        } else if (iprob == 2) {
          value = amp * sin(kx * coords.Xc<1>(i));
        } else if (iprob == 3) {
          value = amp * sin(ky * coords.Xc<2>(j));
        } else if (iprob == 20) {
          value = amp * (sin(ky * coords.Xc<2>(j)) +
                         cos(kx * coords.Xc<1>(i)));
        }
        b3f(k, j, i) = value;
      }
    }
  }

  // The evolved cell field is initialized from the same face averages used by CT.
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        u(IDN, k, j, i) = rho0;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = 0.5 * (b1f(k, j, i) + b1f(k, j, i + 1));
        u(IB2, k, j, i) = 0.5 * (b2f(k, j, i) + b2f(k, j + 1, i));
        u(IB3, k, j, i) =
            ndim >= 3 ? 0.5 * (b3f(k, j, i) + b3f(k + 1, j, i))
                      : b3f(k, j, i);
        u(IEN, k, j, i) =
            p0 / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) +
                   SQR(u(IB3, k, j, i)));
      }
    }
  }

  bface_dev.DeepCopy(bface);
  u_dev.DeepCopy(u);
}

} // namespace resistive_diffusion
