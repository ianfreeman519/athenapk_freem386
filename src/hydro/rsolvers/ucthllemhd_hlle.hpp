// AthenaPK - a performance portable block structured AMR MHD code
#ifndef RSOLVERS_UCTHLLEMHD_HLLE_HPP_
#define RSOLVERS_UCTHLLEMHD_HLLE_HPP_

#include <algorithm>
#include <cmath>

#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"

//! HLLE MHD flux together with the face data required by UCT-HLL.
template <>
struct Riemann<Fluid::ucthllemhd, RiemannSolver::hlle> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const ScratchPad2D<Real> &wl,
        const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        VariablePack<Real> &uct_aux, const AdiabaticCTMHDEOS &eos,
        const Real /*c_h*/) {
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;
    const Real igm1 = 1.0 / (eos.GetGamma() - 1.0);

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      const Real dl = wl(IDN, i);
      const Real dr = wr(IDN, i);
      const Real vxl = wl(ivx, i);
      const Real vxr = wr(ivx, i);
      const Real vyl = wl(ivy, i);
      const Real vyr = wr(ivy, i);
      const Real vzl = wl(ivz, i);
      const Real vzr = wr(ivz, i);
      const Real pl = wl(IPR, i);
      const Real pr = wr(IPR, i);
      // CT guarantees a single normal magnetic field at each face.
      const Real bx = 0.5 * (wl(iBx, i) + wr(iBx, i));
      const Real byl = wl(iBy, i);
      const Real byr = wr(iBy, i);
      const Real bzl = wl(iBz, i);
      const Real bzr = wr(iBz, i);

      const Real pbl = 0.5 * (SQR(bx) + SQR(byl) + SQR(bzl));
      const Real pbr = 0.5 * (SQR(bx) + SQR(byr) + SQR(bzr));
      const Real el = pl * igm1 +
                      0.5 * dl * (SQR(vxl) + SQR(vyl) + SQR(vzl)) + pbl;
      const Real er = pr * igm1 +
                      0.5 * dr * (SQR(vxr) + SQR(vyr) + SQR(vzr)) + pbr;

      const Real cfl = eos.FastMagnetosonicSpeed(dl, pl, bx, byl, bzl);
      const Real cfr = eos.FastMagnetosonicSpeed(dr, pr, bx, byr, bzr);
      const Real sl = std::min(vxl - cfl, vxr - cfr);
      const Real sr = std::max(vxl + cfl, vxr + cfr);
      const Real am = -std::min(sl, 0.0);
      const Real ap = std::max(sr, 0.0);
      const Real asum = ap + am;

      Real ul[8], ur[8], fl[8], fr[8], flux[8];
      ul[IDN] = dl;
      ul[IV1] = dl * vxl;
      ul[IV2] = dl * vyl;
      ul[IV3] = dl * vzl;
      ul[IEN] = el;
      ul[IB1] = bx;
      ul[IB2] = byl;
      ul[IB3] = bzl;
      ur[IDN] = dr;
      ur[IV1] = dr * vxr;
      ur[IV2] = dr * vyr;
      ur[IV3] = dr * vzr;
      ur[IEN] = er;
      ur[IB1] = bx;
      ur[IB2] = byr;
      ur[IB3] = bzr;

      fl[IDN] = dl * vxl;
      fl[IV1] = dl * SQR(vxl) + pl + pbl - SQR(bx);
      fl[IV2] = dl * vyl * vxl - bx * byl;
      fl[IV3] = dl * vzl * vxl - bx * bzl;
      fl[IEN] = vxl * (el + pl + pbl - SQR(bx)) - bx * (vyl * byl + vzl * bzl);
      fl[IB1] = 0.0;
      fl[IB2] = byl * vxl - bx * vyl;
      fl[IB3] = bzl * vxl - bx * vzl;
      fr[IDN] = dr * vxr;
      fr[IV1] = dr * SQR(vxr) + pr + pbr - SQR(bx);
      fr[IV2] = dr * vyr * vxr - bx * byr;
      fr[IV3] = dr * vzr * vxr - bx * bzr;
      fr[IEN] = vxr * (er + pr + pbr - SQR(bx)) - bx * (vyr * byr + vzr * bzr);
      fr[IB1] = 0.0;
      fr[IB2] = byr * vxr - bx * vyr;
      fr[IB3] = bzr * vxr - bx * vzr;

      for (int n = 0; n < 8; ++n) {
        flux[n] = asum > 0.0
                      ? (ap * fl[n] + am * fr[n] - ap * am * (ur[n] - ul[n])) / asum
                      : 0.5 * (fl[n] + fr[n]);
      }
      cons.flux(ivx, IDN, k, j, i) = flux[IDN];
      cons.flux(ivx, ivx, k, j, i) = flux[IV1];
      cons.flux(ivx, ivy, k, j, i) = flux[IV2];
      cons.flux(ivx, ivz, k, j, i) = flux[IV3];
      cons.flux(ivx, IEN, k, j, i) = flux[IEN];
      cons.flux(ivx, iBx, k, j, i) = flux[IB1];
      cons.flux(ivx, iBy, k, j, i) = flux[IB2];
      cons.flux(ivx, iBz, k, j, i) = flux[IB3];

      const Real aL = asum > 0.0 ? ap / asum : 0.5;
      const Real aR = asum > 0.0 ? am / asum : 0.5;
      const Real d = asum > 0.0 ? ap * am / asum : 0.0;
      const Real vT1 = aL * vyl + aR * vyr;
      const Real vT2 = aL * vzl + aR * vzr;
      const auto face = ivx == IV1 ? parthenon::TopologicalElement::F1
                                   : (ivx == IV2 ? parthenon::TopologicalElement::F2
                                                 : parthenon::TopologicalElement::F3);
      uct_aux(face, AL, k, j, i) = aL;
      uct_aux(face, AR, k, j, i) = aR;
      uct_aux(face, DL, k, j, i) = d;
      uct_aux(face, DR, k, j, i) = d;
      uct_aux(face, VBART1, k, j, i) = vT1;
      uct_aux(face, VBART2, k, j, i) = vT2;
    });
  }
};

#endif // RSOLVERS_UCTHLLEMHD_HLLE_HPP_
