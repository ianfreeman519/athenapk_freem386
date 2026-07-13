//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file boundary_conditions_apk.chpp
//  \brief AthenaPK specific boundary conditions
//

#ifndef BVALS_BOUNDARY_CONDITIONS_APK_HPP_
#define BVALS_BOUNDARY_CONDITIONS_APK_HPP_

#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

#include "basic_types.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

#include "../main.hpp"

namespace Hydro {
namespace BoundaryFunction {

using namespace parthenon::package::prelude;
using parthenon::CoordinateDirection;
// using parthenon::MeshBlockData;
// using parthenon::Real;
using parthenon::BoundaryFunction::BCSide;

template <typename PackT, typename CoordsT>
KOKKOS_INLINE_FUNCTION Real ExtrapolateLinearX1(const PackT &cons, const CoordsT &coords,
                                                const int var, const int k, const int j,
                                                const int ref, const int interior_step,
                                                const Real x_ghost) {
  const int i0 = ref;
  const int i1 = ref + interior_step;
  const Real x0 = coords.template Xc<1>(i0);
  const Real x1 = coords.template Xc<1>(i1);
  const Real u0 = cons(var, k, j, i0);
  const Real u1 = cons(var, k, j, i1);
  const Real dx = x1 - x0;
  if (dx == 0.0) {
    return u0;
  }
  return u0 + (u1 - u0) * (x_ghost - x0) / dx;
}

template <typename PackT, typename CoordsT>
KOKKOS_INLINE_FUNCTION Real ExtrapolateQuadraticX1(const PackT &cons, const CoordsT &coords,
                                                   const int var, const int k, const int j,
                                                   const int ref, const int interior_step,
                                                   const Real x_ghost) {
  const int i0 = ref;
  const int i1 = ref + interior_step;
  const int i2 = ref + 2 * interior_step;
  const Real x0 = coords.template Xc<1>(i0);
  const Real x1 = coords.template Xc<1>(i1);
  const Real x2 = coords.template Xc<1>(i2);
  const Real u0 = cons(var, k, j, i0);
  const Real u1 = cons(var, k, j, i1);
  const Real u2 = cons(var, k, j, i2);

  const Real d0 = (x0 - x1) * (x0 - x2);
  const Real d1 = (x1 - x0) * (x1 - x2);
  const Real d2 = (x2 - x0) * (x2 - x1);
  if (d0 == 0.0 || d1 == 0.0 || d2 == 0.0) {
    return ExtrapolateLinearX1(cons, coords, var, k, j, ref, interior_step, x_ghost);
  }

  const Real l0 = ((x_ghost - x1) * (x_ghost - x2)) / d0;
  const Real l1 = ((x_ghost - x0) * (x_ghost - x2)) / d1;
  const Real l2 = ((x_ghost - x0) * (x_ghost - x1)) / d2;
  return l0 * u0 + l1 * u1 + l2 * u2;
}

template <CoordinateDirection DIR, BCSide SIDE>
void ReflectBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  MeshBlock *pmb = mbd->GetBlockPointer();

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  PARTHENON_REQUIRE_THROWS(
      fluid == Fluid::euler,
      "Reflecting boundary conditions for MHD need special treatment.");

  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = (2 * ref) + (INNER ? -1 : 1);

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const bool fine = false; // no usage of fine fields in AthenaPK for now

  const auto nv = IndexRange{0, cons.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectBC", nv, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
        const bool reflect = v == DIR;
        cons(v, k, j, i) =
            (reflect ? -1.0 : 1.0) *
            cons(v, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
      });
}

//========================================================================================
// Linear Extrapolation Boundary Condition (first-order): "LinearBC"
//========================================================================================

template <CoordinateDirection DIR, BCSide SIDE>
void LinearBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  MeshBlock *pmb = mbd->GetBlockPointer();

  // Which axis & side?
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  // Interior range and edge index "ref" (cell adjacent to the ghost region)
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  // Tell Parthenon which ghost face to iterate
  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // Data & coordinates (coarse vs fine)
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const bool fine = false;
  const auto nv = IndexRange{0, cons.GetDim(4) - 1};
  // Capture the right coords object for device
  auto coords = coarse ? pmb->coords : pmb->coords;

  pmb->par_for_bndry(
      "LinearBC_point_slope", nv, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
        // Neighbor one cell deeper into the interior along the active direction
        if (X1) {
          const int nbr = ref + (INNER ? +1 : -1);

          const Real x_ref = coords.Xc<1>(ref);
          const Real x_nbr = coords.Xc<1>(nbr);
          const Real u_ref = cons(v, k, j, ref);
          const Real u_nbr = cons(v, k, j, nbr);
          const Real dx    = x_ref - x_nbr;

          // Guard against degenerate spacing (shouldn't happen in practice)
          const Real m = (dx != 0.0) ? (u_ref - u_nbr) / dx : 0.0;

          const Real x_g = coords.Xc<1>(i);
          cons(v, k, j, i) = m * (x_g - x_ref) + u_ref;

        } else if (X2) {
          const int nbr = ref + (INNER ? +1 : -1);

          const Real x_ref = coords.Xc<2>(ref);
          const Real x_nbr = coords.Xc<2>(nbr);
          const Real u_ref = cons(v, k, ref, i);
          const Real u_nbr = cons(v, k, nbr, i);
          const Real dx    = x_ref - x_nbr;

          const Real m = (dx != 0.0) ? (u_ref - u_nbr) / dx : 0.0;

          const Real x_g = coords.Xc<2>(j);
          cons(v, k, j, i) = m * (x_g - x_ref) + u_ref;

        } else { // X3
          const int nbr = ref + (INNER ? +1 : -1);

          const Real x_ref = coords.Xc<3>(ref);
          const Real x_nbr = coords.Xc<3>(nbr);
          const Real u_ref = cons(v, ref, j, i);
          const Real u_nbr = cons(v, nbr, j, i);
          const Real dx    = x_ref - x_nbr;

          const Real m = (dx != 0.0) ? (u_ref - u_nbr) / dx : 0.0;

          const Real x_g = coords.Xc<3>(k);
          cons(v, k, j, i) = m * (x_g - x_ref) + u_ref;
        }
      });
}

template <BCSide SIDE>
void DiodeX1BC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  MeshBlock *pmb = mbd->GetBlockPointer();

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior);
  const int ref = (SIDE == BCSide::Inner) ? range.s : range.e;
  const int interior_step = (SIDE == BCSide::Inner) ? 1 : -1;
  const int interior_cells = range.e - range.s + 1;

  constexpr IndexDomain domain = (SIDE == BCSide::Inner) ? IndexDomain::inner_x1
                                                         : IndexDomain::outer_x1;

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const int nvar = cons.GetDim(4);
  const bool fine = false;
  const auto nb = IndexRange{0, 0};
  auto coords = pmb->coords;

  pmb->par_for_bndry(
      "DiodeX1BC", nb, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        for (int v = 0; v < nvar; ++v) {
          cons(v, k, j, i) = cons(v, k, j, ref);
        }

        const Real rho = cons(IDN, k, j, i);
        const Real m2 = cons(IM2, k, j, i);
        const Real m3 = cons(IM3, k, j, i);
        const Real copied_m1 = cons(IM1, k, j, i);
        const Real copied_b1 = cons(IB1, k, j, i);
        const Real copied_b2 = cons(IB2, k, j, i);
        const Real copied_b3 = cons(IB3, k, j, i);
        const Real copied_ke =
            rho > 0.0 ? 0.5 * (copied_m1 * copied_m1 + m2 * m2 + m3 * m3) / rho : 0.0;
        const Real copied_me =
            0.5 * (copied_b1 * copied_b1 + copied_b2 * copied_b2 + copied_b3 * copied_b3);
        Real internal_e = cons(IEN, k, j, i) - copied_ke - copied_me;
        if (internal_e < 0.0) {
          internal_e = 0.0;
        }

        Real m1 = copied_m1;
        if ((SIDE == BCSide::Inner && m1 > 0.0) || (SIDE == BCSide::Outer && m1 < 0.0)) {
          m1 = 0.0;
        }
        cons(IM1, k, j, i) = m1;

        const Real x_ghost = coords.Xc<1>(i);
        if (interior_cells >= 3) {
          cons(IB1, k, j, i) =
              ExtrapolateQuadraticX1(cons, coords, IB1, k, j, ref, interior_step, x_ghost);
          cons(IB2, k, j, i) =
              ExtrapolateQuadraticX1(cons, coords, IB2, k, j, ref, interior_step, x_ghost);
          cons(IB3, k, j, i) =
              ExtrapolateQuadraticX1(cons, coords, IB3, k, j, ref, interior_step, x_ghost);
        } else if (interior_cells >= 2) {
          cons(IB1, k, j, i) =
              ExtrapolateLinearX1(cons, coords, IB1, k, j, ref, interior_step, x_ghost);
          cons(IB2, k, j, i) =
              ExtrapolateLinearX1(cons, coords, IB2, k, j, ref, interior_step, x_ghost);
          cons(IB3, k, j, i) =
              ExtrapolateLinearX1(cons, coords, IB3, k, j, ref, interior_step, x_ghost);
        }

        const Real b1 = cons(IB1, k, j, i);
        const Real b2 = cons(IB2, k, j, i);
        const Real b3 = cons(IB3, k, j, i);
        const Real new_ke = rho > 0.0 ? 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / rho : 0.0;
        const Real new_me = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
        cons(IEN, k, j, i) = internal_e + new_ke + new_me;
      });
}

} // namespace BoundaryFunction
} // namespace Hydro

#endif // BVALS_BOUNDARY_CONDITIONS_APK_HPP_
