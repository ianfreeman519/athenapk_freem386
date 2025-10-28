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

} // namespace BoundaryFunction
} // namespace Hydro

#endif // BVALS_BOUNDARY_CONDITIONS_APK_HPP_
