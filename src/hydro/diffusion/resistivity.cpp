//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file resistivity.cpp
//! \brief

// Parthenon headers
#include <cmath>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "config.hpp"
#include "diffusion.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

namespace {

template <bool include_energy>
void OhmicDiffFluxIsoFixedImpl(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;
  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  const auto eta = ohm_diff.Get(0.0, 0.0);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X1 fluxes (ohmic)", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        const auto d3B1 =
            ndim > 2 ? (0.5 * (prim(IB1, k + 1, j, i - 1) + prim(IB1, k + 1, j, i)) -
                        0.5 * (prim(IB1, k - 1, j, i - 1) + prim(IB1, k - 1, j, i))) /
                           (coords.Xf<3, 1>(k + 1, j, i) - coords.Xf<3, 1>(k - 1, j, i))
                     : 0.0;
        const auto d1B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        const auto j2 = d3B1 - d1B3;

        const auto d1B2 =
            (prim(IB2, k, j, i) - prim(IB2, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        const auto d2B1 =
            ndim > 1 ? (0.5 * (prim(IB1, k, j + 1, i - 1) + prim(IB1, k, j + 1, i)) -
                        0.5 * (prim(IB1, k, j - 1, i - 1) + prim(IB1, k, j - 1, i))) /
                           (coords.Xf<2, 1>(k, j + 1, i) - coords.Xf<2, 1>(k, j - 1, i))
                     : 0.0;
        const auto j3 = d1B2 - d2B1;

        cons.flux(X1DIR, IB2, k, j, i) += -eta * j3;
        cons.flux(X1DIR, IB3, k, j, i) += eta * j2;
        if constexpr (include_energy) {
          cons.flux(X1DIR, IEN, k, j, i) +=
              0.5 * eta *
              ((prim(IB3, k, j, i - 1) + prim(IB3, k, j, i)) * j2 -
               (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i)) * j3);
        }
      });

  if (ndim < 2) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X2 fluxes (ohmic)", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        const auto d1B2 = (0.5 * (prim(IB2, k, j - 1, i + 1) + prim(IB2, k, j, i + 1)) -
                           0.5 * (prim(IB2, k, j - 1, i - 1) + prim(IB2, k, j, i - 1))) /
                          (coords.Xf<1, 2>(k, j, i + 1) - coords.Xf<1, 2>(k, j, i - 1));
        const auto d2B1 =
            (prim(IB1, k, j, i) - prim(IB1, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        const auto j3 = d1B2 - d2B1;

        const auto d2B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        const auto d3B2 =
            ndim > 2 ? (0.5 * (prim(IB2, k + 1, j - 1, i) + prim(IB2, k + 1, j, i)) -
                        0.5 * (prim(IB2, k - 1, j - 1, i) + prim(IB2, k - 1, j, i))) /
                           (coords.Xf<3, 2>(k + 1, j, i) - coords.Xf<3, 2>(k - 1, j, i))
                     : 0.0;
        const auto j1 = d2B3 - d3B2;

        cons.flux(X2DIR, IB1, k, j, i) += eta * j3;
        cons.flux(X2DIR, IB3, k, j, i) += -eta * j1;
        if constexpr (include_energy) {
          cons.flux(X2DIR, IEN, k, j, i) +=
              0.5 * eta *
              ((prim(IB1, k, j - 1, i) + prim(IB1, k, j, i)) * j3 -
               (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i)) * j1);
        }
      });

  if (ndim < 3) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X3 fluxes (ohmic)", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        const auto d2B3 = (0.5 * (prim(IB3, k - 1, j + 1, i) + prim(IB3, k, j + 1, i)) -
                           0.5 * (prim(IB3, k - 1, j - 1, i) + prim(IB3, k, j - 1, i))) /
                          (coords.Xf<2, 3>(k, j + 1, i) - coords.Xf<2, 3>(k, j - 1, i));
        const auto d3B2 =
            (prim(IB2, k, j, i) - prim(IB2, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        const auto j1 = d2B3 - d3B2;

        const auto d3B1 =
            (prim(IB1, k, j, i) - prim(IB1, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        const auto d1B3 = (0.5 * (prim(IB3, k - 1, j, i + 1) + prim(IB3, k, j, i + 1)) -
                           0.5 * (prim(IB3, k - 1, j, i - 1) + prim(IB3, k, j, i - 1))) /
                          (coords.Xf<1, 3>(k, j, i + 1) - coords.Xf<1, 3>(k, j, i - 1));
        const auto j2 = d3B1 - d1B3;

        cons.flux(X3DIR, IB1, k, j, i) += -eta * j2;
        cons.flux(X3DIR, IB2, k, j, i) += eta * j1;
        if constexpr (include_energy) {
          cons.flux(X3DIR, IEN, k, j, i) +=
              0.5 * eta *
              ((prim(IB2, k - 1, j, i) + prim(IB2, k, j, i)) * j1 -
               (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i)) * j2);
        }
      });
}

template <bool include_energy, bool use_iterate_thermo>
void OhmicDiffFluxGeneralImpl(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eint_pack = md->PackVariables(std::vector<std::string>{"eint_iter"});

  const int ndim = pmb->pmy_mesh->ndim;
  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  const auto ohm_diff_val = ohm_diff;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X1 fluxes (ohmic)", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const Real rho_at_face = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j, i - 1));
        Real p_at_face = 0.5 * (prim(IPR, k, j, i) + prim(IPR, k, j, i - 1));
        if constexpr (use_iterate_thermo) {
          p_at_face = 0.5 * (prim(IDN, k, j, i) * eint_pack(b, 0, k, j, i) * gm1 +
                             prim(IDN, k, j, i - 1) * eint_pack(b, 0, k, j, i - 1) * gm1);
        }
        const Real eta = ohm_diff_val.Get(p_at_face, rho_at_face);

        const auto d3B1 =
            ndim > 2 ? (0.5 * (prim(IB1, k + 1, j, i - 1) + prim(IB1, k + 1, j, i)) -
                        0.5 * (prim(IB1, k - 1, j, i - 1) + prim(IB1, k - 1, j, i))) /
                           (coords.Xf<3, 1>(k + 1, j, i) - coords.Xf<3, 1>(k - 1, j, i))
                     : 0.0;
        const auto d1B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        const auto j2 = d3B1 - d1B3;

        const auto d1B2 =
            (prim(IB2, k, j, i) - prim(IB2, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        const auto d2B1 =
            ndim > 1 ? (0.5 * (prim(IB1, k, j + 1, i - 1) + prim(IB1, k, j + 1, i)) -
                        0.5 * (prim(IB1, k, j - 1, i - 1) + prim(IB1, k, j - 1, i))) /
                           (coords.Xf<2, 1>(k, j + 1, i) - coords.Xf<2, 1>(k, j - 1, i))
                     : 0.0;
        const auto j3 = d1B2 - d2B1;

        cons.flux(X1DIR, IB2, k, j, i) += -eta * j3;
        cons.flux(X1DIR, IB3, k, j, i) += eta * j2;
        if constexpr (include_energy) {
          cons.flux(X1DIR, IEN, k, j, i) +=
              0.5 * eta *
              ((prim(IB3, k, j, i - 1) + prim(IB3, k, j, i)) * j2 -
               (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i)) * j3);
        }
      });

  if (ndim < 2) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X2 fluxes (ohmic)", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const Real rho_at_face = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j - 1, i));
        Real p_at_face = 0.5 * (prim(IPR, k, j, i) + prim(IPR, k, j - 1, i));
        if constexpr (use_iterate_thermo) {
          p_at_face = 0.5 * (prim(IDN, k, j, i) * eint_pack(b, 0, k, j, i) * gm1 +
                             prim(IDN, k, j - 1, i) * eint_pack(b, 0, k, j - 1, i) * gm1);
        }
        const Real eta = ohm_diff_val.Get(p_at_face, rho_at_face);

        const auto d1B2 = (0.5 * (prim(IB2, k, j - 1, i + 1) + prim(IB2, k, j, i + 1)) -
                           0.5 * (prim(IB2, k, j - 1, i - 1) + prim(IB2, k, j, i - 1))) /
                          (coords.Xf<1, 2>(k, j, i + 1) - coords.Xf<1, 2>(k, j, i - 1));
        const auto d2B1 =
            (prim(IB1, k, j, i) - prim(IB1, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        const auto j3 = d1B2 - d2B1;

        const auto d2B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        const auto d3B2 =
            ndim > 2 ? (0.5 * (prim(IB2, k + 1, j - 1, i) + prim(IB2, k + 1, j, i)) -
                        0.5 * (prim(IB2, k - 1, j - 1, i) + prim(IB2, k - 1, j, i))) /
                           (coords.Xf<3, 2>(k + 1, j, i) - coords.Xf<3, 2>(k - 1, j, i))
                     : 0.0;
        const auto j1 = d2B3 - d3B2;

        cons.flux(X2DIR, IB1, k, j, i) += eta * j3;
        cons.flux(X2DIR, IB3, k, j, i) += -eta * j1;
        if constexpr (include_energy) {
          cons.flux(X2DIR, IEN, k, j, i) +=
              0.5 * eta *
              ((prim(IB1, k, j - 1, i) + prim(IB1, k, j, i)) * j3 -
               (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i)) * j1);
        }
      });

  if (ndim < 3) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X3 fluxes (ohmic)", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const Real rho_at_face = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k - 1, j, i));
        Real p_at_face = 0.5 * (prim(IPR, k, j, i) + prim(IPR, k - 1, j, i));
        if constexpr (use_iterate_thermo) {
          p_at_face = 0.5 * (prim(IDN, k, j, i) * eint_pack(b, 0, k, j, i) * gm1 +
                             prim(IDN, k - 1, j, i) * eint_pack(b, 0, k - 1, j, i) * gm1);
        }
        const Real eta = ohm_diff_val.Get(p_at_face, rho_at_face);

        const auto d2B3 = (0.5 * (prim(IB3, k - 1, j + 1, i) + prim(IB3, k, j + 1, i)) -
                           0.5 * (prim(IB3, k - 1, j - 1, i) + prim(IB3, k, j - 1, i))) /
                          (coords.Xf<2, 3>(k, j + 1, i) - coords.Xf<2, 3>(k, j - 1, i));
        const auto d3B2 =
            (prim(IB2, k, j, i) - prim(IB2, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        const auto j1 = d2B3 - d3B2;

        const auto d3B1 =
            (prim(IB1, k, j, i) - prim(IB1, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        const auto d1B3 = (0.5 * (prim(IB3, k - 1, j, i + 1) + prim(IB3, k, j, i + 1)) -
                           0.5 * (prim(IB3, k - 1, j, i - 1) + prim(IB3, k, j, i - 1))) /
                          (coords.Xf<1, 3>(k, j, i + 1) - coords.Xf<1, 3>(k, j, i - 1));
        const auto j2 = d3B1 - d1B3;

        cons.flux(X3DIR, IB1, k, j, i) += -eta * j2;
        cons.flux(X3DIR, IB2, k, j, i) += eta * j1;
        if constexpr (include_energy) {
          cons.flux(X3DIR, IEN, k, j, i) +=
              0.5 * eta *
              ((prim(IB2, k - 1, j, i) + prim(IB2, k, j, i)) * j1 -
               (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i)) * j2);
        }
      });
}

} // namespace

Real EstimateOhmicHeatingTimestep(MeshData<Real> *md) {
  // Mimicked from EstimateCoolingTimestep but to calculate timestep from heating term eta*j^2
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_heating_time = std::numeric_limits<Real>::infinity();
  Kokkos::Min<Real> reducer_min(min_heating_time);

  Kokkos::parallel_reduce(
      "OhmicHeating::Timestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &thread_min_heating_time) {
            auto &prim = prim_pack(b);
            auto &coords = prim_pack.GetCoords(b);

            const Real rho = prim(IDN, k, j, i);
            const Real pres = prim(IPR, k, j, i);

            const Real internal_e_dens = pres / gm1;
            const Real eta = ohm_diff.Get(pres, rho);

            const auto ndim = prim_pack.GetNdim();

            // Computing cell-centered current density squared j^2 = j1^2 + j2^2 + j3^2
            Real dBzdy, dBydz, dBxdz, dBzdx, dBydx, dBxdy;
            // In 1D, only dBdx exists. In 2D, dBdy is added. In 3D, dBdz is added.
            // curlBx = dBzdy - dBydz)
            dBzdy = ndim > 1 ? (prim(IB3,k,j+1,i) - prim(IB3,k,j-1,i))/(coords.Xc<2>(j+1)-coords.Xc<2>(j-1)) : 0.0;
            dBydz = ndim > 2 ? (prim(IB2,k+1,j,i) - prim(IB2,k-1,j,i))/(coords.Xc<3>(k+1)-coords.Xc<3>(k-1)) : 0.0;
            // curlBy = dBxdz - dBzdx
            dBxdz = ndim > 2 ? (prim(IB1,k+1,j,i) - prim(IB1,k-1,j,i))/(coords.Xc<3>(k+1)-coords.Xc<3>(k-1)) : 0.0;
            dBzdx = (prim(IB3,k,j,i+1) - prim(IB3,k,j,i-1))/(coords.Xc<1>(i+1)-coords.Xc<1>(i-1));
            // curlBz = dBydx - dBxdy
            dBydx = (prim(IB2,k,j,i+1) - prim(IB2,k,j,i-1))/(coords.Xc<1>(i+1)-coords.Xc<1>(i-1));
            dBxdy = ndim > 1 ? (prim(IB1,k,j+1,i) - prim(IB1,k,j-1,i))/(coords.Xc<2>(j+1)-coords.Xc<2>(j-1)) : 0.0;
            
            // Following definitions are for clarity below
            Real jx = dBzdy - dBydz;
            Real jy = dBxdz - dBzdx;
            Real jz = dBydx - dBxdy;

            // Actually calculating heating time from eta*j^2
            const Real j_squared = SQR(jx) + SQR(jy) + SQR(jz);
            const Real de_dt = eta * j_squared; // heating rate
            const Real cooling_time = fabs(internal_e_dens / de_dt);

            thread_min_heating_time = std::min(cooling_time, thread_min_heating_time);
          },
          reducer_min);
      
      return hydro_pkg->Param<Real>("cfl_diff_heat") * min_heating_time;      
}

Real EstimateResistivityTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_resist = std::numeric_limits<Real>::max();
  const auto ndim = prim_pack.GetNdim();

  Real fac = 0.5;
  if (ndim == 2) {
    fac = 0.25;
  } else if (ndim == 3) {
    fac = 1.0 / 6.0;
  }

  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

  if (ohm_diff.GetType() == Resistivity::ohmic &&
      ohm_diff.GetCoeffType() == ResistivityCoeff::fixed) {
    // TODO(pgrete): once mindx is properly calculated before this loop, we can get rid of
    // it entirely.
    // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
    const auto ohm_diff_coeff = ohm_diff.Get(0.0, 0.0);
    Kokkos::parallel_reduce(
        "EstimateResistivityTimestep (ohmic fixed)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          min_dt =
              fmin(min_dt, SQR(coords.Dxc<1>(k, j, i)) / (ohm_diff_coeff + TINY_NUMBER));
          if (ndim >= 2) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<2>(k, j, i)) / (ohm_diff_coeff + TINY_NUMBER));
          }
          if (ndim >= 3) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<3>(k, j, i)) / (ohm_diff_coeff + TINY_NUMBER));
          }
        },
        Kokkos::Min<Real>(min_dt_resist));
  } else if (ohm_diff.GetType() == Resistivity::ohmic &&
              ohm_diff.GetCoeffType() == ResistivityCoeff::spitzer) {
    const auto ohm_diff_val = ohm_diff; // capture by value for the device kernel
    Kokkos::parallel_reduce(
        "EstimateResistivityTimestep (ohmic spitzer)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const Real rho = prim(IDN, k, j, i);
          const Real p = prim(IPR, k, j, i);
          const Real eta = ohm_diff_val.Get(p, rho);

          min_dt =
              fmin(min_dt, SQR(coords.Dxc<1>(k, j, i)) / (eta + TINY_NUMBER));
          if (ndim >= 2) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<2>(k, j, i)) / (eta + TINY_NUMBER));
          }
          if (ndim >= 3) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<3>(k, j, i)) / (eta + TINY_NUMBER));
          }
        },
        Kokkos::Min<Real>(min_dt_resist));
  } else {
    PARTHENON_THROW("Needs impl.");
  }

  const auto &cfl_diff = hydro_pkg->Param<Real>("cfl_diff");
  return cfl_diff * fac * min_dt_resist;
}

//---------------------------------------------------------------------------------------
//! Calculate isotropic resistivity with fixed coefficient

void OhmicDiffFluxIsoFixed(MeshData<Real> *md) {
  OhmicDiffFluxIsoFixedImpl<true>(md);
}

//---------------------------------------------------------------------------------------
//! TODO(pgrete) Calculate Ohmic diffusion, general case, e.g., with varying (Spitzer)
//! coefficient

void OhmicDiffFluxGeneral(MeshData<Real> *md) {
  OhmicDiffFluxGeneralImpl<true, false>(md);
}

void OhmicDiffusionMagneticFlux(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

  if (ohm_diff.GetCoeffType() == ResistivityCoeff::fixed) {
    OhmicDiffFluxIsoFixedImpl<false>(md);
  } else if (ohm_diff.GetCoeffType() == ResistivityCoeff::spitzer) {
    OhmicDiffFluxGeneralImpl<false, false>(md);
  } else {
    PARTHENON_FAIL("Unknown Resistivity Type");
  }
}

void ComputeOhmicHeatingSourceFromFluxDivergence(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto const &eint_pack = md->PackVariables(std::vector<std::string>{"eint_iter"});
  auto const &s_ohm_pack = md->PackVariables(std::vector<std::string>{"s_ohm_iter"});

  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  const auto ohm_diff_val = ohm_diff;
  const int ndim = pmb->pmy_mesh->ndim;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const int nb = prim_pack.GetDim(5);
  const int nk = kb.e - kb.s + 1;
  const int nj = jb.e - jb.s + 1;
  const int ni = ib.e - ib.s + 1;

  parthenon::ParArray4D<Real> flux_x1("ohmic_energy_flux_x1", nb, nk, nj, ni + 1);
  parthenon::ParArray4D<Real> flux_x2("ohmic_energy_flux_x2", nb, nk, nj + 1, ni);
  parthenon::ParArray4D<Real> flux_x3("ohmic_energy_flux_x3", nb, nk + 1, nj, ni);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resistive energy X1 faces", DevExecSpace(), 0,
      prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const Real rho_at_face = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j, i - 1));
        Real eta;
        if (ohm_diff_val.GetCoeffType() == ResistivityCoeff::fixed) {
          eta = ohm_diff_val.Get(0.0, 0.0);
        } else {
          const Real p_at_face =
              0.5 * (prim(IDN, k, j, i) * eint_pack(b, 0, k, j, i) * gm1 +
                     prim(IDN, k, j, i - 1) * eint_pack(b, 0, k, j, i - 1) * gm1);
          eta = ohm_diff_val.Get(p_at_face, rho_at_face);
        }
        const auto d3B1 =
            ndim > 2 ? (0.5 * (prim(IB1, k + 1, j, i - 1) + prim(IB1, k + 1, j, i)) -
                        0.5 * (prim(IB1, k - 1, j, i - 1) + prim(IB1, k - 1, j, i))) /
                           (coords.Xf<3, 1>(k + 1, j, i) - coords.Xf<3, 1>(k - 1, j, i))
                     : 0.0;
        const auto d1B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        const auto j2 = d3B1 - d1B3;
        const auto d1B2 =
            (prim(IB2, k, j, i) - prim(IB2, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        const auto d2B1 =
            ndim > 1 ? (0.5 * (prim(IB1, k, j + 1, i - 1) + prim(IB1, k, j + 1, i)) -
                        0.5 * (prim(IB1, k, j - 1, i - 1) + prim(IB1, k, j - 1, i))) /
                           (coords.Xf<2, 1>(k, j + 1, i) - coords.Xf<2, 1>(k, j - 1, i))
                     : 0.0;
        const auto j3 = d1B2 - d2B1;

        flux_x1(b, k - kb.s, j - jb.s, i - ib.s) =
            0.5 * eta *
            ((prim(IB3, k, j, i - 1) + prim(IB3, k, j, i)) * j2 -
             (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i)) * j3);
      });

  if (ndim >= 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Resistive energy X2 faces", DevExecSpace(), 0,
        prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = prim_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const Real rho_at_face = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j - 1, i));
          Real eta;
          if (ohm_diff_val.GetCoeffType() == ResistivityCoeff::fixed) {
            eta = ohm_diff_val.Get(0.0, 0.0);
          } else {
            const Real p_at_face =
                0.5 * (prim(IDN, k, j, i) * eint_pack(b, 0, k, j, i) * gm1 +
                       prim(IDN, k, j - 1, i) * eint_pack(b, 0, k, j - 1, i) * gm1);
            eta = ohm_diff_val.Get(p_at_face, rho_at_face);
          }
          const auto d1B2 =
              (0.5 * (prim(IB2, k, j - 1, i + 1) + prim(IB2, k, j, i + 1)) -
               0.5 * (prim(IB2, k, j - 1, i - 1) + prim(IB2, k, j, i - 1))) /
              (coords.Xf<1, 2>(k, j, i + 1) - coords.Xf<1, 2>(k, j, i - 1));
          const auto d2B1 =
              (prim(IB1, k, j, i) - prim(IB1, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
          const auto j3 = d1B2 - d2B1;
          const auto d2B3 =
              (prim(IB3, k, j, i) - prim(IB3, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
          const auto d3B2 =
              ndim > 2 ? (0.5 * (prim(IB2, k + 1, j - 1, i) + prim(IB2, k + 1, j, i)) -
                          0.5 * (prim(IB2, k - 1, j - 1, i) + prim(IB2, k - 1, j, i))) /
                             (coords.Xf<3, 2>(k + 1, j, i) - coords.Xf<3, 2>(k - 1, j, i))
                       : 0.0;
          const auto j1 = d2B3 - d3B2;

          flux_x2(b, k - kb.s, j - jb.s, i - ib.s) =
              0.5 * eta *
              ((prim(IB1, k, j - 1, i) + prim(IB1, k, j, i)) * j3 -
               (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i)) * j1);
        });
  }

  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Resistive energy X3 faces", DevExecSpace(), 0,
        prim_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = prim_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const Real rho_at_face = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k - 1, j, i));
          Real eta;
          if (ohm_diff_val.GetCoeffType() == ResistivityCoeff::fixed) {
            eta = ohm_diff_val.Get(0.0, 0.0);
          } else {
            const Real p_at_face =
                0.5 * (prim(IDN, k, j, i) * eint_pack(b, 0, k, j, i) * gm1 +
                       prim(IDN, k - 1, j, i) * eint_pack(b, 0, k - 1, j, i) * gm1);
            eta = ohm_diff_val.Get(p_at_face, rho_at_face);
          }
          const auto d2B3 =
              (0.5 * (prim(IB3, k - 1, j + 1, i) + prim(IB3, k, j + 1, i)) -
               0.5 * (prim(IB3, k - 1, j - 1, i) + prim(IB3, k, j - 1, i))) /
              (coords.Xf<2, 3>(k, j + 1, i) - coords.Xf<2, 3>(k, j - 1, i));
          const auto d3B2 =
              (prim(IB2, k, j, i) - prim(IB2, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
          const auto j1 = d2B3 - d3B2;
          const auto d3B1 =
              (prim(IB1, k, j, i) - prim(IB1, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
          const auto d1B3 =
              (0.5 * (prim(IB3, k - 1, j, i + 1) + prim(IB3, k, j, i + 1)) -
               0.5 * (prim(IB3, k - 1, j, i - 1) + prim(IB3, k, j, i - 1))) /
              (coords.Xf<1, 3>(k, j, i + 1) - coords.Xf<1, 3>(k, j, i - 1));
          const auto j2 = d3B1 - d1B3;

          flux_x3(b, k - kb.s, j - jb.s, i - ib.s) =
              0.5 * eta *
              ((prim(IB2, k - 1, j, i) + prim(IB2, k, j, i)) * j1 -
               (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i)) * j2);
        });
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resistive energy source", DevExecSpace(), 0,
      prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &s_ohm = s_ohm_pack(b);
        Real du = coords.FaceArea<X1DIR>(k, j, i + 1) *
                      flux_x1(b, k - kb.s, j - jb.s, i + 1 - ib.s) -
                  coords.FaceArea<X1DIR>(k, j, i) *
                      flux_x1(b, k - kb.s, j - jb.s, i - ib.s);
        if (ndim >= 2) {
          du += coords.FaceArea<X2DIR>(k, j + 1, i) *
                    flux_x2(b, k - kb.s, j + 1 - jb.s, i - ib.s) -
                coords.FaceArea<X2DIR>(k, j, i) *
                    flux_x2(b, k - kb.s, j - jb.s, i - ib.s);
        }
        if (ndim == 3) {
          du += coords.FaceArea<X3DIR>(k + 1, j, i) *
                    flux_x3(b, k + 1 - kb.s, j - jb.s, i - ib.s) -
                coords.FaceArea<X3DIR>(k, j, i) *
                    flux_x3(b, k - kb.s, j - jb.s, i - ib.s);
        }
        s_ohm(0, k, j, i) = -du / coords.CellVolume(k, j, i);
      });
}
