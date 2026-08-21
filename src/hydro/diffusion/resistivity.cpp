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
using TE = parthenon::TopologicalElement;

KOKKOS_INLINE_FUNCTION
Real OhmicDiffusivity::Get(const Real pres, const Real rho) const {
  if (resistivity_coeff_type_ == ResistivityCoeff::fixed) {
    return coeff_;
  } else if (resistivity_coeff_type_ == ResistivityCoeff::spitzer) {
    // Convert p/rho from code units to temperature in kelvin.
    const Real temperature_kelvin = temperature_from_p_over_rho_ * pres / rho;
    PARTHENON_REQUIRE(temperature_kelvin > 0.0,
                      "Spitzer resistivity requires positive temperature.");

    // Magnetic diffusivity in Heaviside-Lorentz CGS [cm^2/s], then code units.
    Real eta = 1.02688e12 * zbar_ * coeff_ / std::pow(temperature_kelvin, 1.5) *
               eta_cgs_to_code_;
    if (eta_max_ > 0.0) {
      eta = std::min(eta, eta_max_);
    }
    return eta;
  } else {
    PARTHENON_FAIL("Unknown Resistivity coeff");
  }
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

  if (ohm_diff.GetType() == Resistivity::ohmic) {
    Kokkos::parallel_reduce(
        "EstimateResistivityTimestep (ohmic)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const auto eta =
              ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i));
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
  const Real dt_diffusion = cfl_diff * fac * min_dt_resist;

  // Limit the fractional internal-energy increase from Ohmic heating,
  // d(e_int)/dt = eta |curl(B)|^2. This restriction is independent of the
  // dimensional factor used by the parabolic diffusion limit above.
  const auto gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  Real min_heating_time = std::numeric_limits<Real>::infinity();
  const auto ohm_diff_val = ohm_diff;
  Kokkos::parallel_reduce(
      "EstimateResistivityTimestep (ohmic heating)",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    Real &min_time) {
        const auto &coords = prim_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const Real rho = prim(IDN, k, j, i);
        const Real pres = prim(IPR, k, j, i);
        const Real eta = ohm_diff_val.Get(pres, rho);

        const Real d_bz_dy =
            ndim > 1 ? (prim(IB3, k, j + 1, i) - prim(IB3, k, j - 1, i)) /
                           (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                     : 0.0;
        const Real d_by_dz =
            ndim > 2 ? (prim(IB2, k + 1, j, i) - prim(IB2, k - 1, j, i)) /
                           (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                     : 0.0;
        const Real d_bx_dz =
            ndim > 2 ? (prim(IB1, k + 1, j, i) - prim(IB1, k - 1, j, i)) /
                           (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                     : 0.0;
        const Real d_bz_dx =
            (prim(IB3, k, j, i + 1) - prim(IB3, k, j, i - 1)) /
            (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real d_by_dx =
            (prim(IB2, k, j, i + 1) - prim(IB2, k, j, i - 1)) /
            (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real d_bx_dy =
            ndim > 1 ? (prim(IB1, k, j + 1, i) - prim(IB1, k, j - 1, i)) /
                           (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                     : 0.0;

        const Real j1 = d_bz_dy - d_by_dz;
        const Real j2 = d_bx_dz - d_bz_dx;
        const Real j3 = d_by_dx - d_bx_dy;
        const Real heating_rate = eta * (SQR(j1) + SQR(j2) + SQR(j3));
        if (heating_rate > 0.0) {
          const Real internal_energy_density = pres / gm1;
          min_time = fmin(min_time, internal_energy_density / heating_rate);
        }
      },
      Kokkos::Min<Real>(min_heating_time));

  const Real dt_heating = cfl_diff * min_heating_time;
  return std::min(dt_diffusion, dt_heating);
}

//---------------------------------------------------------------------------------------
//! Calculate isotropic Ohmic fluxes with a coefficient interpolated to each face.

void OhmicDiffFlux(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto hydro_pkg = pmb->packages.Get("Hydro");

  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const bool update_cell_centered_b = fluid != Fluid::ucthlldmhd;

  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X1 fluxes (ohmic)", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const auto eta =
            0.5 * (ohm_diff.Get(prim(IPR, k, j, i - 1), prim(IDN, k, j, i - 1)) +
                   ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));

        // Face centered current densities
        // j2 = d3B1 - d1B3
        const auto d3B1 =
            ndim > 2 ? (0.5 * (prim(IB1, k + 1, j, i - 1) + prim(IB1, k + 1, j, i)) -
                        0.5 * (prim(IB1, k - 1, j, i - 1) + prim(IB1, k - 1, j, i))) /
                           (coords.Xf<3, 1>(k + 1, j, i) - coords.Xf<3, 1>(k - 1, j, i))
                     : 0.0;

        const auto d1B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j, i - 1)) / coords.Dxc<1>(k, j, i);

        const auto j2 = d3B1 - d1B3;

        // j3 = d1B2 - d2B1
        const auto d1B2 =
            (prim(IB2, k, j, i) - prim(IB2, k, j, i - 1)) / coords.Dxc<1>(k, j, i);

        const auto d2B1 =
            ndim > 1 ? (0.5 * (prim(IB1, k, j + 1, i - 1) + prim(IB1, k, j + 1, i)) -
                        0.5 * (prim(IB1, k, j - 1, i - 1) + prim(IB1, k, j - 1, i))) /
                           (coords.Xf<2, 1>(k, j + 1, i) - coords.Xf<2, 1>(k, j - 1, i))
                     : 0.0;

        const auto j3 = d1B2 - d2B1;

        if (update_cell_centered_b) {
          cons.flux(X1DIR, IB2, k, j, i) += -eta * j3;
          cons.flux(X1DIR, IB3, k, j, i) += eta * j2;
        }
        cons.flux(X1DIR, IEN, k, j, i) +=
            0.5 * eta *
            ((prim(IB3, k, j, i - 1) + prim(IB3, k, j, i)) * j2 -
             (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i)) * j3);
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
        const auto eta =
            0.5 * (ohm_diff.Get(prim(IPR, k, j - 1, i), prim(IDN, k, j - 1, i)) +
                   ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));

        // Face centered current densities
        // j3 = d1B2 - d2B1
        const auto d1B2 = (0.5 * (prim(IB2, k, j - 1, i + 1) + prim(IB2, k, j, i + 1)) -
                           0.5 * (prim(IB2, k, j - 1, i - 1) + prim(IB2, k, j, i - 1))) /
                          (coords.Xf<1, 2>(k, j, i + 1) - coords.Xf<1, 2>(k, j, i - 1));

        const auto d2B1 =
            (prim(IB1, k, j, i) - prim(IB1, k, j - 1, i)) / coords.Dxc<2>(k, j, i);

        const auto j3 = d1B2 - d2B1;

        // j1 = d2B3 - d3B2
        const auto d2B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j - 1, i)) / coords.Dxc<2>(k, j, i);

        const auto d3B2 =
            ndim > 2 ? (0.5 * (prim(IB2, k + 1, j - 1, i) + prim(IB2, k + 1, j, i)) -
                        0.5 * (prim(IB2, k - 1, j - 1, i) + prim(IB2, k - 1, j, i))) /
                           (coords.Xf<3, 2>(k + 1, j, i) - coords.Xf<3, 2>(k - 1, j, i))
                     : 0.0;

        const auto j1 = d2B3 - d3B2;

        if (update_cell_centered_b) {
          cons.flux(X2DIR, IB1, k, j, i) += eta * j3;
          cons.flux(X2DIR, IB3, k, j, i) += -eta * j1;
        }
        cons.flux(X2DIR, IEN, k, j, i) +=
            0.5 * eta *
            ((prim(IB1, k, j - 1, i) + prim(IB1, k, j, i)) * j3 -
             (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i)) * j1);
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
        const auto eta =
            0.5 * (ohm_diff.Get(prim(IPR, k - 1, j, i), prim(IDN, k - 1, j, i)) +
                   ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));

        // Face centered current densities
        // j1 = d2B3 - d3B2
        const auto d2B3 = (0.5 * (prim(IB3, k - 1, j + 1, i) + prim(IB3, k, j + 1, i)) -
                           0.5 * (prim(IB3, k - 1, j - 1, i) + prim(IB3, k, j - 1, i))) /
                          (coords.Xf<2, 3>(k, j + 1, i) - coords.Xf<2, 3>(k, j - 1, i));

        const auto d3B2 =
            (prim(IB2, k, j, i) - prim(IB2, k - 1, j, i)) / coords.Dxc<3>(k, j, i);

        const auto j1 = d2B3 - d3B2;

        // j2 = d3B1 - d1B3
        const auto d3B1 =
            (prim(IB1, k, j, i) - prim(IB1, k - 1, j, i)) / coords.Dxc<3>(k, j, i);

        const auto d1B3 = (0.5 * (prim(IB3, k - 1, j, i + 1) + prim(IB3, k, j, i + 1)) -
                           0.5 * (prim(IB3, k - 1, j, i - 1) + prim(IB3, k, j, i - 1))) /
                          (coords.Xf<1, 3>(k, j, i + 1) - coords.Xf<1, 3>(k, j, i - 1));

        const auto j2 = d3B1 - d1B3;

        if (update_cell_centered_b) {
          cons.flux(X3DIR, IB1, k, j, i) += -eta * j2;
          cons.flux(X3DIR, IB2, k, j, i) += eta * j1;
        }
        cons.flux(X3DIR, IEN, k, j, i) +=
            0.5 * eta *
            ((prim(IB2, k - 1, j, i) + prim(IB2, k, j, i)) * j1 -
             (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i)) * j2);
      });
}

//---------------------------------------------------------------------------------------
//! Add E_ohm = eta curl(B) to the edge-centered electric field used by CT.

TaskStatus AddOhmicEdgeEMF(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto bface_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

  // z-directed edges. In 1D only the x derivative and the two x-neighboring
  // coefficient values contribute; in 2D/3D eta is bilinearly interpolated.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Add Ohmic Ez edges", DevExecSpace(), 0,
      bface_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &bface = bface_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real dby_dx =
            (bface(TE::F2, 0, k, j, i) - bface(TE::F2, 0, k, j, i - 1)) /
            coords.Dxc<1>(k, j, i);
        const Real dbx_dy =
            ndim > 1 ? (bface(TE::F1, 0, k, j, i) -
                        bface(TE::F1, 0, k, j - 1, i)) /
                           coords.Dxc<2>(k, j, i)
                     : 0.0;
        Real eta =
            0.5 * (ohm_diff.Get(prim(IPR, k, j, i - 1), prim(IDN, k, j, i - 1)) +
                   ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));
        if (ndim > 1) {
          eta = 0.25 *
                (ohm_diff.Get(prim(IPR, k, j - 1, i - 1),
                              prim(IDN, k, j - 1, i - 1)) +
                 ohm_diff.Get(prim(IPR, k, j - 1, i), prim(IDN, k, j - 1, i)) +
                 ohm_diff.Get(prim(IPR, k, j, i - 1), prim(IDN, k, j, i - 1)) +
                 ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));
        }
        bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i) +=
            eta * (dby_dx - dbx_dy);
      });

  if (ndim < 3) {
    return TaskStatus::complete;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Add Ohmic Ey edges", DevExecSpace(), 0,
      bface_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &bface = bface_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real dbx_dz =
            (bface(TE::F1, 0, k, j, i) - bface(TE::F1, 0, k - 1, j, i)) /
            coords.Dxc<3>(k, j, i);
        const Real dbz_dx =
            (bface(TE::F3, 0, k, j, i) - bface(TE::F3, 0, k, j, i - 1)) /
            coords.Dxc<1>(k, j, i);
        const Real eta = 0.25 *
            (ohm_diff.Get(prim(IPR, k - 1, j, i - 1), prim(IDN, k - 1, j, i - 1)) +
             ohm_diff.Get(prim(IPR, k - 1, j, i), prim(IDN, k - 1, j, i)) +
             ohm_diff.Get(prim(IPR, k, j, i - 1), prim(IDN, k, j, i - 1)) +
             ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));
        bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i) +=
            eta * (dbx_dz - dbz_dx);
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Add Ohmic Ex edges", DevExecSpace(), 0,
      bface_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &bface = bface_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real dbz_dy =
            (bface(TE::F3, 0, k, j, i) - bface(TE::F3, 0, k, j - 1, i)) /
            coords.Dxc<2>(k, j, i);
        const Real dby_dz =
            (bface(TE::F2, 0, k, j, i) - bface(TE::F2, 0, k - 1, j, i)) /
            coords.Dxc<3>(k, j, i);
        const Real eta = 0.25 *
            (ohm_diff.Get(prim(IPR, k - 1, j - 1, i), prim(IDN, k - 1, j - 1, i)) +
             ohm_diff.Get(prim(IPR, k - 1, j, i), prim(IDN, k - 1, j, i)) +
             ohm_diff.Get(prim(IPR, k, j - 1, i), prim(IDN, k, j - 1, i)) +
             ohm_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)));
        bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i) +=
            eta * (dbz_dy - dby_dz);
      });

  return TaskStatus::complete;
}
