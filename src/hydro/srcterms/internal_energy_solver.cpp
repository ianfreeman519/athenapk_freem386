//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// C++ headers
#include <algorithm>

// Parthenon headers
#include <interface/update.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../diffusion/diffusion.hpp"
#include "internal_energy_solver.hpp"
#include "tabular_cooling.hpp"

namespace Hydro {

namespace {

InternalEnergySolverConfig GetInternalEnergySolverConfig(StateDescriptor *hydro_pkg) {
  return InternalEnergySolverConfig{
      hydro_pkg->Param<int>("thermal_source_max_iter"),
      hydro_pkg->Param<Real>("thermal_source_temp_rtol"),
      hydro_pkg->Param<Real>("thermal_source_e_rtol"),
      hydro_pkg->Param<Real>("thermal_source_under_relaxation"),
  };
}

void InitializeThermalScratch(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eint_init_pack = md->PackVariables(std::vector<std::string>{"eint_init"});
  const auto &eint_iter_pack = md->PackVariables(std::vector<std::string>{"eint_iter"});
  const auto &eint_next_pack = md->PackVariables(std::vector<std::string>{"eint_next"});
  const auto &temp_iter_pack = md->PackVariables(std::vector<std::string>{"temp_iter"});
  const auto &temp_next_pack = md->PackVariables(std::vector<std::string>{"temp_next"});
  const auto &s_ohm_pack = md->PackVariables(std::vector<std::string>{"s_ohm_iter"});
  const auto &s_ohm_prev_pack = md->PackVariables(std::vector<std::string>{"s_ohm_prev"});
  const auto &iter_count_pack =
      md->PackVariables(std::vector<std::string>{"coupled_iter_count"});
  const auto &temp_err_pack =
      md->PackVariables(std::vector<std::string>{"coupled_temp_err"});
  const auto &e_err_pack = md->PackVariables(std::vector<std::string>{"coupled_e_err"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "InitializeThermalScratch", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        auto &eint_init = eint_init_pack(b);
        auto &eint_iter = eint_iter_pack(b);
        auto &eint_next = eint_next_pack(b);
        auto &temp_iter = temp_iter_pack(b);
        auto &temp_next = temp_next_pack(b);
        auto &s_ohm = s_ohm_pack(b);
        auto &s_ohm_prev = s_ohm_prev_pack(b);
        auto &iter_count = iter_count_pack(b);
        auto &temp_err = temp_err_pack(b);
        auto &e_err = e_err_pack(b);

        const Real rho = cons(IDN, k, j, i);
        Real internal_e =
            cons(IEN, k, j, i) -
            0.5 * (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                   SQR(cons(IM3, k, j, i))) /
                rho;
        if (mhd_enabled) {
          internal_e -= 0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                               SQR(cons(IB3, k, j, i)));
        }
        internal_e /= rho;

        eint_init(0, k, j, i) = internal_e;
        eint_iter(0, k, j, i) = internal_e;
        eint_next(0, k, j, i) = internal_e;
        temp_iter(0, k, j, i) = mbar_gm1_over_kb * internal_e;
        temp_next(0, k, j, i) = mbar_gm1_over_kb * internal_e;
        s_ohm(0, k, j, i) = 0.0;
        s_ohm_prev(0, k, j, i) = 0.0;
        iter_count(0, k, j, i) = 0.0;
        temp_err(0, k, j, i) = 0.0;
        e_err(0, k, j, i) = 0.0;
        prim(IPR, k, j, i) = rho * internal_e * gm1;
      });
}

void CopyCurrentOhmicSource(MeshData<Real> *md) {
  const auto &s_ohm_pack = md->PackVariables(std::vector<std::string>{"s_ohm_iter"});
  const auto &s_ohm_prev_pack = md->PackVariables(std::vector<std::string>{"s_ohm_prev"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CopyCurrentOhmicSource", DevExecSpace(), 0,
      s_ohm_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &s_ohm_prev = s_ohm_prev_pack(b);
        const auto &s_ohm = s_ohm_pack(b);
        s_ohm_prev(0, k, j, i) = s_ohm(0, k, j, i);
      });
}

struct ThermalConvergenceStats {
  Real max_temp_rel;
  Real max_e_rel;
};

ThermalConvergenceStats ComputeConvergenceStats(MeshData<Real> *md) {
  const auto &eint_iter_pack = md->PackVariables(std::vector<std::string>{"eint_iter"});
  const auto &eint_next_pack = md->PackVariables(std::vector<std::string>{"eint_next"});
  const auto &temp_iter_pack = md->PackVariables(std::vector<std::string>{"temp_iter"});
  const auto &temp_next_pack = md->PackVariables(std::vector<std::string>{"temp_next"});
  const auto &temp_err_pack =
      md->PackVariables(std::vector<std::string>{"coupled_temp_err"});
  const auto &e_err_pack = md->PackVariables(std::vector<std::string>{"coupled_e_err"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  Real max_temp_rel = 0.0;
  Real max_e_rel = 0.0;

  Kokkos::parallel_reduce(
      "ComputeConvergenceStats",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {temp_iter_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    Real &thread_max_temp, Real &thread_max_e) {
        auto &temp_err = temp_err_pack(b);
        auto &e_err = e_err_pack(b);
        const Real temp_old = temp_iter_pack(b, 0, k, j, i);
        const Real temp_new = temp_next_pack(b, 0, k, j, i);
        const Real eint_old = eint_iter_pack(b, 0, k, j, i);
        const Real eint_new = eint_next_pack(b, 0, k, j, i);
        const Real temp_rel =
            fabs(temp_new - temp_old) / std::max(fabs(temp_old), static_cast<Real>(1e-30));
        const Real e_rel =
            fabs(eint_new - eint_old) / std::max(fabs(eint_old), static_cast<Real>(1e-30));
        temp_err(0, k, j, i) = temp_rel;
        e_err(0, k, j, i) = e_rel;
        thread_max_temp = std::max(thread_max_temp, temp_rel);
        thread_max_e = std::max(thread_max_e, e_rel);
      },
      Kokkos::Max<Real>(max_temp_rel), Kokkos::Max<Real>(max_e_rel));

  return {max_temp_rel, max_e_rel};
}

bool CheckThermalConvergence(const InternalEnergySolverConfig &cfg,
                             const ThermalConvergenceStats &stats) {
  if (cfg.temp_rtol > 0.0) {
    return stats.max_temp_rel < cfg.temp_rtol;
  }
  return stats.max_e_rel < cfg.e_rtol;
}

void AcceptNextIterate(MeshData<Real> *md, Real under_relaxation) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

  const auto &eint_iter_pack = md->PackVariables(std::vector<std::string>{"eint_iter"});
  const auto &eint_next_pack = md->PackVariables(std::vector<std::string>{"eint_next"});
  const auto &temp_iter_pack = md->PackVariables(std::vector<std::string>{"temp_iter"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AcceptNextIterate", DevExecSpace(), 0,
      eint_iter_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &eint_iter = eint_iter_pack(b);
        auto &temp_iter = temp_iter_pack(b);
        const auto &eint_next = eint_next_pack(b);
        const Real accepted =
            (1.0 - under_relaxation) * eint_iter(0, k, j, i) +
            under_relaxation * eint_next(0, k, j, i);
        eint_iter(0, k, j, i) = accepted;
        temp_iter(0, k, j, i) = mbar_gm1_over_kb * accepted;
      });
}

void CommitCoupledInternalEnergyUpdate(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eint_init_pack = md->PackVariables(std::vector<std::string>{"eint_init"});
  const auto &eint_iter_pack = md->PackVariables(std::vector<std::string>{"eint_iter"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CommitCoupledInternalEnergyUpdate", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const Real rho = cons(IDN, k, j, i);
        const Real delta_eint =
            eint_iter_pack(b, 0, k, j, i) - eint_init_pack(b, 0, k, j, i);
        cons(IEN, k, j, i) += rho * delta_eint;
        prim(IPR, k, j, i) = rho * eint_iter_pack(b, 0, k, j, i) * gm1;
      });
}

void ResetFluxesLocal(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  const int ndim = pmb->pmy_mesh->ndim;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxesLocal X1", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X1DIR, v, k, j, i) = 0.0;
      });

  if (ndim >= 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ResetFluxesLocal X2", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e + 1,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
          auto &cons = cons_pack(b);
          cons.flux(X2DIR, v, k, j, i) = 0.0;
        });
  }

  if (ndim == 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ResetFluxesLocal X3", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e + 1, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
          auto &cons = cons_pack(b);
          cons.flux(X3DIR, v, k, j, i) = 0.0;
        });
  }
}

void ApplyMagneticOnlyResistiveUpdate(MeshData<Real> *md, const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (!hydro_pkg->Param<bool>("thermal_couple_ohmic")) {
    return;
  }

  ResetFluxesLocal(md);
  OhmicDiffusionMagneticFlux(md);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyMagneticOnlyResistiveUpdate", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        cons(IB1, k, j, i) +=
            dt * parthenon::Update::FluxDivHelper(IB1, k, j, i, ndim, coords, cons);
        cons(IB2, k, j, i) +=
            dt * parthenon::Update::FluxDivHelper(IB2, k, j, i, ndim, coords, cons);
        cons(IB3, k, j, i) +=
            dt * parthenon::Update::FluxDivHelper(IB3, k, j, i, ndim, coords, cons);
        prim(IB1, k, j, i) = cons(IB1, k, j, i);
        prim(IB2, k, j, i) = cons(IB2, k, j, i);
        prim(IB3, k, j, i) = cons(IB3, k, j, i);
      });
}

void RecordIterationDiagnostics(MeshData<Real> *md, int iter_count) {
  const auto &iter_count_pack =
      md->PackVariables(std::vector<std::string>{"coupled_iter_count"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RecordIterationDiagnostics", DevExecSpace(), 0,
      iter_count_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &iter_count_field = iter_count_pack(b);
        iter_count_field(0, k, j, i) = static_cast<Real>(iter_count);
      });
}

} // namespace

TaskStatus AddCoupledInternalEnergySources(MeshData<Real> *md, const SimTime &, const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool couple_cooling = hydro_pkg->Param<bool>("thermal_couple_cooling");
  const bool couple_ohmic = hydro_pkg->Param<bool>("thermal_couple_ohmic");
  const bool couple_conduction = hydro_pkg->Param<bool>("thermal_couple_conduction");
  const bool couple_viscous = hydro_pkg->Param<bool>("thermal_couple_viscous");

  if (couple_conduction || couple_viscous) {
    PARTHENON_FAIL("Coupled conduction/viscous heating is not implemented yet.");
  }
  if (!couple_cooling) {
    PARTHENON_FAIL("The current coupled internal-energy solver requires "
                   "thermal_source_solver/couple_cooling = true.");
  }

  InitializeThermalScratch(md);

  const auto cfg = GetInternalEnergySolverConfig(hydro_pkg.get());
  const auto &tabular_cooling = hydro_pkg->Param<cooling::TabularCooling>("tabular_cooling");

  int accepted_iter = 0;
  bool converged = false;
  for (int iter = 0; iter < cfg.max_iter; ++iter) {
    accepted_iter = iter + 1;

    if (couple_ohmic) {
      CopyCurrentOhmicSource(md);
      ComputeOhmicHeatingSourceFromFluxDivergence(md);
    }

    tabular_cooling.CoupledRK12Step(md, dt);
    const auto stats = ComputeConvergenceStats(md);
    converged = CheckThermalConvergence(cfg, stats);
    AcceptNextIterate(md, cfg.under_relaxation);
    if (converged) {
      break;
    }
  }

  RecordIterationDiagnostics(md, accepted_iter);
  CommitCoupledInternalEnergyUpdate(md);
  ApplyMagneticOnlyResistiveUpdate(md, dt);

  return TaskStatus::complete;
}

} // namespace Hydro
