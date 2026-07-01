//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// C++ headers
#include <string>
#include <vector>

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
  return InternalEnergySolverConfig{hydro_pkg->Param<int>("thermal_source_iterations")};
}

bool ThermalSourceSolverEnabled(StateDescriptor *hydro_pkg) {
  return hydro_pkg->Param<bool>("thermal_source_solver_enabled");
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

void ZeroField(MeshData<Real> *md, const std::string &field_name) {
  const auto &field_pack = md->PackVariables(std::vector<std::string>{field_name});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ZeroField", DevExecSpace(), 0, field_pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &field = field_pack(b);
        field(0, k, j, i) = 0.0;
      });
}

void CopyField(MeshData<Real> *md, const std::string &src_field,
               const std::string &dst_field) {
  const auto &src_pack = md->PackVariables(std::vector<std::string>{src_field});
  const auto &dst_pack = md->PackVariables(std::vector<std::string>{dst_field});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CopyField", DevExecSpace(), 0, src_pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &src = src_pack(b);
        auto &dst = dst_pack(b);
        dst(0, k, j, i) = src(0, k, j, i);
      });
}

void BuildAdvectivePlusOhmicThermalSource(MeshData<Real> *md, const std::string &ohmic_field,
                                          const std::string &total_field) {
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &thermal_ae_pack = md->PackVariables(std::vector<std::string>{"thermal_ae"});
  const auto &ohmic_pack = md->PackVariables(std::vector<std::string>{ohmic_field});
  const auto &total_pack = md->PackVariables(std::vector<std::string>{total_field});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "BuildAdvectivePlusOhmicThermalSource", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        const auto &thermal_ae = thermal_ae_pack(b);
        const auto &ohmic = ohmic_pack(b);
        auto &total = total_pack(b);
        total(0, k, j, i) = cons(IDN, k, j, i) * thermal_ae(0, k, j, i) +
                            ohmic(0, k, j, i);
      });
}

void CombineThermalSources(MeshData<Real> *md, const std::string &source_a,
                           const std::string &source_b, const std::string &source_out) {
  const auto &source_a_pack = md->PackVariables(std::vector<std::string>{source_a});
  const auto &source_b_pack = md->PackVariables(std::vector<std::string>{source_b});
  const auto &source_out_pack = md->PackVariables(std::vector<std::string>{source_out});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CombineThermalSources", DevExecSpace(), 0,
      source_a_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &src_a = source_a_pack(b);
        const auto &src_b = source_b_pack(b);
        auto &src_out = source_out_pack(b);
        src_out(0, k, j, i) = src_a(0, k, j, i) + src_b(0, k, j, i);
      });
}

void CommitSimplifiedSDCInternalEnergyUpdate(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eint_adv_pack = md->PackVariables(std::vector<std::string>{"eint_adv"});
  const auto &eint_sdc_pack = md->PackVariables(std::vector<std::string>{"eint_sdc"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CommitSimplifiedSDCInternalEnergyUpdate", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const Real rho = cons(IDN, k, j, i);
        const Real delta_eint =
            eint_sdc_pack(b, 0, k, j, i) - eint_adv_pack(b, 0, k, j, i);
        cons(IEN, k, j, i) += rho * delta_eint;
        prim(IPR, k, j, i) = rho * eint_sdc_pack(b, 0, k, j, i) * gm1;
      });
}

void RecordSDCIterationDiagnostics(MeshData<Real> *md, int iter_count) {
  const auto &iter_count_pack =
      md->PackVariables(std::vector<std::string>{"thermal_sdc_iter_count"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RecordSDCIterationDiagnostics", DevExecSpace(), 0,
      iter_count_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &iter_count_field = iter_count_pack(b);
        iter_count_field(0, k, j, i) = static_cast<Real>(iter_count);
      });
}

TaskStatus RunSimplifiedSDCThermalCoupling(MeshData<Real> *md, const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool couple_cooling = hydro_pkg->Param<bool>("thermal_couple_cooling");
  const bool couple_ohmic = hydro_pkg->Param<bool>("thermal_couple_ohmic");
  const bool couple_conduction = hydro_pkg->Param<bool>("thermal_couple_conduction");
  const bool couple_viscous = hydro_pkg->Param<bool>("thermal_couple_viscous");

  if (couple_conduction || couple_viscous) {
    PARTHENON_FAIL("Coupled conduction/viscous heating is not implemented yet.");
  }
  if (!couple_cooling) {
    PARTHENON_FAIL("The simplified-SDC thermal coupling path currently requires "
                   "thermal_source_solver/couple_cooling = true.");
  }

  const auto cfg = GetInternalEnergySolverConfig(hydro_pkg.get());
  const auto &tabular_cooling = hydro_pkg->Param<cooling::TabularCooling>("tabular_cooling");

  // The accepted stage-start state is the ODE initial condition for every fixed SDC pass.
  CopyField(md, "eint_stage_start", "eint_sdc");
  ZeroField(md, "thermal_src_ohmic");
  ZeroField(md, "thermal_src_total");

  for (int iter = 0; iter < cfg.iterations; ++iter) {
    if (couple_ohmic) {
      BuildOhmicThermalSourceFromFluxDivergence(md, "eint_sdc", "thermal_src_ohmic");
    } else {
      ZeroField(md, "thermal_src_ohmic");
    }

    BuildAdvectivePlusOhmicThermalSource(md, "thermal_src_ohmic", "thermal_src_total");
    tabular_cooling.IntegrateThermalODEWithSource(md, "eint_stage_start",
                                                  "thermal_src_total", "eint_next",
                                                  "temp_next", dt);
    CopyField(md, "eint_next", "eint_sdc");
  }

  if (couple_ohmic) {
    BuildOhmicThermalSourceFromFluxDivergence(md, "eint_sdc", "thermal_src_ohmic");
  } else {
    ZeroField(md, "thermal_src_ohmic");
  }
  tabular_cooling.BuildCoolingThermalSource(md, "eint_sdc", "thermal_src_total");
  // Rebuild the accepted total thermal source from the final stage-local thermal state.
  // This field is consumed only by the pre-flux predictor and is reinitialized at the
  // next stage start by InitializeStageLaggedThermalSource().
  CombineThermalSources(md, "thermal_src_total", "thermal_src_ohmic", "thermal_src_lagged");

  RecordSDCIterationDiagnostics(md, cfg.iterations);
  CommitSimplifiedSDCInternalEnergyUpdate(md);
  ApplyMagneticOnlyResistiveUpdate(md, dt);

  return TaskStatus::complete;
}

} // namespace

TaskStatus InitializeStageLaggedThermalSource(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (!ThermalSourceSolverEnabled(hydro_pkg.get())) {
    return TaskStatus::complete;
  }

  const bool couple_cooling = hydro_pkg->Param<bool>("thermal_couple_cooling");
  const bool couple_ohmic = hydro_pkg->Param<bool>("thermal_couple_ohmic");
  const auto &tabular_cooling = hydro_pkg->Param<cooling::TabularCooling>("tabular_cooling");

  ZeroField(md, "thermal_src_lagged");
  ZeroField(md, "thermal_src_pre_flux");
  ZeroField(md, "thermal_src_ohmic_pre_flux");
  ZeroField(md, "thermal_src_ohmic");
  ZeroField(md, "thermal_src_total");
  CopyField(md, "eint_stage_start", "eint_sdc");

  // thermal_src_lagged is owned by the thermal solver and has stage-local validity:
  // it is initialized from the accepted stage-start state immediately before the
  // pre-flux predictor and consumed exactly once by ApplyPreFluxThermalSource().
  if (couple_cooling) {
    tabular_cooling.BuildCoolingThermalSource(md, "eint_stage_start", "thermal_src_total");
  }
  if (couple_ohmic) {
    BuildOhmicThermalSourceFromFluxDivergence(md, "eint_stage_start", "thermal_src_ohmic");
  }
  CombineThermalSources(md, "thermal_src_total", "thermal_src_ohmic", "thermal_src_lagged");
  CopyField(md, "thermal_src_lagged", "thermal_src_pre_flux");
  CopyField(md, "thermal_src_ohmic", "thermal_src_ohmic_pre_flux");

  return TaskStatus::complete;
}

TaskStatus AddCoupledInternalEnergySources(MeshData<Real> *md, const SimTime &,
                                           const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (!ThermalSourceSolverEnabled(hydro_pkg.get())) {
    return TaskStatus::complete;
  }
  return RunSimplifiedSDCThermalCoupling(md, dt);
}

} // namespace Hydro
