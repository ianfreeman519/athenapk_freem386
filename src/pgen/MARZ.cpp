//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file MARZ.cpp
//! \brief Reduced MARZ-like current-sheet problem generator.
//========================================================================================

#include "Kokkos_MathematicalFunctions.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../bvals/boundary_conditions_apk.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/diffusion/diffusion.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../main.hpp"
#include "../units.hpp"

namespace marz_sheet {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace {

struct MarzSheetParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real B0_cgs;
  Real Bguide_cgs;
  Real rho_in_cgs;
  Real T_in;
  Real v_in_cgs;
  Real delta_B_cgs;
  Real delta_v_cgs;
  Real rho_bump;
  Real delta_rho_cgs;
  Real seed_amplitude_cgs;
  int seed_mode_number;
  Real delta_seed_cgs;
  bool use_driven_x1_boundaries;

  Real B0;
  Real Bguide;
  Real rho_in;
  Real v_in;
  Real delta_B;
  Real delta_v;
  Real delta_rho;
  Real delta_seed;
  Real seed_amplitude;
  Real p_in;
};

MarzSheetParams g_params{};
bool g_params_initialized = false;

MarzSheetParams LoadParams(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                           ParameterInput *pin) {
  MarzSheetParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.B0_cgs = pin->GetOrAddReal("problem/marz_sheet", "B0", 4.0e5);
  params.Bguide_cgs = pin->GetOrAddReal("problem/marz_sheet", "Bguide", 0.0);
  params.rho_in_cgs = pin->GetOrAddReal("problem/marz_sheet", "rho_in", 4.5e-3);
  params.T_in = pin->GetOrAddReal("problem/marz_sheet", "T_in", 1.16e6);
  params.v_in_cgs = pin->GetOrAddReal("problem/marz_sheet", "v_in", 2.0e6);
  params.delta_B_cgs = pin->GetOrAddReal("problem/marz_sheet", "delta_B", 0.02);
  params.delta_v_cgs =
      pin->GetOrAddReal("problem/marz_sheet", "delta_v", params.delta_B_cgs);
  params.rho_bump = pin->GetOrAddReal("problem/marz_sheet", "rho_bump", 0.0);
  params.delta_rho_cgs =
      pin->GetOrAddReal("problem/marz_sheet", "delta_rho", params.delta_B_cgs);
  params.seed_amplitude_cgs =
      pin->GetOrAddReal("problem/marz_sheet", "seed_amplitude", 2.0e2);
  params.seed_mode_number =
      pin->GetOrAddInteger("problem/marz_sheet", "seed_mode_number", 1);
  params.delta_seed_cgs =
      pin->GetOrAddReal("problem/marz_sheet", "delta_seed", 2.5 * params.delta_B_cgs);
  params.use_driven_x1_boundaries =
      pin->GetOrAddBoolean("problem/marz_sheet", "use_driven_x1_boundaries", true);

  PARTHENON_REQUIRE(params.delta_B_cgs > 0.0,
                    "problem/marz_sheet/delta_B must be positive.");
  PARTHENON_REQUIRE(params.delta_v_cgs > 0.0,
                    "problem/marz_sheet/delta_v must be positive.");
  PARTHENON_REQUIRE(params.delta_rho_cgs > 0.0,
                    "problem/marz_sheet/delta_rho must be positive.");
  PARTHENON_REQUIRE(params.delta_seed_cgs > 0.0,
                    "problem/marz_sheet/delta_seed must be positive.");
  PARTHENON_REQUIRE(params.rho_in_cgs > 0.0,
                    "problem/marz_sheet/rho_in must be positive.");
  PARTHENON_REQUIRE(params.T_in > 0.0,
                    "problem/marz_sheet/T_in must be positive.");
  PARTHENON_REQUIRE(params.seed_mode_number >= 0,
                    "problem/marz_sheet/seed_mode_number must be nonnegative.");
  PARTHENON_REQUIRE(1.0 + params.rho_bump > 0.0,
                    "problem/marz_sheet/rho_bump must keep density positive.");

  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar =
      pin->GetReal("hydro", "mean_molecular_weight") * units.atomic_mass_unit();

  params.B0 = params.B0_cgs * units.gauss();
  params.Bguide = params.Bguide_cgs * units.gauss();
  params.rho_in = params.rho_in_cgs * units.g_cm3();
  params.v_in = params.v_in_cgs * units.cm_s();
  params.delta_B = params.delta_B_cgs * units.cm();
  params.delta_v = params.delta_v_cgs * units.cm();
  params.delta_rho = params.delta_rho_cgs * units.cm();
  params.delta_seed = params.delta_seed_cgs * units.cm();
  params.seed_amplitude = params.seed_amplitude_cgs * units.gauss() * units.cm();
  params.p_in = params.T_in * params.k_b * params.rho_in / params.m_bar;

  return params;
}

void EnsureParamsInitialized(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                             ParameterInput *pin) {
  if (!g_params_initialized) {
    g_params = LoadParams(hydro_pkg, pin);
    g_params_initialized = true;
  }
}

template <parthenon::BoundaryFunction::BCSide SIDE>
void DrivenX1Boundary(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  if (!g_params.use_driven_x1_boundaries) {
    Hydro::BoundaryFunction::LinearBC<X1DIR, SIDE>(mbd, coarse);
    return;
  }

  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  constexpr IndexDomain domain =
      (SIDE == parthenon::BoundaryFunction::BCSide::Inner) ? IndexDomain::inner_x1
                                                           : IndexDomain::outer_x1;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  const auto params = g_params;

  pmb->par_for_bndry(
      "marz_sheet::DrivenX1Boundary", nb, domain, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real rho = params.rho_in;
        const Real v1 = (SIDE == parthenon::BoundaryFunction::BCSide::Inner)
                            ? params.v_in
                            : -params.v_in;
        const Real B2 = (SIDE == parthenon::BoundaryFunction::BCSide::Inner)
                            ? -params.B0
                            : params.B0;
        const Real B1 = 0.0;
        const Real B3 = params.Bguide;

        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, i) = rho * v1;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        cons(IB1, k, j, i) = B1;
        cons(IB2, k, j, i) = B2;
        cons(IB3, k, j, i) = B3;
        cons(IEN, k, j, i) =
            params.p_in / params.gm1 +
            0.5 * (SQR(B1) + SQR(B2) + SQR(B3) + rho * SQR(v1));
      });
}

} // namespace

void ProblemInitPackageData(ParameterInput *, parthenon::StateDescriptor *hydro_pkg) {
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  hydro_pkg->AddField("curlBx", m);
  hydro_pkg->AddField("curlBy", m);
  hydro_pkg->AddField("curlBz", m);
  hydro_pkg->AddField("divv", m);
  hydro_pkg->AddField("beta", m);
  hydro_pkg->AddField("eta", m);
  hydro_pkg->AddField("T", m);
  hydro_pkg->AddField("dt_diff_local", m);
  hydro_pkg->AddField("dt_heat_local", m);
  hydro_pkg->AddField("dt_cool_local", m);
  hydro_pkg->AddField("dt_hyp_fms", m);
  hydro_pkg->AddField("dt_hyp_cs", m);
}

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  auto hydro_pkg = mesh->packages.Get("Hydro");
  g_params = LoadParams(hydro_pkg, pin);
  g_params_initialized = true;
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin, const parthenon::SimTime &) {
  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &w = mbd->Get("prim").data;
  auto &data = pmb->meshblock_data.Get();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool has_ohm_diff = hydro_pkg->AllParams().hasKey("ohm_diff");
  OhmicDiffusivity ohm_diff_dev(Resistivity::none, ResistivityCoeff::none, 0.0, 0.0, 0.0,
                                0.0, 0.0, -1.0, 0.0);
  if (has_ohm_diff) {
    ohm_diff_dev = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  }

  auto &curlBx = data->Get("curlBx").data;
  auto &curlBy = data->Get("curlBy").data;
  auto &curlBz = data->Get("curlBz").data;
  auto &divv = data->Get("divv").data;
  auto &eta_field = data->Get("eta").data;
  auto &beta_field = data->Get("beta").data;
  auto &T_field = data->Get("T").data;
  auto &dt_diff_local = data->Get("dt_diff_local").data;
  auto &dt_heat_local = data->Get("dt_heat_local").data;
  auto &dt_cool_local = data->Get("dt_cool_local").data;
  auto &dt_hyp_fms = data->Get("dt_hyp_fms").data;
  auto &dt_hyp_cs = data->Get("dt_hyp_cs").data;
  const Real mbar = hydro_pkg->Param<Real>("mbar");
  const auto units = hydro_pkg->Param<Units>("units");
  const Real k_B = units.k_boltzmann();
  const Real cfl_hyp = hydro_pkg->Param<Real>("cfl");
  const Real cfl_diff = hydro_pkg->Param<Real>("cfl_diff");
  const Real cfl_diff_heat = hydro_pkg->Param<Real>("cfl_diff_heat");
  const Real cfl_cool = pin->GetOrAddReal("cooling", "cfl", 0.1);
  const auto eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const int ndim = pmb->pmy_mesh->ndim;
  const auto resistivity = hydro_pkg->Param<Resistivity>("resistivity");
  const auto enable_cooling = hydro_pkg->Param<Cooling>("enable_cooling");
  cooling::CoolingTableObj cooling_table_obj;
  if (enable_cooling == Cooling::tabular) {
    cooling_table_obj =
        hydro_pkg->Param<cooling::TabularCooling>("tabular_cooling").GetCoolingTableObj();
  }

  Real fac = 0.5;
  if (ndim == 2) {
    fac = 0.25;
  } else if (ndim == 3) {
    fac = 1.0 / 6.0;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "marz_sheet::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inf = std::numeric_limits<Real>::infinity();
        const Real dBzdy = ndim > 1 ? (u(IB3, k, j + 1, i) - u(IB3, k, j - 1, i)) /
                                          (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                                    : 0.0;
        const Real dBydz = ndim > 2 ? (u(IB2, k + 1, j, i) - u(IB2, k - 1, j, i)) /
                                          (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                                    : 0.0;
        const Real dBxdz = ndim > 2 ? (u(IB1, k + 1, j, i) - u(IB1, k - 1, j, i)) /
                                          (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                                    : 0.0;
        const Real dBzdx = (u(IB3, k, j, i + 1) - u(IB3, k, j, i - 1)) /
                           (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real dBydx = (u(IB2, k, j, i + 1) - u(IB2, k, j, i - 1)) /
                           (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real dBxdy = ndim > 1 ? (u(IB1, k, j + 1, i) - u(IB1, k, j - 1, i)) /
                                          (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                                    : 0.0;
        curlBx(k, j, i) = dBzdy - dBydz;
        curlBy(k, j, i) = dBxdz - dBzdx;
        curlBz(k, j, i) = dBydx - dBxdy;

        const Real dvx_dx = (w(IV1, k, j, i + 1) - w(IV1, k, j, i - 1)) /
                            (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real dvy_dy = ndim > 1 ? (w(IV2, k, j + 1, i) - w(IV2, k, j - 1, i)) /
                                           (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                                     : 0.0;
        const Real dvz_dz = ndim > 2 ? (w(IV3, k + 1, j, i) - w(IV3, k - 1, j, i)) /
                                           (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                                     : 0.0;
        divv(k, j, i) = dvx_dx + dvy_dy + dvz_dz;

        const Real rho = u(IDN, k, j, i);
        const Real p = w(IPR, k, j, i);
        T_field(k, j, i) = mbar / k_B * p / rho;
        beta_field(k, j, i) =
            p / (0.5 * 4.0 * M_PI *
                 (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))));

        Real eta_val = 0.0;
        if (has_ohm_diff) {
          eta_val = ohm_diff_dev.Get(p, rho);
        }
        eta_field(k, j, i) = eta_val;

        Real dt_diff_val = inf;
        if (resistivity != Resistivity::none && eta_val > 0.0) {
          dt_diff_val = SQR(coords.Dxc<1>(k, j, i)) / eta_val;
          if (ndim >= 2) {
            dt_diff_val = fmin(dt_diff_val, SQR(coords.Dxc<2>(k, j, i)) / eta_val);
          }
          if (ndim >= 3) {
            dt_diff_val = fmin(dt_diff_val, SQR(coords.Dxc<3>(k, j, i)) / eta_val);
          }
          dt_diff_val = cfl_diff * fac * dt_diff_val;
        }
        dt_diff_local(k, j, i) = dt_diff_val;

        const Real jx = curlBx(k, j, i);
        const Real jy = curlBy(k, j, i);
        const Real jz = curlBz(k, j, i);
        const Real j_squared = SQR(jx) + SQR(jy) + SQR(jz);
        const Real internal_e_dens = p / gm1;
        dt_heat_local(k, j, i) =
            (resistivity != Resistivity::none && eta_val > 0.0 && j_squared > 0.0)
                ? cfl_diff_heat * fabs(internal_e_dens / (eta_val * j_squared))
                : inf;

        const Real internal_e_spec = p / (rho * gm1);
        const Real de_dt_cool = enable_cooling == Cooling::tabular
                                    ? cooling_table_obj.DeDt(internal_e_spec, rho)
                                    : 0.0;
        dt_cool_local(k, j, i) =
            (enable_cooling == Cooling::tabular && de_dt_cool != 0.0 &&
             internal_e_spec >= eos.GetInternalEFloor())
                ? fabs(cfl_cool * internal_e_spec / de_dt_cool)
                : inf;

        Real prim_local[NHYDRO];
        prim_local[IDN] = rho;
        prim_local[IV1] = w(IV1, k, j, i);
        prim_local[IV2] = w(IV2, k, j, i);
        prim_local[IV3] = w(IV3, k, j, i);
        prim_local[IPR] = p;
        const Real cs = eos.SoundSpeed(prim_local);
        Real lambda_fms_x = eos.FastMagnetosonicSpeed(
            rho, p, u(IB1, k, j, i), u(IB2, k, j, i), u(IB3, k, j, i));
        Real dt_hyp_fms_val = coords.Dxc<1>(k, j, i) / (fabs(prim_local[IV1]) + lambda_fms_x);
        Real dt_hyp_cs_val = coords.Dxc<1>(k, j, i) / (fabs(prim_local[IV1]) + cs);
        if (ndim > 1) {
          const Real lambda_fms_y = eos.FastMagnetosonicSpeed(
              rho, p, u(IB2, k, j, i), u(IB3, k, j, i), u(IB1, k, j, i));
          dt_hyp_fms_val = fmin(dt_hyp_fms_val,
                                coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + lambda_fms_y));
          dt_hyp_cs_val =
              fmin(dt_hyp_cs_val, coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + cs));
        }
        if (ndim > 2) {
          const Real lambda_fms_z = eos.FastMagnetosonicSpeed(
              rho, p, u(IB3, k, j, i), u(IB1, k, j, i), u(IB2, k, j, i));
          dt_hyp_fms_val = fmin(dt_hyp_fms_val,
                                coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + lambda_fms_z));
          dt_hyp_cs_val =
              fmin(dt_hyp_cs_val, coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + cs));
        }
        dt_hyp_fms(k, j, i) = cfl_hyp * dt_hyp_fms_val;
        dt_hyp_cs(k, j, i) = cfl_hyp * dt_hyp_cs_val;
      });
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_REQUIRE(pmb->pmy_mesh->ndim >= 2,
                    "marz_sheet requires at least a 2D mesh.");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  EnsureParamsInitialized(hydro_pkg, pin);
  const auto params = g_params;
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real Ly = x2max - x2min;

  PARTHENON_REQUIRE(Ly > 0.0, "parthenon/mesh x2 extent must be positive for marz_sheet.");

  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "B0 [Gauss] ================= " << params.B0_cgs << std::endl;
    std::cout << "Bguide [Gauss] ============= " << params.Bguide_cgs << std::endl;
    std::cout << "rho_in [g/cm^3] ============ " << params.rho_in_cgs << std::endl;
    std::cout << "T_in [K] =================== " << params.T_in << std::endl;
    std::cout << "v_in [cm/s] ================ " << params.v_in_cgs << std::endl;
    std::cout << "delta_B [cm] =============== " << params.delta_B_cgs << std::endl;
    std::cout << "delta_v [cm] =============== " << params.delta_v_cgs << std::endl;
    std::cout << "rho_bump =================== " << params.rho_bump << std::endl;
    std::cout << "delta_rho [cm] ============= " << params.delta_rho_cgs << std::endl;
    std::cout << "seed_amplitude [G cm] ====== " << params.seed_amplitude_cgs << std::endl;
    std::cout << "seed_mode_number =========== " << params.seed_mode_number << std::endl;
    std::cout << "delta_seed [cm] ============ " << params.delta_seed_cgs << std::endl;
    std::cout << "use_driven_x1_boundaries === "
              << params.use_driven_x1_boundaries << std::endl;
    std::cout << "p_in [code units] ========= " << params.p_in << std::endl;
    std::cout << "Ly [code units] =========== " << Ly << std::endl;
  }

  auto &coords = pmb->coords;
  pmb->par_for(
      "ProblemGenerator::marz_sheet", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real tanh_B = tanh(x / params.delta_B);
        const Real tanh_v = tanh(x / params.delta_v);
        const Real sech_rho = 1.0 / cosh(x / params.delta_rho);
        const Real rho = params.rho_in * (1.0 + params.rho_bump * SQR(sech_rho));
        const Real By_base = params.B0 * tanh_B;
        const Real pressure = params.p_in + 0.5 * (SQR(params.B0) - SQR(By_base));

        const Real phase = 2.0 * M_PI * static_cast<Real>(params.seed_mode_number) * y / Ly;
        const Real seed_env = exp(-SQR(x / params.delta_seed));
        const Real dA_dy =
            -params.seed_amplitude * (2.0 * M_PI * static_cast<Real>(params.seed_mode_number) /
                                      Ly) *
            sin(phase) * seed_env;
        const Real dA_dx =
            params.seed_amplitude * cos(phase) * seed_env *
            (-2.0 * x / SQR(params.delta_seed));

        const Real B1 = dA_dy;
        const Real B2 = By_base - dA_dx;
        const Real B3 = params.Bguide;
        const Real v1 = -params.v_in * tanh_v;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * v1;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = B1;
        u(IB2, k, j, i) = B2;
        u(IB3, k, j, i) = B3;
        u(IEN, k, j, i) =
            pressure / params.gm1 +
            0.5 * (SQR(B1) + SQR(B2) + SQR(B3) + rho * SQR(v1));
      });
}

void DrivenInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  DrivenX1Boundary<parthenon::BoundaryFunction::BCSide::Inner>(mbd, coarse);
}

void DrivenOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  DrivenX1Boundary<parthenon::BoundaryFunction::BCSide::Outer>(mbd, coarse);
}

} // namespace marz_sheet
