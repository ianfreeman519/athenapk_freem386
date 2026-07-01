//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file current_sheet_thermal.cpp
//! \brief Minimal static current-sheet problem generator for coupled thermal validation.
//========================================================================================

#include "Kokkos_MathematicalFunctions.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/diffusion/diffusion.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../main.hpp"
#include "../units.hpp"

namespace current_sheet_thermal {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

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
      "current_sheet_thermal::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  const Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  const Real k_b = units.k_boltzmann();
  const Real atomic_mass_unit = units.atomic_mass_unit();
  const Real m_bar =
      pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;

  const Real B0_cgs = pin->GetOrAddReal("problem/current_sheet_thermal", "B0", 4.0e5);
  const Real guide_B_cgs =
      pin->GetOrAddReal("problem/current_sheet_thermal", "guide_B", 0.0);
  const Real delta_cgs = pin->GetOrAddReal("problem/current_sheet_thermal", "delta", 0.02);
  const Real rho_cgs = pin->GetOrAddReal("problem/current_sheet_thermal", "rho0", 4.5e-3);
  const Real T_cgs = pin->GetOrAddReal("problem/current_sheet_thermal", "T0", 1.16e6);
  const Real vx_cgs = pin->GetOrAddReal("problem/current_sheet_thermal", "vx0", 0.0);
  const Real vy_cgs = pin->GetOrAddReal("problem/current_sheet_thermal", "vy0", 0.0);
  const Real pressure_bump_frac =
      pin->GetOrAddReal("problem/current_sheet_thermal", "pressure_bump_frac", 0.0);
  PARTHENON_REQUIRE(delta_cgs > 0.0,
                    "problem/current_sheet_thermal/delta must be positive.");

  const Real B0 = B0_cgs * units.gauss();
  const Real guide_B = guide_B_cgs * units.gauss();
  const Real delta = delta_cgs * units.cm();
  const Real rho0 = rho_cgs * units.g_cm3();
  const Real vx0 = vx_cgs * units.cm_s();
  const Real vy0 = vy_cgs * units.cm_s();
  const Real P0 = T_cgs * k_b * rho0 / m_bar;

  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "B0 [Gauss] ============= " << B0_cgs << std::endl;
    std::cout << "guide_B [Gauss] ======== " << guide_B_cgs << std::endl;
    std::cout << "delta [cm] ============= " << delta_cgs << std::endl;
    std::cout << "rho0 [g/cm^3] ========== " << rho_cgs << std::endl;
    std::cout << "T0 [K] ================= " << T_cgs << std::endl;
    std::cout << "vx0 [cm/s] ============= " << vx_cgs << std::endl;
    std::cout << "vy0 [cm/s] ============= " << vy_cgs << std::endl;
    std::cout << "pressure_bump_frac ===== " << pressure_bump_frac << std::endl;
  }

  auto &coords = pmb->coords;
  pmb->par_for(
      "ProblemGenerator::current_sheet_thermal", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real tanh_x = tanh(x / delta);
        const Real sech_x = 1.0 / cosh(x / delta);
        const Real rho = rho0;
        const Real pressure = P0 * (1.0 + pressure_bump_frac * SQR(sech_x));

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * vx0;
        u(IM2, k, j, i) = rho * vy0;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = 0.0;
        u(IB2, k, j, i) = B0 * tanh_x;
        u(IB3, k, j, i) = guide_B;
        u(IEN, k, j, i) =
            pressure / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      });
}

} // namespace current_sheet_thermal
