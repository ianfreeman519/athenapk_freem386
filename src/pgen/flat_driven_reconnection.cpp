//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file flat_driven_reconnection.cpp
//! \brief Problem generator for a flat_driven_reconnection sheet with localized driving.
//!
//! REFERENCE: Trust Me Bro I made it all up
//========================================================================================

// Parthenon headers
#include "Kokkos_MathematicalFunctions.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "bvals/boundary_conditions.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/diffusion/diffusion.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"

#include <cmath>
#include <limits>
#include <string>

namespace flat_driven_reconnection {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace {

enum class InflowProfileKind { gaussian, cubic, quintic, wendland };

struct FlatDrivenReconnectionParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real rho_inflow;
  Real rho_background;
  Real T_inflow;
  Real T_background;
  Real array_spacing;
  Real half_array_spacing;
  Real h_inflow;
  Real inv_h_inflow;
  Real B_peak;
  Real rise_time;
  Real inv_rise_time;
  InflowProfileKind inflow_profile;
};

struct ProfileEvaluation {
  Real thermo_weight;
  Real magnetic_weight;
};

FlatDrivenReconnectionParams g_params{};
bool g_params_initialized = false;
Real g_boundary_drive_time = 0.0;

InflowProfileKind ParseInflowProfile(const std::string &name) {
  if (name == "gaussian") {
    return InflowProfileKind::gaussian;
  } else if (name == "cubic") {
    return InflowProfileKind::cubic;
  } else if (name == "quintic") {
    return InflowProfileKind::quintic;
  } else if (name == "wendland") {
    return InflowProfileKind::wendland;
  }

  PARTHENON_FAIL("Unknown inflow_profile: " + name);
}

FlatDrivenReconnectionParams LoadParams(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                                        ParameterInput *pin) {
  FlatDrivenReconnectionParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.rho_inflow =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "rho_inflow", 2e-3);
  params.rho_background =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "rho_background", 1e-6);
  params.T_inflow =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "T_inflow", 1e6);
  params.T_background =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "T_background", 1e4);
  params.array_spacing =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "array_spacing", 1.0);
  params.h_inflow =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "h_inflow", 1.0);
  params.B_peak =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "B_peak", 2e5) /
      std::sqrt(4.0 * M_PI);

  const Real rise_time_ns =
      pin->GetOrAddReal("problem/flat_driven_reconnection", "rise_time_ns", 300.0);
  params.rise_time = rise_time_ns * 1.0e-9;
  params.inflow_profile = ParseInflowProfile(
      pin->GetOrAddString("problem/flat_driven_reconnection", "inflow_profile", "gaussian"));

  PARTHENON_REQUIRE(params.array_spacing > 0.0,
                    "problem/flat_driven_reconnection/array_spacing must be positive.");
  PARTHENON_REQUIRE(params.h_inflow > 0.0,
                    "problem/flat_driven_reconnection/h_inflow must be positive.");
  PARTHENON_REQUIRE(params.rise_time > 0.0,
                    "problem/flat_driven_reconnection/rise_time_ns must be positive.");
  PARTHENON_REQUIRE(params.rho_background > 0.0,
                    "problem/flat_driven_reconnection/rho_background must be positive.");
  PARTHENON_REQUIRE(params.T_background > 0.0,
                    "problem/flat_driven_reconnection/T_background must be positive.");
  PARTHENON_REQUIRE(params.T_inflow >= 0.0,
                    "problem/flat_driven_reconnection/T_inflow must be nonnegative.");
  PARTHENON_REQUIRE(params.rho_inflow >= 0.0,
                    "problem/flat_driven_reconnection/rho_inflow must be nonnegative.");

  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar =
      pin->GetReal("hydro", "mean_molecular_weight") * units.atomic_mass_unit();
  params.half_array_spacing = 0.5 * params.array_spacing;
  params.inv_h_inflow = 1.0 / params.h_inflow;
  params.inv_rise_time = 1.0 / params.rise_time;

  return params;
}

KOKKOS_INLINE_FUNCTION
Real InflowProfile(const Real q, const InflowProfileKind kind) {
  if (kind == InflowProfileKind::gaussian) {
    return exp(-SQR(q));
  } else if (kind == InflowProfileKind::cubic) {
    if (q >= 2.0) {
      return 0.0;
    } else if (q <= 1.0) {
      return 1.0 - 1.5 * q * q + 0.75 * q * q * q;
    }
    return 0.25 * pow(2.0 - q, 3);
  } else if (kind == InflowProfileKind::quintic) {
    if (q > 3.0) {
      return 0.0;
    } else if (q > 2.0) {
      return pow(3.0 - q, 5);
    } else if (q > 1.0) {
      return pow(3.0 - q, 5) - 6.0 * pow(2.0 - q, 5);
    }
    return pow(3.0 - q, 5) - 6.0 * pow(2.0 - q, 5) + 15.0 * pow(1.0 - q, 5);
  }

  if (q >= 2.0) {
    return 0.0;
  }
  return pow(1.0 - 0.5 * q, 3) * (1.5 * q + 1.0);
}

KOKKOS_INLINE_FUNCTION
ProfileEvaluation EvaluateProfiles(const FlatDrivenReconnectionParams &params, const Real x,
                                   const Real y) {
  ProfileEvaluation profile{0.0, 0.0};

  for (int A = -1; A <= 1; A += 2) {
    const Real y_local = y - A * params.half_array_spacing;
    const Real q = sqrt(SQR(x) + SQR(y_local)) * params.inv_h_inflow;
    const Real thermo_weight = InflowProfile(q, params.inflow_profile);
    const Real magnetic_weight =
        InflowProfile(fabs(y_local) * params.inv_h_inflow, params.inflow_profile);
    profile.thermo_weight += thermo_weight;
    profile.magnetic_weight += A * magnetic_weight;
  }

  return profile;
}

KOKKOS_INLINE_FUNCTION
Real DriveEnvelope(const FlatDrivenReconnectionParams &params, const Real time) {
  const Real t = fmax(0.0, time);
  const Real ramp = fmin(1.0, t * params.inv_rise_time);
  return params.B_peak * SQR(sin(0.5 * M_PI * ramp));
}

KOKKOS_INLINE_FUNCTION
void RecomputeTotalEnergy(const FlatDrivenReconnectionParams &params, const Real rho,
                          const Real temperature_floor, const Real m1, const Real m2,
                          const Real m3, const Real B1, const Real B2, const Real B3,
                          const Real eint_old, Real &energy) {
  constexpr Real tiny = std::numeric_limits<Real>::min();
  const Real rho_safe = fmax(rho, tiny);
  const Real eint_floor =
      rho_safe * params.k_b * fmax(temperature_floor, 0.0) / (params.m_bar * params.gm1);
  const Real eint_new = fmax(eint_old, eint_floor);
  const Real kinetic = 0.5 * (SQR(m1) + SQR(m2) + SQR(m3)) / rho_safe;
  const Real magnetic = 0.5 * (SQR(B1) + SQR(B2) + SQR(B3));
  energy = eint_new + kinetic + magnetic;
}

template <bool INNER_X2>
void DrivenOutflowX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  if constexpr (INNER_X2) {
    parthenon::BoundaryFunction::OutflowInnerX2(mbd, coarse);
  } else {
    parthenon::BoundaryFunction::OutflowOuterX2(mbd, coarse);
  }

  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  constexpr IndexDomain domain = INNER_X2 ? IndexDomain::inner_x2 : IndexDomain::outer_x2;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  const auto params = g_params;
  const Real drive_amplitude = DriveEnvelope(params, g_boundary_drive_time);
  auto coords = pmb->coords;

  pmb->par_for_bndry(
      "flat_driven_reconnection::DrivenOutflowX2", nb, domain,
      parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto profile = EvaluateProfiles(params, x, y);

        if (profile.thermo_weight <= 0.0 && profile.magnetic_weight == 0.0) {
          return;
        }

        constexpr Real tiny = std::numeric_limits<Real>::min();
        const Real rho_old = cons(IDN, k, j, i);
        const Real m1 = cons(IM1, k, j, i);
        const Real m2 = cons(IM2, k, j, i);
        const Real m3 = cons(IM3, k, j, i);
        const Real B2 = cons(IB2, k, j, i);
        const Real B3 = cons(IB3, k, j, i);
        const Real rho_safe_old = fmax(rho_old, tiny);
        const Real kinetic_old = 0.5 * (SQR(m1) + SQR(m2) + SQR(m3)) / rho_safe_old;
        const Real magnetic_old =
            0.5 * (SQR(cons(IB1, k, j, i)) + SQR(B2) + SQR(B3));
        const Real eint_old =
            fmax(tiny, cons(IEN, k, j, i) - kinetic_old - magnetic_old);

        const Real rho_floor =
            params.rho_background + params.rho_inflow * profile.thermo_weight;
        const Real temperature_floor =
            params.T_background + params.T_inflow * profile.thermo_weight;
        const Real rho_new = fmax(rho_old, rho_floor);
        const Real B1_new = drive_amplitude * profile.magnetic_weight;

        cons(IDN, k, j, i) = rho_new;
        cons(IB1, k, j, i) = B1_new;
        RecomputeTotalEnergy(params, rho_new, temperature_floor, m1, m2, m3, B1_new, B2, B3,
                             eint_old, cons(IEN, k, j, i));
      });
}

} // namespace

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  auto hydro_pkg = mesh->packages.Get("Hydro");
  g_params = LoadParams(hydro_pkg, pin);
  g_params_initialized = true;
  g_boundary_drive_time = 0.0;
}

void PreStepMeshUserWorkInLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  Hydro::PreStepMeshUserWorkInLoop(mesh, pin, tm);
  g_boundary_drive_time = tm.time + tm.dt;
}

// Setting up derived fields:
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

// storing the curls just before output
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/) {
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
  Real mbar = hydro_pkg->Param<Real>("mbar");
  const auto units = hydro_pkg->Param<Units>("units");
  Real k_B = units.k_boltzmann();
  const auto cfl_hyp = hydro_pkg->Param<Real>("cfl");
  const auto cfl_diff = hydro_pkg->Param<Real>("cfl_diff");
  const auto cfl_diff_heat = hydro_pkg->Param<Real>("cfl_diff_heat");
  const auto cfl_cool = pin->GetOrAddReal("cooling", "cfl", 0.1);
  const auto eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const auto ndim = pmb->pmy_mesh->ndim;
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
      "flat_driven_reconnection::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inf = std::numeric_limits<Real>::infinity();
        Real term1, term2;
        term1 =
            (u(IB3, k, j + 1, i) - u(IB3, k, j - 1, i)) / (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        term2 =
            (u(IB2, k + 1, j, i) - u(IB2, k - 1, j, i)) / (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        curlBx(k, j, i) = term1 - term2;
        term1 =
            (u(IB1, k + 1, j, i) - u(IB1, k - 1, j, i)) / (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        term2 =
            (u(IB3, k, j, i + 1) - u(IB3, k, j, i - 1)) / (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        curlBy(k, j, i) = term1 - term2;
        term1 =
            (u(IB2, k, j, i + 1) - u(IB2, k, j, i - 1)) / (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        term2 =
            (u(IB1, k, j + 1, i) - u(IB1, k, j - 1, i)) / (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        curlBz(k, j, i) = term1 - term2;

        Real dvx_dx =
            (w(IV1, k, j, i + 1) - w(IV1, k, j, i - 1)) / (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        Real dvy_dy =
            (w(IV2, k, j + 1, i) - w(IV2, k, j - 1, i)) / (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        Real dvz_dz =
            (w(IV3, k + 1, j, i) - w(IV3, k - 1, j, i)) / (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        divv(k, j, i) = dvx_dx + dvy_dy + dvz_dz;

        Real rho = u(IDN, k, j, i);
        Real p = w(IPR, k, j, i);
        T_field(k, j, i) = mbar / k_B * p / rho;

        beta_field(k, j, i) =
            p / (0.5 * 4 * M_PI *
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

        Real dBzdy = ndim > 1 ? (u(IB3, k, j + 1, i) - u(IB3, k, j - 1, i)) /
                                    (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                              : 0.0;
        Real dBydz = ndim > 2 ? (u(IB2, k + 1, j, i) - u(IB2, k - 1, j, i)) /
                                    (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                              : 0.0;
        Real dBxdz = ndim > 2 ? (u(IB1, k + 1, j, i) - u(IB1, k - 1, j, i)) /
                                    (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                              : 0.0;
        Real dBzdx =
            (u(IB3, k, j, i + 1) - u(IB3, k, j, i - 1)) / (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        Real dBydx =
            (u(IB2, k, j, i + 1) - u(IB2, k, j, i - 1)) / (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        Real dBxdy = ndim > 1 ? (u(IB1, k, j + 1, i) - u(IB1, k, j - 1, i)) /
                                    (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                              : 0.0;
        Real jx = dBzdy - dBydz;
        Real jy = dBxdz - dBzdx;
        Real jz = dBydx - dBxdy;
        Real j_squared = SQR(jx) + SQR(jy) + SQR(jz);
        Real internal_e_dens = p / gm1;
        Real dt_heat_val =
            (resistivity != Resistivity::none && eta_val > 0.0 && j_squared > 0.0)
                ? cfl_diff_heat * fabs(internal_e_dens / (eta_val * j_squared))
                : inf;
        dt_heat_local(k, j, i) = dt_heat_val;

        Real internal_e_spec = p / (rho * gm1);
        Real de_dt_cool = enable_cooling == Cooling::tabular
                              ? cooling_table_obj.DeDt(internal_e_spec, rho)
                              : 0.0;
        Real dt_cool_val = (enable_cooling == Cooling::tabular && de_dt_cool != 0.0 &&
                            internal_e_spec >= eos.GetInternalEFloor())
                               ? fabs(cfl_cool * internal_e_spec / de_dt_cool)
                               : inf;
        dt_cool_local(k, j, i) = dt_cool_val;

        Real prim_local[(NHYDRO)];
        prim_local[IDN] = rho;
        prim_local[IV1] = w(IV1, k, j, i);
        prim_local[IV2] = w(IV2, k, j, i);
        prim_local[IV3] = w(IV3, k, j, i);
        prim_local[IPR] = p;
        Real cs = eos.SoundSpeed(prim_local);
        Real lambda_fms_x =
            eos.FastMagnetosonicSpeed(rho, p, u(IB1, k, j, i), u(IB2, k, j, i), u(IB3, k, j, i));
        Real dt_hyp_fms_val = coords.Dxc<1>(k, j, i) / (fabs(prim_local[IV1]) + lambda_fms_x);
        Real dt_hyp_cs_val = coords.Dxc<1>(k, j, i) / (fabs(prim_local[IV1]) + cs);
        if (ndim > 1) {
          Real lambda_fms_y = eos.FastMagnetosonicSpeed(rho, p, u(IB2, k, j, i),
                                                        u(IB3, k, j, i), u(IB1, k, j, i));
          dt_hyp_fms_val = fmin(dt_hyp_fms_val,
                                coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + lambda_fms_y));
          dt_hyp_cs_val =
              fmin(dt_hyp_cs_val, coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + cs));
        }
        if (ndim > 2) {
          Real lambda_fms_z = eos.FastMagnetosonicSpeed(rho, p, u(IB3, k, j, i),
                                                        u(IB1, k, j, i), u(IB2, k, j, i));
          dt_hyp_fms_val = fmin(dt_hyp_fms_val,
                                coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + lambda_fms_z));
          dt_hyp_cs_val =
              fmin(dt_hyp_cs_val, coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + cs));
        }
        dt_hyp_fms(k, j, i) = cfl_hyp * dt_hyp_fms_val;
        dt_hyp_cs(k, j, i) = cfl_hyp * dt_hyp_cs_val;
      });
}

void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real /*dt*/) {
  const auto params = g_params;
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const Real current_amplitude = DriveEnvelope(params, tm.time + tm.dt);
  const Real previous_amplitude = DriveEnvelope(params, tm.time);
  const Real delta_amplitude = current_amplitude - previous_amplitude;

  if (delta_amplitude == 0.0 && params.rho_inflow == 0.0 && params.T_inflow == 0.0) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "flat_driven_reconnection::Driving",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto profile = EvaluateProfiles(params, x, y);

        if (profile.thermo_weight <= 0.0 && profile.magnetic_weight == 0.0) {
          return;
        }

        constexpr Real tiny = std::numeric_limits<Real>::min();
        const Real rho_old = cons(IDN, k, j, i);
        const Real m1 = cons(IM1, k, j, i);
        const Real m2 = cons(IM2, k, j, i);
        const Real m3 = cons(IM3, k, j, i);
        const Real B1_old = cons(IB1, k, j, i);
        const Real B2 = cons(IB2, k, j, i);
        const Real B3 = cons(IB3, k, j, i);
        const Real rho_safe_old = fmax(rho_old, tiny);
        const Real kinetic_old = 0.5 * (SQR(m1) + SQR(m2) + SQR(m3)) / rho_safe_old;
        const Real magnetic_old = 0.5 * (SQR(B1_old) + SQR(B2) + SQR(B3));
        const Real eint_old = fmax(tiny, cons(IEN, k, j, i) - kinetic_old - magnetic_old);

        const Real rho_floor = params.rho_background + params.rho_inflow * profile.thermo_weight;
        const Real temperature_floor =
            params.T_background + params.T_inflow * profile.thermo_weight;
        const Real rho_new = fmax(rho_old, rho_floor);
        const Real B1_new = B1_old + delta_amplitude * profile.magnetic_weight;

        cons(IDN, k, j, i) = rho_new;
        cons(IB1, k, j, i) = B1_new;
        RecomputeTotalEnergy(params, rho_new, temperature_floor, m1, m2, m3, B1_new, B2, B3,
                             eint_old, cons(IEN, k, j, i));
      });
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput * /*pin*/) {
  PARTHENON_REQUIRE(g_params_initialized,
                    "flat_driven_reconnection::InitUserMeshData must run before ProblemGenerator.");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  const auto params = g_params;

  pmb->par_for(
      "ProblemGenerator::flat_driven_reconnection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto profile = EvaluateProfiles(params, x, y);
        const Real rho = params.rho_background + params.rho_inflow * profile.thermo_weight;
        const Real temperature =
            params.T_background + params.T_inflow * profile.thermo_weight;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = 0.0;
        u(IB2, k, j, i) = 0.0;
        u(IB3, k, j, i) = 0.0;
        u(IEN, k, j, i) =
            rho * params.k_b * temperature / (params.m_bar * params.gm1);
      });
}

void DrivenOutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  DrivenOutflowX2<true>(mbd, coarse);
}

void DrivenOutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  DrivenOutflowX2<false>(mbd, coarse);
}

} // namespace flat_driven_reconnection
