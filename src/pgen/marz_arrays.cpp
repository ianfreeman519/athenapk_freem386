//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file marz_arrays.cpp
//! \brief Source-driven MARZ-like two-array problem generator.
//========================================================================================

#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../main.hpp"
#include "../units.hpp"

namespace marz_arrays {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace {

enum class DriveModel { static_source, ramp };

struct MarzArraysParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real current_peak_MA;
  DriveModel drive_model;
  Real drive_t0;
  Real drive_rise_time;
  Real drive_plateau_fraction;
  Real rho_array_cgs;
  Real rho_background_cgs;
  Real T_array;
  Real T_background;
  Real v_array_cgs;
  Real array_separation_cgs;
  Real width_thermo_cgs;
  Real width_magnetic_cgs;
  Real source_radius_cgs;
  Real source_taper_width_cgs;
  Real source_mass_replenish_time;
  Real source_momentum_replenish_time;
  Real source_energy_replenish_time;
  Real source_field_replenish_time;
  int seed_mode_number;
  Real density_perturb_amplitude;
  Real temperature_perturb_amplitude;
  bool use_driven_x2_boundaries;
  bool use_diode_x1_boundaries;

  Real current_field_prefac_peak;
  Real rho_array;
  Real rho_background;
  Real v_array;
  Real array_separation;
  Real width_thermo;
  Real width_magnetic;
  Real source_radius;
  Real source_taper_width;
};

struct WireProfile {
  Real mask_sum;
  Real thermo_weight_sum;
  Real density_weight_sum;
  Real temperature_weight_sum;
  Real radial_v1_sum;
  Real radial_v2_sum;
  Real B1_sum;
  Real B2_sum;
};

struct InjectedState {
  Real rho;
  Real pressure;
  Real v1;
  Real v2;
  Real v3;
  Real B1;
  Real B2;
  Real B3;
};

MarzArraysParams g_params{};
bool g_params_initialized = false;
Real g_drive_time = 0.0;

KOKKOS_INLINE_FUNCTION
Real GaussianProfile(const Real r, const Real width) {
  const Real exponent = -SQR(r / width);
  return exp(fmax(-700.0, exponent));
}

KOKKOS_INLINE_FUNCTION
Real SmoothTaper(const Real xi) {
  const Real x = fmin(1.0, fmax(0.0, xi));
  return 1.0 - x * x * (3.0 - 2.0 * x);
}

KOKKOS_INLINE_FUNCTION
Real SourceMask(const Real r, const Real source_radius, const Real taper_width) {
  if (r <= source_radius) {
    return 1.0;
  }
  if (taper_width <= 0.0 || r >= source_radius + taper_width) {
    return 0.0;
  }
  return SmoothTaper((r - source_radius) / taper_width);
}

KOKKOS_INLINE_FUNCTION
Real AzimuthalPerturbation(const Real theta, const Real amplitude, const int mode_number) {
  const Real phase = static_cast<Real>(mode_number) * theta;
  return 1.0 + amplitude * cos(phase);
}

DriveModel ParseDriveModel(const std::string &drive_model) {
  if (drive_model == "static") {
    return DriveModel::static_source;
  }
  if (drive_model == "ramp") {
    return DriveModel::ramp;
  }

  std::stringstream msg;
  msg << "problem/marz_arrays/drive_model='" << drive_model
      << "' is not supported. Use 'static' or 'ramp'.";
  PARTHENON_THROW(msg);
}

MarzArraysParams LoadParams(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                            ParameterInput *pin) {
  MarzArraysParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.current_peak_MA =
      pin->GetOrAddReal("problem/marz_arrays", "current_peak_MA", 1.0);
  params.drive_model = ParseDriveModel(
      pin->GetOrAddString("problem/marz_arrays", "drive_model", "ramp"));
  params.drive_t0 = pin->GetOrAddReal("problem/marz_arrays", "drive_t0", 0.0);
  params.drive_rise_time =
      pin->GetOrAddReal("problem/marz_arrays", "drive_rise_time", 2.0e-9);
  params.drive_plateau_fraction =
      pin->GetOrAddReal("problem/marz_arrays", "drive_plateau_fraction", 1.0);
  params.rho_array_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "rho_array", 4.5e-3);
  params.rho_background_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "rho_background", 1.0e-5);
  params.T_array = pin->GetOrAddReal("problem/marz_arrays", "T_array", 3.5e5);
  params.T_background =
      pin->GetOrAddReal("problem/marz_arrays", "T_background", 2.5e4);
  params.v_array_cgs = pin->GetOrAddReal("problem/marz_arrays", "v_array", 2.5e6);
  params.array_separation_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "array_separation", 0.32);
  params.width_thermo_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "width_thermo", 0.05);
  params.width_magnetic_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "width_magnetic", 0.035);
  params.source_radius_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "source_radius", 0.05);
  params.source_taper_width_cgs =
      pin->GetOrAddReal("problem/marz_arrays", "source_taper_width", 0.02);
  params.source_mass_replenish_time =
      pin->GetOrAddReal("problem/marz_arrays", "source_mass_replenish_time", 5.0e-10);
  params.source_momentum_replenish_time =
      pin->GetOrAddReal("problem/marz_arrays", "source_momentum_replenish_time", 2.5e-10);
  params.source_energy_replenish_time =
      pin->GetOrAddReal("problem/marz_arrays", "source_energy_replenish_time", 5.0e-10);
  params.source_field_replenish_time =
      pin->GetOrAddReal("problem/marz_arrays", "source_field_replenish_time", 2.5e-10);
  params.seed_mode_number =
      pin->GetOrAddInteger("problem/marz_arrays", "seed_mode_number", 0);
  params.density_perturb_amplitude = pin->GetOrAddReal(
      "problem/marz_arrays", "density_perturb_amplitude", 0.0);
  params.temperature_perturb_amplitude = pin->GetOrAddReal(
      "problem/marz_arrays", "temperature_perturb_amplitude", 0.0);
  params.use_driven_x2_boundaries =
      pin->GetOrAddBoolean("problem/marz_arrays", "use_driven_x2_boundaries", true);
  params.use_diode_x1_boundaries =
      pin->GetOrAddBoolean("problem/marz_arrays", "use_diode_x1_boundaries", true);

  PARTHENON_REQUIRE(params.current_peak_MA >= 0.0,
                    "problem/marz_arrays/current_peak_MA must be nonnegative.");
  PARTHENON_REQUIRE(params.drive_rise_time >= 0.0,
                    "problem/marz_arrays/drive_rise_time must be nonnegative.");
  PARTHENON_REQUIRE(params.drive_plateau_fraction >= 0.0 &&
                        params.drive_plateau_fraction <= 1.0,
                    "problem/marz_arrays/drive_plateau_fraction must lie in [0, 1].");
  PARTHENON_REQUIRE(params.rho_array_cgs >= 0.0,
                    "problem/marz_arrays/rho_array must be nonnegative.");
  PARTHENON_REQUIRE(params.rho_background_cgs > 0.0,
                    "problem/marz_arrays/rho_background must be positive.");
  PARTHENON_REQUIRE(params.T_array >= 0.0,
                    "problem/marz_arrays/T_array must be nonnegative.");
  PARTHENON_REQUIRE(params.T_background > 0.0,
                    "problem/marz_arrays/T_background must be positive.");
  PARTHENON_REQUIRE(params.v_array_cgs >= 0.0,
                    "problem/marz_arrays/v_array must be nonnegative.");
  PARTHENON_REQUIRE(params.array_separation_cgs > 0.0,
                    "problem/marz_arrays/array_separation must be positive.");
  PARTHENON_REQUIRE(params.width_thermo_cgs > 0.0,
                    "problem/marz_arrays/width_thermo must be positive.");
  PARTHENON_REQUIRE(params.width_magnetic_cgs > 0.0,
                    "problem/marz_arrays/width_magnetic must be positive.");
  PARTHENON_REQUIRE(params.source_radius_cgs > 0.0,
                    "problem/marz_arrays/source_radius must be positive.");
  PARTHENON_REQUIRE(params.source_taper_width_cgs >= 0.0,
                    "problem/marz_arrays/source_taper_width must be nonnegative.");
  PARTHENON_REQUIRE(params.source_mass_replenish_time > 0.0,
                    "problem/marz_arrays/source_mass_replenish_time must be positive.");
  PARTHENON_REQUIRE(params.source_momentum_replenish_time > 0.0,
                    "problem/marz_arrays/source_momentum_replenish_time must be positive.");
  PARTHENON_REQUIRE(params.source_energy_replenish_time > 0.0,
                    "problem/marz_arrays/source_energy_replenish_time must be positive.");
  PARTHENON_REQUIRE(params.source_field_replenish_time > 0.0,
                    "problem/marz_arrays/source_field_replenish_time must be positive.");
  PARTHENON_REQUIRE(params.seed_mode_number >= 0,
                    "problem/marz_arrays/seed_mode_number must be nonnegative.");
  PARTHENON_REQUIRE(1.0 + params.density_perturb_amplitude > 0.0,
                    "problem/marz_arrays/density_perturb_amplitude must keep density positive.");
  PARTHENON_REQUIRE(1.0 + params.temperature_perturb_amplitude > 0.0,
                    "problem/marz_arrays/temperature_perturb_amplitude must keep temperature "
                    "positive.");

  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar =
      pin->GetReal("hydro", "mean_molecular_weight") * units.atomic_mass_unit();
  const Real current_peak_ampere = params.current_peak_MA * 1.0e6;
  params.current_field_prefac_peak =
      0.2 * current_peak_ampere * units.cm() * units.gauss();
  params.rho_array = params.rho_array_cgs * units.g_cm3();
  params.rho_background = params.rho_background_cgs * units.g_cm3();
  params.v_array = params.v_array_cgs * units.cm_s();
  params.array_separation = params.array_separation_cgs * units.cm();
  params.width_thermo = params.width_thermo_cgs * units.cm();
  params.width_magnetic = params.width_magnetic_cgs * units.cm();
  params.source_radius = params.source_radius_cgs * units.cm();
  params.source_taper_width = params.source_taper_width_cgs * units.cm();

  return params;
}

void EnsureParamsInitialized(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                             ParameterInput *pin) {
  if (!g_params_initialized) {
    g_params = LoadParams(hydro_pkg, pin);
    g_params_initialized = true;
  }
}

KOKKOS_INLINE_FUNCTION
Real DriveAmplitude(const MarzArraysParams &params, const Real time) {
  if (params.drive_model == DriveModel::static_source) {
    return 1.0;
  }
  if (time <= params.drive_t0) {
    return 0.0;
  }
  if (params.drive_rise_time <= 0.0) {
    return params.drive_plateau_fraction;
  }

  const Real xi = fmin(1.0, fmax(0.0, (time - params.drive_t0) / params.drive_rise_time));
  const Real smooth = xi * xi * (3.0 - 2.0 * xi);
  return params.drive_plateau_fraction * smooth;
}

KOKKOS_INLINE_FUNCTION
WireProfile EvaluateWireProfile(const MarzArraysParams &params, const Real x, const Real y) {
  WireProfile profile{};
  const Real half_sep = params.array_separation / 2.0;

  for (int wire = 0; wire < 2; ++wire) {
    const Real y_center = (wire == 0 ? -1.0 : 1.0) * half_sep;
    const Real y_local = y - y_center;
    const Real r2 = SQR(x) + SQR(y_local);
    const Real r = sqrt(r2);
    const Real mask = SourceMask(r, params.source_radius, params.source_taper_width);
    if (mask <= 0.0) {
      continue;
    }

    const Real theta = atan2(y_local, x);
    const Real thermo = GaussianProfile(r, params.width_thermo) * mask;
    const Real density_factor = AzimuthalPerturbation(
        theta, params.density_perturb_amplitude, params.seed_mode_number);
    const Real temperature_factor = AzimuthalPerturbation(
        theta, params.temperature_perturb_amplitude, params.seed_mode_number);

    profile.mask_sum += mask;
    profile.thermo_weight_sum += thermo;
    profile.density_weight_sum += thermo * density_factor;
    profile.temperature_weight_sum += thermo * temperature_factor;

    if (r > 0.0) {
      const Real inv_r = 1.0 / r;
      profile.radial_v1_sum += thermo * x * inv_r;
      profile.radial_v2_sum += thermo * y_local * inv_r;
    }

    const Real enclosed_fraction = 1.0 - exp(-r2 / SQR(params.width_magnetic));
    const Real Bphi_over_r =
        r2 > 0.0 ? params.current_field_prefac_peak * enclosed_fraction / r2
                 : params.current_field_prefac_peak / SQR(params.width_magnetic);
    profile.B1_sum += -mask * Bphi_over_r * y_local;
    profile.B2_sum += mask * Bphi_over_r * x;
  }

  profile.mask_sum = fmin(1.0, profile.mask_sum);
  return profile;
}

KOKKOS_INLINE_FUNCTION
InjectedState EvaluateInjectedState(const MarzArraysParams &params, const WireProfile &profile,
                                    const Real time) {
  InjectedState state{};
  const Real amplitude = DriveAmplitude(params, time);
  state.rho =
      fmax(float_min, params.rho_background + params.rho_array * profile.density_weight_sum);
  const Real temperature = fmax(
      float_min, params.T_background + params.T_array * profile.temperature_weight_sum);
  state.pressure = fmax(float_min, temperature * params.k_b * state.rho / params.m_bar);
  state.v1 = amplitude * params.v_array * profile.radial_v1_sum;
  state.v2 = amplitude * params.v_array * profile.radial_v2_sum;
  state.v3 = 0.0;
  state.B1 = amplitude * profile.B1_sum;
  state.B2 = amplitude * profile.B2_sum;
  state.B3 = 0.0;
  return state;
}

KOKKOS_INLINE_FUNCTION
void SetConservedCell(const MarzArraysParams &params, const InjectedState &state, const int k,
                      const int j, const int i, ParArrayND<Real> &u) {
  u(IDN, k, j, i) = state.rho;
  u(IM1, k, j, i) = state.rho * state.v1;
  u(IM2, k, j, i) = state.rho * state.v2;
  u(IM3, k, j, i) = state.rho * state.v3;
  u(IB1, k, j, i) = state.B1;
  u(IB2, k, j, i) = state.B2;
  u(IB3, k, j, i) = state.B3;
  u(IEN, k, j, i) =
      state.pressure / params.gm1 +
      0.5 * (SQR(state.B1) + SQR(state.B2) + SQR(state.B3) +
             state.rho * (SQR(state.v1) + SQR(state.v2) + SQR(state.v3)));
}

template <typename PackAccessor>
KOKKOS_INLINE_FUNCTION
void SetConservedCell(const MarzArraysParams &params, const InjectedState &state, const int k,
                      const int j, const int i, PackAccessor &u) {
  u(IDN, k, j, i) = state.rho;
  u(IM1, k, j, i) = state.rho * state.v1;
  u(IM2, k, j, i) = state.rho * state.v2;
  u(IM3, k, j, i) = state.rho * state.v3;
  u(IB1, k, j, i) = state.B1;
  u(IB2, k, j, i) = state.B2;
  u(IB3, k, j, i) = state.B3;
  u(IEN, k, j, i) =
      state.pressure / params.gm1 +
      0.5 * (SQR(state.B1) + SQR(state.B2) + SQR(state.B3) +
             state.rho * (SQR(state.v1) + SQR(state.v2) + SQR(state.v3)));
}

KOKKOS_INLINE_FUNCTION
Real RelaxationFactor(const Real dt, const Real tau, const Real weight) {
  if (weight <= 0.0) {
    return 0.0;
  }
  return fmin(1.0, weight * dt / tau);
}

template <typename PackAccessor>
KOKKOS_INLINE_FUNCTION
void ApplyLocalizedSource(const MarzArraysParams &params, const WireProfile &profile,
                          const InjectedState &target, const Real dt, const int k,
                          const int j, const int i, PackAccessor &cons) {
  const Real weight = profile.mask_sum;
  if (weight <= 0.0) {
    return;
  }

  const Real alpha_rho =
      RelaxationFactor(dt, params.source_mass_replenish_time, weight);
  const Real alpha_mom =
      RelaxationFactor(dt, params.source_momentum_replenish_time, weight);
  const Real alpha_e =
      RelaxationFactor(dt, params.source_energy_replenish_time, weight);
  const Real alpha_B =
      RelaxationFactor(dt, params.source_field_replenish_time, weight);

  const Real rho_old = cons(IDN, k, j, i);
  const Real m1_old = cons(IM1, k, j, i);
  const Real m2_old = cons(IM2, k, j, i);
  const Real m3_old = cons(IM3, k, j, i);
  const Real B1_old = cons(IB1, k, j, i);
  const Real B2_old = cons(IB2, k, j, i);
  const Real B3_old = cons(IB3, k, j, i);
  const Real E_old = cons(IEN, k, j, i);

  const Real Bsq_old = SQR(B1_old) + SQR(B2_old) + SQR(B3_old);
  const Real kin_old =
      0.5 * (SQR(m1_old) + SQR(m2_old) + SQR(m3_old)) / fmax(rho_old, float_min);
  const Real eint_old = fmax(float_min, E_old - kin_old - 0.5 * Bsq_old);

  const Real rho_new = fmax(float_min, rho_old + alpha_rho * (target.rho - rho_old));
  const Real m1_target = target.rho * target.v1;
  const Real m2_target = target.rho * target.v2;
  const Real m3_target = target.rho * target.v3;
  const Real m1_new = m1_old + alpha_mom * (m1_target - m1_old);
  const Real m2_new = m2_old + alpha_mom * (m2_target - m2_old);
  const Real m3_new = m3_old + alpha_mom * (m3_target - m3_old);
  const Real B1_new = B1_old + alpha_B * (target.B1 - B1_old);
  const Real B2_new = B2_old + alpha_B * (target.B2 - B2_old);
  const Real B3_new = B3_old + alpha_B * (target.B3 - B3_old);
  const Real eint_target = target.pressure / params.gm1;
  const Real eint_new = fmax(float_min, eint_old + alpha_e * (eint_target - eint_old));
  const Real kin_new = 0.5 * (SQR(m1_new) + SQR(m2_new) + SQR(m3_new)) / rho_new;
  const Real Bsq_new = SQR(B1_new) + SQR(B2_new) + SQR(B3_new);

  cons(IDN, k, j, i) = rho_new;
  cons(IM1, k, j, i) = m1_new;
  cons(IM2, k, j, i) = m2_new;
  cons(IM3, k, j, i) = m3_new;
  cons(IB1, k, j, i) = B1_new;
  cons(IB2, k, j, i) = B2_new;
  cons(IB3, k, j, i) = B3_new;
  cons(IEN, k, j, i) = eint_new + kin_new + 0.5 * Bsq_new;
}

template <parthenon::BoundaryFunction::BCSide SIDE>
void DrivenX1Boundary(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  if (!g_params.use_diode_x1_boundaries) {
    Hydro::BoundaryFunction::LinearBC<X1DIR, SIDE>(mbd, coarse);
    return;
  }
  Hydro::BoundaryFunction::DiodeX1BC<SIDE>(mbd, coarse);
}

template <parthenon::BoundaryFunction::BCSide SIDE>
void SourceX2Boundary(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  if (!g_params.use_driven_x2_boundaries) {
    Hydro::BoundaryFunction::LinearBC<X2DIR, SIDE>(mbd, coarse);
    return;
  }

  Hydro::BoundaryFunction::LinearBC<X2DIR, SIDE>(mbd, coarse);

  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  constexpr IndexDomain domain =
      (SIDE == parthenon::BoundaryFunction::BCSide::Inner) ? IndexDomain::inner_x2
                                                           : IndexDomain::outer_x2;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  const auto params = g_params;
  const Real time = g_drive_time;
  auto coords = pmb->coords;

  pmb->par_for_bndry(
      "marz_arrays::SourceX2Boundary", nb, domain, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto profile = EvaluateWireProfile(params, x, y);
        if (profile.mask_sum <= 0.0) {
          return;
        }
        const auto state = EvaluateInjectedState(params, profile, time);
        SetConservedCell(params, state, k, j, i, cons);
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
  g_drive_time = 0.0;
}

void PreStepMeshUserWorkInLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  Hydro::PreStepMeshUserWorkInLoop(mesh, pin, tm);
  g_drive_time = tm.time;
}

void DriveSource(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto params = g_params;
  const Real drive_time = tm.time + dt;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "marz_arrays::DriveSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto profile = EvaluateWireProfile(params, x, y);
        const auto target = EvaluateInjectedState(params, profile, drive_time);
        ApplyLocalizedSource(params, profile, target, dt, k, j, i, cons);
      });
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
      "marz_arrays::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
                    "marz_arrays requires at least a 2D mesh.");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  EnsureParamsInitialized(hydro_pkg, pin);
  const auto params = g_params;
  const Real amplitude0 = DriveAmplitude(params, 0.0);
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");

  PARTHENON_REQUIRE(params.array_separation < (x2max - x2min),
                    "problem/marz_arrays/array_separation must fit inside the x2 domain.");

  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "current_peak_MA ============== " << params.current_peak_MA << std::endl;
    std::cout << "drive_t0 [code] ============= " << params.drive_t0 << std::endl;
    std::cout << "drive_rise_time [code] ====== " << params.drive_rise_time << std::endl;
    std::cout << "drive_plateau_fraction ====== " << params.drive_plateau_fraction
              << std::endl;
    std::cout << "rho_array [g/cm^3] ========== " << params.rho_array_cgs << std::endl;
    std::cout << "rho_background [g/cm^3] ==== " << params.rho_background_cgs << std::endl;
    std::cout << "T_array [K] ================= " << params.T_array << std::endl;
    std::cout << "T_background [K] ============ " << params.T_background << std::endl;
    std::cout << "v_array [cm/s] ============== " << params.v_array_cgs << std::endl;
    std::cout << "array_separation [cm] ======= " << params.array_separation_cgs
              << std::endl;
    std::cout << "width_thermo [cm] =========== " << params.width_thermo_cgs << std::endl;
    std::cout << "width_magnetic [cm] ========= " << params.width_magnetic_cgs
              << std::endl;
    std::cout << "source_radius [cm] ========== " << params.source_radius_cgs << std::endl;
    std::cout << "source_taper_width [cm] ===== " << params.source_taper_width_cgs
              << std::endl;
    std::cout << "tau_mass [code] ============= " << params.source_mass_replenish_time
              << std::endl;
    std::cout << "tau_momentum [code] ========= "
              << params.source_momentum_replenish_time << std::endl;
    std::cout << "tau_energy [code] =========== " << params.source_energy_replenish_time
              << std::endl;
    std::cout << "tau_field [code] ============ " << params.source_field_replenish_time
              << std::endl;
    std::cout << "seed_mode_number ============ " << params.seed_mode_number << std::endl;
    std::cout << "density_perturb_amplitude === " << params.density_perturb_amplitude
              << std::endl;
    std::cout << "temperature_perturb_amplitude " << params.temperature_perturb_amplitude
              << std::endl;
    std::cout << "use_driven_x2_boundaries ==== " << params.use_driven_x2_boundaries
              << std::endl;
    std::cout << "use_diode_x1_boundaries ===== " << params.use_diode_x1_boundaries
              << std::endl;
    std::cout << "drive_amplitude(t=0) ======== " << amplitude0 << std::endl;
    std::cout << "========================================" << std::endl;
  }

  auto &coords = pmb->coords;
  pmb->par_for(
      "ProblemGenerator::marz_arrays", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto profile = EvaluateWireProfile(params, x, y);
        const auto state = EvaluateInjectedState(params, profile, 0.0);
        SetConservedCell(params, state, k, j, i, u);
      });
}

void DrivenInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  DrivenX1Boundary<parthenon::BoundaryFunction::BCSide::Inner>(mbd, coarse);
}

void DrivenOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  DrivenX1Boundary<parthenon::BoundaryFunction::BCSide::Outer>(mbd, coarse);
}

void SourceInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  SourceX2Boundary<parthenon::BoundaryFunction::BCSide::Inner>(mbd, coarse);
}

void SourceOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  SourceX2Boundary<parthenon::BoundaryFunction::BCSide::Outer>(mbd, coarse);
}

} // namespace marz_arrays
