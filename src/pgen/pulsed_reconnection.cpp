//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file pulsed_reconnection.cpp
//! \brief Problem generator for pulsed reconnection with configurable thermal and
//! magnetic radial profiles.
//========================================================================================

// Parthenon headers
#include "Kokkos_MathematicalFunctions.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

namespace pulsed_reconnection {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

enum class ProfileShape { none, gaussian, tophat, cubic, wendland, quintic };

struct RadialProfileParams {
  ProfileShape shape = ProfileShape::none;
  Real width = 0.0;
  Real tophat_core_width = 0.0;
  Real tophat_falloff_width = 0.0;
};

struct SupportTable {
  static constexpr int kMaxPoints = 2049;
  int num_points = 0;
  Real r_max = 0.0;
  Real dr = 0.0;
  Real values[kMaxPoints] = {};
};

KOKKOS_INLINE_FUNCTION
void EvaluateGaussianProfileAndDerivative(const Real r, const Real width, Real &profile,
                                          Real &dprofile_dr);

KOKKOS_INLINE_FUNCTION
void EvaluateRadialProfileAndDerivative(const RadialProfileParams &params, const Real r,
                                        Real &profile, Real &dprofile_dr);

KOKKOS_INLINE_FUNCTION
Real EvaluateRadialProfile(const RadialProfileParams &params, const Real r);

KOKKOS_INLINE_FUNCTION
Real EvaluateSupportTable(const SupportTable &table, const Real r);

// Only density and temperature inherit the azimuthal modulation.
KOKKOS_INLINE_FUNCTION
Real AzimuthalThermoPerturbation(const Real theta, const Real p, const int mode_number);

namespace {

Real PeakNormalizedDerivativeMagnitude(const RadialProfileParams &params);
Real ProfileSupportRadius(const RadialProfileParams &params);
SupportTable BuildUnitAmplitudeSupportTable(const RadialProfileParams &params,
                                            const std::string &label);

struct PulsedGaussianParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real current_peak_MA;
  bool drive_enable;
  Real drive_peak_current_MA;
  Real drive_t_peak_ns;
  Real drive_t_peak;
  Real drive_rho_floor_cgs;
  Real drive_rho_floor;
  Real drive_T_floor;
  Real rho_wire_cgs;
  Real rho_background_cgs;
  Real T_wire;
  Real T_background;
  Real v0_cgs;
  Real array_separation_cgs;
  int azimuthal_mode_number;
  Real density_perturb_amplitude;
  Real temperature_perturb_amplitude;
  Real current_field_prefac;
  Real rho_wire;
  Real rho_background;
  Real v0;
  Real array_separation;
  Real velocity_normalization;
  Real peak_magnetic_field_strength;
  Real amr_magnetic_field_reference;
  Real initial_magnetic_profile_amplitude;
  Real drive_peak_magnetic_profile_amplitude;
  bool initial_force_balance;
  bool drive_force_balance;
  RadialProfileParams initial_thermal_profile;
  RadialProfileParams initial_magnetic_profile;
  RadialProfileParams drive_thermal_profile;
  RadialProfileParams drive_magnetic_profile;
  SupportTable initial_support_table;
  SupportTable drive_support_table;
};

struct PulsedGaussianState {
  Real rho;
  Real pressure;
  Real v1;
  Real v2;
  Real v3;
};

struct DriveFloorState {
  Real rho_floor;
  Real T_floor;
};

KOKKOS_INLINE_FUNCTION
Real EvaluateMagneticSupportSum(const PulsedGaussianParams &params,
                                const SupportTable &table, const Real x, const Real y,
                                const Real amplitude) {
  if (table.num_points < 2 || amplitude == 0.0) {
    return 0.0;
  }
  Real support = 0.0;
  const Real half_sep = 0.5 * params.array_separation;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    support += EvaluateSupportTable(table, sqrt(SQR(x) + SQR(y_local)));
  }
  return SQR(amplitude) * support;
}

PulsedGaussianParams g_source_params{};
bool g_source_params_initialized = false;

// The legacy single-wire ampere-loop field peaks at q = (r / w_B)^2 satisfying
// exp(-q) * (2 q + 1) = 1. These constants keep the current input roughly aligned with
// the old peak-field scaling even as the profile families change.
constexpr Real kOldAmpereLoopPeakQ = 1.2564312086261696770;
constexpr Real kOldAmpereLoopPeakFactor = 0.6381726863389509616;
constexpr Real kGaussianGradientPeakFactor = 0.8577638849607067968; // sqrt(2 / e)

KOKKOS_INLINE_FUNCTION
Real DriveCurrentAtTime(const PulsedGaussianParams &params, const Real time) {
  if (time <= 0.0 || params.drive_t_peak <= 0.0 || time >= 2.0 * params.drive_t_peak) {
    return 0.0;
  }
  const Real phase = M_PI * time / (2.0 * params.drive_t_peak);
  const Real s = sin(phase);
  return params.drive_peak_current_MA * 1.0e6 * s * s;
}

KOKKOS_INLINE_FUNCTION
Real DrivePotentialAmplitudeFromCurrent(const PulsedGaussianParams &params,
                                        const Real current_ampere) {
  const Real peak_current_ampere = params.drive_peak_current_MA * 1.0e6;
  if (peak_current_ampere <= 0.0) {
    return 0.0;
  }
  return params.drive_peak_magnetic_profile_amplitude * current_ampere /
         peak_current_ampere;
}

KOKKOS_INLINE_FUNCTION
Real EvaluateDriveAz(const PulsedGaussianParams &params, const Real x, const Real y,
                     const Real amplitude) {
  if (params.drive_magnetic_profile.shape == ProfileShape::none || amplitude == 0.0) {
    return 0.0;
  }
  const Real half_sep = 0.5 * params.array_separation;
  Real az = 0.0;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    az += amplitude *
          EvaluateRadialProfile(params.drive_magnetic_profile, sqrt(SQR(x) + SQR(y_local)));
  }
  return az;
}

KOKKOS_INLINE_FUNCTION
Real EvaluateInitialAz(const PulsedGaussianParams &params, const Real x, const Real y,
                       const Real /*z*/) {
  if (params.initial_magnetic_profile.shape == ProfileShape::none ||
      params.initial_magnetic_profile_amplitude == 0.0) {
    return 0.0;
  }
  const Real half_sep = 0.5 * params.array_separation;
  Real az = 0.0;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    az += params.initial_magnetic_profile_amplitude *
          EvaluateRadialProfile(params.initial_magnetic_profile,
                                sqrt(SQR(x) + SQR(y_local)));
  }
  return az;
}

template <typename B1Face>
KOKKOS_INLINE_FUNCTION Real CellCenteredB1(const B1Face &b1f, const int k, const int j,
                                           const int i) {
  return 0.5 * (b1f(k, j, i) + b1f(k, j, i + 1));
}

template <typename B2Face>
KOKKOS_INLINE_FUNCTION Real CellCenteredB2(const B2Face &b2f, const int k, const int j,
                                           const int i) {
  return 0.5 * (b2f(k, j, i) + b2f(k, j + 1, i));
}

template <typename B3Face>
KOKKOS_INLINE_FUNCTION Real CellCenteredB3(const B3Face &b3f, const int ndim,
                                           const int k, const int j, const int i) {
  return ndim > 2 ? 0.5 * (b3f(k, j, i) + b3f(k + 1, j, i))
                  : 0.0;
}

KOKKOS_INLINE_FUNCTION
DriveFloorState EvaluateDriveFloorState(const PulsedGaussianParams &params, const Real x,
                                        const Real y) {
  DriveFloorState floors{0.0, 0.0};
  if (params.drive_thermal_profile.shape == ProfileShape::none) {
    return floors;
  }
  const Real half_sep = 0.5 * params.array_separation;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    const Real profile =
        EvaluateRadialProfile(params.drive_thermal_profile, sqrt(SQR(x) + SQR(y_local)));
    floors.rho_floor += params.drive_rho_floor * profile;
    floors.T_floor += params.drive_T_floor * profile;
  }
  return floors;
}

ProfileShape ParseProfileShape(const std::string &name, const bool allow_none,
                               const std::string &input_name) {
  if (allow_none && name == "none") {
    return ProfileShape::none;
  }
  if (name == "gaussian") {
    return ProfileShape::gaussian;
  }
  if (name == "tophat") {
    return ProfileShape::tophat;
  }
  if (name == "cubic") {
    return ProfileShape::cubic;
  }
  if (name == "wendland") {
    return ProfileShape::wendland;
  }
  if (name == "quintic") {
    return ProfileShape::quintic;
  }
  const std::string allowed =
      allow_none ? "'none', 'gaussian', 'tophat', 'cubic', 'wendland', or 'quintic'"
                 : "'gaussian', 'tophat', 'cubic', 'wendland', or 'quintic'";
  PARTHENON_FAIL("problem/pulsed_reconnection/" + input_name +
                 " must be one of " + allowed + ".");
}

const char *ProfileShapeName(const ProfileShape shape) {
  switch (shape) {
  case ProfileShape::none:
    return "none";
  case ProfileShape::gaussian:
    return "gaussian";
  case ProfileShape::tophat:
    return "tophat";
  case ProfileShape::cubic:
    return "cubic";
  case ProfileShape::wendland:
    return "wendland";
  case ProfileShape::quintic:
    return "quintic";
  }
  return "unknown";
}

void RejectLegacyInputKeys(ParameterInput *pin) {
  const char *block = "problem/pulsed_reconnection";
  for (const auto &key : std::vector<std::string>{"thermal_profile", "force_balance",
                                                  "drive_hydro_support_enable",
                                                  "drive_cutoff_radius_factor",
                                                  "core_width", "falloff_width"}) {
    if (pin->DoesParameterExist(block, key)) {
      PARTHENON_FAIL("problem/pulsed_reconnection/" + key +
                     " has been replaced by the new profile schema.");
    }
  }
}

PulsedGaussianParams LoadSourceParams(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                                      ParameterInput *pin) {
  RejectLegacyInputKeys(pin);
  PulsedGaussianParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.current_peak_MA = pin->GetOrAddReal("problem/pulsed_reconnection",
                                             "current_peak_MA", 1.0);
  params.drive_enable =
      pin->GetOrAddBoolean("problem/pulsed_reconnection", "drive_enable", false);
  params.drive_peak_current_MA = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_peak_current_MA",
      params.current_peak_MA);
  params.drive_t_peak_ns = pin->GetOrAddReal("problem/pulsed_reconnection",
                                             "drive_t_peak_ns", 500.0);
  params.rho_wire_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "rho_wire", 1e-3);
  params.rho_background_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "rho_background", 1e-6);
  params.T_wire =
      pin->GetOrAddReal("problem/pulsed_reconnection", "T_wire", 1.1e4);
  params.T_background =
      pin->GetOrAddReal("problem/pulsed_reconnection", "T_background", 1e2);
  params.v0_cgs = pin->GetOrAddReal("problem/pulsed_reconnection", "v0", 1.0e6);
  params.array_separation_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "array_separation", 4.0);
  params.initial_thermal_profile.width =
      pin->GetOrAddReal("problem/pulsed_reconnection", "w", 1.0);
  params.initial_thermal_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "initial_thermal_profile", "gaussian"),
                        false, "initial_thermal_profile");
  params.initial_thermal_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_thermal_core_width",
      params.initial_thermal_profile.width);
  params.initial_thermal_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_thermal_falloff_width",
      params.initial_thermal_profile.width);
  params.initial_magnetic_profile.width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_B",
      pin->GetOrAddReal("problem/pulsed_reconnection", "w_magnetic",
                        params.initial_thermal_profile.width));
  params.initial_magnetic_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "initial_magnetic_profile", "gaussian"),
                        true, "initial_magnetic_profile");
  params.initial_magnetic_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_magnetic_core_width",
      params.initial_magnetic_profile.width);
  params.initial_magnetic_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_magnetic_falloff_width",
      params.initial_magnetic_profile.width);
  params.initial_force_balance = pin->GetOrAddBoolean(
      "problem/pulsed_reconnection", "initial_force_balance", true);
  params.drive_thermal_profile.width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_drive",
      params.initial_magnetic_profile.width);
  params.drive_thermal_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "drive_thermal_profile", "none"),
                        true, "drive_thermal_profile");
  params.drive_thermal_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_thermal_core_width",
      params.drive_thermal_profile.width);
  params.drive_thermal_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_thermal_falloff_width",
      params.drive_thermal_profile.width);
  params.drive_magnetic_profile = params.drive_thermal_profile;
  params.drive_magnetic_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "drive_magnetic_profile", "gaussian"),
                        true, "drive_magnetic_profile");
  params.drive_magnetic_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_magnetic_core_width",
      params.drive_thermal_profile.width);
  params.drive_magnetic_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_magnetic_falloff_width",
      params.drive_thermal_profile.width);
  params.drive_force_balance = pin->GetOrAddBoolean(
      "problem/pulsed_reconnection", "drive_force_balance", false);
  params.drive_rho_floor_cgs = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_rho_floor", params.rho_wire_cgs);
  params.drive_T_floor = pin->GetOrAddReal("problem/pulsed_reconnection",
                                           "drive_T_floor", params.T_wire);
  params.azimuthal_mode_number =
      pin->GetOrAddInteger("problem/pulsed_reconnection", "N", 0);
  params.density_perturb_amplitude = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "density_perturb_amplitude", 0.0);
  params.temperature_perturb_amplitude = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "temperature_perturb_amplitude", 0.0);
  PARTHENON_REQUIRE(params.initial_thermal_profile.width > 0.0,
                    "problem/pulsed_reconnection/w must be positive.");
  PARTHENON_REQUIRE(
      params.initial_thermal_profile.tophat_core_width > 0.0,
      "problem/pulsed_reconnection/initial_thermal_core_width must be positive.");
  PARTHENON_REQUIRE(
      params.initial_thermal_profile.tophat_falloff_width > 0.0,
      "problem/pulsed_reconnection/initial_thermal_falloff_width must be "
      "positive.");
  PARTHENON_REQUIRE(params.initial_magnetic_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_B must be positive.");
  PARTHENON_REQUIRE(params.drive_thermal_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_drive must be positive.");
  PARTHENON_REQUIRE(params.array_separation_cgs > 0.0,
                    "problem/pulsed_reconnection/array_separation must be "
                    "positive.");
  PARTHENON_REQUIRE(params.current_peak_MA >= 0.0,
                    "problem/pulsed_reconnection/current_peak_MA must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_peak_current_MA >= 0.0,
                    "problem/pulsed_reconnection/drive_peak_current_MA must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_t_peak_ns > 0.0,
                    "problem/pulsed_reconnection/drive_t_peak_ns must be "
                    "positive.");
  PARTHENON_REQUIRE(params.drive_rho_floor_cgs >= 0.0,
                    "problem/pulsed_reconnection/drive_rho_floor must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_T_floor >= 0.0,
                    "problem/pulsed_reconnection/drive_T_floor must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.azimuthal_mode_number >= 0,
                    "problem/pulsed_reconnection/N must be nonnegative.");

  PARTHENON_REQUIRE(hydro_pkg->AllParams().hasKey("units") &&
                        hydro_pkg->AllParams().hasKey("mbar") &&
                        hydro_pkg->AllParams().hasKey("mbar_over_kb"),
                    "pulsed_reconnection requires a <units> block and "
                    "hydro/He_mass_fraction.");
  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar = hydro_pkg->Param<Real>("mbar");
  const Real current_peak_ampere = params.current_peak_MA * 1.0e6;
  params.current_field_prefac = 0.2 * current_peak_ampere * units.cm() * units.gauss();
  params.rho_wire = params.rho_wire_cgs * units.g_cm3();
  params.rho_background = params.rho_background_cgs * units.g_cm3();
  params.drive_rho_floor = params.drive_rho_floor_cgs * units.g_cm3();
  params.v0 = params.v0_cgs * units.cm_s();
  params.array_separation = params.array_separation_cgs * units.cm();
  params.initial_thermal_profile.width *= units.cm();
  params.initial_thermal_profile.tophat_core_width *= units.cm();
  params.initial_thermal_profile.tophat_falloff_width *= units.cm();
  params.initial_magnetic_profile.width *= units.cm();
  params.initial_magnetic_profile.tophat_core_width *= units.cm();
  params.initial_magnetic_profile.tophat_falloff_width *= units.cm();
  params.drive_thermal_profile.width *= units.cm();
  params.drive_thermal_profile.tophat_core_width *= units.cm();
  params.drive_thermal_profile.tophat_falloff_width *= units.cm();
  params.drive_magnetic_profile.width *= units.cm();
  params.drive_magnetic_profile.tophat_core_width *= units.cm();
  params.drive_magnetic_profile.tophat_falloff_width *= units.cm();
  // drive_t_peak_ns is a physical time. units.s() is the number of code-time
  // units per physical second, so multiply to convert seconds to code time.
  params.drive_t_peak = params.drive_t_peak_ns * 1.0e-9 * units.s();

  const Real drive_peak_current_ampere = params.drive_peak_current_MA * 1.0e6;
  const Real drive_current_field_prefac =
      0.2 * drive_peak_current_ampere * units.cm() * units.gauss();
  params.peak_magnetic_field_strength =
      params.initial_magnetic_profile.width > 0.0
          ? params.current_field_prefac * kOldAmpereLoopPeakFactor /
                params.initial_magnetic_profile.width
          : 0.0;
  const Real initial_peak_grad =
      PeakNormalizedDerivativeMagnitude(params.initial_magnetic_profile);
  params.initial_magnetic_profile_amplitude =
      initial_peak_grad > 0.0 ? params.peak_magnetic_field_strength / initial_peak_grad
                              : 0.0;
  const Real drive_peak_field =
      params.drive_magnetic_profile.width > 0.0
          ? drive_current_field_prefac * kOldAmpereLoopPeakFactor /
                params.drive_magnetic_profile.width
          : 0.0;
  params.amr_magnetic_field_reference =
      fmax(params.peak_magnetic_field_strength, drive_peak_field);
  if (pin->GetString("refinement", "type") == "user") {
    PARTHENON_REQUIRE(
        params.amr_magnetic_field_reference > 0.0,
        "Current-based AMR for pulsed_reconnection requires a positive "
        "initial or drive peak magnetic-field strength.");
  }
  const Real drive_peak_grad =
      PeakNormalizedDerivativeMagnitude(params.drive_magnetic_profile);
  params.drive_peak_magnetic_profile_amplitude =
      drive_peak_grad > 0.0 ? drive_peak_field / drive_peak_grad : 0.0;
  params.velocity_normalization =
      params.initial_thermal_profile.width > 0.0
          ? params.v0 / (kGaussianGradientPeakFactor / params.initial_thermal_profile.width)
          : 0.0;
  params.initial_support_table =
      BuildUnitAmplitudeSupportTable(params.initial_magnetic_profile, "initial_support");
  params.drive_support_table =
      BuildUnitAmplitudeSupportTable(params.drive_magnetic_profile, "drive_support");

  return params;
}

void EnsureSourceParamsInitialized(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                                   ParameterInput *pin) {
  if (!g_source_params_initialized) {
    g_source_params = LoadSourceParams(hydro_pkg, pin);
    g_source_params_initialized = true;
  }
}

KOKKOS_INLINE_FUNCTION
PulsedGaussianState EvaluateSourceState(const PulsedGaussianParams &params, const Real x,
                                        const Real y) {
  PulsedGaussianState state{};
  const Real d = params.array_separation / 2.0;
  Real thermo_profile_sum = 0.0;
  Real density_profile_sum = 0.0;
  Real magnetic_support_sum = 0.0;

  for (int A = -1; A <= 1; A += 2) {
    const Real y_center = A * d;
    const Real y_local = y - y_center;
    const Real r2 = SQR(x) + SQR(y_local);
    const Real r = sqrt(r2);
    const Real theta = atan2(y_local, x);

    const Real thermo_profile = EvaluateRadialProfile(params.initial_thermal_profile, r);
    Real gaussian_velocity_profile_unused = 0.0;
    Real dthermo_profile_dr = 0.0;
    EvaluateGaussianProfileAndDerivative(r, params.initial_thermal_profile.width,
                                         gaussian_velocity_profile_unused,
                                         dthermo_profile_dr);
    const Real density_perturbation = AzimuthalThermoPerturbation(
        theta, params.density_perturb_amplitude, params.azimuthal_mode_number);
    const Real temperature_perturbation = AzimuthalThermoPerturbation(
        theta, params.temperature_perturb_amplitude, params.azimuthal_mode_number);
    thermo_profile_sum += thermo_profile * temperature_perturbation;
    density_profile_sum += thermo_profile * density_perturbation;
    magnetic_support_sum += EvaluateSupportTable(params.initial_support_table, r);

    if (r > 0.0) {
      const Real inv_r = 1.0 / r;
      const Real xhat = x * inv_r;
      const Real yhat = y_local * inv_r;

      const Real radial_velocity =
          params.drive_enable ? 0.0 : -params.velocity_normalization * dthermo_profile_dr;
      state.v1 += radial_velocity * xhat;
      state.v2 += radial_velocity * yhat;

    }
  }

  state.rho = params.rho_background + params.rho_wire * density_profile_sum;
  const Real T = params.T_background + params.T_wire * thermo_profile_sum;
  state.pressure =
      T * params.k_b * state.rho / params.m_bar +
      (params.initial_force_balance
           ? SQR(params.initial_magnetic_profile_amplitude) * magnetic_support_sum
           : 0.0);
  return state;
}

} // namespace

KOKKOS_INLINE_FUNCTION
void EvaluateGaussianProfileAndDerivative(const Real r, const Real width, Real &profile,
                                          Real &dprofile_dr) {
  const Real exponent = -SQR(r / width);
  profile = exp(fmax(-700.0, exponent));
  dprofile_dr = (-2.0 * r / SQR(width)) * profile;
}

KOKKOS_INLINE_FUNCTION
void EvaluateRadialProfileAndDerivative(const RadialProfileParams &params, const Real r,
                                        Real &profile, Real &dprofile_dr) {
  if (params.shape == ProfileShape::none) {
    profile = 0.0;
    dprofile_dr = 0.0;
    return;
  }
  if (params.shape == ProfileShape::gaussian) {
    EvaluateGaussianProfileAndDerivative(r, params.width, profile, dprofile_dr);
    return;
  }
  if (params.shape == ProfileShape::tophat) {
    if (r <= params.tophat_core_width) {
      profile = 1.0;
      dprofile_dr = 0.0;
    } else if (r >= params.tophat_core_width + params.tophat_falloff_width) {
      profile = 0.0;
      dprofile_dr = 0.0;
    } else {
      const Real x = (r - params.tophat_core_width) / params.tophat_falloff_width;
      const Real x2 = x * x;
      const Real x3 = x2 * x;
      const Real x4 = x3 * x;
      const Real x5 = x4 * x;
      profile = 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
      dprofile_dr =
          (-30.0 * x2 + 60.0 * x3 - 30.0 * x4) / params.tophat_falloff_width;
    }
    return;
  }
  const Real q = r / params.width;
  const Real inv_width = 1.0 / params.width;
  if (params.shape == ProfileShape::cubic) {
    if (q >= 2.0) {
      profile = 0.0;
      dprofile_dr = 0.0;
    } else if (q <= 1.0) {
      profile = 1.0 - 1.5 * q * q + 0.75 * q * q * q;
      dprofile_dr = (-3.0 * q + 2.25 * q * q) * inv_width;
    } else {
      const Real s = 2.0 - q;
      profile = 0.25 * s * s * s;
      dprofile_dr = -0.75 * s * s * inv_width;
    }
    return;
  }
  if (params.shape == ProfileShape::wendland) {
    if (q >= 2.0) {
      profile = 0.0;
      dprofile_dr = 0.0;
    } else {
      const Real s = 1.0 - 0.5 * q;
      profile = s * s * s * (1.0 + 1.5 * q);
      dprofile_dr = -3.0 * q * s * s * inv_width;
    }
    return;
  }
  if (q >= 3.0) {
    profile = 0.0;
    dprofile_dr = 0.0;
  } else if (q <= 1.0) {
    profile = (pow(3.0 - q, 5) - 6.0 * pow(2.0 - q, 5) + 15.0 * pow(1.0 - q, 5)) /
              66.0;
    dprofile_dr =
        (-5.0 * pow(3.0 - q, 4) + 30.0 * pow(2.0 - q, 4) -
         75.0 * pow(1.0 - q, 4)) *
        inv_width / 66.0;
  } else if (q <= 2.0) {
    profile = (pow(3.0 - q, 5) - 6.0 * pow(2.0 - q, 5)) / 66.0;
    dprofile_dr =
        (-5.0 * pow(3.0 - q, 4) + 30.0 * pow(2.0 - q, 4)) * inv_width / 66.0;
  } else {
    profile = pow(3.0 - q, 5) / 66.0;
    dprofile_dr = -5.0 * pow(3.0 - q, 4) * inv_width / 66.0;
  }
}

KOKKOS_INLINE_FUNCTION
Real EvaluateRadialProfile(const RadialProfileParams &params, const Real r) {
  Real profile = 0.0;
  Real dprofile_dr = 0.0;
  EvaluateRadialProfileAndDerivative(params, r, profile, dprofile_dr);
  return profile;
}

KOKKOS_INLINE_FUNCTION
Real EvaluateSupportTable(const SupportTable &table, const Real r) {
  if (table.num_points < 2 || r >= table.r_max) {
    return 0.0;
  }
  const Real idx = r / table.dr;
  const int i0 = static_cast<int>(
      fmin(static_cast<Real>(table.num_points - 2), floor(idx)));
  const Real frac = idx - static_cast<Real>(i0);
  return (1.0 - frac) * table.values[i0] + frac * table.values[i0 + 1];
}

KOKKOS_INLINE_FUNCTION
Real AzimuthalThermoPerturbation(const Real theta, const Real p, const int mode_number) {
  const Real phase = static_cast<Real>(mode_number) * theta;
  const Real cos_phase = cos(phase);
  return 1 + p * cos_phase;
}

namespace {

Real ProfileSupportRadius(const RadialProfileParams &params) {
  switch (params.shape) {
  case ProfileShape::none:
    return 0.0;
  case ProfileShape::gaussian:
    return 6.0 * params.width;
  case ProfileShape::tophat:
    return params.tophat_core_width + params.tophat_falloff_width;
  case ProfileShape::cubic:
  case ProfileShape::wendland:
    return 2.0 * params.width;
  case ProfileShape::quintic:
    return 3.0 * params.width;
  }
  return 0.0;
}

Real PeakNormalizedDerivativeMagnitude(const RadialProfileParams &params) {
  if (params.shape == ProfileShape::none) {
    return 0.0;
  }
  if (params.shape == ProfileShape::gaussian) {
    return kGaussianGradientPeakFactor / params.width;
  }
  const int samples = 20000;
  const Real r_max = ProfileSupportRadius(params);
  Real peak = 0.0;
  for (int i = 0; i <= samples; ++i) {
    const Real r = r_max * static_cast<Real>(i) / static_cast<Real>(samples);
    Real profile = 0.0;
    Real dprofile_dr = 0.0;
    EvaluateRadialProfileAndDerivative(params, r, profile, dprofile_dr);
    peak = std::max(peak, std::abs(dprofile_dr));
  }
  return peak;
}

SupportTable BuildUnitAmplitudeSupportTable(const RadialProfileParams &params,
                                            const std::string &label) {
  SupportTable table;
  if (params.shape == ProfileShape::none) {
    return table;
  }
  table.num_points = SupportTable::kMaxPoints;
  table.r_max = ProfileSupportRadius(params);
  table.dr = table.r_max / static_cast<Real>(table.num_points - 1);
  std::vector<Real> bphi(table.num_points, 0.0);
  std::vector<Real> rhs(table.num_points, 0.0);
  std::vector<Real> support(table.num_points, 0.0);

  for (int i = 0; i < table.num_points; ++i) {
    const Real r = table.dr * static_cast<Real>(i);
    Real profile = 0.0;
    Real dprofile_dr = 0.0;
    EvaluateRadialProfileAndDerivative(params, r, profile, dprofile_dr);
    bphi[i] = dprofile_dr;
  }
  for (int i = 1; i < table.num_points; ++i) {
    const Real db_dr =
        i == table.num_points - 1 ? (bphi[i] - bphi[i - 1]) / table.dr
                                  : (bphi[i + 1] - bphi[i - 1]) / (2.0 * table.dr);
    const Real r = table.dr * static_cast<Real>(i);
    rhs[i] = -(SQR(bphi[i]) / r + bphi[i] * db_dr);
  }
  support[table.num_points - 1] = 0.0;
  for (int i = table.num_points - 2; i >= 0; --i) {
    support[i] = support[i + 1] - 0.5 * (rhs[i] + rhs[i + 1]) * table.dr;
  }
  for (int i = 0; i < table.num_points; ++i) {
    table.values[i] = support[i];
  }
  return table;
}

} // namespace

void ProblemInitPackageData(ParameterInput *pin,
                            parthenon::StateDescriptor *hydro_pkg) {
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  PARTHENON_REQUIRE(fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd,
                    "pulsed_reconnection requires ctmhd or ucthlldmhd.");

  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  hydro_pkg->AddField("curlBx", m);
  hydro_pkg->AddField("curlBy", m);
  hydro_pkg->AddField("curlBz", m);
  hydro_pkg->AddField("divv", m);
  hydro_pkg->AddField("beta", m);
  hydro_pkg->AddField("eta", m);
  hydro_pkg->AddField("T", m);

  if (pin->GetString("refinement", "type") == "user") {
    const Real refine_tol =
        pin->GetOrAddReal("refinement", "current_refine_tol", 0.20);
    const Real derefine_tol =
        pin->GetOrAddReal("refinement", "current_derefine_tol", 0.08);
    PARTHENON_REQUIRE(refine_tol > 0.0,
                      "refinement/current_refine_tol must be positive.");
    PARTHENON_REQUIRE(derefine_tol >= 0.0,
                      "refinement/current_derefine_tol must be nonnegative.");
    PARTHENON_REQUIRE(
        derefine_tol < 0.5 * refine_tol,
        "refinement/current_derefine_tol must be less than half of "
        "refinement/current_refine_tol to avoid factor-two AMR level oscillation.");
    hydro_pkg->AddParam<Real>("refinement/current_refine_tol", refine_tol);
    hydro_pkg->AddParam<Real>("refinement/current_derefine_tol", derefine_tol);
  }
}

parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto &bface = mbd->Get("Bface").data;
  const auto b1f = bface.Get(IBF1, 0, 0, 0);
  const auto b2f = bface.Get(IBF2, 0, 0, 0);
  auto &coords = pmb->coords;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const Real refine_tol =
      hydro_pkg->Param<Real>("refinement/current_refine_tol");
  const Real derefine_tol =
      hydro_pkg->Param<Real>("refinement/current_derefine_tol");
  const Real magnetic_field_reference = g_source_params.amr_magnetic_field_reference;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real max_chi_j = 0.0;
  pmb->par_reduce(
      "pulsed_reconnection::CurrentRefinement", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &local_max) {
        const int im = i == ib.s ? i : i - 1;
        const int ip = i == ib.e ? i : i + 1;
        const int jm = j == jb.s ? j : j - 1;
        const int jp = j == jb.e ? j : j + 1;
        const Real by_im1 =
            0.5 * (b2f(k, j, im) + b2f(k, j + 1, im));
        const Real by_ip1 =
            0.5 * (b2f(k, j, ip) + b2f(k, j + 1, ip));
        const Real bx_jm1 =
            0.5 * (b1f(k, jm, i) + b1f(k, jm, i + 1));
        const Real bx_jp1 =
            0.5 * (b1f(k, jp, i) + b1f(k, jp, i + 1));
        const Real dBy_dx =
            (by_ip1 - by_im1) /
            (coords.Xc<1>(ip) - coords.Xc<1>(im));
        const Real dBx_dy =
            (bx_jp1 - bx_jm1) /
            (coords.Xc<2>(jp) - coords.Xc<2>(jm));
        const Real dx_eff =
            fmax(coords.Dxc<1>(k, j, i), coords.Dxc<2>(k, j, i));
        const Real chi_j = dx_eff * fabs(dBy_dx - dBx_dy) /
                           magnetic_field_reference;
        local_max = fmax(local_max, chi_j);
      },
      Kokkos::Max<Real>(max_chi_j));

  if (max_chi_j > refine_tol) return parthenon::AmrTag::refine;
  if (max_chi_j < derefine_tol) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/) {
  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &w = mbd->Get("prim").data;
  auto &bface = mbd->Get("Bface").data;
  const auto b1f = bface.Get(IBF1, 0, 0, 0);
  const auto b2f = bface.Get(IBF2, 0, 0, 0);
  const auto b3f = bface.Get(IBF3, 0, 0, 0);
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool has_fixed_resistivity =
      hydro_pkg->Param<Resistivity>("resistivity") == Resistivity::ohmic;
  const auto units = hydro_pkg->Param<Units>("units");
  const Real eta_cgs =
      has_fixed_resistivity
          ? pin->GetReal("diffusion", "ohm_diff_coeff_code") *
                SQR(units.code_length_cgs()) / units.code_time_cgs()
          : 0.0;

  auto &curlBx = mbd->Get("curlBx").data;
  auto &curlBy = mbd->Get("curlBy").data;
  auto &curlBz = mbd->Get("curlBz").data;
  auto &divv = mbd->Get("divv").data;
  auto &eta_field = mbd->Get("eta").data;
  auto &beta_field = mbd->Get("beta").data;
  auto &T_field = mbd->Get("T").data;
  const Real mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  const auto ndim = pmb->pmy_mesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  pmb->par_for(
      "pulsed_reconnection::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int im = i == ib.s ? i : i - 1;
        const int ip = i == ib.e ? i : i + 1;
        const int jm = j == jb.s ? j : j - 1;
        const int jp = j == jb.e ? j : j + 1;
        const int km = k == kb.s ? k : k - 1;
        const int kp = k == kb.e ? k : k + 1;
        const Real dBz_dy =
            ndim > 1 ? (CellCenteredB3(b3f, ndim, k, jp, i) -
                        CellCenteredB3(b3f, ndim, k, jm, i)) /
                           (coords.Xc<2>(jp) - coords.Xc<2>(jm))
                     : 0.0;
        const Real dBy_dz =
            ndim > 2 ? (CellCenteredB2(b2f, kp, j, i) -
                        CellCenteredB2(b2f, km, j, i)) /
                           (coords.Xc<3>(kp) - coords.Xc<3>(km))
                     : 0.0;
        const Real dBx_dz =
            ndim > 2 ? (CellCenteredB1(b1f, kp, j, i) -
                        CellCenteredB1(b1f, km, j, i)) /
                           (coords.Xc<3>(kp) - coords.Xc<3>(km))
                     : 0.0;
        const Real dBz_dx = (CellCenteredB3(b3f, ndim, k, j, ip) -
                             CellCenteredB3(b3f, ndim, k, j, im)) /
                             (coords.Xc<1>(ip) - coords.Xc<1>(im));
        const Real dBy_dx = (CellCenteredB2(b2f, k, j, ip) -
                             CellCenteredB2(b2f, k, j, im)) /
                             (coords.Xc<1>(ip) - coords.Xc<1>(im));
        const Real dBx_dy =
            ndim > 1 ? (CellCenteredB1(b1f, k, jp, i) -
                        CellCenteredB1(b1f, k, jm, i)) /
                           (coords.Xc<2>(jp) - coords.Xc<2>(jm))
                     : 0.0;
        curlBx(k, j, i) = dBz_dy - dBy_dz;
        curlBy(k, j, i) = dBx_dz - dBz_dx;
        curlBz(k, j, i) = dBy_dx - dBx_dy;

        const Real dvx_dx = (w(IV1, k, j, ip) - w(IV1, k, j, im)) /
                            (coords.Xc<1>(ip) - coords.Xc<1>(im));
        const Real dvy_dy =
            ndim > 1 ? (w(IV2, k, jp, i) - w(IV2, k, jm, i)) /
                           (coords.Xc<2>(jp) - coords.Xc<2>(jm))
                     : 0.0;
        const Real dvz_dz =
            ndim > 2 ? (w(IV3, kp, j, i) - w(IV3, km, j, i)) /
                           (coords.Xc<3>(kp) - coords.Xc<3>(km))
                     : 0.0;
        divv(k, j, i) = dvx_dx + dvy_dy + dvz_dz;

        const Real rho = u(IDN, k, j, i);
        const Real p = w(IPR, k, j, i);
        T_field(k, j, i) = mbar_over_kb * p / rho;
        const Real b_squared = SQR(CellCenteredB1(b1f, k, j, i)) +
                               SQR(CellCenteredB2(b2f, k, j, i)) +
                               SQR(CellCenteredB3(b3f, ndim, k, j, i));
        beta_field(k, j, i) = b_squared > 0.0 ? 2.0 * p / b_squared : 0.0;

        eta_field(k, j, i) = eta_cgs;
      });
}

void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  const auto params = g_source_params;
  if (!params.drive_enable || dt <= 0.0) {
    return;
  }

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const Real amplitude_old =
      DrivePotentialAmplitudeFromCurrent(params, DriveCurrentAtTime(params, tm.time));
  const Real amplitude_new = DrivePotentialAmplitudeFromCurrent(
      params, DriveCurrentAtTime(params, tm.time + dt));

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "pulsed_reconnection::DriveB1Faces",
      parthenon::DevExecSpace(), 0, bface_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &bface = bface_pack(b);
        const auto &coords = bface_pack.GetCoords(b);
        const Real x = coords.X<1, TE::NN>(k, j, i);
        const Real y0 = coords.X<2, TE::NN>(k, j, i);
        const Real y1 = coords.X<2, TE::NN>(k, j + 1, i);
        const Real delta_az0 = EvaluateDriveAz(params, x, y0, amplitude_new) -
                               EvaluateDriveAz(params, x, y0, amplitude_old);
        const Real delta_az1 = EvaluateDriveAz(params, x, y1, amplitude_new) -
                               EvaluateDriveAz(params, x, y1, amplitude_old);
        bface(TE::F1, 0, k, j, i) +=
            (delta_az1 - delta_az0) / coords.Dxc<2>(k, j, i);
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "pulsed_reconnection::DriveB2Faces",
      parthenon::DevExecSpace(), 0, bface_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &bface = bface_pack(b);
        const auto &coords = bface_pack.GetCoords(b);
        const Real y = coords.X<2, TE::NN>(k, j, i);
        const Real x0 = coords.X<1, TE::NN>(k, j, i);
        const Real x1 = coords.X<1, TE::NN>(k, j, i + 1);
        const Real delta_az0 = EvaluateDriveAz(params, x0, y, amplitude_new) -
                               EvaluateDriveAz(params, x0, y, amplitude_old);
        const Real delta_az1 = EvaluateDriveAz(params, x1, y, amplitude_new) -
                               EvaluateDriveAz(params, x1, y, amplitude_old);
        bface(TE::F2, 0, k, j, i) -=
            (delta_az1 - delta_az0) / coords.Dxc<1>(k, j, i);
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "pulsed_reconnection::DriveCellState",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &bface = bface_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real old_B1 = cons(IB1, k, j, i);
        const Real old_B2 = cons(IB2, k, j, i);
        const Real old_B3 = cons(IB3, k, j, i);
        const Real new_B1 = 0.5 * (bface(TE::F1, 0, k, j, i) +
                                   bface(TE::F1, 0, k, j, i + 1));
        const Real new_B2 = 0.5 * (bface(TE::F2, 0, k, j, i) +
                                   bface(TE::F2, 0, k, j + 1, i));
        const Real new_B3 = old_B3;
        cons(IB1, k, j, i) = new_B1;
        cons(IB2, k, j, i) = new_B2;
        cons(IB3, k, j, i) = new_B3;
        cons(IEN, k, j, i) +=
            0.5 * (SQR(new_B1) + SQR(new_B2) + SQR(new_B3) - SQR(old_B1) -
                   SQR(old_B2) - SQR(old_B3));

        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);

        const auto floors = EvaluateDriveFloorState(params, x, y);
        const Real magnetic_support =
            params.drive_force_balance
                ? fmax(0.0, EvaluateMagneticSupportSum(params, params.drive_support_table, x,
                                                       y, amplitude_new))
                : 0.0;
        if (floors.rho_floor <= 0.0 && floors.T_floor <= 0.0 &&
            magnetic_support <= 0.0) {
          return;
        }

        const Real rho_old = cons(IDN, k, j, i);
        const Real delta_rho = fmax(0.0, floors.rho_floor - rho_old);
        const Real rho_new = rho_old + delta_rho;
        cons(IDN, k, j, i) = rho_new;

        const Real momentum_sq = SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                 SQR(cons(IM3, k, j, i));
        const Real magnetic_energy =
            0.5 * (SQR(new_B1) + SQR(new_B2) + SQR(new_B3));
        const Real kinetic_energy =
            rho_new > 0.0 ? 0.5 * momentum_sq / rho_new : 0.0;
        const Real internal_energy =
            fmax(0.0, cons(IEN, k, j, i) - kinetic_energy - magnetic_energy);
        const Real pressure = params.gm1 * internal_energy;
        // The existing hydro support option already acts like a density/temperature
        // floor inside the driven layer. Extend that logic so the total gas pressure
        // also includes the analytic magnetic support required to counter the Lorentz
        // force from the field that was just injected during this source-term step.
        //
        // This is intentionally framed as a floor on the thermal pressure rather than
        // a direct overwrite. Cells that are already hotter or more pressurized keep
        // their excess internal energy, while under-supported cells are lifted to the
        // minimum pressure implied by:
        //   1. the requested temperature floor at the updated density, and
        //   2. the instantaneous analytic magnetic support profile.
        const Real thermal_pressure_floor =
            rho_new > 0.0 ? floors.T_floor * params.k_b * rho_new / params.m_bar : 0.0;
        const Real target_pressure = thermal_pressure_floor + magnetic_support;
        const Real delta_internal_energy =
            target_pressure > pressure ? (target_pressure - pressure) / params.gm1 : 0.0;
        cons(IEN, k, j, i) += delta_internal_energy;
      });
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &bface = mbd->Get("Bface").data;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  PARTHENON_REQUIRE(pmb->pmy_mesh->ndim >= 2,
                    "pulsed_reconnection requires at least two dimensions.");
  EnsureSourceParamsInitialized(hydro_pkg, pin);
  const auto params = g_source_params;

  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "gamma ================== " << pin->GetReal("hydro", "gamma")
              << std::endl;
    std::cout << "current_peak [MA] ====== " << params.current_peak_MA << std::endl;
    std::cout << "drive_enable =========== " << params.drive_enable << std::endl;
    std::cout << "drive_peak_current [MA]  " << params.drive_peak_current_MA << std::endl;
    std::cout << "drive_t_peak [ns] ====== " << params.drive_t_peak_ns << std::endl;
    std::cout << "rho_wire(core) [g/cm^3]= " << params.rho_wire_cgs << std::endl;
    std::cout << "rho_background [g/cm^3]= " << params.rho_background_cgs << std::endl;
    std::cout << "T_wire(core) [K] ======= " << params.T_wire << std::endl;
    std::cout << "T_background [K] ======= " << params.T_background << std::endl;
    std::cout << "v0(peak) [cm/s] ======== " << params.v0_cgs << std::endl;
    std::cout << "array_separation [cm] == " << params.array_separation_cgs << std::endl;
    std::cout << "initial thermo profile == "
              << ProfileShapeName(params.initial_thermal_profile.shape) << std::endl;
    std::cout << "initial magnetic profile "
              << ProfileShapeName(params.initial_magnetic_profile.shape) << std::endl;
    std::cout << "drive thermal profile == "
              << ProfileShapeName(params.drive_thermal_profile.shape) << std::endl;
    std::cout << "drive magnetic profile = "
              << ProfileShapeName(params.drive_magnetic_profile.shape) << std::endl;
    std::cout << "initial_force_balance == " << params.initial_force_balance
              << std::endl;
    std::cout << "drive_force_balance === " << params.drive_force_balance << std::endl;
    std::cout << "drive rho floor [g/cm^3] " << params.drive_rho_floor_cgs << std::endl;
    std::cout << "drive T floor [K] ====== " << params.drive_T_floor << std::endl;
    std::cout << "azimuthal mode N ======= " << params.azimuthal_mode_number
              << std::endl;
    std::cout << "dens. perturb. amplitude=" << params.density_perturb_amplitude
              << std::endl;
    std::cout << "temp perturb. amplitude =" << params.temperature_perturb_amplitude
              << std::endl;
    std::cout << "Converted code units:" << std::endl;
    std::cout << "matched |B|_peak [code] = " << params.peak_magnetic_field_strength
              << std::endl;
    std::cout << "initial mag amp [code] = " << params.initial_magnetic_profile_amplitude
              << std::endl;
    std::cout << "drive mag amp [code] === "
              << params.drive_peak_magnetic_profile_amplitude
              << std::endl;
    std::cout << "rho_wire(core) [code] == " << params.rho_wire << std::endl;
    std::cout << "rho_background [code] == " << params.rho_background << std::endl;
    std::cout << "v0(peak) [code] ======== " << params.v0 << std::endl;
    std::cout << "array_separation [code]  " << params.array_separation << std::endl;
    std::cout << "thermo width w [code] == " << params.initial_thermal_profile.width
              << std::endl;
    std::cout << "initial thermal core === "
              << params.initial_thermal_profile.tophat_core_width
              << std::endl;
    std::cout << "initial thermal falloff "
              << params.initial_thermal_profile.tophat_falloff_width << std::endl;
    std::cout << "magnetic width w_B [code]" << params.initial_magnetic_profile.width
              << std::endl;
    std::cout << "thermo perturbation ==== 1 + p*cos(N*theta)" << std::endl;
    std::cout << "velocity =============== "
              << (params.drive_enable ? "disabled in driven mode"
                                      : "normalized -grad(gaussian thermo profile)")
              << std::endl;
    std::cout << "magnetic field ========= "
              << "B = z_hat x grad(profile)"
              << std::endl;
    std::cout << "old |B| peak match at q = " << kOldAmpereLoopPeakQ << std::endl;
  }

  auto &coords = pmb->coords;
  auto u_host = u.GetHostMirrorAndCopy();
  auto bface_host = bface.GetHostMirrorAndCopy();
  auto b1f = bface_host.Get(IBF1, 0, 0, 0);
  auto b2f = bface_host.Get(IBF2, 0, 0, 0);
  auto b3f = bface_host.Get(IBF3, 0, 0, 0);

  // Athena++'s pbval->nblevel equivalent. Multiple finer neighbors can occupy the
  // same offset, but they are all at the same level on a properly nested mesh.
  const int level = pmb->loc.level();
  std::array<int, 27> neighbor_levels;
  neighbor_levels.fill(level);
  for (const auto &neighbor : pmb->GetNeighbors()) {
    const int ox1 = neighbor.offsets(X1DIR);
    const int ox2 = neighbor.offsets(X2DIR);
    const int ox3 = neighbor.offsets(X3DIR);
    const int idx = (ox1 + 1) + 3 * ((ox2 + 1) + 3 * (ox3 + 1));
    neighbor_levels[idx] = std::max(neighbor_levels[idx], neighbor.loc.level());
  }

  const bool two_d = pmb->pmy_mesh->ndim < 3;
  auto finer_neighbor = [&](const int ox1, const int ox2, const int ox3) {
    const int idx = (ox1 + 1) + 3 * ((ox2 + 1) + 3 * (ox3 + 1));
    return neighbor_levels[idx] > level;
  };

  // Return true when an edge lies on a coarse/fine interface. The neighbor offset
  // along the edge must be zero, while each nonzero transverse offset must coincide
  // with the corresponding lower or upper block boundary.
  auto edge_touches_finer_neighbor = [&](const int edge_dir, const int k, const int j,
                                         const int i) {
    if (two_d) return false;

    const std::array<int, 3> edge_idx{i, j, k};
    const std::array<int, 3> lower{ib.s, jb.s, kb.s};
    const std::array<int, 3> upper{ib.e + 1, jb.e + 1, kb.e + 1};
    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          const std::array<int, 3> offset{ox1, ox2, ox3};
          if (offset[edge_dir - 1] != 0) continue;
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          if (!finer_neighbor(ox1, ox2, ox3)) continue;

          bool touches = true;
          for (int dir = 0; dir < 3; ++dir) {
            if (dir == edge_dir - 1 || offset[dir] == 0) continue;
            if ((offset[dir] < 0 && edge_idx[dir] != lower[dir]) ||
                (offset[dir] > 0 && edge_idx[dir] != upper[dir])) {
              touches = false;
              break;
            }
          }
          if (touches) return true;
        }
      }
    }
    return false;
  };

  // Build the edge-centered vector potential first. On a coarse edge adjacent to a
  // finer neighbor, use the average over the two child-edge centers. The present
  // pulsed-reconnection potential is invariant along x3, so this average is
  // analytically identical, but the refinement-compatible discretization is explicit.
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "pulsed_reconnection::az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  const int kl = two_d ? kb.s : kb.s - 1;
  const int ku = two_d ? kb.e : kb.e + 1;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
      for (int i = ib.s - 1; i <= ib.e + 1; ++i) {
        const Real x = coords.X<1, TE::E3>(k, j, i);
        const Real y = coords.X<2, TE::E3>(k, j, i);
        const Real z = coords.X<3, TE::E3>(k, j, i);
        if (edge_touches_finer_neighbor(X3DIR, k, j, i)) {
          const Real quarter_dx3 = 0.25 * coords.Dxf<3>(k);
          az(k, j, i) =
              0.5 * (EvaluateInitialAz(params, x, y, z - quarter_dx3) +
                     EvaluateInitialAz(params, x, y, z + quarter_dx3));
        } else {
          az(k, j, i) = EvaluateInitialAz(params, x, y, z);
        }
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        const Real y0 = coords.Xf<2>(j);
        const Real y1 = coords.Xf<2>(j + 1);
        b1f(k, j, i) = (az(k, j + 1, i) - az(k, j, i)) / (y1 - y0);
      }
    }
  }
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real x0 = coords.Xf<1>(i);
        const Real x1 = coords.Xf<1>(i + 1);
        b2f(k, j, i) = -(az(k, j, i + 1) - az(k, j, i)) / (x1 - x0);
      }
    }
  }
  for (int k = kb.s; k <= kb.e + (pmb->pmy_mesh->ndim >= 3 ? 1 : 0); ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        b3f(k, j, i) = 0.0;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const auto state = EvaluateSourceState(params, coords.Xc<1>(i), coords.Xc<2>(j));
        const Real B1 = CellCenteredB1(b1f, k, j, i);
        const Real B2 = CellCenteredB2(b2f, k, j, i);
        const Real B3 = 0.0;
        u_host(IDN, k, j, i) = state.rho;
        u_host(IM1, k, j, i) = state.rho * state.v1;
        u_host(IM2, k, j, i) = state.rho * state.v2;
        u_host(IM3, k, j, i) = state.rho * state.v3;
        u_host(IB1, k, j, i) = B1;
        u_host(IB2, k, j, i) = B2;
        u_host(IB3, k, j, i) = B3;
        u_host(IEN, k, j, i) =
            state.pressure / params.gm1 +
            0.5 * (SQR(B1) + SQR(B2) + SQR(B3) +
                   state.rho *
                       (SQR(state.v1) + SQR(state.v2) + SQR(state.v3)));
      }
    }
  }
  bface.DeepCopy(bface_host);
  u.DeepCopy(u_host);
}

} // namespace pulsed_reconnection
