//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file pulsed_reconnection.cpp
//! \brief Problem generator for pulsed reconnection with independent density,
//! temperature, and magnetic radial profiles.
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
#include <utility>
#include <vector>

// AthenaPK headers
#include "../hydro/diffusion/diffusion.hpp"
#include "../main.hpp"
#include "../units.hpp"

namespace pulsed_reconnection {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

enum class ProfileShape { none, gaussian, tophat, cubic, wendland, quintic };
enum class TimeProfile { fixed, sin2 };

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

// Only the initial density and temperature profiles inherit azimuthal modulation.
KOKKOS_INLINE_FUNCTION
Real AzimuthalProfilePerturbation(const Real theta, const Real p, const int mode_number);

namespace {

Real PeakNormalizedDerivativeMagnitude(const RadialProfileParams &params);
Real ProfileSupportRadius(const RadialProfileParams &params);
SupportTable BuildUnitAmplitudeSupportTable(const RadialProfileParams &params,
                                            const std::string &label);

// All input-derived state used by initialization and the per-step source. Keeping it in
// one trivially-copyable object allows the same definitions to be captured by device
// kernels without repeatedly querying the package.
struct PulsedReconnectionParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real B_peak_gauss;
  bool drive_enable;
  Real drive_B_peak_gauss;
  Real drive_t_peak_ns;
  Real drive_t_peak;
  Real drive_rho_profile_floor_cgs;
  Real drive_rho_profile_floor;
  Real drive_T_profile_floor;
  Real rho_wire_cgs;
  Real rho_background_cgs;
  Real T_wire;
  Real T_background;
  Real v0_cgs;
  Real array_separation_cgs;
  int azimuthal_mode_number;
  Real density_perturb_amplitude;
  Real temperature_perturb_amplitude;
  Real rho_wire;
  Real rho_background;
  Real v0;
  Real array_separation;
  Real velocity_normalization;
  Real initial_peak_magnetic_field_strength;
  Real amr_magnetic_field_reference;
  Real initial_magnetic_profile_amplitude;
  Real drive_peak_magnetic_profile_amplitude;
  bool initial_force_balance;
  bool drive_force_balance;
  TimeProfile drive_rho_time_profile;
  TimeProfile drive_T_time_profile;
  RadialProfileParams initial_rho_profile;
  RadialProfileParams initial_T_profile;
  RadialProfileParams initial_magnetic_profile;
  RadialProfileParams drive_rho_profile;
  RadialProfileParams drive_T_profile;
  RadialProfileParams drive_magnetic_profile;
  SupportTable initial_support_table;
  SupportTable drive_support_table;
};

struct PulsedReconnectionState {
  Real rho;
  Real pressure;
  Real v1;
  Real v2;
  Real v3;
};

struct DriveSupportState {
  Real rho_target;
  Real T_floor;
};

// These flags are derived from explicit output-variable requests. A field that is not
// enrolled must never be retrieved by UserWorkBeforeOutput.
struct DiagnosticSelection {
  bool curlBx = false;
  bool curlBy = false;
  bool curlBz = false;
  bool divB = false;
  bool divv = false;
  bool beta = false;
  bool eta = false;
  bool T = false;

  bool Any() const {
    return curlBx || curlBy || curlBz || divB || divv || beta || eta || T;
  }
};

KOKKOS_INLINE_FUNCTION
Real EvaluateMagneticSupportSum(const PulsedReconnectionParams &params,
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

PulsedReconnectionParams g_source_params{};
bool g_source_params_initialized = false;

// Exact peak of |d exp[-(r/w)^2]/dr|, used only for the Gaussian kernel.
constexpr Real kGaussianGradientPeakFactor = 0.8577638849607067968; // sqrt(2 / e)
KOKKOS_INLINE_FUNCTION
Real PulseEnvelopeAtTime(const PulsedReconnectionParams &params, const Real time) {
  if (time <= 0.0 || params.drive_t_peak <= 0.0 || time >= 2.0 * params.drive_t_peak) {
    return 0.0;
  }
  const Real phase = M_PI * time / (2.0 * params.drive_t_peak);
  const Real s = sin(phase);
  return s * s;
}

KOKKOS_INLINE_FUNCTION
Real TimeProfileEnvelope(const PulsedReconnectionParams &params,
                         const TimeProfile profile, const Real time) {
  return profile == TimeProfile::fixed ? 1.0 : PulseEnvelopeAtTime(params, time);
}

KOKKOS_INLINE_FUNCTION
Real DrivePotentialAmplitudeAtTime(const PulsedReconnectionParams &params,
                                   const Real time) {
  return params.drive_peak_magnetic_profile_amplitude * PulseEnvelopeAtTime(params, time);
}

KOKKOS_INLINE_FUNCTION
Real EvaluateDriveAz(const PulsedReconnectionParams &params, const Real x, const Real y,
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
Real EvaluateInitialAz(const PulsedReconnectionParams &params, const Real x, const Real y,
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

// Density support and temperature support are intentionally independent: they may use
// different kernels, widths, and time envelopes.
KOKKOS_INLINE_FUNCTION
DriveSupportState EvaluateDriveSupportState(const PulsedReconnectionParams &params,
                                            const Real x, const Real y,
                                            const Real time) {
  DriveSupportState support{0.0, 0.0};
  const Real rho_envelope =
      TimeProfileEnvelope(params, params.drive_rho_time_profile, time);
  const Real T_envelope =
      TimeProfileEnvelope(params, params.drive_T_time_profile, time);
  const Real half_sep = 0.5 * params.array_separation;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    const Real r = sqrt(SQR(x) + SQR(y_local));
    support.rho_target += params.drive_rho_profile_floor * rho_envelope *
                          EvaluateRadialProfile(params.drive_rho_profile, r);
    support.T_floor += params.drive_T_profile_floor * T_envelope *
                       EvaluateRadialProfile(params.drive_T_profile, r);
  }
  return support;
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

TimeProfile ParseTimeProfile(const std::string &name, const std::string &input_name) {
  if (name == "fixed") return TimeProfile::fixed;
  if (name == "sin2") return TimeProfile::sin2;
  PARTHENON_FAIL("problem/pulsed_reconnection/" + input_name +
                 " must be either 'fixed' or 'sin2'.");
  return TimeProfile::sin2;
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

const char *TimeProfileName(const TimeProfile profile) {
  return profile == TimeProfile::fixed ? "fixed" : "sin2";
}

void RejectLegacyInputKeys(ParameterInput *pin) {
  const char *block = "problem/pulsed_reconnection";
  const std::array<std::pair<const char *, const char *>, 11> replacements{{
      {"w", "w_initial_rho and w_initial_T"},
      {"initial_thermal_profile", "initial_rho_profile and initial_T_profile"},
      {"initial_thermal_core_width",
       "initial_rho_core_width and initial_T_core_width"},
      {"initial_thermal_falloff_width",
       "initial_rho_falloff_width and initial_T_falloff_width"},
      {"w_drive", "w_drive_rho, w_drive_T, and w_drive_magnetic"},
      {"w_drive_thermal", "w_drive_rho and w_drive_T"},
      {"drive_thermal_profile", "drive_rho_profile and drive_T_profile"},
      {"drive_thermal_core_width", "drive_rho_core_width and drive_T_core_width"},
      {"drive_thermal_falloff_width",
       "drive_rho_falloff_width and drive_T_falloff_width"},
      {"drive_rho_floor", "drive_rho_profile_floor"},
      {"drive_T_floor", "drive_T_profile_floor"},
  }};
  for (const auto &[old_key, new_keys] : replacements) {
    if (pin->DoesParameterExist(block, old_key)) {
      PARTHENON_FAIL("problem/pulsed_reconnection/" + std::string(old_key) +
                     " has been replaced by " + new_keys + ".");
    }
  }
  for (const auto &key : std::vector<std::string>{"current_peak_MA",
                                                  "drive_peak_current_MA"}) {
    if (pin->DoesParameterExist(block, key)) {
      const std::string replacement =
          key == "current_peak_MA" ? "B_peak_gauss" : "drive_B_peak_gauss";
      PARTHENON_FAIL("problem/pulsed_reconnection/" + key +
                     " has been replaced by " + replacement + ".");
    }
  }
  for (const auto &key : std::vector<std::string>{"thermal_profile", "force_balance",
                                                  "drive_hydro_support_enable",
                                                  "drive_cutoff_radius_factor",
                                                  "core_width", "falloff_width"}) {
    if (pin->DoesParameterExist(block, key))
      PARTHENON_FAIL("problem/pulsed_reconnection/" + key +
                     " belongs to an unsupported legacy schema.");
  }
}

PulsedReconnectionParams
LoadSourceParams(const std::shared_ptr<StateDescriptor> &hydro_pkg, ParameterInput *pin) {
  RejectLegacyInputKeys(pin);
  PulsedReconnectionParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.B_peak_gauss =
      pin->GetOrAddReal("problem/pulsed_reconnection", "B_peak_gauss", 0.0);
  params.drive_enable =
      pin->GetOrAddBoolean("problem/pulsed_reconnection", "drive_enable", false);
  params.drive_B_peak_gauss = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_B_peak_gauss", params.B_peak_gauss);
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
  // Initial density and temperature are independent fields. Their widths are required
  // so no hidden length scale enters the physical initial condition.
  params.initial_rho_profile.width =
      pin->GetReal("problem/pulsed_reconnection", "w_initial_rho");
  params.initial_rho_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "initial_rho_profile", "wendland"),
                        false, "initial_rho_profile");
  params.initial_rho_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_rho_core_width",
      params.initial_rho_profile.width);
  params.initial_rho_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_rho_falloff_width",
      params.initial_rho_profile.width);
  params.initial_T_profile.width =
      pin->GetReal("problem/pulsed_reconnection", "w_initial_T");
  params.initial_T_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "initial_T_profile", "wendland"),
                        false, "initial_T_profile");
  params.initial_T_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_T_core_width",
      params.initial_T_profile.width);
  params.initial_T_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "initial_T_falloff_width",
      params.initial_T_profile.width);
  params.initial_magnetic_profile.width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_B",
      pin->GetOrAddReal("problem/pulsed_reconnection", "w_magnetic",
                        params.initial_rho_profile.width));
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
  params.drive_rho_profile.width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_drive_rho", params.initial_rho_profile.width);
  params.drive_rho_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "drive_rho_profile", "none"),
                        true, "drive_rho_profile");
  params.drive_rho_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_rho_core_width",
      params.drive_rho_profile.width);
  params.drive_rho_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_rho_falloff_width",
      params.drive_rho_profile.width);
  params.drive_T_profile.width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_drive_T", params.initial_T_profile.width);
  params.drive_T_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "drive_T_profile", "none"),
                        true, "drive_T_profile");
  params.drive_T_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_T_core_width", params.drive_T_profile.width);
  params.drive_T_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_T_falloff_width",
      params.drive_T_profile.width);
  params.drive_magnetic_profile.width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_drive_magnetic",
      params.initial_magnetic_profile.width);
  params.drive_magnetic_profile.shape =
      ParseProfileShape(pin->GetOrAddString("problem/pulsed_reconnection",
                                            "drive_magnetic_profile", "gaussian"),
                        true, "drive_magnetic_profile");
  params.drive_magnetic_profile.tophat_core_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_magnetic_core_width",
      params.drive_magnetic_profile.width);
  params.drive_magnetic_profile.tophat_falloff_width = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "drive_magnetic_falloff_width",
      params.drive_magnetic_profile.width);
  params.drive_force_balance = pin->GetOrAddBoolean(
      "problem/pulsed_reconnection", "drive_force_balance", false);
  params.drive_rho_time_profile = ParseTimeProfile(
      pin->GetOrAddString("problem/pulsed_reconnection", "drive_rho_time_profile",
                          "sin2"),
      "drive_rho_time_profile");
  params.drive_T_time_profile = ParseTimeProfile(
      pin->GetOrAddString("problem/pulsed_reconnection", "drive_T_time_profile", "sin2"),
      "drive_T_time_profile");
  if (params.drive_enable) {
    params.drive_rho_profile_floor_cgs =
        pin->GetReal("problem/pulsed_reconnection", "drive_rho_profile_floor");
    params.drive_T_profile_floor =
        pin->GetReal("problem/pulsed_reconnection", "drive_T_profile_floor");
  } else {
    params.drive_rho_profile_floor_cgs = 0.0;
    params.drive_T_profile_floor = 0.0;
  }
  params.azimuthal_mode_number =
      pin->GetOrAddInteger("problem/pulsed_reconnection", "N", 0);
  params.density_perturb_amplitude = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "density_perturb_amplitude", 0.0);
  params.temperature_perturb_amplitude = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "temperature_perturb_amplitude", 0.0);
  PARTHENON_REQUIRE(params.initial_rho_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_initial_rho must be positive.");
  PARTHENON_REQUIRE(
      params.initial_rho_profile.tophat_core_width > 0.0,
      "problem/pulsed_reconnection/initial_rho_core_width must be positive.");
  PARTHENON_REQUIRE(
      params.initial_rho_profile.tophat_falloff_width > 0.0,
      "problem/pulsed_reconnection/initial_rho_falloff_width must be "
      "positive.");
  PARTHENON_REQUIRE(params.initial_T_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_initial_T must be positive.");
  PARTHENON_REQUIRE(params.initial_T_profile.tophat_core_width > 0.0,
                    "problem/pulsed_reconnection/initial_T_core_width must be positive.");
  PARTHENON_REQUIRE(
      params.initial_T_profile.tophat_falloff_width > 0.0,
      "problem/pulsed_reconnection/initial_T_falloff_width must be positive.");
  PARTHENON_REQUIRE(params.initial_magnetic_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_B must be positive.");
  PARTHENON_REQUIRE(params.drive_rho_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_drive_rho must be positive.");
  PARTHENON_REQUIRE(params.drive_rho_profile.tophat_core_width > 0.0,
                    "problem/pulsed_reconnection/drive_rho_core_width must be positive.");
  PARTHENON_REQUIRE(
      params.drive_rho_profile.tophat_falloff_width > 0.0,
      "problem/pulsed_reconnection/drive_rho_falloff_width must be positive.");
  PARTHENON_REQUIRE(params.drive_T_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_drive_T must be positive.");
  PARTHENON_REQUIRE(params.drive_T_profile.tophat_core_width > 0.0,
                    "problem/pulsed_reconnection/drive_T_core_width must be positive.");
  PARTHENON_REQUIRE(params.drive_T_profile.tophat_falloff_width > 0.0,
                    "problem/pulsed_reconnection/drive_T_falloff_width must be positive.");
  PARTHENON_REQUIRE(params.drive_magnetic_profile.width > 0.0,
                    "problem/pulsed_reconnection/w_drive_magnetic must be positive.");
  PARTHENON_REQUIRE(params.array_separation_cgs > 0.0,
                    "problem/pulsed_reconnection/array_separation must be "
                    "positive.");
  PARTHENON_REQUIRE(params.B_peak_gauss >= 0.0,
                    "problem/pulsed_reconnection/B_peak_gauss must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_B_peak_gauss >= 0.0,
                    "problem/pulsed_reconnection/drive_B_peak_gauss must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_t_peak_ns > 0.0,
                    "problem/pulsed_reconnection/drive_t_peak_ns must be "
                    "positive.");
  PARTHENON_REQUIRE(params.drive_rho_profile_floor_cgs >= 0.0,
                    "problem/pulsed_reconnection/drive_rho_profile_floor must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_T_profile_floor >= 0.0,
                    "problem/pulsed_reconnection/drive_T_profile_floor must be "
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
  params.rho_wire = params.rho_wire_cgs * units.g_cm3();
  params.rho_background = params.rho_background_cgs * units.g_cm3();
  params.drive_rho_profile_floor =
      params.drive_rho_profile_floor_cgs * units.g_cm3();
  params.v0 = params.v0_cgs * units.cm_s();
  params.array_separation = params.array_separation_cgs * units.cm();
  params.initial_rho_profile.width *= units.cm();
  params.initial_rho_profile.tophat_core_width *= units.cm();
  params.initial_rho_profile.tophat_falloff_width *= units.cm();
  params.initial_T_profile.width *= units.cm();
  params.initial_T_profile.tophat_core_width *= units.cm();
  params.initial_T_profile.tophat_falloff_width *= units.cm();
  params.initial_magnetic_profile.width *= units.cm();
  params.initial_magnetic_profile.tophat_core_width *= units.cm();
  params.initial_magnetic_profile.tophat_falloff_width *= units.cm();
  params.drive_rho_profile.width *= units.cm();
  params.drive_rho_profile.tophat_core_width *= units.cm();
  params.drive_rho_profile.tophat_falloff_width *= units.cm();
  params.drive_T_profile.width *= units.cm();
  params.drive_T_profile.tophat_core_width *= units.cm();
  params.drive_T_profile.tophat_falloff_width *= units.cm();
  params.drive_magnetic_profile.width *= units.cm();
  params.drive_magnetic_profile.tophat_core_width *= units.cm();
  params.drive_magnetic_profile.tophat_falloff_width *= units.cm();
  // drive_t_peak_ns is a physical time. units.s() is the number of code-time
  // units per physical second, so multiply to convert seconds to code time.
  params.drive_t_peak = params.drive_t_peak_ns * 1.0e-9 * units.s();

  // A_z is the selected radial profile multiplied by an amplitude, so
  // |B_phi| = |dA_z/dr|. Normalize independently for the initial and driven
  // profiles so the requested values are their actual peak fields.
  params.initial_peak_magnetic_field_strength = params.B_peak_gauss * units.gauss();
  const Real initial_peak_grad =
      PeakNormalizedDerivativeMagnitude(params.initial_magnetic_profile);
  params.initial_magnetic_profile_amplitude =
      initial_peak_grad > 0.0
          ? params.initial_peak_magnetic_field_strength / initial_peak_grad
          : 0.0;
  const Real drive_peak_field = params.drive_B_peak_gauss * units.gauss();
  params.amr_magnetic_field_reference =
      fmax(params.initial_peak_magnetic_field_strength, drive_peak_field);
  if (pin->GetString("refinement", "type") == "user") {
    PARTHENON_REQUIRE(
        params.amr_magnetic_field_reference > 0.0,
        "Magnetic-field-based AMR for pulsed_reconnection requires a positive "
        "initial or drive peak magnetic-field strength.");
  }
  const Real drive_peak_grad =
      PeakNormalizedDerivativeMagnitude(params.drive_magnetic_profile);
  params.drive_peak_magnetic_profile_amplitude =
      drive_peak_grad > 0.0 ? drive_peak_field / drive_peak_grad : 0.0;
  // Normalize against the chosen rho kernel so v0 remains the peak speed for every
  // supported profile family rather than silently retaining a Gaussian velocity shape.
  const Real initial_rho_peak_grad =
      PeakNormalizedDerivativeMagnitude(params.initial_rho_profile);
  params.velocity_normalization =
      initial_rho_peak_grad > 0.0 ? params.v0 / initial_rho_peak_grad : 0.0;
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
PulsedReconnectionState EvaluateSourceState(const PulsedReconnectionParams &params,
                                            const Real x, const Real y) {
  PulsedReconnectionState state{};
  const Real d = params.array_separation / 2.0;
  Real T_profile_sum = 0.0;
  Real rho_profile_sum = 0.0;
  Real magnetic_support_sum = 0.0;

  for (int A = -1; A <= 1; A += 2) {
    const Real y_center = A * d;
    const Real y_local = y - y_center;
    const Real r2 = SQR(x) + SQR(y_local);
    const Real r = sqrt(r2);
    const Real theta = atan2(y_local, x);

    Real rho_profile = 0.0;
    Real drho_profile_dr = 0.0;
    EvaluateRadialProfileAndDerivative(params.initial_rho_profile, r, rho_profile,
                                       drho_profile_dr);
    const Real T_profile = EvaluateRadialProfile(params.initial_T_profile, r);
    const Real density_perturbation = AzimuthalProfilePerturbation(
        theta, params.density_perturb_amplitude, params.azimuthal_mode_number);
    const Real temperature_perturbation = AzimuthalProfilePerturbation(
        theta, params.temperature_perturb_amplitude, params.azimuthal_mode_number);
    T_profile_sum += T_profile * temperature_perturbation;
    rho_profile_sum += rho_profile * density_perturbation;
    magnetic_support_sum += EvaluateSupportTable(params.initial_support_table, r);

    if (r > 0.0) {
      const Real inv_r = 1.0 / r;
      const Real xhat = x * inv_r;
      const Real yhat = y_local * inv_r;

      const Real radial_velocity =
          params.drive_enable ? 0.0 : -params.velocity_normalization * drho_profile_dr;
      state.v1 += radial_velocity * xhat;
      state.v2 += radial_velocity * yhat;

    }
  }

  state.rho = params.rho_background + params.rho_wire * rho_profile_sum;
  const Real T = params.T_background + params.T_wire * T_profile_sum;
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
Real AzimuthalProfilePerturbation(const Real theta, const Real p, const int mode_number) {
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

DiagnosticSelection RequestedDiagnostics(ParameterInput *pin) {
  DiagnosticSelection requested;
  // Only an explicit appearance in an output block enrolls a problem diagnostic.
  // Empty variable lists intentionally do not imply all optional diagnostics.
  for (const auto &block : pin->GetBlockNamesWithPrefix("parthenon/output")) {
    if (!pin->DoesParameterExist(block, "variables")) continue;
    for (const auto &name : pin->GetVector<std::string>(block, "variables")) {
      if (name == "curlBx") requested.curlBx = true;
      if (name == "curlBy") requested.curlBy = true;
      if (name == "curlBz") requested.curlBz = true;
      if (name == "divB") requested.divB = true;
      if (name == "divv") requested.divv = true;
      if (name == "beta") requested.beta = true;
      if (name == "eta") requested.eta = true;
      if (name == "T") requested.T = true;
    }
  }
  return requested;
}

} // namespace

void ProblemInitPackageData(ParameterInput *pin,
                            parthenon::StateDescriptor *hydro_pkg) {
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  PARTHENON_REQUIRE(fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd,
                    "pulsed_reconnection requires ctmhd or ucthlldmhd.");

  const auto diagnostics = RequestedDiagnostics(pin);
  hydro_pkg->AddParam<DiagnosticSelection>("pulsed_reconnection/diagnostics", diagnostics);
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  if (diagnostics.curlBx) hydro_pkg->AddField("curlBx", m);
  if (diagnostics.curlBy) hydro_pkg->AddField("curlBy", m);
  if (diagnostics.curlBz) hydro_pkg->AddField("curlBz", m);
  if (diagnostics.divB) hydro_pkg->AddField("divB", m);
  if (diagnostics.divv) hydro_pkg->AddField("divv", m);
  if (diagnostics.beta) hydro_pkg->AddField("beta", m);
  if (diagnostics.eta) hydro_pkg->AddField("eta", m);
  if (diagnostics.T) hydro_pkg->AddField("T", m);

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

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput * /*pin*/,
                          const parthenon::SimTime & /*tm*/) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto diagnostics =
      hydro_pkg->Param<DiagnosticSelection>("pulsed_reconnection/diagnostics");
  if (!diagnostics.Any()) return;

  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  const int ndim = pmb->pmy_mesh->ndim;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  if (diagnostics.curlBx) {
    auto &out = mbd->Get("curlBx").data;
    auto &bface = mbd->Get("Bface").data;
    const auto b2f = bface.Get(IBF2, 0, 0, 0);
    const auto b3f = bface.Get(IBF3, 0, 0, 0);
    pmb->par_for("pulsed_reconnection::curlBx", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
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
                   out(k, j, i) = dBz_dy - dBy_dz;
                 });
  }

  if (diagnostics.curlBy) {
    auto &out = mbd->Get("curlBy").data;
    auto &bface = mbd->Get("Bface").data;
    const auto b1f = bface.Get(IBF1, 0, 0, 0);
    const auto b3f = bface.Get(IBF3, 0, 0, 0);
    pmb->par_for("pulsed_reconnection::curlBy", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   const int im = i == ib.s ? i : i - 1;
                   const int ip = i == ib.e ? i : i + 1;
                   const int km = k == kb.s ? k : k - 1;
                   const int kp = k == kb.e ? k : k + 1;
                   const Real dBx_dz =
                       ndim > 2 ? (CellCenteredB1(b1f, kp, j, i) -
                                   CellCenteredB1(b1f, km, j, i)) /
                                      (coords.Xc<3>(kp) - coords.Xc<3>(km))
                                : 0.0;
                   const Real dBz_dx =
                       (CellCenteredB3(b3f, ndim, k, j, ip) -
                        CellCenteredB3(b3f, ndim, k, j, im)) /
                       (coords.Xc<1>(ip) - coords.Xc<1>(im));
                   out(k, j, i) = dBx_dz - dBz_dx;
                 });
  }

  if (diagnostics.curlBz) {
    auto &out = mbd->Get("curlBz").data;
    auto &bface = mbd->Get("Bface").data;
    const auto b1f = bface.Get(IBF1, 0, 0, 0);
    const auto b2f = bface.Get(IBF2, 0, 0, 0);
    pmb->par_for("pulsed_reconnection::curlBz", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   const int im = i == ib.s ? i : i - 1;
                   const int ip = i == ib.e ? i : i + 1;
                   const int jm = j == jb.s ? j : j - 1;
                   const int jp = j == jb.e ? j : j + 1;
                   const Real dBy_dx =
                       (CellCenteredB2(b2f, k, j, ip) -
                        CellCenteredB2(b2f, k, j, im)) /
                       (coords.Xc<1>(ip) - coords.Xc<1>(im));
                   const Real dBx_dy =
                       ndim > 1 ? (CellCenteredB1(b1f, k, jp, i) -
                                   CellCenteredB1(b1f, k, jm, i)) /
                                      (coords.Xc<2>(jp) - coords.Xc<2>(jm))
                                : 0.0;
                   out(k, j, i) = dBy_dx - dBx_dy;
                 });
  }

  if (diagnostics.divB) {
    auto &out = mbd->Get("divB").data;
    auto &bface = mbd->Get("Bface").data;
    const auto b1f = bface.Get(IBF1, 0, 0, 0);
    const auto b2f = bface.Get(IBF2, 0, 0, 0);
    const auto b3f = bface.Get(IBF3, 0, 0, 0);
    pmb->par_for("pulsed_reconnection::divB", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   const Real dBx_dx =
                       (b1f(k, j, i + 1) - b1f(k, j, i)) / coords.Dxc<1>(k, j, i);
                   const Real dBy_dy =
                       ndim > 1 ? (b2f(k, j + 1, i) - b2f(k, j, i)) /
                                      coords.Dxc<2>(k, j, i)
                                : 0.0;
                   const Real dBz_dz =
                       ndim > 2 ? (b3f(k + 1, j, i) - b3f(k, j, i)) /
                                      coords.Dxc<3>(k, j, i)
                                : 0.0;
                   out(k, j, i) = dBx_dx + dBy_dy + dBz_dz;
                 });
  }

  if (diagnostics.divv) {
    auto &out = mbd->Get("divv").data;
    auto &w = mbd->Get("prim").data;
    pmb->par_for("pulsed_reconnection::divv", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   const int im = i == ib.s ? i : i - 1;
                   const int ip = i == ib.e ? i : i + 1;
                   const int jm = j == jb.s ? j : j - 1;
                   const int jp = j == jb.e ? j : j + 1;
                   const int km = k == kb.s ? k : k - 1;
                   const int kp = k == kb.e ? k : k + 1;
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
                   out(k, j, i) = dvx_dx + dvy_dy + dvz_dz;
                 });
  }

  if (diagnostics.T) {
    auto &out = mbd->Get("T").data;
    auto &u = mbd->Get("cons").data;
    auto &w = mbd->Get("prim").data;
    const Real mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
    pmb->par_for("pulsed_reconnection::T", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   out(k, j, i) = mbar_over_kb * w(IPR, k, j, i) / u(IDN, k, j, i);
                 });
  }

  if (diagnostics.beta) {
    auto &out = mbd->Get("beta").data;
    auto &w = mbd->Get("prim").data;
    auto &bface = mbd->Get("Bface").data;
    const auto b1f = bface.Get(IBF1, 0, 0, 0);
    const auto b2f = bface.Get(IBF2, 0, 0, 0);
    const auto b3f = bface.Get(IBF3, 0, 0, 0);
    pmb->par_for("pulsed_reconnection::beta", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   const Real b_squared = SQR(CellCenteredB1(b1f, k, j, i)) +
                                          SQR(CellCenteredB2(b2f, k, j, i)) +
                                          SQR(CellCenteredB3(b3f, ndim, k, j, i));
                   out(k, j, i) =
                       b_squared > 0.0 ? 2.0 * w(IPR, k, j, i) / b_squared : 0.0;
                 });
  }

  if (diagnostics.eta) {
    auto &out = mbd->Get("eta").data;
    auto &u = mbd->Get("cons").data;
    auto &w = mbd->Get("prim").data;
    const bool has_resistivity =
        hydro_pkg->Param<Resistivity>("resistivity") == Resistivity::ohmic;
    const auto ohm_diff =
        has_resistivity
            ? hydro_pkg->Param<OhmicDiffusivity>("ohm_diff")
            : OhmicDiffusivity(Resistivity::none, ResistivityCoeff::none, 0.0, 0.0,
                               0.0, 0.0, -1.0);
    const auto units = hydro_pkg->Param<Units>("units");
    const Real eta_code_to_cgs =
        SQR(units.code_length_cgs()) / units.code_time_cgs();
    pmb->par_for("pulsed_reconnection::eta", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                 KOKKOS_LAMBDA(const int k, const int j, const int i) {
                   out(k, j, i) = has_resistivity
                                      ? ohm_diff.Get(w(IPR, k, j, i), u(IDN, k, j, i)) *
                                            eta_code_to_cgs
                                      : 0.0;
                 });
  }
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
  const Real amplitude_old = DrivePotentialAmplitudeAtTime(params, tm.time);
  const Real amplitude_new = DrivePotentialAmplitudeAtTime(params, tm.time + dt);
  // The first-order source installs the end-of-step magnetic state, so rho and T
  // support use the same end-of-step time when evaluating their independent envelopes.
  const Real support_time = tm.time + dt;

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

        const auto support = EvaluateDriveSupportState(params, x, y, support_time);
        const Real magnetic_support =
            params.drive_force_balance
                ? fmax(0.0, EvaluateMagneticSupportSum(params, params.drive_support_table, x,
                                                       y, amplitude_new))
                : 0.0;
        if (support.rho_target <= 0.0 && support.T_floor <= 0.0 &&
            magnetic_support <= 0.0) {
          return;
        }

        const Real rho_old = cons(IDN, k, j, i);
        // The envelope modulates a one-sided target. On the falling side of a sin2
        // pulse, previously supplied mass remains and is transported by the equations.
        const Real delta_rho = fmax(0.0, support.rho_target - rho_old);
        const Real rho_new = rho_old + delta_rho;

        // Density support represents co-moving plasma, not stationary ballast. Add the
        // matching momentum so top-up changes density without changing the cell velocity.
        const Real v1_old = rho_old > 0.0 ? cons(IM1, k, j, i) / rho_old : 0.0;
        const Real v2_old = rho_old > 0.0 ? cons(IM2, k, j, i) / rho_old : 0.0;
        const Real v3_old = rho_old > 0.0 ? cons(IM3, k, j, i) / rho_old : 0.0;
        cons(IDN, k, j, i) = rho_new;
        cons(IM1, k, j, i) += delta_rho * v1_old;
        cons(IM2, k, j, i) += delta_rho * v2_old;
        cons(IM3, k, j, i) += delta_rho * v3_old;

        // Inject both the kinetic energy required by co-motion and the specific internal
        // energy corresponding to the instantaneous local T profile floor. This term is
        // separate from the minimum-pressure correction below.
        const Real injected_kinetic_energy =
            0.5 * delta_rho * (SQR(v1_old) + SQR(v2_old) + SQR(v3_old));
        const Real injected_internal_energy =
            delta_rho * params.k_b * support.T_floor / (params.m_bar * params.gm1);
        cons(IEN, k, j, i) += injected_kinetic_energy + injected_internal_energy;

        const Real momentum_sq = SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                 SQR(cons(IM3, k, j, i));
        const Real magnetic_energy =
            0.5 * (SQR(new_B1) + SQR(new_B2) + SQR(new_B3));
        const Real kinetic_energy =
            rho_new > 0.0 ? 0.5 * momentum_sq / rho_new : 0.0;
        const Real internal_energy =
            fmax(0.0, cons(IEN, k, j, i) - kinetic_energy - magnetic_energy);
        const Real pressure = params.gm1 * internal_energy;
        // Enforce, but never overwrite downward to, the sum of the independently
        // profiled temperature floor and instantaneous magnetic force-balance support.
        const Real thermal_pressure_floor =
            rho_new > 0.0 ? support.T_floor * params.k_b * rho_new / params.m_bar : 0.0;
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
    std::cout << "B_peak [gauss] ========= " << params.B_peak_gauss << std::endl;
    std::cout << "drive_enable =========== " << params.drive_enable << std::endl;
    std::cout << "drive_B_peak [gauss] === " << params.drive_B_peak_gauss << std::endl;
    std::cout << "drive_t_peak [ns] ====== " << params.drive_t_peak_ns << std::endl;
    std::cout << "rho_wire(core) [g/cm^3]= " << params.rho_wire_cgs << std::endl;
    std::cout << "rho_background [g/cm^3]= " << params.rho_background_cgs << std::endl;
    std::cout << "T_wire(core) [K] ======= " << params.T_wire << std::endl;
    std::cout << "T_background [K] ======= " << params.T_background << std::endl;
    std::cout << "v0(peak) [cm/s] ======== " << params.v0_cgs << std::endl;
    std::cout << "array_separation [cm] == " << params.array_separation_cgs << std::endl;
    std::cout << "initial rho profile ==== "
              << ProfileShapeName(params.initial_rho_profile.shape) << std::endl;
    std::cout << "initial T profile ====== "
              << ProfileShapeName(params.initial_T_profile.shape) << std::endl;
    std::cout << "initial magnetic profile "
              << ProfileShapeName(params.initial_magnetic_profile.shape) << std::endl;
    std::cout << "drive rho profile ====== "
              << ProfileShapeName(params.drive_rho_profile.shape) << std::endl;
    std::cout << "drive T profile ======== "
              << ProfileShapeName(params.drive_T_profile.shape) << std::endl;
    std::cout << "drive magnetic profile = "
              << ProfileShapeName(params.drive_magnetic_profile.shape) << std::endl;
    std::cout << "initial_force_balance == " << params.initial_force_balance
              << std::endl;
    std::cout << "drive_force_balance === " << params.drive_force_balance << std::endl;
    std::cout << "drive rho profile floor  " << params.drive_rho_profile_floor_cgs
              << " g/cm^3" << std::endl;
    std::cout << "drive T profile floor == " << params.drive_T_profile_floor << " K"
              << std::endl;
    std::cout << "drive rho time profile = "
              << TimeProfileName(params.drive_rho_time_profile) << std::endl;
    std::cout << "drive T time profile === "
              << TimeProfileName(params.drive_T_time_profile) << std::endl;
    std::cout << "azimuthal mode N ======= " << params.azimuthal_mode_number
              << std::endl;
    std::cout << "dens. perturb. amplitude=" << params.density_perturb_amplitude
              << std::endl;
    std::cout << "temp perturb. amplitude =" << params.temperature_perturb_amplitude
              << std::endl;
    std::cout << "Converted code units:" << std::endl;
    std::cout << "initial |B|_peak [code] = "
              << params.initial_peak_magnetic_field_strength
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
    std::cout << "initial rho width [code] " << params.initial_rho_profile.width
              << std::endl;
    std::cout << "initial T width [code] == " << params.initial_T_profile.width
              << std::endl;
    std::cout << "magnetic width w_B [code]" << params.initial_magnetic_profile.width
              << std::endl;
    std::cout << "rho/T perturbation ===== 1 + p*cos(N*theta)" << std::endl;
    std::cout << "velocity =============== "
              << (params.drive_enable ? "disabled in driven mode"
                                      : "normalized -grad(initial rho profile)")
              << std::endl;
    std::cout << "magnetic field ========= "
              << "B = z_hat x grad(profile), peak-normalized"
              << std::endl;
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
