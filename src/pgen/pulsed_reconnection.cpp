//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file pulsed_reconnection.cpp
//! \brief Problem generator for pulsed reconnection with independent density,
//! temperature, and magnetic radial profiles.
// Two arrays at (0, +/- array_separation/2) supply the initial plasma profiles.
// Their edge-centered A_z defines the face magnetic field through a discrete curl.
// The drive changes A_z with a sin^2 pulse and replenishes plasma to independent
// density/temperature targets; only injected annular mass gets the driven velocity.
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
#include <cmath>
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

namespace {

// Radial profiles and physical state ---------------------------------------------------
enum class ProfileShape {
  none, gaussian, tophat, annulus, confined_current, cubic, wendland, quintic
};
enum class TimeProfile { fixed, sin2, none };
enum class VelocityDriveMode { none, injected_mass };

struct RadialProfileParams {
  ProfileShape shape = ProfileShape::none;
  Real width = 0.0;
  Real tophat_core_width = 0.0;
  Real tophat_falloff_width = 0.0;
  Real annulus_inner_radius = 0.0;
  Real annulus_outer_radius = 0.0;
  Real annulus_transition_width = 0.0;
  Real conductor_radius = 0.0;
  Real conductor_transition_width = 0.0;
  Real wire_inner_radius = 0.0;
  Real wire_outer_radius = 0.0;
  Real wire_transition_width = 0.0;
  // Derived once on the host for the analytic, unit-peak confined-current potential.
  Real confined_scale = 0.0;
  Real confined_inner_integral = 0.0;
  Real confined_outer_integral = 0.0;
};

struct SupportTable {
  static constexpr int kMaxPoints = 2049;
  int num_points = 0;
  Real r_max = 0.0;
  Real dr = 0.0;
  Real values[kMaxPoints] = {};
};

// Runtime state in code units (temperatures in kelvin), shared by initialization and
// driving. Physical input values used only for reporting stay local to setup.
struct PulsedReconnectionParams {
  // Thermodynamics and array geometry
  Real gm1;
  Real k_b;
  Real m_bar;
  Real array_separation;

  // Initial plasma and field
  Real rho_background;
  Real T_background;
  Real rho_wire;
  Real T_wire;
  Real rho_inner_reservoir;
  Real T_inner_reservoir;
  Real inner_reservoir_radius;
  Real v0;
  Real velocity_normalization;
  int azimuthal_mode_number;
  Real density_perturb_amplitude;
  Real temperature_perturb_amplitude;
  bool initial_force_balance;
  Real initial_peak_magnetic_field_strength;
  Real initial_magnetic_profile_amplitude;
  RadialProfileParams initial_rho_profile;
  RadialProfileParams initial_T_profile;
  RadialProfileParams initial_magnetic_profile;
  SupportTable initial_support_table;

  // Pulsed magnetic drive and plasma replenishment
  bool drive_enable;
  Real drive_t_peak;
  Real drive_peak_magnetic_profile_amplitude;
  bool drive_force_balance;
  Real drive_rho_profile_floor;
  Real drive_T_profile_floor;
  TimeProfile drive_rho_time_profile;
  TimeProfile drive_T_time_profile;
  RadialProfileParams drive_rho_profile;
  RadialProfileParams drive_T_profile;
  RadialProfileParams drive_magnetic_profile;
  SupportTable drive_support_table;
  VelocityDriveMode drive_velocity_mode;
  Real drive_velocity_peak;
  Real drive_velocity_inner_radius;
  Real drive_velocity_outer_radius;
  Real drive_velocity_transition_width;
  TimeProfile drive_velocity_time_profile;

  // Inner reservoir support and refinement
  bool inner_reservoir_support_enabled;
  TimeProfile inner_reservoir_time_profile;
  Real drive_rho_inner_reservoir_floor;
  Real drive_T_inner_reservoir_floor;
  Real amr_magnetic_field_reference;
};

struct InitialState {
  Real rho;
  Real pressure;
  Real v1;
  Real v2;
  Real v3;
};

struct DriveSupportState {
  Real rho_target;
  Real rho_inner_target;
  Real T_floor;
  Real T_inner_floor;
  Real v1_target;
  Real v2_target;
  Real velocity_weight;
  Real inner_reservoir_weight;
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
  if (profile == TimeProfile::none) return 0.0;
  return profile == TimeProfile::fixed ? 1.0 : PulseEnvelopeAtTime(params, time);
}

KOKKOS_INLINE_FUNCTION
Real SmootherStep01(const Real x) {
  const Real clamped = fmin(1.0, fmax(0.0, x));
  return clamped * clamped * clamped *
         (clamped * (clamped * 6.0 - 15.0) + 10.0);
}

KOKKOS_INLINE_FUNCTION
Real EvaluateInnerReservoirProfile(const RadialProfileParams &params,
                                   const Real reservoir_radius, const Real r) {
  if (params.shape != ProfileShape::annulus || r >= reservoir_radius) {
    return 0.0;
  }
  const Real transition_start = reservoir_radius - params.annulus_transition_width;
  if (r <= transition_start) return 1.0;
  return 1.0 - SmootherStep01((r - transition_start) /
                              params.annulus_transition_width);
}

KOKKOS_INLINE_FUNCTION
Real EvaluateVelocityAnnulus(const PulsedReconnectionParams &params, const Real r) {
  if (params.drive_velocity_mode != VelocityDriveMode::injected_mass ||
      r <= params.drive_velocity_inner_radius ||
      r >= params.drive_velocity_outer_radius) {
    return 0.0;
  }
  if (r < params.drive_velocity_inner_radius +
              params.drive_velocity_transition_width) {
    return SmootherStep01((r - params.drive_velocity_inner_radius) /
                          params.drive_velocity_transition_width);
  }
  if (r > params.drive_velocity_outer_radius -
              params.drive_velocity_transition_width) {
    return SmootherStep01((params.drive_velocity_outer_radius - r) /
                          params.drive_velocity_transition_width);
  }
  return 1.0;
}

// Integral from 0 to q of w*S(t)/(a + b*t) dt, where b is +w or -w.
// The logarithmic primitive is evaluated as dimensionless moments. A convergent
// series avoids cancellation for narrow ramps / small q; no radial interpolation
// or quadrature is used at evaluation time.
KOKKOS_INLINE_FUNCTION
Real SmootherStepOverRadiusIntegral(const Real q, const Real a, const Real b,
                                   const Real w) {
  if (q <= 0.0) return 0.0;
  if (a == 0.0) {
    // Allowed conductor ramp starting at the axis: S(t)/t is nonsingular.
    return (w / b) * q * q * q *
           (10.0 / 3.0 - 15.0 * q / 4.0 + 6.0 * q * q / 5.0);
  }
  const Real z = b * q / a;
  Real moments[6] = {};
  if (fabs(z) < 0.5) {
    Real power = z;
    for (int k = 0; k < 64; ++k) {
      moments[3] += power / static_cast<Real>(k + 4);
      moments[4] += power / static_cast<Real>(k + 5);
      moments[5] += power / static_cast<Real>(k + 6);
      power *= -z;
      if (fabs(power) <= 1.0e-18 * fabs(z)) break;
    }
  } else {
    moments[0] = log1p(z);
    for (int n = 1; n <= 5; ++n) {
      moments[n] = 1.0 / static_cast<Real>(n) - moments[n - 1] / z;
    }
  }
  return (w / b) * q * q * q *
         (10.0 * moments[3] - 15.0 * q * moments[4] +
          6.0 * q * q * moments[5]);
}

KOKKOS_INLINE_FUNCTION
Real EvaluateConfinedCurrentPotential(const RadialProfileParams &params, const Real r) {
  const Real rc = params.conductor_radius;
  const Real wc = params.conductor_transition_width;
  const Real ro = params.wire_outer_radius;
  const Real wo = params.wire_transition_width;
  const Real a = rc - wc;
  const Real outer_start = ro - wo;
  if (r >= ro) return 0.0;
  if (r >= outer_start) {
    // Integrate backwards from the cutoff: 1-S(q) = S(1-q).
    return params.confined_scale *
           SmootherStepOverRadiusIntegral((ro - r) / wo, ro, -wo, wo);
  }
  const Real outer = params.confined_outer_integral;
  if (r >= rc) {
    return params.confined_scale * (outer + log1p((outer_start - r) / r));
  }
  const Real inner =
      r <= a ? 0.0 : SmootherStepOverRadiusIntegral((r - a) / wc, a, wc, wc);
  // A_z = integral_r^ro B_phi(s) ds; psi = -A_z for z-hat cross grad(psi).
  return params.confined_scale *
         (outer + log1p((outer_start - rc) / rc) +
          params.confined_inner_integral - inner);
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
    const Real exponent = -SQR(r / params.width);
    profile = exp(fmax(-700.0, exponent));
    dprofile_dr = (-2.0 * r / SQR(params.width)) * profile;
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
      dprofile_dr = (-30.0 * x2 + 60.0 * x3 - 30.0 * x4) / params.tophat_falloff_width;
    }
    return;
  }
  if (params.shape == ProfileShape::annulus) {
    const Real inner = params.annulus_inner_radius;
    const Real outer = params.annulus_outer_radius;
    const Real transition = params.annulus_transition_width;
    if (r <= inner || r >= outer) {
      profile = 0.0;
      dprofile_dr = 0.0;
    } else if (r < inner + transition) {
      const Real q = (r - inner) / transition;
      profile = SmootherStep01(q);
      dprofile_dr = 30.0 * q * q * SQR(1.0 - q) / transition;
    } else if (r <= outer - transition) {
      profile = 1.0;
      dprofile_dr = 0.0;
    } else {
      const Real q = (outer - r) / transition;
      profile = SmootherStep01(q);
      dprofile_dr = -30.0 * q * q * SQR(1.0 - q) / transition;
    }
    return;
  }
  if (params.shape == ProfileShape::confined_current) {
    // Confined-current A_z is evaluated analytically by the magnetic potential
    // routines, using the derived normalization in RadialProfileParams.
    profile = 0.0;
    dprofile_dr = 0.0;
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
    dprofile_dr = (-5.0 * pow(3.0 - q, 4) + 30.0 * pow(2.0 - q, 4) -
         75.0 * pow(1.0 - q, 4)) *
        inv_width / 66.0;
  } else if (q <= 2.0) {
    profile = (pow(3.0 - q, 5) - 6.0 * pow(2.0 - q, 5)) / 66.0;
    dprofile_dr = (-5.0 * pow(3.0 - q, 4) + 30.0 * pow(2.0 - q, 4)) * inv_width / 66.0;
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

// Only initial density and temperature receive azimuthal modulation.
KOKKOS_INLINE_FUNCTION
Real AzimuthalProfilePerturbation(const Real theta, const Real p, const int mode_number) {
  const Real phase = static_cast<Real>(mode_number) * theta;
  const Real cos_phase = cos(phase);
  return 1 + p * cos_phase;
}

Real ProfileSupportRadius(const RadialProfileParams &params) {
  switch (params.shape) {
  case ProfileShape::none:
    return 0.0;
  case ProfileShape::gaussian:
    return 6.0 * params.width;
  case ProfileShape::tophat:
    return params.tophat_core_width + params.tophat_falloff_width;
  case ProfileShape::annulus:
    return params.annulus_outer_radius;
  case ProfileShape::confined_current:
    return params.wire_outer_radius;
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

void InitializeConfinedCurrentPotential(RadialProfileParams &params) {
  if (params.shape != ProfileShape::confined_current) return;
  const Real rc = params.conductor_radius;
  const Real wc = params.conductor_transition_width;
  const Real a = rc - wc;
  // B_phi is proportional to S(q)/r in the conductor ramp and 1/r outside.
  // Its unique interior maximum satisfies r*S'(q) - wc*S(q) = 0.
  Real lo = 0.0;
  Real hi = 1.0;
  for (int n = 0; n < 64; ++n) {
    const Real q = 0.5 * (lo + hi);
    const Real derivative = 30.0 * q * q * SQR(1.0 - q);
    if ((a + wc * q) * derivative > wc * SmootherStep01(q)) {
      lo = q;
    } else {
      hi = q;
    }
  }
  const Real q_peak = 0.5 * (lo + hi);
  // Units of length: multiplying S(q)/r by this yields a unit peak field.
  params.confined_scale = (a + wc * q_peak) / SmootherStep01(q_peak);
  params.confined_inner_integral =
      SmootherStepOverRadiusIntegral(1.0, a, wc, wc);
  params.confined_outer_integral = SmootherStepOverRadiusIntegral(
      1.0, params.wire_outer_radius, -params.wire_transition_width,
      params.wire_transition_width);
}

SupportTable BuildUnitAmplitudeSupportTable(const RadialProfileParams &params) {
  SupportTable table;
  if (params.shape == ProfileShape::none) {
    return table;
  }
  table.num_points = SupportTable::kMaxPoints;
  table.r_max = ProfileSupportRadius(params);
  PARTHENON_REQUIRE(std::isfinite(table.r_max) && table.r_max != 0.0,
                    "Enabled force-balance support requires a finite, nonzero support "
                    "radius (used as a divisor in its pressure table).");
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

KOKKOS_INLINE_FUNCTION
Real EvaluateAz(const RadialProfileParams &profile, const Real amplitude,
                const Real array_separation, const Real x, const Real y) {
  if (profile.shape == ProfileShape::none || amplitude == 0.0) {
    return 0.0;
  }
  const Real half_sep = 0.5 * array_separation;
  Real az = 0.0;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    const Real r = sqrt(SQR(x) + SQR(y_local));
    az += amplitude *
          (profile.shape == ProfileShape::confined_current
               ? EvaluateConfinedCurrentPotential(profile, r)
               : EvaluateRadialProfile(profile, r));
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

// Density and temperature supports use independent radial profiles. Injected velocity
// is restricted to a smooth annulus around each array and applies only to replenished
// mass, leaving density top-up elsewhere co-moving with the existing reservoir.
KOKKOS_INLINE_FUNCTION
DriveSupportState EvaluateDriveSupportState(const PulsedReconnectionParams &params,
                                            const Real x, const Real y, const Real time) {
  DriveSupportState support{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const Real rho_envelope =
      TimeProfileEnvelope(params, params.drive_rho_time_profile, time);
  const Real T_envelope =
      TimeProfileEnvelope(params, params.drive_T_time_profile, time);
  const Real velocity_envelope =
      TimeProfileEnvelope(params, params.drive_velocity_time_profile, time);
  const Real inner_reservoir_envelope =
      params.inner_reservoir_support_enabled
          ? TimeProfileEnvelope(params, params.inner_reservoir_time_profile, time)
          : 0.0;
  const Real half_sep = 0.5 * params.array_separation;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    const Real r = sqrt(SQR(x) + SQR(y_local));
    support.rho_target += params.drive_rho_profile_floor * rho_envelope *
                          EvaluateRadialProfile(params.drive_rho_profile, r);
    support.T_floor += params.drive_T_profile_floor * T_envelope *
                       EvaluateRadialProfile(params.drive_T_profile, r);
    if (params.inner_reservoir_support_enabled) {
      const Real rho_inner_weight =
          EvaluateInnerReservoirProfile(params.initial_rho_profile,
                                        params.inner_reservoir_radius, r);
      const Real T_inner_weight =
          EvaluateInnerReservoirProfile(params.initial_T_profile,
                                        params.inner_reservoir_radius, r);
      support.rho_inner_target += params.drive_rho_inner_reservoir_floor *
                                  inner_reservoir_envelope * rho_inner_weight;
      support.T_inner_floor += params.drive_T_inner_reservoir_floor *
                               inner_reservoir_envelope * T_inner_weight;
      support.inner_reservoir_weight += fmax(rho_inner_weight, T_inner_weight);
    }
    if (r > 0.0) {
      const Real velocity_weight = EvaluateVelocityAnnulus(params, r);
      const Real radial_velocity =
          params.drive_velocity_peak * velocity_envelope * velocity_weight;
      support.v1_target += radial_velocity * x / r;
      support.v2_target += radial_velocity * y_local / r;
      support.velocity_weight += velocity_weight;
    }
  }
  return support;
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

// Input setup -------------------------------------------------------------------------
constexpr const char *kInputBlock = "problem/pulsed_reconnection";

// Advisory checks never modify the requested physical values.
void WarnUnless(const bool condition, const std::string &message) {
  if (!condition && parthenon::Globals::my_rank == 0) {
    PARTHENON_WARN(message);
  }
}

void ConvertProfileLengths(RadialProfileParams &profile, const Real cm) {
  profile.width *= cm;
  profile.tophat_core_width *= cm;
  profile.tophat_falloff_width *= cm;
  profile.annulus_inner_radius *= cm;
  profile.annulus_outer_radius *= cm;
  profile.annulus_transition_width *= cm;
  profile.conductor_radius *= cm;
  profile.conductor_transition_width *= cm;
  profile.wire_inner_radius *= cm;
  profile.wire_outer_radius *= cm;
  profile.wire_transition_width *= cm;
}

ProfileShape ParseProfileShape(const std::string &name, const std::string &input_name) {
  if (name == "none") return ProfileShape::none;
  if (name == "gaussian") return ProfileShape::gaussian;
  if (name == "tophat") return ProfileShape::tophat;
  if (name == "annulus") return ProfileShape::annulus;
  if (name == "confined_current") return ProfileShape::confined_current;
  if (name == "cubic") return ProfileShape::cubic;
  if (name == "wendland") return ProfileShape::wendland;
  if (name == "quintic") return ProfileShape::quintic;
  PARTHENON_FAIL("problem/pulsed_reconnection/" + input_name +
                 " must be 'none', 'gaussian', 'tophat', 'annulus', "
                 "'confined_current', 'cubic', 'wendland', or 'quintic'.");
}

TimeProfile ParseTimeProfile(const std::string &name, const std::string &input_name) {
  if (name == "none") return TimeProfile::none;
  if (name == "fixed") return TimeProfile::fixed;
  if (name == "sin2") return TimeProfile::sin2;
  PARTHENON_FAIL("problem/pulsed_reconnection/" + input_name +
                 " must be 'none', 'fixed', or 'sin2'.");
  return TimeProfile::sin2;
}

// Select the shape (and optional driving envelope) before reading its parameters.
// A disabled profile leaves all dependent inputs untouched, including malformed ones.
RadialProfileParams ReadProfile(
    ParameterInput *pin, const std::string &name, const std::string &width_key,
    const std::string &default_shape, const Real default_width,
    const bool require_width = false, TimeProfile *time_profile = nullptr) {
  RadialProfileParams profile;
  const auto shape = pin->GetOrAddString(kInputBlock, name + "_profile", default_shape);
  if (shape == "none") return profile;
  if (time_profile != nullptr) {
    *time_profile = ParseTimeProfile(
        pin->GetOrAddString(kInputBlock, name + "_time_profile", "sin2"),
        name + "_time_profile");
    if (*time_profile == TimeProfile::none) return RadialProfileParams{};
  }
  profile.shape = ParseProfileShape(shape, name + "_profile");

  const bool uses_width = profile.shape == ProfileShape::gaussian ||
                          profile.shape == ProfileShape::cubic ||
                          profile.shape == ProfileShape::wendland ||
                          profile.shape == ProfileShape::quintic;
  if (width_key == "w_B") {
    // Preserve the existing w_B precedence over the w_magnetic alias.
    profile.width = pin->GetOrAddReal(
        kInputBlock, "w_B", pin->GetOrAddReal(kInputBlock, "w_magnetic", default_width));
  } else {
    profile.width = require_width && uses_width
                        ? pin->GetReal(kInputBlock, width_key)
                        : pin->GetOrAddReal(kInputBlock, width_key, default_width);
  }
  if (profile.shape == ProfileShape::tophat) {
    profile.tophat_core_width =
        pin->GetOrAddReal(kInputBlock, name + "_core_width", profile.width);
    profile.tophat_falloff_width =
        pin->GetOrAddReal(kInputBlock, name + "_falloff_width", profile.width);
  } else if (profile.shape == ProfileShape::annulus) {
    profile.annulus_inner_radius =
        pin->GetOrAddReal(kInputBlock, name + "_inner_radius", 0.0);
    profile.annulus_outer_radius =
        pin->GetOrAddReal(kInputBlock, name + "_outer_radius", 0.0);
    profile.annulus_transition_width =
        pin->GetOrAddReal(kInputBlock, name + "_transition_width", 0.0);
  } else if (profile.shape == ProfileShape::confined_current &&
             (name == "initial_magnetic" || name == "drive_magnetic")) {
    profile.conductor_radius =
        pin->GetOrAddReal(kInputBlock, name + "_conductor_radius", 0.0);
    profile.conductor_transition_width = pin->GetOrAddReal(
        kInputBlock, name + "_conductor_transition_width", profile.conductor_radius);
    profile.wire_inner_radius =
        pin->GetOrAddReal(kInputBlock, name + "_wire_inner_radius", 0.0);
    profile.wire_outer_radius =
        pin->GetOrAddReal(kInputBlock, name + "_wire_outer_radius", 0.0);
    profile.wire_transition_width = pin->GetOrAddReal(
        kInputBlock, name + "_wire_transition_width",
        profile.wire_outer_radius - profile.wire_inner_radius);
  }
  return profile;
}

void CheckProfile(const RadialProfileParams &profile, const std::string &name,
                  const bool magnetic) {
  const std::string label = "problem/pulsed_reconnection/" + name;
  switch (profile.shape) {
  case ProfileShape::none:
    return;
  case ProfileShape::gaussian:
  case ProfileShape::cubic:
  case ProfileShape::wendland:
  case ProfileShape::quintic:
    PARTHENON_REQUIRE(std::isfinite(profile.width) && profile.width != 0.0,
                      label + " requires a finite, nonzero width (used as a divisor).");
    WarnUnless(profile.width > 0.0, label + " has a negative radial width.");
    return;
  case ProfileShape::tophat:
    WarnUnless(profile.tophat_core_width >= 0.0, label + " has a negative core width.");
    // The evaluator never enters the ramp when the falloff width is nonpositive.
    WarnUnless(profile.tophat_falloff_width > 0.0,
               label + " has no smooth falloff; a nonpositive width gives a step.");
    return;
  case ProfileShape::annulus:
    WarnUnless(profile.annulus_inner_radius >= 0.0,
               label + " has a negative inner radius.");
    WarnUnless(profile.annulus_outer_radius > profile.annulus_inner_radius,
               label + " has an empty or reversed annulus; its radial profile is zero.");
    WarnUnless(profile.annulus_transition_width > 0.0,
               label + " has no smooth transition; a nonpositive width gives a step.");
    WarnUnless(2.0 * profile.annulus_transition_width <=
                   profile.annulus_outer_radius - profile.annulus_inner_radius,
               label + " has overlapping transition ramps; the inner ramp takes "
                       "precedence where they overlap.");
    return;
  case ProfileShape::confined_current:
    if (!magnetic) {
      WarnUnless(false, label + " uses confined_current, whose density/temperature "
                                "evaluator is zero; only its magnetic potential is "
                                "implemented.");
      return;
    }
    // These bounds protect the analytic integrals' divisors and logarithm domains.
    PARTHENON_REQUIRE(
        std::isfinite(profile.conductor_radius) && profile.conductor_radius > 0.0 &&
            profile.conductor_transition_width > 0.0 &&
            profile.conductor_transition_width <= profile.conductor_radius,
        label + " requires a positive finite conductor radius and a transition in "
                "(0, conductor_radius] for its analytic potential.");
    PARTHENON_REQUIRE(
        std::isfinite(profile.wire_outer_radius) && profile.wire_outer_radius > 0.0 &&
            profile.wire_transition_width > 0.0 &&
            profile.wire_transition_width < profile.wire_outer_radius,
        label + " requires 0 < wire_transition_width < wire_outer_radius with a "
                "finite outer radius to keep its analytic logarithms finite.");
    WarnUnless(profile.wire_inner_radius > profile.conductor_radius,
               label + " places the wire inner radius inside the conductor.");
    WarnUnless(profile.wire_outer_radius > profile.wire_inner_radius,
               label + " has reversed wire radii.");
    WarnUnless(profile.wire_transition_width <=
                   profile.wire_outer_radius - profile.wire_inner_radius,
               label + " extends the wire transition inside the wire inner radius.");
    WarnUnless(profile.wire_outer_radius - profile.wire_transition_width >=
                   profile.conductor_radius,
               label + " has overlapping conductor and wire transitions.");
    return;
  }
}

VelocityDriveMode ParseVelocityDriveMode(const std::string &name) {
  if (name == "none") return VelocityDriveMode::none;
  if (name == "injected_mass") return VelocityDriveMode::injected_mass;
  PARTHENON_FAIL("problem/pulsed_reconnection/drive_velocity_mode must be either "
                 "'none' or 'injected_mass'.");
  return VelocityDriveMode::none;
}

const char *ProfileShapeName(const ProfileShape shape) {
  switch (shape) {
  case ProfileShape::none:
    return "none";
  case ProfileShape::gaussian:
    return "gaussian";
  case ProfileShape::tophat:
    return "tophat";
  case ProfileShape::annulus:
    return "annulus";
  case ProfileShape::confined_current:
    return "confined_current";
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
  if (profile == TimeProfile::none) return "none";
  return profile == TimeProfile::fixed ? "fixed" : "sin2";
}

void WarnLegacyInputKeys(ParameterInput *pin) {
  const char *block = "problem/pulsed_reconnection";
  const std::array<std::pair<const char *, const char *>, 15> replacements{{
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
      {"w_drive_velocity", "drive_velocity_inner_radius and drive_velocity_outer_radius"},
      {"drive_velocity_profile", "drive_velocity_mode=injected_mass"},
      {"drive_velocity_core_width", "drive_velocity_inner_radius"},
      {"drive_velocity_falloff_width",
       "drive_velocity_outer_radius and drive_velocity_transition_width"},
  }};
  for (const auto &[old_key, new_keys] : replacements) {
    if (pin->DoesParameterExist(block, old_key)) {
      WarnUnless(false, "problem/pulsed_reconnection/" + std::string(old_key) +
                     " has been replaced by " + new_keys + ". This legacy key is ignored.");
    }
  }
  for (const auto &key : std::vector<std::string>{"current_peak_MA",
                                                  "drive_peak_current_MA"}) {
    if (pin->DoesParameterExist(block, key)) {
      const std::string replacement =
          key == "current_peak_MA" ? "B_peak_gauss" : "drive_B_peak_gauss";
      WarnUnless(false, "problem/pulsed_reconnection/" + key +
                     " has been replaced by " + replacement + ". This legacy key is ignored.");
    }
  }
  for (const auto &key : std::vector<std::string>{"thermal_profile", "force_balance",
                                                  "drive_hydro_support_enable",
                                                  "drive_cutoff_radius_factor",
                                                  "core_width", "falloff_width"}) {
    if (pin->DoesParameterExist(block, key))
      WarnUnless(false, "problem/pulsed_reconnection/" + key +
                     " belongs to an unsupported legacy schema and is ignored.");
  }
}

PulsedReconnectionParams
LoadSourceParams(const std::shared_ptr<StateDescriptor> &hydro_pkg, ParameterInput *pin) {
  WarnLegacyInputKeys(pin);
  PulsedReconnectionParams params{};

  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.drive_enable = pin->GetOrAddBoolean(kInputBlock, "drive_enable", false);

  // Initial profiles may be independently disabled. Only width-based kernels need
  // a required width; annuli and confined-current profiles use their own radii.
  params.initial_rho_profile =
      ReadProfile(pin, "initial_rho", "w_initial_rho", "wendland", 0.0, true);
  params.initial_T_profile =
      ReadProfile(pin, "initial_T", "w_initial_T", "wendland", 0.0, true);
  params.initial_magnetic_profile = ReadProfile(
      pin, "initial_magnetic", "w_B", "gaussian", params.initial_rho_profile.width);
  const bool initial_rho_enabled = params.initial_rho_profile.shape != ProfileShape::none;
  const bool initial_T_enabled = params.initial_T_profile.shape != ProfileShape::none;
  const bool initial_B_enabled =
      params.initial_magnetic_profile.shape != ProfileShape::none;

  const Real rho_background_cgs = pin->GetOrAddReal(kInputBlock, "rho_background", 1e-6);
  params.T_background = pin->GetOrAddReal(kInputBlock, "T_background", 1e2);
  const Real array_separation_cgs = pin->GetOrAddReal(kInputBlock, "array_separation", 4.0);
  const Real rho_wire_cgs =
      initial_rho_enabled ? pin->GetOrAddReal(kInputBlock, "rho_wire", 1e-3) : 0.0;
  params.T_wire = initial_T_enabled ? pin->GetOrAddReal(kInputBlock, "T_wire", 1.1e4) : 0.0;
  const Real rho_inner_reservoir_cgs =
      initial_rho_enabled ? pin->GetOrAddReal(kInputBlock, "rho_inner_reservoir", 0.0) : 0.0;
  params.T_inner_reservoir =
      initial_T_enabled ? pin->GetOrAddReal(kInputBlock, "T_inner_reservoir", 0.0) : 0.0;
  const Real B_peak_gauss =
      initial_B_enabled ? pin->GetOrAddReal(kInputBlock, "B_peak_gauss", 0.0) : 0.0;
  params.initial_force_balance =
      initial_B_enabled && pin->GetOrAddBoolean(kInputBlock, "initial_force_balance", true);
  const Real v0_cgs =
      initial_rho_enabled && (!params.drive_enable ||
                             params.initial_rho_profile.shape == ProfileShape::annulus)
          ? pin->GetOrAddReal(kInputBlock, "v0", 1.0e6)
          : 0.0;
  params.azimuthal_mode_number = initial_rho_enabled || initial_T_enabled
                                    ? pin->GetOrAddInteger(kInputBlock, "N", 0)
                                    : 0;
  params.density_perturb_amplitude =
      initial_rho_enabled
          ? pin->GetOrAddReal(kInputBlock, "density_perturb_amplitude", 0.0)
          : 0.0;
  params.temperature_perturb_amplitude =
      initial_T_enabled
          ? pin->GetOrAddReal(kInputBlock, "temperature_perturb_amplitude", 0.0)
          : 0.0;

  Real drive_B_peak_gauss = 0.0;
  Real drive_velocity_peak_cgs = 0.0;
  Real drive_velocity_inner_radius_cgs = 0.0;
  Real drive_velocity_outer_radius_cgs = 0.0;
  Real drive_velocity_transition_width_cgs = 0.0;
  Real drive_t_peak_ns = 0.0;
  Real drive_rho_profile_floor_cgs = 0.0;
  Real drive_rho_inner_reservoir_floor_cgs = 0.0;
  params.drive_rho_time_profile = TimeProfile::none;
  params.drive_T_time_profile = TimeProfile::none;
  params.drive_velocity_time_profile = TimeProfile::none;
  params.inner_reservoir_time_profile = TimeProfile::none;
  if (params.drive_enable) {
    params.drive_rho_profile = ReadProfile(
        pin, "drive_rho", "w_drive_rho", "none", params.initial_rho_profile.width,
        false, &params.drive_rho_time_profile);
    if (params.drive_rho_profile.shape != ProfileShape::none) {
      drive_rho_profile_floor_cgs = pin->GetReal(kInputBlock, "drive_rho_profile_floor");
    }
    params.drive_T_profile = ReadProfile(
        pin, "drive_T", "w_drive_T", "none", params.initial_T_profile.width,
        false, &params.drive_T_time_profile);
    if (params.drive_T_profile.shape != ProfileShape::none) {
      params.drive_T_profile_floor = pin->GetReal(kInputBlock, "drive_T_profile_floor");
    }
    params.drive_magnetic_profile = ReadProfile(
        pin, "drive_magnetic", "w_drive_magnetic", "gaussian",
        params.initial_magnetic_profile.width);
    if (params.drive_magnetic_profile.shape != ProfileShape::none) {
      drive_B_peak_gauss = pin->GetOrAddReal(kInputBlock, "drive_B_peak_gauss", B_peak_gauss);
      params.drive_force_balance =
          pin->GetOrAddBoolean(kInputBlock, "drive_force_balance", false);
    }

    const auto velocity_mode =
        pin->GetOrAddString(kInputBlock, "drive_velocity_mode", "none");
    if (velocity_mode != "none") {
      params.drive_velocity_time_profile = ParseTimeProfile(
          pin->GetOrAddString(kInputBlock, "drive_velocity_time_profile", "sin2"),
          "drive_velocity_time_profile");
      if (params.drive_velocity_time_profile != TimeProfile::none) {
        params.drive_velocity_mode = ParseVelocityDriveMode(velocity_mode);
        drive_velocity_peak_cgs = pin->GetOrAddReal(kInputBlock, "drive_velocity_peak", 0.0);
        drive_velocity_inner_radius_cgs =
            pin->GetOrAddReal(kInputBlock, "drive_velocity_inner_radius", 0.0);
        drive_velocity_outer_radius_cgs =
            pin->GetOrAddReal(kInputBlock, "drive_velocity_outer_radius", 0.0);
        drive_velocity_transition_width_cgs =
            pin->GetOrAddReal(kInputBlock, "drive_velocity_transition_width", 0.0);
      }
    }

    const auto reservoir_support =
        pin->GetOrAddString(kInputBlock, "inner_reservoir_support", "none");
    if (params.initial_rho_profile.shape == ProfileShape::annulus ||
        params.initial_T_profile.shape == ProfileShape::annulus) {
      params.inner_reservoir_time_profile =
          ParseTimeProfile(reservoir_support, "inner_reservoir_support");
    } else {
      WarnUnless(reservoir_support == "none",
                 "No initial annulus supplies a reservoir mask; inner reservoir "
                 "support is inactive and its parameters are ignored.");
    }
    params.inner_reservoir_support_enabled =
        params.inner_reservoir_time_profile != TimeProfile::none;
    if (params.inner_reservoir_support_enabled) {
      // Reservoir masks come from the initial profiles, independently for rho and T.
      if (params.initial_rho_profile.shape == ProfileShape::annulus) {
        drive_rho_inner_reservoir_floor_cgs = pin->GetOrAddReal(
            kInputBlock, "drive_rho_inner_reservoir_floor", rho_inner_reservoir_cgs);
      }
      if (params.initial_T_profile.shape == ProfileShape::annulus) {
        params.drive_T_inner_reservoir_floor = pin->GetOrAddReal(
            kInputBlock, "drive_T_inner_reservoir_floor", params.T_inner_reservoir);
      }
    }

    const bool uses_pulse = params.drive_magnetic_profile.shape != ProfileShape::none ||
                            params.drive_rho_time_profile == TimeProfile::sin2 ||
                            params.drive_T_time_profile == TimeProfile::sin2 ||
                            params.drive_velocity_time_profile == TimeProfile::sin2 ||
                            params.inner_reservoir_time_profile == TimeProfile::sin2;
    if (uses_pulse) {
      drive_t_peak_ns = pin->GetOrAddReal(kInputBlock, "drive_t_peak_ns", 500.0);
      WarnUnless(drive_t_peak_ns > 0.0,
                 "A nonpositive drive_t_peak_ns makes the sin2 pulse identically zero.");
    }
  }
  const bool reservoir_enabled = rho_inner_reservoir_cgs != 0.0 ||
                                 params.T_inner_reservoir != 0.0 ||
                                 params.inner_reservoir_support_enabled;
  const Real inner_reservoir_radius_cgs =
      reservoir_enabled
          ? pin->GetOrAddReal(kInputBlock, "inner_reservoir_radius",
                             params.initial_rho_profile.annulus_inner_radius)
          : 0.0;

  CheckProfile(params.initial_rho_profile, "initial_rho", false);
  CheckProfile(params.initial_T_profile, "initial_T", false);
  CheckProfile(params.initial_magnetic_profile, "initial_magnetic", true);
  CheckProfile(params.drive_rho_profile, "drive_rho", false);
  CheckProfile(params.drive_T_profile, "drive_T", false);
  CheckProfile(params.drive_magnetic_profile, "drive_magnetic", true);
  if (reservoir_enabled) {
    WarnUnless(inner_reservoir_radius_cgs >= 0.0,
               "A negative inner_reservoir_radius produces no reservoir mask.");
    // Overlap is permitted: the existing initialization/source sums both profiles.
    const auto warn_reservoir = [&](const RadialProfileParams &profile,
                                    const std::string &name, const bool enabled) {
      if (!enabled || profile.shape == ProfileShape::none) return;
      WarnUnless(profile.shape == ProfileShape::annulus,
                 name + " has no annular reservoir mask; its reservoir contribution "
                        "will be zero.");
      if (profile.shape != ProfileShape::annulus) return;
      WarnUnless(inner_reservoir_radius_cgs >= profile.annulus_transition_width,
                 "inner_reservoir_radius is smaller than the " + name +
                     " transition width; the reservoir has no flat central region.");
      WarnUnless(inner_reservoir_radius_cgs <= profile.annulus_inner_radius,
                 "The inner reservoir overlaps " + name + "; their contributions add.");
    };
    warn_reservoir(params.initial_rho_profile, "initial_rho",
                   rho_inner_reservoir_cgs != 0.0 || params.inner_reservoir_support_enabled);
    warn_reservoir(params.initial_T_profile, "initial_T",
                   params.T_inner_reservoir != 0.0 || params.inner_reservoir_support_enabled);
    WarnUnless(rho_inner_reservoir_cgs >= 0.0 && params.T_inner_reservoir >= 0.0,
               "Negative initial reservoir density/temperature subtracts from the "
               "background and may produce an unphysical state.");
  }
  if (params.inner_reservoir_support_enabled) {
    WarnUnless(drive_rho_inner_reservoir_floor_cgs >= 0.0 &&
                   params.drive_T_inner_reservoir_floor >= 0.0,
               "Negative inner reservoir floors may provide no replenishment.");
  }
  if (params.drive_velocity_mode != VelocityDriveMode::none) {
    WarnUnless(drive_velocity_peak_cgs > 0.0,
               "A nonpositive drive_velocity_peak gives zero or inward injected flow.");
    WarnUnless(drive_velocity_inner_radius_cgs >= 0.0,
               "The driven velocity annulus has a negative inner radius.");
    WarnUnless(drive_velocity_outer_radius_cgs > drive_velocity_inner_radius_cgs,
               "The driven velocity annulus is empty or reversed; its velocity is zero.");
    WarnUnless(drive_velocity_transition_width_cgs > 0.0,
               "A nonpositive drive_velocity_transition_width gives a sharp annulus.");
    WarnUnless(2.0 * drive_velocity_transition_width_cgs <=
                   drive_velocity_outer_radius_cgs - drive_velocity_inner_radius_cgs,
               "The velocity transition ramps overlap; the inner ramp takes precedence.");
  }
  WarnUnless(array_separation_cgs > 0.0,
             "Nonpositive array_separation coincides or exchanges the two array centers.");
  WarnUnless(B_peak_gauss >= 0.0 && drive_B_peak_gauss >= 0.0,
             "Negative magnetic peak amplitudes reverse the corresponding field.");
  WarnUnless(drive_rho_profile_floor_cgs >= 0.0 && params.drive_T_profile_floor >= 0.0,
             "Negative drive density/temperature floors may provide no replenishment.");
  WarnUnless(params.azimuthal_mode_number >= 0,
             "Negative N reverses the azimuthal phase; the cosine perturbation is unchanged.");
  WarnUnless(!(params.initial_force_balance &&
               params.initial_magnetic_profile.shape == ProfileShape::confined_current) &&
                 !(params.drive_force_balance &&
                   params.drive_magnetic_profile.shape == ProfileShape::confined_current),
             "Force-balance support is not implemented for confined_current; that "
             "profile contributes no magnetic pressure support.");

  PARTHENON_REQUIRE(hydro_pkg->AllParams().hasKey("units") &&
                        hydro_pkg->AllParams().hasKey("mbar") &&
                        hydro_pkg->AllParams().hasKey("mbar_over_kb"),
                    "pulsed_reconnection requires a <units> block and "
                    "hydro/He_mass_fraction.");
  // Convert once; kernels use only code units and temperatures in kelvin.
  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar = hydro_pkg->Param<Real>("mbar");
  params.rho_wire = rho_wire_cgs * units.g_cm3();
  params.rho_inner_reservoir = rho_inner_reservoir_cgs * units.g_cm3();
  params.inner_reservoir_radius = inner_reservoir_radius_cgs * units.cm();
  params.drive_rho_inner_reservoir_floor =
      drive_rho_inner_reservoir_floor_cgs * units.g_cm3();
  params.rho_background = rho_background_cgs * units.g_cm3();
  params.drive_rho_profile_floor = drive_rho_profile_floor_cgs * units.g_cm3();
  params.v0 = v0_cgs * units.cm_s();
  params.drive_velocity_peak = drive_velocity_peak_cgs * units.cm_s();
  params.drive_velocity_inner_radius = drive_velocity_inner_radius_cgs * units.cm();
  params.drive_velocity_outer_radius = drive_velocity_outer_radius_cgs * units.cm();
  params.drive_velocity_transition_width =
      drive_velocity_transition_width_cgs * units.cm();
  params.array_separation = array_separation_cgs * units.cm();
  ConvertProfileLengths(params.initial_rho_profile, units.cm());
  ConvertProfileLengths(params.initial_T_profile, units.cm());
  ConvertProfileLengths(params.initial_magnetic_profile, units.cm());
  ConvertProfileLengths(params.drive_rho_profile, units.cm());
  ConvertProfileLengths(params.drive_T_profile, units.cm());
  ConvertProfileLengths(params.drive_magnetic_profile, units.cm());
  // drive_t_peak_ns is a physical time. units.s() is the number of code-time
  // units per physical second, so multiply to convert seconds to code time.
  params.drive_t_peak = drive_t_peak_ns * 1.0e-9 * units.s();

  // A_z is the selected radial profile multiplied by an amplitude, so
  // |B_phi| = |dA_z/dr|. Normalize independently for the initial and driven
  // profiles so the requested values are their actual peak fields.
  params.initial_peak_magnetic_field_strength = B_peak_gauss * units.gauss();
  if (params.initial_magnetic_profile.shape == ProfileShape::confined_current) {
    // The analytic potential is normalized so max(|dA_z/dr|) = 1.
    params.initial_magnetic_profile_amplitude =
        params.initial_peak_magnetic_field_strength;
  } else {
    const Real initial_peak_grad =
        PeakNormalizedDerivativeMagnitude(params.initial_magnetic_profile);
    params.initial_magnetic_profile_amplitude =
        initial_peak_grad > 0.0
            ? params.initial_peak_magnetic_field_strength / initial_peak_grad
            : 0.0;
  }
  const Real drive_peak_field = drive_B_peak_gauss * units.gauss();
  params.amr_magnetic_field_reference =
      fmax(fabs(params.initial_peak_magnetic_field_strength), fabs(drive_peak_field));
  if (pin->GetString("refinement", "type") == "user") {
    WarnUnless(params.amr_magnetic_field_reference > 0.0,
               "No active magnetic peak supplies an AMR reference; current-based "
               "refinement will leave block levels unchanged.");
  }
  if (params.drive_magnetic_profile.shape == ProfileShape::confined_current) {
    params.drive_peak_magnetic_profile_amplitude = drive_peak_field;
  } else {
    const Real drive_peak_grad =
        PeakNormalizedDerivativeMagnitude(params.drive_magnetic_profile);
    params.drive_peak_magnetic_profile_amplitude =
        drive_peak_grad > 0.0 ? drive_peak_field / drive_peak_grad : 0.0;
  }
  // Normalize against the chosen rho kernel so v0 remains the peak speed for every
  // supported profile family rather than silently retaining a Gaussian velocity shape.
  const Real initial_rho_peak_grad =
      PeakNormalizedDerivativeMagnitude(params.initial_rho_profile);
  params.velocity_normalization =
      initial_rho_peak_grad > 0.0 ? params.v0 / initial_rho_peak_grad : 0.0;
  if (params.initial_magnetic_profile.shape == ProfileShape::confined_current) {
    InitializeConfinedCurrentPotential(params.initial_magnetic_profile);
  } else if (params.initial_force_balance) {
    params.initial_support_table = BuildUnitAmplitudeSupportTable(
        params.initial_magnetic_profile);
  }
  if (params.drive_magnetic_profile.shape == ProfileShape::confined_current) {
    InitializeConfinedCurrentPotential(params.drive_magnetic_profile);
  } else if (params.drive_force_balance) {
    params.drive_support_table =
        BuildUnitAmplitudeSupportTable(params.drive_magnetic_profile);
  }

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "========================================" << '\n'
              << "Input parameters:" << '\n'
              << "gamma ================== " << pin->GetReal("hydro", "gamma") << '\n'
              << "B_peak [gauss] ========= " << B_peak_gauss << '\n'
              << "drive_enable =========== " << params.drive_enable << '\n'
              << "drive_B_peak [gauss] === " << drive_B_peak_gauss << '\n'
              << "drive velocity [cm/s] == " << drive_velocity_peak_cgs << '\n'
              << "drive velocity mode ==== "
              << (params.drive_velocity_mode == VelocityDriveMode::injected_mass
                      ? "injected_mass"
                      : "none") << '\n'
              << "drive velocity annulus = ["
              << drive_velocity_inner_radius_cgs << ", "
              << drive_velocity_outer_radius_cgs << "] cm, transition "
              << drive_velocity_transition_width_cgs << " cm" << '\n'
              << "drive_t_peak [ns] ====== " << drive_t_peak_ns << '\n'
              << "rho_wire(core) [g/cm^3]= " << rho_wire_cgs << '\n'
              << "rho inner reservoir ==== " << rho_inner_reservoir_cgs
              << " g/cm^3" << '\n'
              << "inner reservoir radius = " << inner_reservoir_radius_cgs
              << " cm" << '\n'
              << "rho_background [g/cm^3]= " << rho_background_cgs << '\n'
              << "T_wire(core) [K] ======= " << params.T_wire << '\n'
              << "T inner reservoir [K] == " << params.T_inner_reservoir << '\n'
              << "inner reservoir support  "
              << (params.inner_reservoir_support_enabled
                      ? TimeProfileName(params.inner_reservoir_time_profile)
                      : "none") << '\n'
              << "inner rho support floor  "
              << drive_rho_inner_reservoir_floor_cgs << " g/cm^3" << '\n'
              << "inner T support floor == "
              << params.drive_T_inner_reservoir_floor << " K" << '\n'
              << "T_background [K] ======= " << params.T_background << '\n'
              << "v0(peak) [cm/s] ======== " << v0_cgs << '\n'
              << "array_separation [cm] == " << array_separation_cgs << '\n'
              << "initial rho profile ==== "
              << ProfileShapeName(params.initial_rho_profile.shape) << '\n'
              << "initial T profile ====== "
              << ProfileShapeName(params.initial_T_profile.shape) << '\n'
              << "initial magnetic profile "
              << ProfileShapeName(params.initial_magnetic_profile.shape) << '\n'
              << "drive rho profile ====== "
              << ProfileShapeName(params.drive_rho_profile.shape) << '\n'
              << "drive T profile ======== "
              << ProfileShapeName(params.drive_T_profile.shape) << '\n'
              << "drive magnetic profile = "
              << ProfileShapeName(params.drive_magnetic_profile.shape) << '\n';
    if (params.drive_magnetic_profile.shape == ProfileShape::confined_current) {
      std::cout << "drive magnetic potential analytic; B_peak is the profile maximum"
                << '\n';
      std::cout << "drive B(rc)/B_peak ===== "
                << params.drive_magnetic_profile.confined_scale /
                       params.drive_magnetic_profile.conductor_radius << '\n';
      std::cout << "drive current radii [code] "
                << params.drive_magnetic_profile.conductor_radius << ", "
                << params.drive_magnetic_profile.wire_inner_radius << ", "
                << params.drive_magnetic_profile.wire_outer_radius
                << " (conductor, wire inner, wire outer)" << '\n';
      std::cout << "drive current transitions "
                << params.drive_magnetic_profile.conductor_transition_width << ", "
                << params.drive_magnetic_profile.wire_transition_width
                << " (conductor, wire)" << '\n';
    }
    std::cout << "initial_force_balance == " << params.initial_force_balance << '\n'
              << "drive_force_balance === " << params.drive_force_balance << '\n'
              << "drive rho profile floor  " << drive_rho_profile_floor_cgs
              << " g/cm^3" << '\n'
              << "drive T profile floor == " << params.drive_T_profile_floor << " K"
              << '\n'
              << "drive rho time profile = "
              << TimeProfileName(params.drive_rho_time_profile) << '\n'
              << "drive T time profile === "
              << TimeProfileName(params.drive_T_time_profile) << '\n'
              << "drive velocity time ==== "
              << TimeProfileName(params.drive_velocity_time_profile) << '\n'
              << "azimuthal mode N ======= " << params.azimuthal_mode_number << '\n'
              << "dens. perturb. amplitude=" << params.density_perturb_amplitude << '\n'
              << "temp perturb. amplitude =" << params.temperature_perturb_amplitude
              << '\n'
              << "Converted code units:" << '\n'
              << "initial |B|_peak [code] = "
              << params.initial_peak_magnetic_field_strength << '\n'
              << "initial mag amp [code] = " << params.initial_magnetic_profile_amplitude
              << '\n'
              << "drive mag amp [code] === "
              << params.drive_peak_magnetic_profile_amplitude << '\n'
              << "rho_wire(core) [code] == " << params.rho_wire << '\n'
              << "rho inner res. [code] == " << params.rho_inner_reservoir << '\n'
              << "rho_background [code] == " << params.rho_background << '\n'
              << "v0(peak) [code] ======== " << params.v0 << '\n'
              << "drive velocity [code] == " << params.drive_velocity_peak << '\n'
              << "array_separation [code]  " << params.array_separation << '\n'
              << "initial rho width [code] " << params.initial_rho_profile.width << '\n'
              << "initial T width [code] == " << params.initial_T_profile.width << '\n'
              << "magnetic width w_B [code]" << params.initial_magnetic_profile.width
              << '\n'
              << "rho/T perturbation ===== 1 + p*cos(N*theta)" << '\n'
              << "velocity =============== "
              << (params.initial_rho_profile.shape == ProfileShape::annulus
                      ? "outer initial rho transition plus driven injected mass"
                      : (params.drive_enable ? "driven mass injected radially outward"
                                             : "normalized -grad(initial rho profile)"))
              << '\n'
              << "magnetic field ========= "
              << "B = z_hat x grad(profile), peak-normalized" << '\n';
  }

  return params;
}

KOKKOS_INLINE_FUNCTION
InitialState EvaluateInitialState(const PulsedReconnectionParams &params,
                                  const Real x, const Real y) {
  InitialState state{};
  const Real d = params.array_separation / 2.0;
  Real T_profile_sum = 0.0;
  Real rho_profile_sum = 0.0;
  Real T_inner_reservoir_sum = 0.0;
  Real rho_inner_reservoir_sum = 0.0;
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
    T_inner_reservoir_sum +=
        EvaluateInnerReservoirProfile(params.initial_T_profile,
                                      params.inner_reservoir_radius, r);
    rho_inner_reservoir_sum +=
        EvaluateInnerReservoirProfile(params.initial_rho_profile,
                                      params.inner_reservoir_radius, r);
    magnetic_support_sum += EvaluateSupportTable(params.initial_support_table, r);

    if (r > 0.0) {
      const Real inv_r = 1.0 / r;
      const Real xhat = x * inv_r;
      const Real yhat = y_local * inv_r;

      Real radial_velocity = 0.0;
      if (params.initial_rho_profile.shape == ProfileShape::annulus) {
        // For an annulus, -grad(rho) points inward on the inner transition and
        // outward on the outer transition. Initialize only the outward ablation flow.
        radial_velocity = fmax(0.0, -params.velocity_normalization * drho_profile_dr);
      } else if (!params.drive_enable) {
        radial_velocity = -params.velocity_normalization * drho_profile_dr;
      }
      state.v1 += radial_velocity * xhat;
      state.v2 += radial_velocity * yhat;

    }
  }

  state.rho = params.rho_background + params.rho_wire * rho_profile_sum +
              params.rho_inner_reservoir * rho_inner_reservoir_sum;
  const Real T = params.T_background + params.T_wire * T_profile_sum +
                 params.T_inner_reservoir * T_inner_reservoir_sum;
  state.pressure =
      T * params.k_b * state.rho / params.m_bar +
      (params.initial_force_balance
           ? SQR(params.initial_magnetic_profile_amplitude) * magnetic_support_sum
           : 0.0);
  return state;
}

} // namespace

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  PARTHENON_REQUIRE(IsCTFluid(fluid),
                    "pulsed_reconnection requires a constrained-transport fluid.");

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
    WarnUnless(refine_tol > 0.0,
               "Nonpositive current_refine_tol may request refinement everywhere.");
    WarnUnless(derefine_tol >= 0.0,
               "Negative current_derefine_tol prevents current-based derefinement.");
    WarnUnless(
        derefine_tol < 0.5 * refine_tol,
        "current_derefine_tol is not less than half of current_refine_tol; "
        "factor-two AMR level oscillation is possible.");
    hydro_pkg->AddParam<Real>("refinement/current_refine_tol", refine_tol);
    hydro_pkg->AddParam<Real>("refinement/current_derefine_tol", derefine_tol);
  }
}

// Initialization ----------------------------------------------------------------------
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
  if (!g_source_params_initialized) {
    g_source_params = LoadSourceParams(hydro_pkg, pin);
    g_source_params_initialized = true;
  }
  const auto params = g_source_params;

  auto &coords = pmb->coords;
  auto u_host = u.GetHostMirrorAndCopy();
  auto bface_host = bface.GetHostMirrorAndCopy();
  auto b1f = bface_host.Get(IBF1, 0, 0, 0);
  auto b2f = bface_host.Get(IBF2, 0, 0, 0);
  auto b3f = bface_host.Get(IBF3, 0, 0, 0);

  const bool two_d = pmb->pmy_mesh->ndim < 3;
  // A_z is independent of z: its value at a coarse edge center equals the
  // average at its two child-edge centers. No neighbor-level correction is needed.
  // Construct face fields from the discrete curl to preserve zero divergence.
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
        az(k, j, i) = EvaluateAz(params.initial_magnetic_profile,
                               params.initial_magnetic_profile_amplitude,
                               params.array_separation, x, y);
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
        const auto state = EvaluateInitialState(params, coords.Xc<1>(i), coords.Xc<2>(j));
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
        u_host(IEN, k, j, i) = state.pressure / params.gm1 +
            0.5 * (SQR(B1) + SQR(B2) + SQR(B3) +
                   state.rho *
                       (SQR(state.v1) + SQR(state.v2) + SQR(state.v3)));
      }
    }
  }
  bface.DeepCopy(bface_host);
  u.DeepCopy(u_host);
}

// Driving -----------------------------------------------------------------------------
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
      params.drive_peak_magnetic_profile_amplitude * PulseEnvelopeAtTime(params, tm.time);
  const Real amplitude_new =
      params.drive_peak_magnetic_profile_amplitude *
      PulseEnvelopeAtTime(params, tm.time + dt);
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
        const Real delta_az0 =
            EvaluateAz(params.drive_magnetic_profile, amplitude_new,
                       params.array_separation, x, y0) -
            EvaluateAz(params.drive_magnetic_profile, amplitude_old,
                       params.array_separation, x, y0);
        const Real delta_az1 =
            EvaluateAz(params.drive_magnetic_profile, amplitude_new,
                       params.array_separation, x, y1) -
            EvaluateAz(params.drive_magnetic_profile, amplitude_old,
                       params.array_separation, x, y1);
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
        const Real delta_az0 =
            EvaluateAz(params.drive_magnetic_profile, amplitude_new,
                       params.array_separation, x0, y) -
            EvaluateAz(params.drive_magnetic_profile, amplitude_old,
                       params.array_separation, x0, y);
        const Real delta_az1 =
            EvaluateAz(params.drive_magnetic_profile, amplitude_new,
                       params.array_separation, x1, y) -
            EvaluateAz(params.drive_magnetic_profile, amplitude_old,
                       params.array_separation, x1, y);
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
        if (support.rho_target <= 0.0 && support.rho_inner_target <= 0.0 &&
            support.inner_reservoir_weight <= 0.0 &&
            support.T_floor <= 0.0 && support.T_inner_floor <= 0.0 &&
            magnetic_support <= 0.0) {
          return;
        }

        const Real rho_old = cons(IDN, k, j, i);
        const Real v1_old = rho_old > 0.0 ? cons(IM1, k, j, i) / rho_old : 0.0;
        const Real v2_old = rho_old > 0.0 ? cons(IM2, k, j, i) / rho_old : 0.0;
        const Real v3_old = rho_old > 0.0 ? cons(IM3, k, j, i) / rho_old : 0.0;

        // Maintain the inner reservoir first. Its replacement mass is co-moving and
        // never receives the annular ablation velocity. The background contribution
        // is included only where an inner-reservoir mask has support.
        const Real inner_rho_target =
            support.inner_reservoir_weight > 0.0
                ? params.rho_background + support.rho_inner_target
                : 0.0;
        const Real delta_rho_inner = fmax(0.0, inner_rho_target - rho_old);
        const Real rho_after_inner = rho_old + delta_rho_inner;

        // The annular target is additive to the supported inner reservoir, matching
        // initialization across their complementary transition masks. On the falling
        // side of a sin2 pulse, previously supplied mass is never removed.
        const Real total_rho_target = inner_rho_target + support.rho_target;
        const Real delta_rho_annulus = fmax(0.0, total_rho_target - rho_after_inner);
        const Real delta_rho = delta_rho_inner + delta_rho_annulus;
        const Real rho_new = rho_old + delta_rho;

        // Only annular replacement mass receives the prescribed radial velocity.
        const bool inject_driven_velocity =
            params.drive_velocity_mode == VelocityDriveMode::injected_mass &&
            support.velocity_weight > 0.0;
        const Real v1_injected =
            inject_driven_velocity ? support.v1_target : v1_old;
        const Real v2_injected =
            inject_driven_velocity ? support.v2_target : v2_old;
        const Real v3_injected = inject_driven_velocity ? 0.0 : v3_old;
        cons(IDN, k, j, i) = rho_new;
        cons(IM1, k, j, i) +=
            delta_rho_inner * v1_old + delta_rho_annulus * v1_injected;
        cons(IM2, k, j, i) +=
            delta_rho_inner * v2_old + delta_rho_annulus * v2_injected;
        cons(IM3, k, j, i) +=
            delta_rho_inner * v3_old + delta_rho_annulus * v3_injected;

        // Inject both the kinetic energy required by co-motion and the specific internal
        // energy corresponding to the instantaneous local T profile floor. This term is
        // separate from the minimum-pressure correction below.
        const Real injected_kinetic_energy =
            0.5 * delta_rho_inner *
                (SQR(v1_old) + SQR(v2_old) + SQR(v3_old)) +
            0.5 * delta_rho_annulus *
                (SQR(v1_injected) + SQR(v2_injected) + SQR(v3_injected));
        const Real supported_T_floor =
            support.T_floor +
            (support.inner_reservoir_weight > 0.0
                 ? params.T_background + support.T_inner_floor
                 : 0.0);
        const Real injected_internal_energy =
            delta_rho * params.k_b * supported_T_floor /
            (params.m_bar * params.gm1);
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
            rho_new > 0.0
                ? supported_T_floor * params.k_b * rho_new / params.m_bar
                : 0.0;
        const Real target_pressure = thermal_pressure_floor + magnetic_support;
        const Real delta_internal_energy =
            target_pressure > pressure ? (target_pressure - pressure) / params.gm1 : 0.0;
        cons(IEN, k, j, i) += delta_internal_energy;
      });
}

// Refinement --------------------------------------------------------------------------
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
  if (magnetic_field_reference <= 0.0) return parthenon::AmrTag::same;

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

// Diagnostics -------------------------------------------------------------------------
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

} // namespace pulsed_reconnection
