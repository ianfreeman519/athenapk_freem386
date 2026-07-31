//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file pulsed_reconnection_gaussian.cpp
//! \brief Problem generator for pulsed reconnection with Gaussian thermodynamics and a
//! divergence-free Gaussian magnetic field.
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

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/diffusion/diffusion.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"

namespace pulsed_reconnection_gaussian {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Evaluate a Gaussian profile and its radial derivative in one place so the thermal and
// magnetic initializers stay algebraically consistent.
KOKKOS_INLINE_FUNCTION
void GaussianProfileAndDerivative(const Real r, const Real width, Real &profile,
                                  Real &dprofile_dr);

// Analytic pressure support that balances the Lorentz force from
// B = z_hat x grad(A * exp(-(r / w_B)^2)) for a single wire, with the integration
// constant chosen so the support vanishes at infinity.
KOKKOS_INLINE_FUNCTION
Real GaussianMagneticSupportProfile(const Real r, const Real width_magnetic,
                                    const Real magnetic_profile_amplitude);

// Only density and temperature inherit the azimuthal modulation.
KOKKOS_INLINE_FUNCTION
Real AzimuthalThermoPerturbation(const Real theta, const Real p, const int mode_number);

namespace {

// Store both raw inputs and the derived code-unit quantities that the hot loops use.
// In particular, `magnetic_profile_amplitude` is the amplitude A of the scalar profile
// whose curl gives B; it is deliberately not the peak magnetic field itself.
struct PulsedGaussianParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real current_peak_MA;
  bool drive_enable;
  Real drive_peak_current_MA;
  Real drive_t_peak_ns;
  Real drive_t_peak;
  Real drive_width_cgs;
  Real drive_width;
  Real drive_cutoff_radius_factor;
  bool drive_hydro_support_enable;
  Real drive_rho_floor_cgs;
  Real drive_rho_floor;
  Real drive_T_floor;
  Real rho_wire_cgs;
  Real rho_background_cgs;
  Real T_wire;
  Real T_background;
  Real v0_cgs;
  Real array_separation_cgs;
  Real width_thermo_cgs;
  Real width_magnetic_cgs;
  int azimuthal_mode_number;
  Real density_perturb_amplitude;
  Real temperature_perturb_amplitude;
  Real force_balance;
  Real current_field_prefac;
  Real rho_wire;
  Real rho_background;
  Real v0;
  Real array_separation;
  Real width_thermo;
  Real width_magnetic;
  Real peak_magnetic_field_strength;
  Real magnetic_profile_amplitude;
  Real velocity_normalization;
};

struct PulsedGaussianState {
  Real rho;
  Real pressure;
  Real v1;
  Real v2;
  Real v3;
  Real B1;
  Real B2;
  Real B3;
};

struct DriveFloorState {
  Real rho_floor;
  Real T_floor;
};

KOKKOS_INLINE_FUNCTION
Real EvaluateDriveMagneticSupport(const PulsedGaussianParams &params, const Real x,
                                  const Real y, const Real amplitude) {
  // In driven mode we inject the magnetic field through a time-dependent Gaussian
  // vector potential. The matching analytic pressure support must therefore be
  // evaluated from that same instantaneous Gaussian amplitude so that the gas
  // pressure grows in step with the magnetic pressure/tension.
  Real support = 0.0;
  const Real half_sep = 0.5 * params.array_separation;
  const Real cutoff_radius = params.drive_cutoff_radius_factor * params.drive_width;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    const Real r2 = SQR(x) + SQR(y_local);
    if (r2 > SQR(cutoff_radius)) {
      continue;
    }
    support += GaussianMagneticSupportProfile(sqrt(r2), params.drive_width, amplitude);
  }
  return support;
}

PulsedGaussianParams g_source_params{};
bool g_source_params_initialized = false;

// The legacy single-wire ampere-loop field peaks at q = (r / w_B)^2 satisfying
// exp(-q) * (2 q + 1) = 1. These constants let the new Gaussian curl field match that
// old peak |B| exactly for the same `current_peak_MA`.
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
  return params.magnetic_profile_amplitude * current_ampere / peak_current_ampere;
}

KOKKOS_INLINE_FUNCTION
Real EvaluateDriveAzSingleWire(const Real x, const Real y_local, const Real drive_width,
                               const Real cutoff_radius, const Real amplitude) {
  const Real r2 = SQR(x) + SQR(y_local);
  if (r2 > SQR(cutoff_radius)) {
    return 0.0;
  }
  return amplitude * exp(fmax(-700.0, -r2 / SQR(drive_width)));
}

KOKKOS_INLINE_FUNCTION
Real EvaluateDriveAz(const PulsedGaussianParams &params, const Real x, const Real y,
                     const Real amplitude) {
  const Real half_sep = 0.5 * params.array_separation;
  const Real cutoff_radius = params.drive_cutoff_radius_factor * params.drive_width;
  return EvaluateDriveAzSingleWire(x, y - half_sep, params.drive_width, cutoff_radius,
                                   amplitude) +
         EvaluateDriveAzSingleWire(x, y + half_sep, params.drive_width, cutoff_radius,
                                   amplitude);
}

KOKKOS_INLINE_FUNCTION
DriveFloorState EvaluateDriveFloorState(const PulsedGaussianParams &params, const Real x,
                                        const Real y) {
  DriveFloorState floors{0.0, 0.0};
  const Real half_sep = 0.5 * params.array_separation;
  const Real cutoff_radius = params.drive_cutoff_radius_factor * params.drive_width;
  for (int sign = -1; sign <= 1; sign += 2) {
    const Real y_local = y - sign * half_sep;
    const Real r2 = SQR(x) + SQR(y_local);
    if (r2 > SQR(cutoff_radius)) {
      continue;
    }
    const Real profile = exp(fmax(-700.0, -r2 / SQR(params.drive_width)));
    floors.rho_floor += params.drive_rho_floor * profile;
    floors.T_floor += params.drive_T_floor * profile;
  }
  return floors;
}

PulsedGaussianParams LoadSourceParams(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                                      ParameterInput *pin) {
  PulsedGaussianParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.current_peak_MA = pin->GetOrAddReal("problem/pulsed_reconnection_gaussian",
                                             "current_peak_MA", 1.0);
  params.drive_enable =
      pin->GetOrAddBoolean("problem/pulsed_reconnection_gaussian", "drive_enable", false);
  params.drive_peak_current_MA = pin->GetOrAddReal(
      "problem/pulsed_reconnection_gaussian", "drive_peak_current_MA",
      params.current_peak_MA);
  params.drive_t_peak_ns = pin->GetOrAddReal("problem/pulsed_reconnection_gaussian",
                                             "drive_t_peak_ns", 500.0);
  params.rho_wire_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "rho_wire", 1e-3);
  params.rho_background_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "rho_background", 1e-6);
  params.T_wire =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "T_wire", 1.1e4);
  params.T_background =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "T_background", 1e2);
  params.v0_cgs = pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "v0", 1.0e6);
  params.array_separation_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "array_separation", 4.0);
  params.width_thermo_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "w", 1.0);
  params.width_magnetic_cgs = pin->GetOrAddReal(
      "problem/pulsed_reconnection_gaussian", "w_B",
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "w_magnetic",
                        params.width_thermo_cgs));
  params.drive_width_cgs = pin->GetOrAddReal("problem/pulsed_reconnection_gaussian",
                                             "w_drive", params.width_magnetic_cgs);
  params.drive_cutoff_radius_factor = pin->GetOrAddReal(
      "problem/pulsed_reconnection_gaussian", "drive_cutoff_radius_factor", 3.0);
  params.drive_hydro_support_enable = pin->GetOrAddBoolean(
      "problem/pulsed_reconnection_gaussian", "drive_hydro_support_enable", false);
  params.drive_rho_floor_cgs = pin->GetOrAddReal(
      "problem/pulsed_reconnection_gaussian", "drive_rho_floor", params.rho_wire_cgs);
  params.drive_T_floor = pin->GetOrAddReal("problem/pulsed_reconnection_gaussian",
                                           "drive_T_floor", params.T_wire);
  params.azimuthal_mode_number =
      pin->GetOrAddInteger("problem/pulsed_reconnection_gaussian", "N", 0);
  params.density_perturb_amplitude = pin->GetOrAddReal(
      "problem/pulsed_reconnection_gaussian", "density_perturb_amplitude", 0.0);
  params.temperature_perturb_amplitude = pin->GetOrAddReal(
      "problem/pulsed_reconnection_gaussian", "temperature_perturb_amplitude", 0.0);
  params.force_balance =
      pin->GetOrAddReal("problem/pulsed_reconnection_gaussian", "force_balance", 1.0);

  PARTHENON_REQUIRE(params.width_thermo_cgs > 0.0,
                    "problem/pulsed_reconnection_gaussian/w must be positive.");
  PARTHENON_REQUIRE(params.width_magnetic_cgs > 0.0,
                    "problem/pulsed_reconnection_gaussian/w_B must be positive.");
  PARTHENON_REQUIRE(params.drive_width_cgs > 0.0,
                    "problem/pulsed_reconnection_gaussian/w_drive must be positive.");
  PARTHENON_REQUIRE(params.array_separation_cgs > 0.0,
                    "problem/pulsed_reconnection_gaussian/array_separation must be "
                    "positive.");
  PARTHENON_REQUIRE(params.current_peak_MA >= 0.0,
                    "problem/pulsed_reconnection_gaussian/current_peak_MA must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_peak_current_MA >= 0.0,
                    "problem/pulsed_reconnection_gaussian/drive_peak_current_MA must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_t_peak_ns > 0.0,
                    "problem/pulsed_reconnection_gaussian/drive_t_peak_ns must be "
                    "positive.");
  PARTHENON_REQUIRE(
      params.drive_cutoff_radius_factor > 0.0,
      "problem/pulsed_reconnection_gaussian/drive_cutoff_radius_factor must be positive.");
  PARTHENON_REQUIRE(params.drive_rho_floor_cgs >= 0.0,
                    "problem/pulsed_reconnection_gaussian/drive_rho_floor must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.drive_T_floor >= 0.0,
                    "problem/pulsed_reconnection_gaussian/drive_T_floor must be "
                    "nonnegative.");
  PARTHENON_REQUIRE(params.azimuthal_mode_number >= 0,
                    "problem/pulsed_reconnection_gaussian/N must be nonnegative.");
  PARTHENON_REQUIRE(params.force_balance >= 0.0,
                    "problem/pulsed_reconnection_gaussian/force_balance must be "
                    "nonnegative.");

  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar =
      pin->GetReal("hydro", "mean_molecular_weight") * units.atomic_mass_unit();
  const Real current_peak_ampere = params.current_peak_MA * 1.0e6;
  params.current_field_prefac = 0.2 * current_peak_ampere * units.cm() * units.gauss();
  params.rho_wire = params.rho_wire_cgs * units.g_cm3();
  params.rho_background = params.rho_background_cgs * units.g_cm3();
  params.drive_rho_floor = params.drive_rho_floor_cgs * units.g_cm3();
  params.v0 = params.v0_cgs * units.cm_s();
  params.array_separation = params.array_separation_cgs * units.cm();
  params.width_thermo = params.width_thermo_cgs * units.cm();
  params.width_magnetic = params.width_magnetic_cgs * units.cm();
  params.drive_width = params.drive_width_cgs * units.cm();
  params.drive_t_peak = params.drive_t_peak_ns * 1.0e-9 / units.s();

  // Match the peak magnitude of the legacy ampere-loop field, then back out the scalar
  // Gaussian amplitude A required by B = z_hat x grad(A * exp(-(r / w_B)^2)).
  const Real drive_peak_current_ampere = params.drive_peak_current_MA * 1.0e6;
  const Real drive_current_field_prefac =
      0.2 * drive_peak_current_ampere * units.cm() * units.gauss();
  params.peak_magnetic_field_strength =
      params.width_magnetic > 0.0
          ? drive_current_field_prefac * kOldAmpereLoopPeakFactor / params.width_magnetic
          : 0.0;
  params.magnetic_profile_amplitude =
      params.peak_magnetic_field_strength * params.width_magnetic /
      kGaussianGradientPeakFactor;

  // The thermal Gaussian gradient peaks at sqrt(2 / e) / w. This normalization keeps
  // the per-wire outward expansion speed at the requested peak value v0.
  params.velocity_normalization =
      params.width_thermo > 0.0
          ? params.v0 / (kGaussianGradientPeakFactor / params.width_thermo)
          : 0.0;

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

  // Each wire contributes a thermodynamic Gaussian, a radial expansion, an azimuthal
  // magnetic field from the curl construction, and an analytic magnetic support term.
  // The total state is the simple sum of the two wire-centered contributions plus the
  // global background density/temperature.
  for (int A = -1; A <= 1; A += 2) {
    const Real y_center = A * d;
    const Real y_local = y - y_center;
    const Real r2 = SQR(x) + SQR(y_local);
    const Real r = sqrt(r2);
    const Real theta = atan2(y_local, x);

    Real thermo_profile = 0.0;
    Real dthermo_profile_dr = 0.0;
    GaussianProfileAndDerivative(r, params.width_thermo, thermo_profile,
                                 dthermo_profile_dr);
    const Real density_perturbation = AzimuthalThermoPerturbation(
        theta, params.density_perturb_amplitude, params.azimuthal_mode_number);
    const Real temperature_perturbation = AzimuthalThermoPerturbation(
        theta, params.temperature_perturb_amplitude, params.azimuthal_mode_number);
    thermo_profile_sum += thermo_profile * temperature_perturbation;
    density_profile_sum += thermo_profile * density_perturbation;
    magnetic_support_sum += GaussianMagneticSupportProfile(
        r, params.width_magnetic, params.magnetic_profile_amplitude);

    if (r > 0.0) {
      const Real inv_r = 1.0 / r;
      const Real xhat = x * inv_r;
      const Real yhat = y_local * inv_r;

      // `dthermo_profile_dr` is negative, so `-dthermo_profile_dr` points outward.
      const Real radial_velocity =
          params.drive_enable ? 0.0 : -params.velocity_normalization * dthermo_profile_dr;
      state.v1 += radial_velocity * xhat;
      state.v2 += radial_velocity * yhat;

      // The magnetic field comes from B = z_hat x grad(phi). For a radial phi(r),
      // that produces a purely azimuthal field with no imposed azimuthal perturbation.
      if (!params.drive_enable) {
        Real magnetic_profile_unused = 0.0;
        Real dmagnetic_profile_dr = 0.0;
        GaussianProfileAndDerivative(r, params.width_magnetic, magnetic_profile_unused,
                                     dmagnetic_profile_dr);
        dmagnetic_profile_dr *= params.magnetic_profile_amplitude;
        state.B1 += -dmagnetic_profile_dr * yhat;
        state.B2 += dmagnetic_profile_dr * xhat;
      }
    }
  }

  state.rho = params.rho_background + params.rho_wire * density_profile_sum;
  const Real T = params.T_background + params.T_wire * thermo_profile_sum;
  state.pressure =
      T * params.k_b * state.rho / params.m_bar +
      (params.drive_enable ? 0.0 : params.force_balance * magnetic_support_sum);
  return state;
}

} // namespace

KOKKOS_INLINE_FUNCTION
void GaussianProfileAndDerivative(const Real r, const Real width, Real &profile,
                                  Real &dprofile_dr) {
  const Real exponent = -SQR(r / width);
  profile = exp(fmax(-700.0, exponent));
  dprofile_dr = (-2.0 * r / SQR(width)) * profile;
}

KOKKOS_INLINE_FUNCTION
Real GaussianMagneticSupportProfile(const Real r, const Real width_magnetic,
                                    const Real magnetic_profile_amplitude) {
  // For phi = A exp(-q), q = (r / w_B)^2:
  //   B_phi = dphi/dr = -2 A r / w_B^2 * exp(-q)
  // Integrating radial force balance with p_support(infinity) = 0 gives:
  //   p_support = (A^2 / w_B^2) * (1 - 2 q) * exp(-2 q)
  const Real q = SQR(r / width_magnetic);
  const Real profile = exp(fmax(-700.0, -2.0 * q));
  return SQR(magnetic_profile_amplitude / width_magnetic) * (1.0 - 2.0 * q) * profile;
}

KOKKOS_INLINE_FUNCTION
Real AzimuthalThermoPerturbation(const Real theta, const Real p, const int mode_number) {
  const Real phase = static_cast<Real>(mode_number) * theta;
  const Real cos_phase = cos(phase);
  return 1 + p * cos_phase;
}

void ProblemInitPackageData(ParameterInput * /*pin*/,
                            parthenon::StateDescriptor *hydro_pkg) {
  // Mirror the legacy pulsed-reconnection diagnostic fields so existing analysis and
  // local test decks can be reused against the new initializer.
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

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/) {
  // This is intentionally kept close to the legacy pulsed-reconnection diagnostics.
  // The new initializer changes only the setup physics, not the derived-field outputs
  // that downstream scripts expect to find.
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
      "pulsed_reconnection_gaussian::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inf = std::numeric_limits<Real>::infinity();
        Real term1, term2;

        // Store the local current density and kinematic diagnostics directly into
        // output fields so the test deck can inspect them without postprocessing.
        term1 = (u(IB3, k, j + 1, i) - u(IB3, k, j - 1, i)) /
                (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        term2 = (u(IB2, k + 1, j, i) - u(IB2, k - 1, j, i)) /
                (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        curlBx(k, j, i) = term1 - term2;

        term1 = (u(IB1, k + 1, j, i) - u(IB1, k - 1, j, i)) /
                (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        term2 = (u(IB3, k, j, i + 1) - u(IB3, k, j, i - 1)) /
                (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        curlBy(k, j, i) = term1 - term2;

        term1 = (u(IB2, k, j, i + 1) - u(IB2, k, j, i - 1)) /
                (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        term2 = (u(IB1, k, j + 1, i) - u(IB1, k, j - 1, i)) /
                (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        curlBz(k, j, i) = term1 - term2;

        const Real dvx_dx = (w(IV1, k, j, i + 1) - w(IV1, k, j, i - 1)) /
                            (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real dvy_dy = (w(IV2, k, j + 1, i) - w(IV2, k, j - 1, i)) /
                            (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        const Real dvz_dz = (w(IV3, k + 1, j, i) - w(IV3, k - 1, j, i)) /
                            (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        divv(k, j, i) = dvx_dx + dvy_dy + dvz_dz;

        const Real rho = u(IDN, k, j, i);
        const Real p = w(IPR, k, j, i);
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

        const Real dBzdy =
            ndim > 1 ? (u(IB3, k, j + 1, i) - u(IB3, k, j - 1, i)) /
                           (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                     : 0.0;
        const Real dBydz =
            ndim > 2 ? (u(IB2, k + 1, j, i) - u(IB2, k - 1, j, i)) /
                           (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                     : 0.0;
        const Real dBxdz =
            ndim > 2 ? (u(IB1, k + 1, j, i) - u(IB1, k - 1, j, i)) /
                           (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                     : 0.0;
        const Real dBzdx = (u(IB3, k, j, i + 1) - u(IB3, k, j, i - 1)) /
                           (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real dBydx = (u(IB2, k, j, i + 1) - u(IB2, k, j, i - 1)) /
                           (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        const Real dBxdy =
            ndim > 1 ? (u(IB1, k, j + 1, i) - u(IB1, k, j - 1, i)) /
                           (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                     : 0.0;
        const Real jx = dBzdy - dBydz;
        const Real jy = dBxdz - dBzdx;
        const Real jz = dBydx - dBxdy;
        const Real j_squared = SQR(jx) + SQR(jy) + SQR(jz);
        const Real internal_e_dens = p / gm1;
        const Real dt_heat_val =
            (resistivity != Resistivity::none && eta_val > 0.0 && j_squared > 0.0)
                ? cfl_diff_heat * fabs(internal_e_dens / (eta_val * j_squared))
                : inf;
        dt_heat_local(k, j, i) = dt_heat_val;

        const Real internal_e_spec = p / (rho * gm1);
        const Real de_dt_cool = enable_cooling == Cooling::tabular
                                    ? cooling_table_obj.DeDt(internal_e_spec, rho)
                                    : 0.0;
        const Real dt_cool_val =
            (enable_cooling == Cooling::tabular && de_dt_cool != 0.0 &&
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
        const Real cs = eos.SoundSpeed(prim_local);
        const Real lambda_fms_x = eos.FastMagnetosonicSpeed(
            rho, p, u(IB1, k, j, i), u(IB2, k, j, i), u(IB3, k, j, i));
        Real dt_hyp_fms_val =
            coords.Dxc<1>(k, j, i) / (fabs(prim_local[IV1]) + lambda_fms_x);
        Real dt_hyp_cs_val = coords.Dxc<1>(k, j, i) / (fabs(prim_local[IV1]) + cs);
        if (ndim > 1) {
          const Real lambda_fms_y = eos.FastMagnetosonicSpeed(
              rho, p, u(IB2, k, j, i), u(IB3, k, j, i), u(IB1, k, j, i));
          dt_hyp_fms_val = fmin(
              dt_hyp_fms_val,
              coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + lambda_fms_y));
          dt_hyp_cs_val = fmin(dt_hyp_cs_val,
                               coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + cs));
        }
        if (ndim > 2) {
          const Real lambda_fms_z = eos.FastMagnetosonicSpeed(
              rho, p, u(IB3, k, j, i), u(IB1, k, j, i), u(IB2, k, j, i));
          dt_hyp_fms_val = fmin(
              dt_hyp_fms_val,
              coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + lambda_fms_z));
          dt_hyp_cs_val = fmin(dt_hyp_cs_val,
                               coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + cs));
        }
        dt_hyp_fms(k, j, i) = cfl_hyp * dt_hyp_fms_val;
        dt_hyp_cs(k, j, i) = cfl_hyp * dt_hyp_cs_val;
      });
}

void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  const auto params = g_source_params;
  if (!params.drive_enable || dt <= 0.0) {
    return;
  }

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const Real amplitude_old =
      DrivePotentialAmplitudeFromCurrent(params, DriveCurrentAtTime(params, tm.time));
  const Real amplitude_new = DrivePotentialAmplitudeFromCurrent(
      params, DriveCurrentAtTime(params, tm.time + dt));

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "pulsed_reconnection_gaussian::Driving",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real az_xm_old = EvaluateDriveAz(params, coords.Xc<1>(i - 1), y, amplitude_old);
        const Real az_xp_old = EvaluateDriveAz(params, coords.Xc<1>(i + 1), y, amplitude_old);
        const Real az_ym_old = EvaluateDriveAz(params, x, coords.Xc<2>(j - 1), amplitude_old);
        const Real az_yp_old = EvaluateDriveAz(params, x, coords.Xc<2>(j + 1), amplitude_old);
        const Real az_xm_new = EvaluateDriveAz(params, coords.Xc<1>(i - 1), y, amplitude_new);
        const Real az_xp_new = EvaluateDriveAz(params, coords.Xc<1>(i + 1), y, amplitude_new);
        const Real az_ym_new = EvaluateDriveAz(params, x, coords.Xc<2>(j - 1), amplitude_new);
        const Real az_yp_new = EvaluateDriveAz(params, x, coords.Xc<2>(j + 1), amplitude_new);

        const Real delta_B1 =
            ((az_yp_new - az_ym_new) - (az_yp_old - az_ym_old)) /
            (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        const Real delta_B2 =
            -((az_xp_new - az_xm_new) - (az_xp_old - az_xm_old)) /
            (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));

        const Real old_B1 = cons(IB1, k, j, i);
        const Real old_B2 = cons(IB2, k, j, i);
        const Real old_B3 = cons(IB3, k, j, i);
        cons(IB1, k, j, i) += delta_B1;
        cons(IB2, k, j, i) += delta_B2;
        cons(IEN, k, j, i) += old_B1 * delta_B1 + old_B2 * delta_B2 +
                              0.5 * (SQR(delta_B1) + SQR(delta_B2));

        if (!params.drive_hydro_support_enable) {
          return;
        }

        const auto floors = EvaluateDriveFloorState(params, x, y);
        const Real magnetic_support =
            params.force_balance > 0.0
                ? fmax(0.0, params.force_balance *
                                  EvaluateDriveMagneticSupport(params, x, y, amplitude_new))
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
            0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) + SQR(old_B3));
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
  auto hydro_pkg = pmb->packages.Get("Hydro");
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
    std::cout << "thermo width w [cm] ==== " << params.width_thermo_cgs << std::endl;
    std::cout << "magnetic width w_B [cm]  " << params.width_magnetic_cgs << std::endl;
    std::cout << "drive width w_drive [cm] " << params.drive_width_cgs << std::endl;
    std::cout << "drive cutoff radius ==== "
              << params.drive_cutoff_radius_factor * params.drive_width_cgs << std::endl;
    std::cout << "drive hydro support ==== " << params.drive_hydro_support_enable
              << std::endl;
    std::cout << "drive rho floor [g/cm^3] " << params.drive_rho_floor_cgs << std::endl;
    std::cout << "drive T floor [K] ====== " << params.drive_T_floor << std::endl;
    std::cout << "azimuthal mode N ======= " << params.azimuthal_mode_number
              << std::endl;
    std::cout << "dens. perturb. amplitude=" << params.density_perturb_amplitude
              << std::endl;
    std::cout << "temp perturb. amplitude =" << params.temperature_perturb_amplitude
              << std::endl;
    std::cout << "force_balance ========= " << params.force_balance << std::endl;
    std::cout << "Converted code units:" << std::endl;
    std::cout << "matched |B|_peak [code] = " << params.peak_magnetic_field_strength
              << std::endl;
    std::cout << "mag profile amp [code] = " << params.magnetic_profile_amplitude
              << std::endl;
    std::cout << "rho_wire(core) [code] == " << params.rho_wire << std::endl;
    std::cout << "rho_background [code] == " << params.rho_background << std::endl;
    std::cout << "v0(peak) [code] ======== " << params.v0 << std::endl;
    std::cout << "array_separation [code]  " << params.array_separation << std::endl;
    std::cout << "thermo width w [code] == " << params.width_thermo << std::endl;
    std::cout << "magnetic width w_B [code]" << params.width_magnetic << std::endl;
    std::cout << "thermo profile ========= exp(-(r / w)^2)" << std::endl;
    std::cout << "thermo perturbation ==== 1 + p*cos(N*theta)" << std::endl;
    std::cout << "velocity =============== "
              << (params.drive_enable ? "disabled in driven mode"
                                      : "normalized -grad(T_profile)")
              << std::endl;
    std::cout << "magnetic field ========= "
              << (params.drive_enable ? "source-built from Az(t)"
                                      : "B = z_hat x grad(A*exp(-(r/w_B)^2))")
              << std::endl;
    std::cout << "old |B| peak match at q = " << kOldAmpereLoopPeakQ << std::endl;
  }

  auto &coords = pmb->coords;
  pmb->par_for(
      "ProblemGenerator::pulsed_reconnection_gaussian", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);

        // Evaluate the full double-wire state at each cell center and write the
        // conservative variables directly.
        const auto state = EvaluateSourceState(params, x, y);

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
                   state.rho *
                       (SQR(state.v1) + SQR(state.v2) + SQR(state.v3)));
      });
}

} // namespace pulsed_reconnection_gaussian
