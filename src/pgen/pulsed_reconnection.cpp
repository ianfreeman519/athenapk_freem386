//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file pulsed_reconnection.cpp
//! \brief Problem generator mimicking pulsed power reconnection experiments.
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
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../bvals/boundary_conditions_apk.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/diffusion/diffusion.hpp" // For storing eta later
#include "../hydro/srcterms/tabular_cooling.hpp"

namespace pulsed_reconnection {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

KOKKOS_INLINE_FUNCTION
Real GaussianProfile(const Real r, const Real width);

KOKKOS_INLINE_FUNCTION
Real AzimuthalThermoPerturbation(const Real theta, const Real p, const int mode_number);

namespace {

struct PulsedSourceParams {
  Real gm1;
  Real k_b;
  Real m_bar;
  Real current_peak_MA;
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
  Real current_field_prefac;
  Real rho_wire;
  Real rho_background;
  Real v0;
  Real array_separation;
  Real width_thermo;
  Real width_magnetic;
};

struct PulsedSourceState {
  Real rho;
  Real pressure;
  Real v1;
  Real v2;
  Real v3;
  Real B1;
  Real B2;
  Real B3;
};

PulsedSourceParams g_source_params{};
bool g_source_params_initialized = false;

PulsedSourceParams LoadSourceParams(const std::shared_ptr<StateDescriptor> &hydro_pkg,
                                    ParameterInput *pin) {
  PulsedSourceParams params{};
  params.gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  params.current_peak_MA =
      pin->GetOrAddReal("problem/pulsed_reconnection", "current_peak_MA", 1.0);
  params.rho_wire_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "rho_wire", 1e-3);
  params.rho_background_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "rho_background", 1e-6);
  params.T_wire = pin->GetOrAddReal("problem/pulsed_reconnection", "T_wire", 1.1e4);
  params.T_background =
      pin->GetOrAddReal("problem/pulsed_reconnection", "T_background", 1e2);
  params.v0_cgs = pin->GetOrAddReal("problem/pulsed_reconnection", "v0", 1.0e6);
  params.array_separation_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "array_separation", 4.0);
  params.width_thermo_cgs =
      pin->GetOrAddReal("problem/pulsed_reconnection", "w", 1.0);
  params.width_magnetic_cgs = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_B",
      pin->GetOrAddReal("problem/pulsed_reconnection", "w_magnetic",
                        params.width_thermo_cgs));
  params.azimuthal_mode_number =
      pin->GetOrAddInteger("problem/pulsed_reconnection", "N", 0);
  params.density_perturb_amplitude =
      pin->GetOrAddReal("problem/pulsed_reconnection", "density_perturb_amplitude", 0.0);
  params.temperature_perturb_amplitude =
      pin->GetOrAddReal("problem/pulsed_reconnection", "temperature_perturb_amplitude",
                        0.0);

  PARTHENON_REQUIRE(params.width_thermo_cgs > 0.0,
                    "problem/pulsed_reconnection/w must be positive.");
  PARTHENON_REQUIRE(params.width_magnetic_cgs > 0.0,
                    "problem/pulsed_reconnection/w_B must be positive.");
  PARTHENON_REQUIRE(params.array_separation_cgs > 0.0,
                    "problem/pulsed_reconnection/array_separation must be positive.");
  PARTHENON_REQUIRE(params.current_peak_MA >= 0.0,
                    "problem/pulsed_reconnection/current_peak_MA must be nonnegative.");
  PARTHENON_REQUIRE(params.azimuthal_mode_number >= 0,
                    "problem/pulsed_reconnection/N must be nonnegative.");

  const auto units = hydro_pkg->Param<Units>("units");
  params.k_b = units.k_boltzmann();
  params.m_bar =
      pin->GetReal("hydro", "mean_molecular_weight") * units.atomic_mass_unit();
  const Real current_peak_ampere = params.current_peak_MA * 1.0e6;
  params.current_field_prefac = 0.2 * current_peak_ampere * units.cm() * units.gauss();
  params.rho_wire = params.rho_wire_cgs * units.g_cm3();
  params.rho_background = params.rho_background_cgs * units.g_cm3();
  params.v0 = params.v0_cgs * units.cm_s();
  params.array_separation = params.array_separation_cgs * units.cm();
  params.width_thermo = params.width_thermo_cgs * units.cm();
  params.width_magnetic = params.width_magnetic_cgs * units.cm();

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
PulsedSourceState EvaluateSourceState(const PulsedSourceParams &params, const Real x,
                                      const Real y) {
  PulsedSourceState state{};
  const Real d = params.array_separation / 2.0;
  Real thermo_profile_sum = 0.0;
  Real density_profile_sum = 0.0;

  for (int A = -1; A <= 1; A += 2) {
    const Real y_center = A * d;
    const Real y_local = y - y_center;
    const Real r2 = SQR(x) + SQR(y_local);
    const Real r = sqrt(r2);
    const Real theta = atan2(y_local, x);
    const Real thermo_profile = GaussianProfile(r, params.width_thermo);
    const Real density_perturbation = AzimuthalThermoPerturbation(
        theta, params.density_perturb_amplitude, params.azimuthal_mode_number);
    const Real temperature_perturbation = AzimuthalThermoPerturbation(
        theta, params.temperature_perturb_amplitude, params.azimuthal_mode_number);
    thermo_profile_sum += thermo_profile * temperature_perturbation;
    density_profile_sum += thermo_profile * density_perturbation;

    if (r > 0.0) {
      const Real inv_r = 1.0 / r;
      const Real xhat = x * inv_r;
      const Real yhat = y_local * inv_r;
      state.v1 += params.v0 * xhat * thermo_profile;
      state.v2 += params.v0 * yhat * thermo_profile;
    }

    const Real enclosed_fraction = 1.0 - exp(-r2 / SQR(params.width_magnetic));
    const Real Bphi_over_r =
        r2 > 0.0 ? params.current_field_prefac * enclosed_fraction / r2
                 : params.current_field_prefac / SQR(params.width_magnetic);
    state.B1 += -Bphi_over_r * y_local;
    state.B2 += Bphi_over_r * x;
  }

  state.rho = params.rho_background + params.rho_wire * density_profile_sum;
  const Real T = params.T_background + params.T_wire * thermo_profile_sum;
  state.pressure = T * params.k_b * state.rho / params.m_bar;
  return state;
}

template <bool INNER_X2>
void PulsedSourceX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto params = g_source_params;
  const bool fine = false;
  const auto nb = IndexRange{0, 0};
  constexpr auto domain = INNER_X2 ? IndexDomain::inner_x2 : IndexDomain::outer_x2;
  auto coords = pmb->coords;

  pmb->par_for_bndry(
      "PulsedSourceX2", nb, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const auto state = EvaluateSourceState(params, x, y);
        cons(IDN, k, j, i) = state.rho;
        cons(IM1, k, j, i) = state.rho * state.v1;
        cons(IM2, k, j, i) = state.rho * state.v2;
        cons(IM3, k, j, i) = state.rho * state.v3;
        cons(IB1, k, j, i) = state.B1;
        cons(IB2, k, j, i) = state.B2;
        cons(IB3, k, j, i) = state.B3;
        cons(IEN, k, j, i) =
            state.pressure / params.gm1 +
            0.5 * (SQR(state.B1) + SQR(state.B2) + SQR(state.B3) +
                   state.rho *
                       (SQR(state.v1) + SQR(state.v2) + SQR(state.v3)));
      });
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
Real GaussianProfile(const Real r, const Real width) {
  const Real exponent = -SQR(r / width);
  return exp(fmax(-700.0, exponent));
}

KOKKOS_INLINE_FUNCTION
Real GaussianProfileDerivative(const Real r, const Real width) {
  return (-2.0 * r / SQR(width)) * GaussianProfile(r, width);
}

KOKKOS_INLINE_FUNCTION
Real AzimuthalThermoPerturbation(const Real theta, const Real p, const int mode_number) {
  const Real phase = static_cast<Real>(mode_number) * theta;
  const Real cos_phase = cos(phase);
  return 1 + p * cos_phase;
}

// Setting up derived fields:
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  // Defining m to pass to the field definition
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  // Field definitions
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
  g_source_params = LoadSourceParams(hydro_pkg, pin);
  g_source_params_initialized = true;
}

// storing the curls just before output
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm) {
  auto &coords = pmb->coords;
  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &w = mbd->Get("prim").data;
  auto &data = pmb->meshblock_data.Get(); // This is for grabbing the meshblocks defined above
  auto hydro_pkg = pmb->packages.Get("Hydro"); // This is for grabbing the calculated diffusivity
  const bool has_ohm_diff = hydro_pkg->AllParams().hasKey("ohm_diff");
  OhmicDiffusivity ohm_diff_dev(Resistivity::none, ResistivityCoeff::none, 0.0, 0.0, 0.0,
                                0.0, 0.0, -1.0, 0.0); // Dummy init
  if (has_ohm_diff) {
    ohm_diff_dev = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  }

  // Get derived fields
  auto &curlBx = data->Get("curlBx").data;
  auto &curlBy = data->Get("curlBy").data;
  auto &curlBz = data->Get("curlBz").data;
  auto &divv = data->Get("divv").data;
  auto &eta_field    = data->Get("eta").data;
  auto &beta_field   = data->Get("beta").data;
  auto &T_field      = data->Get("T").data;
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

  // Getting indices
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Actually computing and storing curl data
  pmb->par_for(
      "pulsed_reconnection::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real inf = std::numeric_limits<Real>::infinity();
        Real term1, term2;
        // curlBx = dBzdy - dBydz
        term1 = (u(IB3,k,j+1,i) - u(IB3,k,j-1,i))/(coords.Xc<2>(j+1)-coords.Xc<2>(j-1));
        term2 = (u(IB2,k+1,j,i) - u(IB2,k-1,j,i))/(coords.Xc<3>(k+1)-coords.Xc<3>(k-1));
        curlBx(k, j, i) = term1 - term2;
        // curlBy = dBxdz - dBzdx
        term1 = (u(IB1,k+1,j,i) - u(IB1,k-1,j,i))/(coords.Xc<3>(k+1)-coords.Xc<3>(k-1));
        term2 = (u(IB3,k,j,i+1) - u(IB3,k,j,i-1))/(coords.Xc<1>(i+1)-coords.Xc<1>(i-1));
        curlBy(k, j, i) = term1 - term2;
        // curlBz = dBydx - dBxdy
        term1 = (u(IB2,k,j,i+1) - u(IB2,k,j,i-1))/(coords.Xc<1>(i+1)-coords.Xc<1>(i-1));
        term2 = (u(IB1,k,j+1,i) - u(IB1,k,j-1,i))/(coords.Xc<2>(j+1)-coords.Xc<2>(j-1));
        curlBz(k, j, i) = term1 - term2;
        // divv = dvx/dx + dvy/dy + dvz/dz
        Real dvx_dx = (w(IV1, k, j, i+1) - w(IV1, k, j, i-1)) / (coords.Xc<1>(i+1) - coords.Xc<1>(i-1));
        Real dvy_dy = (w(IV2, k, j+1, i) - w(IV2, k, j-1, i)) / (coords.Xc<2>(j+1) - coords.Xc<2>(j-1));
        Real dvz_dz = (w(IV3, k+1, j, i) - w(IV3, k-1, j, i)) / (coords.Xc<3>(k+1) - coords.Xc<3>(k-1));
        divv(k, j, i) = dvx_dx + dvy_dy + dvz_dz;

        // Calculating temperature 
        Real rho = u(IDN, k, j, i);
        Real p = w(IPR, k, j, i);
        T_field(k, j, i) = mbar / k_B * p / rho;

        // beta = p / (B^2 / 2) - in Heaviside Lorentz units, this is p / (0.5 * 4pi * B^2)
        beta_field(k, j, i) = p / (0.5 * 4 * M_PI * (SQR(u(IB1,k,j,i)) + SQR(u(IB2,k,j,i)) + SQR(u(IB3,k,j,i))));
        Real eta_val = 0.0;
        if (has_ohm_diff) {
          eta_val = ohm_diff_dev.Get(p, rho);
        }
        eta_field(k, j, i) = eta_val;

        // Resistive diffusion timestep: cfl * fac * min(dx^2 / eta)
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

        // Ohmic heating timestep: cfl * e_int / (eta j^2)
        Real dBzdy = ndim > 1 ? (u(IB3, k, j + 1, i) - u(IB3, k, j - 1, i)) /
                                    (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1))
                              : 0.0;
        Real dBydz = ndim > 2 ? (u(IB2, k + 1, j, i) - u(IB2, k - 1, j, i)) /
                                    (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                              : 0.0;
        Real dBxdz = ndim > 2 ? (u(IB1, k + 1, j, i) - u(IB1, k - 1, j, i)) /
                                    (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1))
                              : 0.0;
        Real dBzdx = (u(IB3, k, j, i + 1) - u(IB3, k, j, i - 1)) /
                     (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
        Real dBydx = (u(IB2, k, j, i + 1) - u(IB2, k, j, i - 1)) /
                     (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1));
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

        // Cooling timestep: cfl * |e / de_dt|
        Real internal_e_spec = p / (rho * gm1);
        Real de_dt_cool = enable_cooling == Cooling::tabular
                              ? cooling_table_obj.DeDt(internal_e_spec, rho)
                              : 0.0;
        Real dt_cool_val = (enable_cooling == Cooling::tabular && de_dt_cool != 0.0 &&
                            internal_e_spec >= eos.GetInternalEFloor())
                               ? fabs(cfl_cool * internal_e_spec / de_dt_cool)
                               : inf;
        dt_cool_local(k, j, i) = dt_cool_val;

        // Hyperbolic timestep estimates using fast magnetosonic and sound speeds
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
          Real lambda_fms_y =
              eos.FastMagnetosonicSpeed(rho, p, u(IB2, k, j, i), u(IB3, k, j, i), u(IB1, k, j, i));
          dt_hyp_fms_val =
              fmin(dt_hyp_fms_val, coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + lambda_fms_y));
          dt_hyp_cs_val =
              fmin(dt_hyp_cs_val, coords.Dxc<2>(k, j, i) / (fabs(prim_local[IV2]) + cs));
        }
        if (ndim > 2) {
          Real lambda_fms_z =
              eos.FastMagnetosonicSpeed(rho, p, u(IB3, k, j, i), u(IB1, k, j, i), u(IB2, k, j, i));
          dt_hyp_fms_val =
              fmin(dt_hyp_fms_val, coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + lambda_fms_z));
          dt_hyp_cs_val =
              fmin(dt_hyp_cs_val, coords.Dxc<3>(k, j, i) / (fabs(prim_local[IV3]) + cs));
        }
        dt_hyp_fms(k, j, i) = cfl_hyp * dt_hyp_fms_val;
        dt_hyp_cs(k, j, i) = cfl_hyp * dt_hyp_cs_val;
      }
  );
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

  // Printing out input values for slurm records
  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "gamma ================== " << pin->GetReal("hydro", "gamma") << std::endl;
    std::cout << "current_peak [MA] ====== " << params.current_peak_MA << std::endl;
    std::cout << "rho_wire(core) [g/cm^3]= " << params.rho_wire_cgs << std::endl;
    std::cout << "rho_background [g/cm^3]= " << params.rho_background_cgs << std::endl;
    std::cout << "T_wire(core) [K] ======= " << params.T_wire << std::endl;
    std::cout << "T_background [K] ======= " << params.T_background << std::endl;
    std::cout << "v0(core) [cm/s] ======== " << params.v0_cgs << std::endl;
    std::cout << "array_separation [cm] == " << params.array_separation_cgs << std::endl;
    std::cout << "thermo width w [cm] ==== " << params.width_thermo_cgs << std::endl;
    std::cout << "magnetic width w_B [cm]  " << params.width_magnetic_cgs << std::endl;
    std::cout << "azimuthal mode N ======= " << params.azimuthal_mode_number << std::endl;
    std::cout << "dens. perturb. amplitude=" << params.density_perturb_amplitude << std::endl;
    std::cout << "temp perturb. amplitude =" << params.temperature_perturb_amplitude
              << std::endl;
    std::cout << "Converted code units:" << std::endl;
    std::cout << "current field prefac === " << params.current_field_prefac << std::endl;
    std::cout << "rho_wire(core) [code] == " << params.rho_wire << std::endl;
    std::cout << "rho_background [code] == " << params.rho_background << std::endl;
    std::cout << "v0(core) [code] ======== " << params.v0 << std::endl;
    std::cout << "array_separation [code]  " << params.array_separation << std::endl;
    std::cout << "thermo width w [code] == " << params.width_thermo << std::endl;
    std::cout << "magnetic width w_B [code]" << params.width_magnetic << std::endl;
    std::cout << "thermo profile ========= exp(-(r / w)^2)" << std::endl;
    std::cout << "thermo perturbation ==== 1 + p*cos(N*theta)" << std::endl;
    std::cout << "magnetic current ======= Jz = I/(pi*w_B^2)*exp(-(r/w_B)^2)"
              << std::endl;
    std::cout << "magnetic field ========= Bphi = 0.2*I[A]/r[cm]*(1-exp(-(r/w_B)^2))"
              << std::endl;
  }

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator::pulsed_reconnection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
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

void PulsedSourceInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PulsedSourceX2<true>(mbd, coarse);
}

void PulsedSourceOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PulsedSourceX2<false>(mbd, coarse);
}

void PulsedDiodeInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  Hydro::BoundaryFunction::DiodeX1BC<parthenon::BoundaryFunction::BCSide::Inner>(mbd,
                                                                                  coarse);
}

void PulsedDiodeOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  Hydro::BoundaryFunction::DiodeX1BC<parthenon::BoundaryFunction::BCSide::Outer>(mbd,
                                                                                  coarse);
}

} // namespace pulsed_reconnection
