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
#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/diffusion/diffusion.hpp" // For storing eta later
#include "../hydro/srcterms/tabular_cooling.hpp"

namespace pulsed_reconnection {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

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
                                0.0, 0.0, -1.0); // Dummy init
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

  Real gm1  = pin->GetReal("hydro", "gamma") - 1.0;
  Real B0 = pin->GetOrAddReal("problem/pulsed_reconnection", "B0", 5.0e5);
  Real rho_wire = pin->GetOrAddReal("problem/pulsed_reconnection", "rho_wire", 1e-3);
  Real rho_background = pin->GetOrAddReal("problem/pulsed_reconnection", "rho_background", 1e-6);
  Real T_wire = pin->GetOrAddReal("problem/pulsed_reconnection", "T_wire", 1.1e4);
  Real T_background = pin->GetOrAddReal("problem/pulsed_reconnection", "T_background", 1e2);
  Real v0 = pin->GetOrAddReal("problem/pulsed_reconnection", "v0", 1.0e6);
  Real array_separation =
      pin->GetOrAddReal("problem/pulsed_reconnection", "array_separation", 4.0);
  Real width_thermo = pin->GetOrAddReal("problem/pulsed_reconnection", "w", 1.0);
  Real width_magnetic = pin->GetOrAddReal(
      "problem/pulsed_reconnection", "w_B",
      pin->GetOrAddReal("problem/pulsed_reconnection", "w_magnetic", width_thermo));
  int azimuthal_mode_number =
      pin->GetOrAddInteger("problem/pulsed_reconnection", "N", 0);
  Real perturb_amplitude = pin->GetOrAddReal("problem/pulsed_reconnection", "perturb_amplitude", 0.0);
  PARTHENON_REQUIRE(width_thermo > 0.0, "problem/pulsed_reconnection/w must be positive.");
  PARTHENON_REQUIRE(width_magnetic > 0.0,
                    "problem/pulsed_reconnection/w_B must be positive.");
  PARTHENON_REQUIRE(array_separation > 0.0,
                    "problem/pulsed_reconnection/array_separation must be positive.");
  PARTHENON_REQUIRE(azimuthal_mode_number >= 0,
                    "problem/pulsed_reconnection/N must be nonnegative.");
  

  Real k_b, atomic_mass_unit, m_bar;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  k_b = units.k_boltzmann();
  atomic_mass_unit = units.atomic_mass_unit();
  m_bar = pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;

  // Printing out input values for slurm records
  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "gamma ================== " << pin->GetReal("hydro", "gamma") << std::endl;
    std::cout << "B0 ===================== " << B0 << std::endl;
    std::cout << "rho_wire(core) [g/cm^3] " << rho_wire << std::endl;
    std::cout << "rho_background [g/cm^3]= " << rho_background << std::endl;
    std::cout << "T_wire(core) [K] ======= " << T_wire << std::endl;
    std::cout << "T_background [K] ======= " << T_background << std::endl;
    std::cout << "v0(core) [cm/s] ======== " << v0 << std::endl;
    std::cout << "array_separation [cm] == " << array_separation << std::endl;
    std::cout << "thermo width w [cm] ==== " << width_thermo << std::endl;
    std::cout << "magnetic width w_B [cm]  " << width_magnetic << std::endl;
    std::cout << "azimuthal mode N ======= " << azimuthal_mode_number << std::endl;
    std::cout << "perturb. amplitude ===== " << perturb_amplitude << std::endl;
    std::cout << "thermo profile ========= exp(-(r / w)^2)" << std::endl;
    std::cout << "thermo perturbation ==== 1 + p*cos(N*theta)" << std::endl;
    std::cout << "magnetic profile ======= exp(-(r / w_B)^2)" << std::endl;
    std::cout << "B = cross(zhat, B0 * grad(phi)) with phi built from the magnetic Gaussian"
              << std::endl;
  }

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator::pulsed_reconnection", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Gaussian exploding-wire profile centered on each array element.
        Real v1 = 0.0;
        Real v2 = 0.0;
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real d = array_separation / 2.0;
        Real thermo_profile_sum = 0.0;
        Real dphi_dx = 0.0;
        Real dphi_dy = 0.0;

        // Two wire arrays centered at x=0, y=+/-array_separation/2 with radial outflow.
        for (int A = -1; A <= 1; A += 2) {
          Real y_center = A * d;
          Real y_local = y - y_center;
          Real r = sqrt(SQR(x) + SQR(y_local));
          Real theta = atan2(y_local, x);
          Real thermo_profile = GaussianProfile(r, width_thermo);
          Real dmagnetic_profile_dr = GaussianProfileDerivative(r, width_magnetic);
          Real thermo_perturbation =
              AzimuthalThermoPerturbation(theta, perturb_amplitude, azimuthal_mode_number);

          thermo_profile_sum += thermo_profile * thermo_perturbation;

          if (r > 0.0) {
            const Real inv_r = 1.0 / r;
            const Real xhat = x * inv_r;
            const Real yhat = y_local * inv_r;
            v1 += v0 * xhat * thermo_profile;
            v2 += v0 * yhat * thermo_profile;
            dphi_dx += dmagnetic_profile_dr * xhat;
            dphi_dy += dmagnetic_profile_dr * yhat;
          }
        }

        Real rho = rho_background + rho_wire * thermo_profile_sum;
        Real T = T_background + T_wire * thermo_profile_sum;
        Real P = T * k_b * rho / m_bar;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * v1;
        u(IM2, k, j, i) = rho * v2;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = -B0 * dphi_dy;
        u(IB2, k, j, i) = B0 * dphi_dx;
        u(IB3, k, j, i) = 0.0;

        u(IEN, k, j, i) = P / gm1 +
                          0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) +
                                 SQR(u(IB3, k, j, i)) +
                                 (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) +
                                  SQR(u(IM3, k, j, i))) /
                                     u(IDN, k, j, i));
      });
}
 // namespace pulsed_reconnection
}
