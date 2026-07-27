//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file tophat.cpp
//! \brief Problem generator for a pulsed-reconnection-like current sheet 

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

namespace tophat {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

template <typename PackT, typename CoordsT>
KOKKOS_INLINE_FUNCTION Real LinearExtrapolateX1(const PackT &cons, const CoordsT &coords,
                                                const int var, const int k, const int j,
                                                const int ref, const int interior_step,
                                                const Real x_ghost) {
  const int i0 = ref;
  const int i1 = ref + interior_step;
  const Real x0 = coords.template Xc<1>(i0);
  const Real x1 = coords.template Xc<1>(i1);
  const Real u0 = cons(var, k, j, i0);
  const Real u1 = cons(var, k, j, i1);
  const Real dx = x1 - x0;
  return (dx == 0.0) ? u0 : u0 + (u1 - u0) * (x_ghost - x0) / dx;
}

KOKKOS_INLINE_FUNCTION
Real TophatProfile(const Real r, const Real width, const Real falloff) {
  if (r <= width) {
    return 1.0;
  } else if (r >= width + falloff) {
    return 0.0;
  } else {
    const Real x = (r - width) / falloff;
    const Real x3 = x * x * x;
    const Real x4 = x3 * x;
    const Real x5 = x4 * x;
    return 1.0 - 10.0*x3 + 15.0*x4 - 6.0*x5;
  }
}

KOKKOS_INLINE_FUNCTION
Real TophatProfileDerivative(const Real r, const Real width, const Real falloff) {
  if (r <= width || r >= width + falloff) {
    return 0.0;
  } else {
    const Real x = (r - width) / falloff;
    const Real one_minus_x = 1.0 - x;
    return (-30.0 / falloff) * x * x * one_minus_x * one_minus_x;  
  }
}


KOKKOS_INLINE_FUNCTION
Real TophatProfileEnclosedCurrentFraction(const Real r, const Real width, const Real falloff) {
  const Real total_integral =
      0.5 * SQR(width) + 0.5 * width * falloff + (1.0 / 7.0) * SQR(falloff);

  if (r <= 0.0) {
    return 0.0;
  } else if (r <= width) {
    return 0.5 * SQR(r) / total_integral;
  } else if (r >= width + falloff) {
    return 1.0;
  } else {
    const Real x = (r - width) / falloff;
    const Real x2 = x * x;
    const Real x4 = x2 * x2;
    const Real x5 = x4 * x;
    const Real x6 = x5 * x;
    const Real x7 = x6 * x;
    const Real int_profile_dx = x - 2.5 * x4 + 3.0 * x5 - x6;
    return (0.5 * SQR(width) + width * falloff * int_profile_dx +
            (1.0 / 7.0) * SQR(falloff)) /
           total_integral;
  }
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
      "harris::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
  auto &cons = mbd->Get("cons").data;
  auto detected_resistivity_type = pin->GetOrAddString("diffusion", "resistivity_coeff", "none");

  Real gm1  = pin->GetReal("hydro", "gamma") - 1.0;
  Real rho_background = pin->GetOrAddReal("problem/tophat", "rho_background", 1e-6);
  Real T_background = pin->GetOrAddReal("problem/tophat", "T_background", 1e3);
  Real array_separation = pin->GetOrAddReal("problem/tophat", "array_separation", 4.0);
  Real d = 0.5 * array_separation;

  Real core_width = pin->GetOrAddReal("problem/tophat", "core_width", 1.0);
  Real initial_expansion_width = pin->GetOrAddReal("problem/tophat", "initial_expansion_width", 0.5);
  Real rho_array = pin->GetOrAddReal("problem/tophat", "rho_array", 1e-3);
  Real T_array = pin->GetOrAddReal("problem/tophat", "T_array", 1.1e4);

  Real peak_current_MA = pin->GetOrAddReal("problem/tophat", "peak_current_MA", 1.0);
  Real v0 = pin->GetOrAddReal("problem/tophat", "v0", 1.0e6);
  Real azimuthal_mode_number = pin->GetOrAddReal("problem/tophat", "N", 0.0);
  Real density_perturb_amplitude = pin->GetOrAddReal("problem/tophat", "density_perturb_amplitude", 0.0);
  Real temperature_perturb_amplitude = pin->GetOrAddReal("problem/tophat", "temperature_perturb_amplitude", 0.0);

  Real k_b, atomic_mass_unit, m_bar;
  // Grabbing hydro pkg and units objects now...
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  k_b = units.k_boltzmann();
  atomic_mass_unit = units.atomic_mass_unit();
  m_bar = pin->GetReal("hydro", "mean_molecular_weight") * atomic_mass_unit;

  // Printing out input values for slurm records
  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "gamma ..... " << pin->GetReal("hydro", "gamma") << std::endl;
    std::cout << "rho_background [g/cm^3] = " << rho_background << std::endl;
    std::cout << "T_background [K] = " << T_background << std::endl;
    std::cout << "array_separation [cm] = " << array_separation << std::endl;
    std::cout << "core_width [cm] = " << core_width << std::endl;
    std::cout << "initial_expansion [cm] = " << initial_expansion_width << std::endl;
    std::cout << "rho_array [g/cm^3] = " << rho_array << std::endl;
    std::cout << "T_array [K] = " << T_array << std::endl;
    std::cout << "peak_current [MA] = " << peak_current_MA << std::endl;
    std::cout << "v0 [cm/s] = " << v0 << std::endl;
    std::cout << "azimuthal mode N = " << azimuthal_mode_number << std::endl;
    std::cout << "density perturbation amplitude = " << density_perturb_amplitude << std::endl;
    std::cout << "temperature perturbation amplitude = " << temperature_perturb_amplitude << std::endl;
  } 

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator::tophat", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);  // Grabbing local x and y coordinates for the cell
        const Real y = coords.Xc<2>(j);

        // Initializing local field data to the background
        cons(IDN, k, j, i) = rho_background;
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        cons(IB1, k, j, i) = 0.0;
        cons(IB2, k, j, i) = 0.0;
        cons(IB3, k, j, i) = 0.0;
        cons(IEN, k, j, i) = rho_background * k_b * T_background / m_bar / gm1;
        
        // Looping over both arrays (one at y=-1*d, one at y=+1*d)
        for (int A = -1; A <= 1; A += 2) {
          const Real array_y = A * d;
          const Real y_local = y - array_y;
          const Real r2 = SQR(x) + SQR(y_local);
          const Real r = std::sqrt(r2);
          const Real theta = atan2(y_local, x);
          const Real tophat_profile = TophatProfile(r, core_width, initial_expansion_width);
          const Real dtophat_dr = TophatProfileDerivative(r, core_width, initial_expansion_width);
          const Real density_perturb = AzimuthalThermoPerturbation(theta, density_perturb_amplitude, azimuthal_mode_number);
          const Real temperature_perturb = AzimuthalThermoPerturbation(theta, temperature_perturb_amplitude, azimuthal_mode_number);

          const Real local_density = rho_array * tophat_profile * density_perturb;
          const Real local_temperature = T_array * tophat_profile * temperature_perturb;

          // Finding r_hat in cartesian for velocity
          const Real inv_r = (r > 0.0) ? 1.0 / r : 0.0;
          const Real xhat = x * inv_r;
          const Real yhat = y_local * inv_r;

          // Magnetic field - finding enclosed current from the tophat profile
          const Real enclosed_fraction = TophatProfileEnclosedCurrentFraction(r, core_width, initial_expansion_width);
          const Real magnetic_prefactor = 0.2 * peak_current_MA * 1e6;
          const Real Bphi = (r > 0.0) ? magnetic_prefactor * enclosed_fraction / r : 0.0;

          // Adding contributions from this array to the local cell values
          cons(IDN, k, j, i) += local_density;
          cons(IM1, k, j, i) += local_density * v0 * dtophat_dr * xhat;
          cons(IM2, k, j, i) += local_density * v0 * dtophat_dr * yhat;
          cons(IB1, k, j, i) += -Bphi * yhat;  // Bx = -Bphi * sin(theta)
          cons(IB2, k, j, i) += Bphi * xhat;  // By = Bphi * cos(theta)
          cons(IEN, k, j, i) += local_density * k_b * local_temperature / m_bar / gm1;    // Kinetic and magnetic will be included later
        }
        cons(IEN, k, j, i) +=
            0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                   SQR(cons(IB3, k, j, i)) +
                   (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                    SQR(cons(IM3, k, j, i))) /
                       cons(IDN, k, j, i));
      });
}

template <bool INNER_X1>
void TophatDiodeX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior);
  const int ref = INNER_X1 ? range.s : range.e;
  const int interior_step = INNER_X1 ? 1 : -1;

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  auto coords = pmb->coords;
  constexpr auto domain = INNER_X1 ? IndexDomain::inner_x1 : IndexDomain::outer_x1;

  pmb->par_for_bndry(
      "TophatDiodeX1", IndexRange{0, 0}, domain, parthenon::TopologicalElement::CC, coarse,
      false, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real x_ghost = coords.Xc<1>(i);

        const Real rho = LinearExtrapolateX1(cons, coords, IDN, k, j, ref, interior_step, x_ghost);
        Real m1 = LinearExtrapolateX1(cons, coords, IM1, k, j, ref, interior_step, x_ghost);
        const Real m2 = LinearExtrapolateX1(cons, coords, IM2, k, j, ref, interior_step, x_ghost);
        const Real m3 = LinearExtrapolateX1(cons, coords, IM3, k, j, ref, interior_step, x_ghost);
        const Real B1 = LinearExtrapolateX1(cons, coords, IB1, k, j, ref, interior_step, x_ghost);
        const Real B2 = LinearExtrapolateX1(cons, coords, IB2, k, j, ref, interior_step, x_ghost);
        const Real B3 = LinearExtrapolateX1(cons, coords, IB3, k, j, ref, interior_step, x_ghost);

        const int i0 = ref;
        const int i1 = ref + interior_step;
        const Real rho0 = cons(IDN, k, j, i0);
        const Real rho1 = cons(IDN, k, j, i1);
        const Real ke0 =
            rho0 > 0.0
                ? 0.5 * (SQR(cons(IM1, k, j, i0)) + SQR(cons(IM2, k, j, i0)) +
                         SQR(cons(IM3, k, j, i0))) /
                      rho0
                : 0.0;
        const Real ke1 =
            rho1 > 0.0
                ? 0.5 * (SQR(cons(IM1, k, j, i1)) + SQR(cons(IM2, k, j, i1)) +
                         SQR(cons(IM3, k, j, i1))) /
                      rho1
                : 0.0;
        const Real me0 = 0.5 * (SQR(cons(IB1, k, j, i0)) + SQR(cons(IB2, k, j, i0)) +
                                 SQR(cons(IB3, k, j, i0)));
        const Real me1 = 0.5 * (SQR(cons(IB1, k, j, i1)) + SQR(cons(IB2, k, j, i1)) +
                                 SQR(cons(IB3, k, j, i1)));
        const Real eint0 = cons(IEN, k, j, i0) - ke0 - me0;
        const Real eint1 = cons(IEN, k, j, i1) - ke1 - me1;

        const Real x0 = coords.Xc<1>(i0);
        const Real x1 = coords.Xc<1>(i1);
        const Real dx = x1 - x0;
        const Real eint =
            (dx == 0.0) ? eint0 : eint0 + (eint1 - eint0) * (x_ghost - x0) / dx;

        if constexpr (INNER_X1) {
          m1 = fmin(m1, 0.0);
        } else {
          m1 = fmax(m1, 0.0);
        }

        const Real ke = rho > 0.0 ? 0.5 * (SQR(m1) + SQR(m2) + SQR(m3)) / rho : 0.0;
        const Real me = 0.5 * (SQR(B1) + SQR(B2) + SQR(B3));

        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, i) = m1;
        cons(IM2, k, j, i) = m2;
        cons(IM3, k, j, i) = m3;
        cons(IB1, k, j, i) = B1;
        cons(IB2, k, j, i) = B2;
        cons(IB3, k, j, i) = B3;
        cons(IEN, k, j, i) = fmax(0.0, eint) + ke + me;
      });
}

void DiodeInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  TophatDiodeX1<true>(mbd, coarse);
}

void DiodeOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  TophatDiodeX1<false>(mbd, coarse);
}

} // namespace tophat
