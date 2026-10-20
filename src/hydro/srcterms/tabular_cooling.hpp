//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tabular_cooling.hpp
//  \brief Class for defining tabular cooling
#ifndef HYDRO_SRCTERMS_TABULAR_COOLING_HPP_
#define HYDRO_SRCTERMS_TABULAR_COOLING_HPP_

// C++ headers
#include <fstream>   // stringstream
#include <iterator>  // istream_iterator
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // string
#include <vector>    // vector

// Parthenon headers
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <mesh/domain.hpp>
#include <mesh/meshblock_pack.hpp>
#include <outputs/io_wrapper.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../../eos/tabulated_eos.hpp"
#include "../../units.hpp"
#include "cooling_integrator.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace cooling {

class CoolingTableObj {
  /************************************************************
   *  Cooling Table Object, for interpolating a cooling rate out of a cooling
   *  table. Currently assumes evenly space log_temperatures in cooling table
   *
   *  Lightweight object intended for inlined computation within kernels
   ************************************************************/
 private:
  // Log cooling coefficient
  parthenon::ParArray1D<parthenon::Real> log_lambdas_;

  // Spacing of cooling table
  // TODO: assumes evenly spaced cooling table
  parthenon::Real log_temp_start_, log_temp_final_, d_log_temp_;
  unsigned int n_temp_;

  // Mean molecular mass * ( adiabatic_index -1) / boltzmann_constant
  parthenon::Real mbar_gm1_over_k_B_;

  // Square of the selected number density per unit mass, e.g. (n_e / rho)^2
  parthenon::Real number_density_per_mass_squared_;
  TabulatedEOS eos_;
  parthenon::Real atomic_mass_unit_;

 public:
  CoolingTableObj()
      : log_lambdas_(), log_temp_start_(NAN), log_temp_final_(NAN), d_log_temp_(NAN),
        n_temp_(0), mbar_gm1_over_k_B_(NAN), number_density_per_mass_squared_(NAN),
        eos_(), atomic_mass_unit_(NAN) {}
  CoolingTableObj(const parthenon::ParArray1D<parthenon::Real> log_lambdas,
                  const parthenon::Real log_temp_start,
                  const parthenon::Real log_temp_final, const parthenon::Real d_log_temp,
                  const unsigned int n_temp, const parthenon::Real mbar_over_kb,
                  const parthenon::Real adiabatic_index,
                  const parthenon::Real number_density_per_mass_squared)
      : log_lambdas_(log_lambdas), log_temp_start_(log_temp_start),
        log_temp_final_(log_temp_final), d_log_temp_(d_log_temp), n_temp_(n_temp),
        mbar_gm1_over_k_B_(mbar_over_kb * (adiabatic_index - 1)),
        number_density_per_mass_squared_(number_density_per_mass_squared), eos_(),
        atomic_mass_unit_(NAN) {}
  CoolingTableObj(const parthenon::ParArray1D<parthenon::Real> log_lambdas,
                  const parthenon::Real log_temp_start,
                  const parthenon::Real log_temp_final, const parthenon::Real d_log_temp,
                  const unsigned int n_temp, const TabulatedEOS eos,
                  const parthenon::Real atomic_mass_unit)
      : log_lambdas_(log_lambdas), log_temp_start_(log_temp_start),
        log_temp_final_(log_temp_final), d_log_temp_(d_log_temp), n_temp_(n_temp),
        mbar_gm1_over_k_B_(NAN), number_density_per_mass_squared_(NAN), eos_(eos),
        atomic_mass_unit_(atomic_mass_unit) {}

  // Interpolate a cooling rate from the table
  // from internal energy density and density
  KOKKOS_INLINE_FUNCTION parthenon::Real
  DeDt(const parthenon::Real &e, const parthenon::Real &rho, bool &is_valid) const {
    using namespace parthenon;

    if (e < 0 || std::isnan(e)) {
      is_valid = false;
      return 0;
    }

    const auto state = eos_.IsInitialized()
                           ? eos_.StateFromDensityInternalEnergy(rho, e)
                           : TabulatedEOSState{NAN, mbar_gm1_over_k_B_ * e, NAN, NAN,
                                               NAN, NAN};
    const Real temp = state.temperature;
    const Real log_temp = log10(temp);
    Real log_lambda;
    if (log_temp < log_temp_start_) {
      return 0;
    } else if (log_temp > log_temp_final_) {
      // Above table
      // Return de/dt
      // TODO(forrestglines):Currently free-free cooling is used for
      // temperatures above the table. This behavior could be generalized via
      // templates
      log_lambda = 0.5 * log_temp - 0.5 * log_temp_final_ + log_lambdas_(n_temp_ - 1);
    } else {
      // Inside table, interpolate assuming log spaced temperatures

      // Determine where temp is in the table
      const unsigned int i_temp = std::min(
          n_temp_ - 2,
          static_cast<unsigned int>((log_temp - log_temp_start_) / d_log_temp_));
      const Real log_temp_i = log_temp_start_ + d_log_temp_ * i_temp;

      // log_temp should be between log_temps[i_temp] and log_temps[i_temp+1]
      PARTHENON_REQUIRE(log_temp >= log_temp_i && log_temp <= log_temp_i + d_log_temp_,
                        "FATAL ERROR in [CoolingTable::DeDt]: Failed to find log_temp");

      const Real log_lambda_i = log_lambdas_(i_temp);
      const Real log_lambda_ip1 = log_lambdas_(i_temp + 1);

      // Linearly interpolate lambda at log_temp
      log_lambda = log_lambda_i + (log_temp - log_temp_i) *
                                      (log_lambda_ip1 - log_lambda_i) / d_log_temp_;
    }
    // Return de/dt
    const Real lambda = pow(10., log_lambda);
    // Cooling tables are normalized as n_e^2 Lambda(T).  For a general EOS mu_e is
    // state dependent, so n_e/rho must be evaluated at every RHS call.
    const Real ne_per_mass = eos_.IsInitialized()
                                 ? 1.0 / (state.mu_e * atomic_mass_unit_)
                                 : std::sqrt(number_density_per_mass_squared_);
    const Real de_dt = -lambda * SQR(ne_per_mass) * rho;
    return de_dt;
  }

  KOKKOS_INLINE_FUNCTION parthenon::Real DeDt(const parthenon::Real &e,
                                              const parthenon::Real &rho) const {
    bool is_valid = true;
    return DeDt(e, rho, is_valid);
  }
};

class TabularCooling {
 private:
  // Defines uniformly spaced log temperature range of the table
  unsigned int n_temp_;
  parthenon::Real log_temp_start_, log_temp_final_, d_log_temp_, lambda_final_;

  // Table of log cooling rates
  // TODO(forrestglines): Make log_lambdas_ explicitly a texture cache array, use CUDA to
  // interpolate directly
  // Log versions are used in subcyling cooling where cooling rates are interpolated
  // Non-log versions are used for Townsend cooling
  parthenon::ParArray1D<parthenon::Real> log_lambdas_;
  parthenon::ParArray1D<parthenon::Real> lambdas_;
  parthenon::ParArray1D<parthenon::Real> temps_;
  // Townsend cooling temporal evolution function
  parthenon::ParArray1D<parthenon::Real> townsend_Y_k_;
  // Townsend cooling power law indices
  parthenon::ParArray1D<parthenon::Real> townsend_alpha_k_;

  CoolIntegrator integrator_;

  // Temperature floor (assumed in Kelvin and only used in cooling function)
  // This is either the temperature floor used by the hydro method or the
  // lowest temperature in the cooling table (assuming zero cooling below the
  // table), whichever temperature is higher
  parthenon::Real T_floor_;

  // Square of the selected number density per unit mass. This converts the tabulated
  // volumetric loss, n^2 Lambda(T), to a specific-energy loss rate.
  parthenon::Real number_density_per_mass_squared_;

  // Maximum number of iterations/subcycles
  unsigned int max_iter_;

  // Cooling CFL
  parthenon::Real cooling_time_cfl_;

  // Minimum timestep that the cooling may limit the simulation timestep
  // Use nonpositive values to disable
  parthenon::Real min_cooling_timestep_;

  // Tolerances
  parthenon::Real d_log_temp_tol_, d_e_tol_;

  CoolingTableObj cooling_table_obj_;
  bool general_eos_ = false;
  TabulatedEOS eos_;

  CoolingSettings IntegrationSettings() const {
    return {general_eos_, eos_, std::max(T_floor_, std::pow(10.0, log_temp_start_)),
            max_iter_, d_e_tol_, cooling_time_cfl_};
  }

 public:
  TabularCooling(parthenon::ParameterInput *pin,
                 std::shared_ptr<parthenon::StateDescriptor> hydro_pkg);

  void SrcTerm(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt) const;

  // Townsend 2009 exact integration scheme
  void TownsendSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real dt) const;

  // (Adaptive) subcyling using a fixed integration scheme
  template <typename RKStepper>
  void SubcyclingFixedIntSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                 const parthenon::Real dt,
                                 const RKStepper rk_stepper) const;

  parthenon::Real EstimateTimeStep(parthenon::MeshData<parthenon::Real> *md) const;

  // Get a lightweight object for computing cooling rate from the cooling table
  const CoolingTableObj GetCoolingTableObj() const { return cooling_table_obj_; }

  void TestCoolingTable(parthenon::ParameterInput *pin) const;
};

} // namespace cooling

#endif // HYDRO_SRCTERMS_TABULAR_COOLING_HPP_
