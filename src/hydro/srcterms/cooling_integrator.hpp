// Shared adaptive specific-energy integration for radiative cooling.
#ifndef HYDRO_SRCTERMS_COOLING_INTEGRATOR_HPP_
#define HYDRO_SRCTERMS_COOLING_INTEGRATOR_HPP_

#include "../../eos/tabulated_eos.hpp"
#include "../../main.hpp"
#include "../reservoir_mask.hpp"
#include <algorithm>
#include <cmath>
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <limits>
#include <mesh/domain.hpp>
#include <vector>

namespace cooling {
using namespace parthenon;

// Struct to take one RK step using heun's method to compute 2nd and 1st order estimations
// in y1_h and y1_l
struct RK12Stepper {
  template <typename Function>
  static KOKKOS_INLINE_FUNCTION void
  Step(const parthenon::Real t0, const parthenon::Real h, const parthenon::Real y0,
       Function f, parthenon::Real &y1_h, parthenon::Real &y1_l, bool &valid) {
    const parthenon::Real f_t0_y0 = f(t0, y0, valid);
    y1_l = y0 + h * f_t0_y0;                                 // 1st order
    y1_h = y0 + h / 2. * (f_t0_y0 + f(t0 + h, y1_l, valid)); // 2nd order
  }
  static KOKKOS_INLINE_FUNCTION parthenon::Real OptimalStep(const parthenon::Real h,
                                                            const parthenon::Real err,
                                                            const parthenon::Real tol) {
    return 0.95 * h * pow(tol / err, 2);
  }
};

// Struct to take a 5th and 4th order RK step to compute 5th and 4th order estimations in
// y1_h and y1_l
struct RK45Stepper {
  template <typename Function>
  static KOKKOS_INLINE_FUNCTION void
  Step(const parthenon::Real t0, const parthenon::Real h, const parthenon::Real y0,
       Function f, parthenon::Real &y1_h, parthenon::Real &y1_l, bool &valid) {
    const parthenon::Real k1 = h * f(t0, y0, valid);
    const parthenon::Real k2 = h * f(t0 + 1. / 4. * h, y0 + 1. / 4. * k1, valid);
    const parthenon::Real k3 =
        h * f(t0 + 3. / 8. * h, y0 + 3. / 32. * k1 + 9. / 32. * k2, valid);
    const parthenon::Real k4 =
        h * f(t0 + 12. / 13. * h,
              y0 + 1932. / 2197. * k1 - 7200. / 2197. * k2 + 7296. / 2197. * k3, valid);
    const parthenon::Real k5 =
        h * f(t0 + h,
              y0 + 439. / 216. * k1 - 8. * k2 + 3680. / 513. * k3 - 845. / 4104. * k4,
              valid);
    const parthenon::Real k6 = h * f(t0 + 1. / 2. * h,
                                     y0 - 8. / 27. * k1 + 2. * k2 - 3544. / 2565. * k3 +
                                         1859. / 4104. * k4 - 11. / 40. * k5,
                                     valid); // TODO(forrestglines): Check k2?
    y1_l = y0 + 25. / 216. * k1 + 1408. / 2565. * k3 + 2197. / 4104. * k4 -
           1. / 5. * k5; // 4th order
    y1_h = y0 + 16. / 135. * k1 + 6656. / 12825. * k3 + 28561. / 56430. * k4 -
           9. / 50. * k5 + 2. / 55. * k6; // 5th order
  }
  static KOKKOS_INLINE_FUNCTION parthenon::Real OptimalStep(const parthenon::Real h,
                                                            const parthenon::Real err,
                                                            const parthenon::Real tol) {
    return 0.95 * h * pow(tol / err, 5);
  }
};

enum class CoolIntegrator { undefined, rk12, rk45, townsend };

struct CoolingSettings {
  bool general_eos;
  TabulatedEOS eos;
  Real temperature_floor;
  unsigned int max_iter;
  Real d_e_tol;
  Real cfl;
};

template <typename RKStepper, typename Rate>
void IntegrateCooling(MeshData<Real> *md, const Real dt_, const RKStepper rk_stepper,
                      const Rate cooling_rate, const CoolingSettings settings) {

  const auto dt = dt_; // HACK capturing parameters still broken with Cuda 11.6 ...
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto mask = GetReservoirMask(*hydro_pkg);
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const bool mhd_enabled = fluid != Fluid::euler;
  // Grab member variables for compiler

  const bool general_eos = settings.general_eos;
  const auto eos = settings.eos;
  const auto gm1 = general_eos ? NAN : hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const auto mbar_gm1_over_kb =
      general_eos ? NAN : hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

  const unsigned int max_iter = settings.max_iter;

  const Real min_sub_dt = dt / max_iter;

  const Real d_e_tol = settings.d_e_tol;

  // The caller combines the model and hydro temperature floors.
  const Real temp_floor = settings.temperature_floor;

  // For a general EOS the floor depends on density and is evaluated in the kernel.
  const Real ideal_internal_e_floor = temp_floor / mbar_gm1_over_kb;

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // need to include ghost zones as this source is called prior to the other fluxes when
  // split
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  par_for(
      DEFAULT_LOOP_PATTERN, "Cooling::AdaptiveSource", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real cooling_weight = 1.0 - mask.Weight(coords.Xc<1>(i), coords.Xc<2>(j));
        // A prescribed core has no radiative ODE or cooling-floor update. Return
        // before querying the EOS or computing a rate in that region.
        if (cooling_weight == 0.0) return;
        // Need to use `cons` here as prim may still contain state at t_0;
        const Real rho = cons(IDN, k, j, i);
        // TODO(pgrete) with potentially more EOS, a separate get_pressure (or similar)
        // function could be useful.
        Real internal_e =
            cons(IEN, k, j, i) - 0.5 *
                                     (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                      SQR(cons(IM3, k, j, i))) /
                                     rho;
        if (mhd_enabled) {
          internal_e -= 0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                               SQR(cons(IB3, k, j, i)));
        }
        internal_e /= rho;
        const Real internal_e_initial = internal_e;
        const Real internal_e_floor = general_eos
                                          ? eos.InternalEnergyFromDensityTemperature(
                                                rho, temp_floor)
                                          : ideal_internal_e_floor;

        bool dedt_valid = true;

        // Wrap DeDt into a functor for the RKStepper
        auto DeDt_wrapper = [&](const Real t, const Real e, bool &valid) {
          // Once a trial stage fails, avoid further EOS queries in that RK step.
          if (!valid) return Real(0.0);
          // A cooling-only trial cannot gain energy. Large explicit trials can
          // otherwise overshoot the EOS domain before the error estimate rejects them.
          if (!std::isfinite(e) || e <= 0.0 ||
              e > internal_e_initial * (1.0 + 1.e-12)) {
            valid = false;
            return Real(0.0);
          }
          if (e <= internal_e_floor) return Real(0.0);
          return cooling_weight * cooling_rate.DeDt(e, rho, valid);
        };

        Real sub_t = 0; // current subcycle time
        // Try full dt. If error is too large adaptive timestepping will reduce sub_dt
        Real sub_dt = dt;

        // Apply the floor before asking the cooling table to convert internal energy to
        // temperature. Apart from avoiding an out-of-domain EOS query, this ensures that
        // an already under-floored state is repaired even when no cooling is performed.
        if (internal_e_initial <= internal_e_floor) {
          internal_e = internal_e_floor;
          cons(IEN, k, j, i) += rho * (internal_e - internal_e_initial);
          prim(IPR, k, j, i) =
              general_eos
                  ? eos.StateFromDensityInternalEnergy(rho, internal_e).pressure
                  : rho * internal_e * gm1;
          return;
        }

        // Check if cooling is actually happening, e.g., when T is below T_cool_min.
        const Real dedt_initial = DeDt_wrapper(0.0, internal_e_initial, dedt_valid);
        if (dedt_initial == 0.0) {
          return;
        }

        // Use minumum subcycle timestep when d_e_tol == 0
        if (d_e_tol == 0) {
          sub_dt = min_sub_dt;
        }

        unsigned int sub_iter = 0;
        // check for dedt != 0.0 required in case cooling floor it hit during subcycling
        while ((sub_t * (1 + 1e-12) < dt) &&
               (DeDt_wrapper(sub_t, internal_e, dedt_valid) != 0.0)) {

          if (sub_iter > max_iter) {
            // Due to sub_dt >= min_dt, this error should never happen
            PARTHENON_FAIL(
                "FATAL ERROR in [IntegrateCooling]: Sub "
                "cycles exceed max_iter (This should be impossible)");
          }

          // Next higher order estimate
          Real internal_e_next_h;
          // Error in estimate of higher order
          Real d_e_err;
          // Number of attempts on this subcycle
          unsigned int sub_attempt = 0;
          // Whether to reattempt this subcycle
          bool reattempt_sub = true;
          do {
            // Next lower order estimate
            Real internal_e_next_l;
            d_e_err = 0.0;
            // Do one dual order RK step
            dedt_valid = true;
            RKStepper::Step(sub_t, sub_dt, internal_e, DeDt_wrapper, internal_e_next_h,
                            internal_e_next_l, dedt_valid);

            sub_attempt++;

            if (!dedt_valid) {
              if (sub_dt == min_sub_dt) {
                // Cooling is so fast that even the minimum subcycle dt would lead to
                // negative internal energy -- so just cool to the floor of the cooling
                // table
                sub_dt = (dt - sub_t);
                internal_e_next_h = internal_e_floor;
                reattempt_sub = false;
              } else {
                reattempt_sub = true;
                sub_dt = min_sub_dt;
              }
            } else {

              // Compute error
              d_e_err = fabs((internal_e_next_h - internal_e_next_l) / internal_e_next_h);

              reattempt_sub = false;
              // Accepting or reattempting the subcycle:
              //
              // -If the error is small, accept the subcycle
              //
              // -If the error on the subcycle is too high, compute a new time
              // step to reattempt the subcycle
              //   -But if the new time step is smaller than the minimum subcycle
              //   time step (total step duration/ max iterations), just use the
              //   minimum subcycle time step instead

              if (std::isnan(d_e_err)) {
                reattempt_sub = true;
                sub_dt = min_sub_dt;
              } else if (d_e_err >= d_e_tol && sub_dt > min_sub_dt) {
                // Reattempt this subcycle
                reattempt_sub = true;
                // Error was too high, shrink the timestep
                if (d_e_tol == 0) {
                  sub_dt = min_sub_dt;
                } else {
                  sub_dt = RKStepper::OptimalStep(sub_dt, d_e_err, d_e_tol);
                }
                // Don't drop timestep under maximum iteration count
                if (sub_dt < min_sub_dt || sub_attempt >= max_iter) {
                  sub_dt = min_sub_dt;
                }
              }
            }

          } while (reattempt_sub);
          // Accept this subcycle
          sub_t += sub_dt;

          internal_e = internal_e_next_h;

          // skip to the end of subcycling if error is 0 (very unlikely)
          if (d_e_err == 0) {
            sub_dt = dt - sub_t;
          } else {
            // Grow the timestep
            // (or shrink in case d_e_err >= d_e_tol and sub_dt is already at min_sub_dt)
            sub_dt = RKStepper::OptimalStep(sub_dt, d_e_err, d_e_tol);
          }

          if (d_e_tol == 0) {
            sub_dt = min_sub_dt;
          }

          // Don't drop timestep under the minimum step size
          sub_dt = std::max(sub_dt, min_sub_dt);

          // Limit by end time
          sub_dt = std::min(sub_dt, dt - sub_t);

          sub_iter++;
        }

        // If cooled below floor, reset to floor value.
        // This could happen if the floor value is larger than the lower end of the
        // cooling table or if they are close and the last subcycle in the cooling above
        // the lower end pushed the temperature below the lower end (and the floor).
        internal_e = (internal_e > internal_e_floor) ? internal_e : internal_e_floor;

        // Remove the cooling from the total energy density
        cons(IEN, k, j, i) += rho * (internal_e - internal_e_initial);
        // Latter technically not required if no other tasks follows before
        // ConservedToPrim conversion, but keeping it for now (better safe than sorry).
        prim(IPR, k, j, i) = general_eos
                                 ? eos.StateFromDensityInternalEnergy(rho, internal_e)
                                       .pressure
                                 : rho * internal_e * gm1;
      });
  if (general_eos) eos.CheckDiagnostics("radiative cooling source");
}

template <typename Rate>
Real EstimateCoolingTimeStep(MeshData<Real> *md, const Rate cooling_rate,
                             const CoolingSettings settings) {
  if (settings.cfl <= 0.0) {
    return std::numeric_limits<Real>::max();
  }

  if (isnan(settings.cfl) || isinf(settings.cfl)) {
    return std::numeric_limits<Real>::infinity();
  }

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto mask = GetReservoirMask(*hydro_pkg);
  const auto mask_diag = md->PackVariables(
      std::vector<std::string>{mask.enabled ? "reservoir_mask_diagnostics" : "prim"});

  const bool general_eos = settings.general_eos;
  const auto eos = settings.eos;
  const auto gm1 = general_eos ? NAN : hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const auto mbar_gm1_over_kb =
      general_eos ? NAN : hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

  // Determine the cooling floor, whichever is higher of the cooling table floor
  // or fluid solver floor
  const Real temp_floor = settings.temperature_floor;

  const Real ideal_internal_e_floor = temp_floor / mbar_gm1_over_kb;

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_cooling_time = std::numeric_limits<Real>::infinity();
  Kokkos::Min<Real> reducer_min(min_cooling_time);

  Kokkos::parallel_reduce(
      "Cooling::TimeStep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &thread_min_cooling_time) {
        auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real cooling_weight = 1.0 - mask.Weight(coords.Xc<1>(i), coords.Xc<2>(j));
        if (mask.enabled) mask_diag(b, 3, k, j, i) = std::numeric_limits<Real>::max();
        if (cooling_weight == 0.0) return;

        const Real rho = prim(IDN, k, j, i);
        const Real pres = prim(IPR, k, j, i);

        const Real internal_e = general_eos
                                    ? eos.InternalEnergyFromDensityPressure(rho, pres)
                                    : pres / (rho * gm1);
        const Real internal_e_floor = general_eos
                                          ? eos.InternalEnergyFromDensityTemperature(
                                                rho, temp_floor)
                                          : ideal_internal_e_floor;

        // Do not sample the cooling table below the configured temperature floor. The
        // general-EOS DeDt path converts internal energy through the rho-e table.
        const Real de_dt = internal_e > internal_e_floor
                               ? cooling_weight * cooling_rate.DeDt(internal_e, rho)
                               : 0.0;

        // Compute cooling time
        // If de_dt is zero (temperature is smaller than lower end of cooling table) or
        // current temp is below floor, use infinite cooling time
        const Real cooling_time = ((de_dt == 0) || (internal_e < internal_e_floor))
                                      ? std::numeric_limits<Real>::infinity()
                                      : fabs(internal_e / de_dt);

        if (mask.enabled && std::isfinite(cooling_time))
          mask_diag(b, 3, k, j, i) = settings.cfl * cooling_time;
        thread_min_cooling_time = std::min(cooling_time, thread_min_cooling_time);
      },
      reducer_min);

  if (general_eos) eos.CheckDiagnostics("radiative cooling timestep estimate");

  return settings.cfl * min_cooling_time;
}

} // namespace cooling
#endif
