#ifndef EOS_TABULATED_EOS_HPP_
#define EOS_TABULATED_EOS_HPP_

#include <cmath>
#include <cstdint>
#include <string>

#include <parthenon/parthenon.hpp>

struct TabulatedEOSState {
  parthenon::Real pressure;
  parthenon::Real temperature;
  parthenon::Real sound_speed;
  parthenon::Real zbar;
  parthenon::Real mu;
  parthenon::Real mu_e;
};

// Three-chart, device-copyable EOS object. All logarithmic data are normalized to
// AthenaPK code units at load time; temperature is normalized to kelvin.
class TabulatedEOS {
 public:
  enum Diagnostic : int {
    rho_low = 0,
    rho_high,
    energy_low,
    energy_high,
    pressure_low,
    pressure_high,
    temperature_low,
    temperature_high,
    invalid_input,
    far_out_of_bounds,
    ndiagnostics
  };

  TabulatedEOS() = default;

  static TabulatedEOS Load(const std::string &filename, parthenon::Real density_factor,
                           parthenon::Real energy_factor,
                           parthenon::Real pressure_factor,
                           parthenon::Real sound_speed_factor,
                           parthenon::Real temperature_factor,
                           parthenon::Real mean_ion_mass);

  KOKKOS_INLINE_FUNCTION
  bool IsInitialized() const { return nrho_e_ > 1; }

  KOKKOS_INLINE_FUNCTION
  TabulatedEOSState StateFromDensityInternalEnergy(const parthenon::Real rho,
                                                   const parthenon::Real eps) const {
    const auto lr = SafeLog(rho, rho_e_lr0_);
    const auto le = SafeLog(eps, rho_e_lx0_);
    RecordBounds(lr, rho_e_lr0_, rho_e_dlr_, nrho_e_, rho_low, rho_high,
                 "rho-e/density");
    RecordBounds(le, rho_e_lx0_, rho_e_dlx_, neps_, energy_low, energy_high,
                 "rho-e/energy");
    const auto lp = Interpolate(log_pressure_, lr, le, rho_e_lr0_, rho_e_dlr_, nrho_e_,
                                rho_e_lx0_, rho_e_dlx_, neps_);
    const auto lt = Interpolate(log_temperature_, lr, le, rho_e_lr0_, rho_e_dlr_,
                                nrho_e_, rho_e_lx0_, rho_e_dlx_, neps_);
    const auto lc = Interpolate(log_sound_speed_, lr, le, rho_e_lr0_, rho_e_dlr_,
                                nrho_e_, rho_e_lx0_, rho_e_dlx_, neps_);
    const auto z = Interpolate(zbar_, lr, le, rho_e_lr0_, rho_e_dlr_, nrho_e_,
                               rho_e_lx0_, rho_e_dlx_, neps_);
    const auto zsafe = z > 1.e-30 ? z : 1.e-30;
    return {Pow10(lp), Pow10(lt), Pow10(lc), z,
            mean_ion_mass_ / (1.0 + zsafe), mean_ion_mass_ / zsafe};
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real InternalEnergyFromDensityPressure(const parthenon::Real rho,
                                                    const parthenon::Real pressure) const {
    const auto lr = SafeLog(rho, rho_p_lr0_);
    const auto lp = SafeLog(pressure, rho_p_lx0_);
    RecordBounds(lr, rho_p_lr0_, rho_p_dlr_, nrho_p_, rho_low, rho_high,
                 "rho-p/density");
    RecordBounds(lp, rho_p_lx0_, rho_p_dlx_, npres_, pressure_low, pressure_high,
                 "rho-p/pressure");
    const auto le = Interpolate(log_energy_rho_p_, lr, lp, rho_p_lr0_, rho_p_dlr_,
                                nrho_p_, rho_p_lx0_, rho_p_dlx_, npres_);
    RecordBounds(le, rho_e_lx0_, rho_e_dlx_, neps_, energy_low, energy_high,
                 "rho-p/result-energy");
    return Pow10(le);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real InternalEnergyFromDensityTemperature(
      const parthenon::Real rho, const parthenon::Real temperature_kelvin) const {
    const auto lr = SafeLog(rho, rho_t_lr0_);
    const auto lt = SafeLog(temperature_kelvin, rho_t_lx0_);
    RecordBounds(lr, rho_t_lr0_, rho_t_dlr_, nrho_t_, rho_low, rho_high,
                 "rho-T/density");
    RecordBounds(lt, rho_t_lx0_, rho_t_dlx_, ntemp_, temperature_low,
                 temperature_high, "rho-T/temperature");
    const auto le = Interpolate(log_energy_rho_t_, lr, lt, rho_t_lr0_, rho_t_dlr_,
                                nrho_t_, rho_t_lx0_, rho_t_dlx_, ntemp_);
    RecordBounds(le, rho_e_lx0_, rho_e_dlx_, neps_, energy_low, energy_high,
                 "rho-T/result-energy");
    return Pow10(le);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real FastMagnetosonicSpeed(const parthenon::Real rho,
                                       const parthenon::Real eps,
                                       const parthenon::Real bx,
                                       const parthenon::Real by,
                                       const parthenon::Real bz) const {
    const auto cs = StateFromDensityInternalEnergy(rho, eps).sound_speed;
    const auto cs2 = cs * cs;
    const auto va2 = (bx * bx + by * by + bz * bz) / rho;
    const auto vax2 = bx * bx / rho;
    const auto sum = cs2 + va2;
    const auto disc = Kokkos::fmax(sum * sum - 4.0 * cs2 * vax2, 0.0);
    return Kokkos::sqrt(0.5 * (sum + Kokkos::sqrt(disc)));
  }

  void ResetDiagnostics() const;
  void CheckDiagnostics(const std::string &where) const;
  std::string Fingerprint() const;

  KOKKOS_INLINE_FUNCTION
  bool IsEnergyMoreThanOneDecadeLow(const parthenon::Real eps) const {
    return eps > 0.0 && Kokkos::isfinite(eps) &&
           Kokkos::log10(eps) < rho_e_lx0_ - 1.0;
  }

 private:
  using View1D = parthenon::ParArray1D<parthenon::Real>;
  using View2D = parthenon::ParArray2D<parthenon::Real>;

  KOKKOS_INLINE_FUNCTION
  parthenon::Real SafeLog(const parthenon::Real x,
                          const parthenon::Real fallback) const {
    if (!(x > 0.0) || !Kokkos::isfinite(x)) {
      const int occurrence = Kokkos::atomic_fetch_add(&diagnostics_(invalid_input), 1);
      if (occurrence < 32) {
        Kokkos::printf("TABULATED_EOS_INVALID value=%.17e fallback_log10=%.17e\n", x,
                       fallback);
      }
      // The host checkpoint will terminate after this kernel. Returning an in-domain
      // coordinate keeps the remainder of the active kernel numerically bounded.
      return fallback;
    }
    return Kokkos::log10(x);
  }

  KOKKOS_INLINE_FUNCTION
  static parthenon::Real Pow10(const parthenon::Real x) {
    return Kokkos::pow(10.0, x);
  }

  KOKKOS_INLINE_FUNCTION
  void RecordBounds(const parthenon::Real x, const parthenon::Real x0,
                    const parthenon::Real dx, const int n, const int ilo,
                    const int ihi, const char *coordinate) const {
    const auto xmax = x0 + dx * (n - 1);
    if (x < x0) {
      Kokkos::atomic_add(&diagnostics_(ilo), 1);
      if (x < x0 - 1.0) {
        const int occurrence =
            Kokkos::atomic_fetch_add(&diagnostics_(far_out_of_bounds), 1);
        if (occurrence < 32) {
          Kokkos::printf(
              "TABULATED_EOS_FAR coordinate=%s side=low log10_query=%.17e "
              "log10_bound=%.17e decades_outside=%.17e\n",
              coordinate, x, x0, x0 - x);
        }
      }
    } else if (x > xmax) {
      Kokkos::atomic_add(&diagnostics_(ihi), 1);
      if (x > xmax + 1.0) {
        const int occurrence =
            Kokkos::atomic_fetch_add(&diagnostics_(far_out_of_bounds), 1);
        if (occurrence < 32) {
          Kokkos::printf(
              "TABULATED_EOS_FAR coordinate=%s side=high log10_query=%.17e "
              "log10_bound=%.17e decades_outside=%.17e\n",
              coordinate, x, xmax, x - xmax);
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  static parthenon::Real Interpolate(const View2D &q, const parthenon::Real x,
                                    const parthenon::Real y,
                                    const parthenon::Real x0,
                                    const parthenon::Real dx, const int nx,
                                    const parthenon::Real y0,
                                    const parthenon::Real dy, const int ny) {
    const auto fx = (x - x0) / dx;
    const auto fy = (y - y0) / dy;
    int i = static_cast<int>(Kokkos::floor(fx));
    int j = static_cast<int>(Kokkos::floor(fy));
    i = i < 0 ? 0 : (i > nx - 2 ? nx - 2 : i);
    j = j < 0 ? 0 : (j > ny - 2 ? ny - 2 : j);
    const auto tx = fx - i;
    const auto ty = fy - j;
    return (1.0 - tx) * ((1.0 - ty) * q(i, j) + ty * q(i, j + 1)) +
           tx * ((1.0 - ty) * q(i + 1, j) + ty * q(i + 1, j + 1));
  }

  View2D log_pressure_, log_temperature_, log_sound_speed_, zbar_;
  View2D log_energy_rho_p_, log_energy_rho_t_;
  parthenon::ParArray1D<int> diagnostics_;

  int nrho_e_ = 0, neps_ = 0, nrho_p_ = 0, npres_ = 0, nrho_t_ = 0, ntemp_ = 0;
  parthenon::Real rho_e_lr0_ = 0, rho_e_dlr_ = 0, rho_e_lx0_ = 0, rho_e_dlx_ = 0;
  parthenon::Real rho_p_lr0_ = 0, rho_p_dlr_ = 0, rho_p_lx0_ = 0, rho_p_dlx_ = 0;
  parthenon::Real rho_t_lr0_ = 0, rho_t_dlr_ = 0, rho_t_lx0_ = 0, rho_t_dlx_ = 0;
  parthenon::Real mean_ion_mass_ = 0;
  // Keep the device-captured object trivially composed of views and scalar values.
  // Formatting the host-side restart fingerprint is deliberately done on demand.
  std::uint64_t fingerprint_ = 0;
};

#endif
