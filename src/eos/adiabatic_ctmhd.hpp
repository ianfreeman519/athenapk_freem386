#ifndef EOS_ADIABATIC_CTMHD_HPP_
#define EOS_ADIABATIC_CTMHD_HPP_
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers
#include <limits> // std::numeric_limits<float>

// Parthenon headers
#include "mesh/mesh.hpp"

// Athena headers
#include "../main.hpp"
#include "eos.hpp"
#include "tabulated_eos.hpp"

using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::Real;

class AdiabaticCTMHDEOS : public EquationOfState {
 public:
  AdiabaticCTMHDEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
                     Real velocity_ceiling, Real internal_e_ceiling, Real gamma)
      : EquationOfState(pressure_floor, density_floor, internal_e_floor, velocity_ceiling,
                        internal_e_ceiling),
        gamma_{gamma} {}

  AdiabaticCTMHDEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
                    Real velocity_ceiling, Real internal_e_ceiling,
                    const TabulatedEOS &table, Real temperature_floor = -1.0,
                    Real temperature_ceiling = std::numeric_limits<Real>::infinity())
      : EquationOfState(pressure_floor, density_floor, internal_e_floor, velocity_ceiling,
                        internal_e_ceiling, temperature_floor, temperature_ceiling),
        gamma_{NAN}, table_{table} {}

  void ConservedToPrimitive(MeshData<Real> *md) const override;

  KOKKOS_INLINE_FUNCTION
  Real GetGamma() const { return gamma_; }
  KOKKOS_INLINE_FUNCTION bool IsTabulated() const { return table_.IsInitialized(); }
  KOKKOS_INLINE_FUNCTION TabulatedEOSState StateFromDensityInternalEnergy(
      Real rho, Real eps) const {
    if (IsTabulated()) return table_.StateFromDensityInternalEnergy(rho, eps);
    const Real p = (gamma_ - 1.0) * rho * eps;
    return {p, NAN, std::sqrt(gamma_ * p / rho), NAN, NAN, NAN};
  }
  KOKKOS_INLINE_FUNCTION Real InternalEnergyFromDensityPressure(Real rho, Real p) const {
    return IsTabulated() ? table_.InternalEnergyFromDensityPressure(rho, p)
                         : p / ((gamma_ - 1.0) * rho);
  }
  KOKKOS_INLINE_FUNCTION Real InternalEnergyFromDensityTemperature(Real rho,
                                                                   Real T) const {
    return table_.InternalEnergyFromDensityTemperature(rho, T);
  }
  void CheckDiagnostics(const std::string &where) const {
    if (IsTabulated()) table_.CheckDiagnostics(where);
  }

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
  // \brief returns adiabatic sound speed given vector of primitive variables
  // TODO(pgrete): need to fix idx defs
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(const Real prim[NHYDRO]) const {
    if (!IsTabulated()) return std::sqrt(gamma_ * prim[IPR] / prim[IDN]);
    const auto eps = InternalEnergyFromDensityPressure(prim[IDN], prim[IPR]);
    return StateFromDensityInternalEnergy(prim[IDN], eps).sound_speed;
  }
  // fast magnetosonic speed function for adiabatic EOS
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(const Real d, const Real p, const Real bx, const Real by,
                             const Real bz) const {
    if (IsTabulated()) {
      const auto eps = InternalEnergyFromDensityPressure(d, p);
      return table_.FastMagnetosonicSpeed(d, eps, bx, by, bz);
    }
    Real asq = gamma_ * p;
    Real ct2 = by * by + bz * bz;
    Real qsq = bx * bx + ct2 + asq;
    Real tmp = bx * bx + ct2 - asq;
    return std::sqrt(0.5 * (qsq + std::sqrt(tmp * tmp + 4.0 * asq * ct2)) / d);
  }
  //

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::ConsToPrim(View4D cons, View4D prim, const int& k, const
  // int& j, const int& i) \brief Fills an array of primitives given an array of
  // conserveds, potentially updating the conserved with floors
  template <typename View4D>
  KOKKOS_INLINE_FUNCTION void ConsToPrim(View4D cons, View4D prim, const int &nhydro,
                                         const int &nscalars, const int &k, const int &j,
                                         const int &i, const int rank = -1,
                                         const int block = -1,
                                         const bool is_ghost = false) const {
    auto gam = GetGamma();
    auto gm1 = gam - 1.0;
    auto density_floor_ = GetDensityFloor();
    auto pressure_floor_ = GetPressureFloor();
    auto e_floor_ = GetInternalEFloor();

    auto velocity_ceiling_ = GetVelocityCeiling();
    auto e_ceiling_ = GetInternalECeiling();

    Real &u_d = cons(IDN, k, j, i);
    Real &u_m1 = cons(IM1, k, j, i);
    Real &u_m2 = cons(IM2, k, j, i);
    Real &u_m3 = cons(IM3, k, j, i);
    Real &u_e = cons(IEN, k, j, i);
    Real &u_b1 = cons(IB1, k, j, i);
    Real &u_b2 = cons(IB2, k, j, i);
    Real &u_b3 = cons(IB3, k, j, i);

    Real &w_d = prim(IDN, k, j, i);
    Real &w_vx = prim(IV1, k, j, i);
    Real &w_vy = prim(IV2, k, j, i);
    Real &w_vz = prim(IV3, k, j, i);
    Real &w_p = prim(IPR, k, j, i);
    Real &w_Bx = prim(IB1, k, j, i);
    Real &w_By = prim(IB2, k, j, i);
    Real &w_Bz = prim(IB3, k, j, i);

    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative density is encountered.
    PARTHENON_REQUIRE(u_d > 0.0 || density_floor_ > 0.0,
                      "Got negative density. Consider enabling first-order flux "
                      "correction or setting a reasonble density floor.");
    // apply density floor, without changing momentum or energy
    u_d = (u_d > density_floor_) ? u_d : density_floor_;
    w_d = u_d;

    Real di = 1.0 / u_d;
    w_vx = u_m1 * di;
    w_vy = u_m2 * di;
    w_vz = u_m3 * di;

    w_Bx = u_b1;
    w_By = u_b2;
    w_Bz = u_b3;

    Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    Real e_B = 0.5 * (SQR(u_b1) + SQR(u_b2) + SQR(u_b3));
    Real eps = (u_e - e_k - e_B) * di;
    const Real raw_eps = eps;
    const bool raw_energy_far_low =
        IsTabulated() && table_.IsEnergyMoreThanOneDecadeLow(raw_eps);
    if (IsTabulated()) {
      if (!(eps > 0.0) && pressure_floor_ > 0.0)
        eps = InternalEnergyFromDensityPressure(u_d, pressure_floor_);
      if (GetTemperatureFloor() > 0.0) {
        const Real target =
            InternalEnergyFromDensityTemperature(u_d, GetTemperatureFloor());
        if (!(eps >= target)) eps = target;
      }
      u_e = u_d * eps + e_k + e_B;
      if (raw_energy_far_low) {
        const Real reconstructed_eps = (u_e - e_k - e_B) * di;
        Kokkos::printf(
            "CTMHD_EOS_LOW rank=%d block=%d k=%d j=%d i=%d ghost=%d "
            "rho=%.17e raw_eps=%.17e floor_eps=%.17e reconstructed_eps=%.17e "
            "Etot=%.17e Ek=%.17e EB=%.17e Eth=%.17e "
            "momentum=(%.17e,%.17e,%.17e) B=(%.17e,%.17e,%.17e)\n",
            rank, block, k, j, i, static_cast<int>(is_ghost), u_d, raw_eps, eps,
            reconstructed_eps, u_e, e_k, e_B, u_e - e_k - e_B, u_m1, u_m2, u_m3,
            u_b1, u_b2, u_b3);
      }
      w_p = StateFromDensityInternalEnergy(u_d, eps).pressure;
    } else {
      w_p = gm1 * (u_e - e_k - e_B);
    }

    // apply velocity ceiling. By default ceiling is std::numeric_limits<Real>::infinity()
    const Real w_v2 = SQR(w_vx) + SQR(w_vy) + SQR(w_vz);
    if (w_v2 > SQR(velocity_ceiling_)) {
      const Real w_v = sqrt(w_v2);
      w_vx *= velocity_ceiling_ / w_v;
      w_vy *= velocity_ceiling_ / w_v;
      w_vz *= velocity_ceiling_ / w_v;

      u_m1 *= velocity_ceiling_ / w_v;
      u_m2 *= velocity_ceiling_ / w_v;
      u_m3 *= velocity_ceiling_ / w_v;

      Real e_k_new = 0.5 * u_d * SQR(velocity_ceiling_);
      u_e -= e_k - e_k_new;
      e_k = e_k_new;
    }

    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative pressure is encountered.
    PARTHENON_REQUIRE(w_p > 0.0 || pressure_floor_ > 0.0 || e_floor_ > 0.0,
                      "Got negative pressure. Consider enabling first-order flux "
                      "correction or setting a reasonble pressure or temperature floor.");

    // Pressure floor (if present) takes precedence over temperature floor
    if ((pressure_floor_ > 0.0) && (w_p < pressure_floor_)) {
      // apply pressure floor, correct total energy
      eps = IsTabulated() ? InternalEnergyFromDensityPressure(u_d, pressure_floor_)
                          : pressure_floor_ / (gm1 * u_d);
      u_e = u_d * eps + e_k + e_B;
      w_p = pressure_floor_;
    }

    if (IsTabulated() && GetTemperatureFloor() > 0.0) {
      const Real target = InternalEnergyFromDensityTemperature(u_d, GetTemperatureFloor());
      if (eps < target) {
        eps = target;
        u_e = u_d * eps + e_k + e_B;
        w_p = StateFromDensityInternalEnergy(u_d, eps).pressure;
      }
    }

    // temperature (internal energy) based pressure floor
    const Real eff_pressure_floor =
        IsTabulated() && e_floor_ > 0.0
            ? StateFromDensityInternalEnergy(u_d, e_floor_).pressure
            : gm1 * u_d * e_floor_;
    if (w_p < eff_pressure_floor) {
      // apply temperature floor, correct total energy
      u_e = (u_d * e_floor_) + e_k + e_B;
      w_p = eff_pressure_floor;
    }

    // temperature (internal energy) based pressure ceiling
    const Real eff_pressure_ceiling =
        IsTabulated() && Kokkos::isfinite(e_ceiling_)
            ? StateFromDensityInternalEnergy(u_d, e_ceiling_).pressure
            : gm1 * u_d * e_ceiling_;
    if (w_p > eff_pressure_ceiling) {
      // apply temperature ceiling, correct total energy
      u_e = (u_d * e_ceiling_) + e_k + e_B;
      w_p = eff_pressure_ceiling;
    }

    if (IsTabulated() && Kokkos::isfinite(GetTemperatureCeiling())) {
      const Real target =
          InternalEnergyFromDensityTemperature(u_d, GetTemperatureCeiling());
      if ((u_e - e_k - e_B) * di > target) {
        u_e = u_d * target + e_k + e_B;
        w_p = StateFromDensityInternalEnergy(u_d, target).pressure;
      }
    }

    // Convert passive scalars
    for (auto n = nhydro; n < nhydro + nscalars; ++n) {
      prim(n, k, j, i) = cons(n, k, j, i) * di;
    }
  }

 private:
  Real gamma_; // ratio of specific heats
  TabulatedEOS table_;
};

#endif // EOS_ADIABATIC_CTMHD_HPP_
