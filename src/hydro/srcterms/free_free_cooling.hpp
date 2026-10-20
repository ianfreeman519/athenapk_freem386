// Optically thin, single-temperature free-free cooling using the EOS mean charge.
#ifndef HYDRO_SRCTERMS_FREE_FREE_COOLING_HPP_
#define HYDRO_SRCTERMS_FREE_FREE_COOLING_HPP_

#include "cooling_integrator.hpp"
#include "../../units.hpp"

namespace cooling {

struct FreeFreeRate {
  TabulatedEOS eos;
  parthenon::Real coefficient; // C * gB in code units; temperature stays in K
  parthenon::Real ion_mass;    // A * m_u in code units

  KOKKOS_INLINE_FUNCTION
  parthenon::Real DeDt(const parthenon::Real e, const parthenon::Real rho,
                       bool &valid) const {
    if (!std::isfinite(e) || e <= 0.0 || !std::isfinite(rho) || rho <= 0.0) {
      valid = false;
      return 0.0;
    }
    const auto state = eos.StateFromDensityInternalEnergy(rho, e);
    if (!std::isfinite(state.temperature) || state.temperature <= 0.0 ||
        !std::isfinite(state.zbar) || state.zbar < 0.0) {
      valid = false;
      return 0.0;
    }
    const auto ni = rho / ion_mass;
    const auto ne = state.zbar * ni;
    // Q = C gB Zbar^2 n_e n_i sqrt(T); return specific-energy loss.
    return -coefficient * state.zbar * state.zbar * ne * (ni / rho) *
           std::sqrt(state.temperature);
  }

  KOKKOS_INLINE_FUNCTION
  parthenon::Real DeDt(const parthenon::Real e, const parthenon::Real rho) const {
    bool valid = true;
    return DeDt(e, rho, valid);
  }
};

class FreeFreeCooling {
  FreeFreeRate rate_;
  CoolingSettings settings_;
  CoolIntegrator integrator_;

 public:
  FreeFreeCooling(parthenon::ParameterInput *pin,
                  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg);
  void SrcTerm(parthenon::MeshData<parthenon::Real> *md,
               parthenon::Real dt) const;
  parthenon::Real EstimateTimeStep(parthenon::MeshData<parthenon::Real> *md) const;
};

} // namespace cooling
#endif
