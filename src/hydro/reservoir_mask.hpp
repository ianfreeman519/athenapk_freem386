// Prescribed apparatus mask shared by the pulsed problem and resistive operator.
#ifndef HYDRO_RESERVOIR_MASK_HPP_
#define HYDRO_RESERVOIR_MASK_HPP_
#include "../main.hpp"
#include <cmath>

struct ReservoirMask {
  bool enabled = false;
  parthenon::Real core_radius = 0.0, outer_radius = 0.0;
  parthenon::Real half_separation = 0.0, relaxation_time = 0.0;

  KOKKOS_INLINE_FUNCTION
  parthenon::Real Weight(const parthenon::Real x, const parthenon::Real y) const {
    if (!enabled) return 0.0;
    const auto r = sqrt(x * x + pow(fabs(y) - half_separation, 2));
    if (r <= core_radius) return 1.0;
    if (r >= outer_radius) return 0.0;
    const auto q = (r - core_radius) / (outer_radius - core_radius);
    return fmax(0.0, fmin(1.0, 1.0 - q * q * q * (10.0 + q * (-15.0 + 6.0 * q))));
  }

  // A deliberately conservative exclusion: all points of the cell's one-cell
  // halo must be strictly inside the same prescribed core. This covers the
  // energy-flux, edge-current, and heating-estimator stencils in 2D.
  template <typename Coordinates>
  KOKKOS_INLINE_FUNCTION bool ContainsStencil(const Coordinates &c, int k, int j,
                                              int i) const {
    if (!enabled) return false;
    for (int dj = -1; dj <= 1; ++dj)
      for (int di = -1; di <= 1; ++di)
        if (Weight(c.template Xc<1>(i + di), c.template Xc<2>(j + dj)) != 1.0)
          return false;
    return true;
  }
};

inline ReservoirMask GetReservoirMask(parthenon::StateDescriptor &pkg) {
  return pkg.AllParams().hasKey("reservoir_mask")
             ? pkg.Param<ReservoirMask>("reservoir_mask")
             : ReservoirMask{};
}
#endif
