// Opt-in, unsplit 2D UCT energy coupling. References are per-stage scratch.
#ifndef HYDRO_CTMHD_COMPATIBLE_ENERGY_HPP_
#define HYDRO_CTMHD_COMPATIBLE_ENERGY_HPP_

#include "../../main.hpp"
#include <parthenon/package.hpp>

namespace Hydro::CTMHD {
using namespace parthenon::package::prelude;

void InitializeCompatibleEnergy(ParameterInput *pin, StateDescriptor *pkg);
TaskStatus PrepareCompatibleEnergy(MeshData<Real> *md, MeshData<Real> *initial,
                                   Real gam0, Real gam1);
TaskStatus CompatibleResistiveEnergyFlux(MeshData<Real> *md);
TaskStatus CorrectCompatibleEnergy(MeshData<Real> *md, Real beta_dt);
} // namespace Hydro::CTMHD
#endif
