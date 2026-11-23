// Opt-in, unsplit/RKL2 2D UCT energy coupling. References are per-stage scratch.
#ifndef HYDRO_CTMHD_COMPATIBLE_ENERGY_HPP_
#define HYDRO_CTMHD_COMPATIBLE_ENERGY_HPP_

#include "../../main.hpp"
#include <parthenon/package.hpp>

namespace Hydro::CTMHD {
using namespace parthenon::package::prelude;

void InitializeCompatibleEnergy(ParameterInput *pin, StateDescriptor *pkg);
TaskStatus PrepareCompatibleEnergy(MeshData<Real> *md, MeshData<Real> *initial,
                                   Real gam0, Real gam1, bool diffusion_only = false);
TaskStatus CompatibleResistiveEnergyFlux(MeshData<Real> *md);
TaskStatus CorrectCompatibleEnergy(MeshData<Real> *md, Real beta_dt);
// Optional interface probes: 0 capture update input, 1 capture updated state,
// 2 after boundary exchange, 3 after centering, 4/5 before/after EMF sync,
// 6 synchronized resistive energy operator. Values describe the latest stage.
TaskStatus InterfaceEnergyProbe(MeshData<Real> *md, int action);
} // namespace Hydro::CTMHD
#endif
