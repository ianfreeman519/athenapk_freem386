#ifndef HYDRO_SRCTERMS_INTERNAL_ENERGY_SOLVER_HPP_
#define HYDRO_SRCTERMS_INTERNAL_ENERGY_SOLVER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../hydro.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

struct InternalEnergySolverConfig {
  int iterations;
};

TaskStatus InitializeStageLaggedThermalSource(MeshData<Real> *md);
TaskStatus AddCoupledInternalEnergySources(MeshData<Real> *md, const SimTime &tm,
                                           const Real dt);

} // namespace Hydro

#endif // HYDRO_SRCTERMS_INTERNAL_ENERGY_SOLVER_HPP_
