#ifndef HYDRO_CTMHD_UCTHLLDMHD_HPP_
#define HYDRO_CTMHD_UCTHLLDMHD_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::UCT {

TaskStatus Assemble_Edge_EMF(MeshData<Real> *md);

} // namespace Hydro::UCT

#endif // HYDRO_CTMHD_UCTHLLDMHD_HPP_
