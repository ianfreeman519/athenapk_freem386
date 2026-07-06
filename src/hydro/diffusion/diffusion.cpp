//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusion.cpp
//! \brief

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "diffusion.hpp"

using namespace parthenon::package::prelude;

TaskStatus CalcDiffFluxes(StateDescriptor *hydro_pkg, MeshData<Real> *md,
                          bool magnetic_only_resistive_flux) {
  const bool coupled_ohmic =
      hydro_pkg->Param<bool>("thermal_source_solver_enabled") &&
      hydro_pkg->Param<bool>("thermal_couple_ohmic");
  const auto &conduction = hydro_pkg->Param<Conduction>("conduction");
  if (conduction != Conduction::none) {
    const auto &thermal_diff = hydro_pkg->Param<ThermalDiffusivity>("thermal_diff");

    if (conduction == Conduction::isotropic &&
        thermal_diff.GetCoeffType() == ConductionCoeff::fixed) {
      ThermalFluxIsoFixed(md);
    } else {
      ThermalFluxGeneral(md);
    }
  }
  const auto &viscosity = hydro_pkg->Param<Viscosity>("viscosity");
  if (viscosity != Viscosity::none) {
    const auto &mom_diff = hydro_pkg->Param<MomentumDiffusivity>("mom_diff");

    if (viscosity == Viscosity::isotropic &&
        mom_diff.GetCoeffType() == ViscosityCoeff::fixed) {
      MomentumDiffFluxIsoFixed(md);
    } else {
      MomentumDiffFluxGeneral(md);
    }
  }
  const auto &resistivity = hydro_pkg->Param<Resistivity>("resistivity");
  if (resistivity != Resistivity::none) {
    const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

    if (coupled_ohmic) {
      if (magnetic_only_resistive_flux) {
        // STS still owns the magnetic diffusion update, but coupled ohmic heating keeps
        // the legacy resistive IEN flux disabled so the thermal solver remains the sole
        // owner of the heat source.
        AddMagneticOnlyResistiveFlux(md);
      } else {
        // Outside STS, the coupled internal-energy solver owns the lagged ohmic source
        // iteration and the final magnetic-only resistive commit. Skip the standard
        // resistive diffusion path here to avoid double-applying B or IEN.
      }
    } else if (resistivity == Resistivity::ohmic &&
               (ohm_diff.GetCoeffType() == ResistivityCoeff::fixed ||
                ohm_diff.GetCoeffType() == ResistivityCoeff::spitzer)) {
      AddOhmicResistiveFlux(md);
    } else {
      PARTHENON_FAIL("Unknown Resistivity Type")
    }
  }
  return TaskStatus::complete;
}
