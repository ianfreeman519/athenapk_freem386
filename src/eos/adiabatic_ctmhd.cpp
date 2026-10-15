//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements functions in class EquationOfState for adiabatic
//  hydrodynamics`

// C headers

// C++ headers
#include <cmath> // sqrt()

// Parthenon headers
#include "../eos/adiabatic_ctmhd.hpp"
#include "../main.hpp"
#include "config.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
using parthenon::IndexDomain;
using parthenon::MeshBlockVarPack;
using parthenon::ParArray4D;

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticCTMHDEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  // before coming into this function, the mag field in cons_pack should
  // be the correct centered version. Then, this function properly 
  // computes the primitive E variable with the correct derived cc B field
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  auto iib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  auto ijb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  auto ikb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const int rank = parthenon::Globals::my_rank;

  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  auto this_on_device = (*this);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        // auto &nu = entropy_pack(b);

        const bool is_ghost = i < iib.s || i > iib.e || j < ijb.s || j > ijb.e ||
                              k < ikb.s || k > ikb.e;
        return this_on_device.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, rank, b,
                                         is_ghost);
      });
  if (IsTabulated()) {
    CheckDiagnostics("CT-MHD conserved-to-primitive state conversion");
    parthenon::PackIndexMap eos_map;
    auto eos_pack = md->PackVariables(std::vector<std::string>{
        "eos_temperature", "eos_sound_speed", "eos_zbar", "eos_mu", "eos_mu_e"}, eos_map);
    const int iT = eos_map["eos_temperature"].first;
    const int ics = eos_map["eos_sound_speed"].first;
    const int iz = eos_map["eos_zbar"].first;
    const int imu = eos_map["eos_mu"].first;
    const int imue = eos_map["eos_mu_e"].first;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FillTabulatedEOSFieldsCT", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons_pack(b);
          auto &out = eos_pack(b);
          const Real rho = u(IDN, k, j, i);
          const Real ek = 0.5 / rho *
                          (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) +
                           SQR(u(IM3, k, j, i)));
          const Real eb = 0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) +
                                 SQR(u(IB3, k, j, i)));
          const Real eps = (u(IEN, k, j, i) - ek - eb) / rho;
          if (this_on_device.table_.IsEnergyMoreThanOneDecadeLow(eps)) {
            const bool is_ghost = i < iib.s || i > iib.e || j < ijb.s || j > ijb.e ||
                                  k < ikb.s || k > ikb.e;
            Kokkos::printf(
                "CTMHD_EOS_FIELD_LOW rank=%d block=%d k=%d j=%d i=%d ghost=%d "
                "rho=%.17e eps=%.17e Etot=%.17e Ek=%.17e EB=%.17e Eth=%.17e "
                "momentum=(%.17e,%.17e,%.17e) B=(%.17e,%.17e,%.17e)\n",
                rank, b, k, j, i, static_cast<int>(is_ghost), rho, eps,
                u(IEN, k, j, i), ek, eb, u(IEN, k, j, i) - ek - eb,
                u(IM1, k, j, i), u(IM2, k, j, i), u(IM3, k, j, i),
                u(IB1, k, j, i), u(IB2, k, j, i), u(IB3, k, j, i));
          }
          const auto s = this_on_device.StateFromDensityInternalEnergy(rho, eps);
          out(iT, k, j, i) = s.temperature;
          out(ics, k, j, i) = s.sound_speed;
          out(iz, k, j, i) = s.zbar;
          out(imu, k, j, i) = s.mu;
          out(imue, k, j, i) = s.mu_e;
        });
  }
  CheckDiagnostics("CT-MHD EOS output-field reconstruction");
}
