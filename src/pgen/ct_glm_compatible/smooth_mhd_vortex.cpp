//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file smooth_mhd_vortex.cpp
//! \brief Problem generator for (currently 2D) smooth mhd vortex convergence test
// 
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../main.hpp"
#include "outputs/outputs.hpp"

namespace smooth_mhd_vortex {
  using namespace parthenon::package::prelude;
  using TE = parthenon::TopologicalElement;

  template <typename ConsHost, typename BfaceHost>
  void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface);


Real B0_ = 0.0;

// Relative divergence of B error, i.e., L * |div(B)| / |B_0|
// This is different from the standard package one because it uses
// a fixed B0, which is required in this pgen to get sensible results
// as some fraction of the domain has |B| = 0
Real RelDivBHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  auto B0 = B0_;

  pmb->par_reduce(
      "RelDivBHst", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real divb =
            (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) / coords.Dxc<1>(k, j, i) +
            (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        if (three_d) {
          divb +=
              (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        }
        lsum += 0.5 *
                (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) + SQR(coords.Dxc<2>(k, j, i)) +
                           SQR(coords.Dxc<3>(k, j, i)))) *
                std::abs(divb) / B0 * coords.CellVolume(k, j, i);
      },
      sum);

  return sum;
}

// If using CT, this will be the relevant metric to check
Real FaceDivBHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &Bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  auto B0 = B0_;

  pmb->par_reduce(
      "FaceDivBHst", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &Bface = Bface_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real facedivb =
            (Bface(TE::F1,  0, k, j, i + 1) - Bface(TE::F1,  0, k, j, i)) / coords.Dxc<1>(k, j, i) +
            (Bface(TE::F2,  0, k, j + 1, i) - Bface(TE::F2,  0, k, j, i)) / coords.Dxc<2>(k, j, i);
        if (three_d) {
          facedivb +=
              (Bface(TE::F3,  0, k + 1, j, i) - Bface(TE::F3,  0, k, j, i)) / coords.Dxc<3>(k, j, i);
        }
        lsum += 0.5 *
                (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) + SQR(coords.Dxc<2>(k, j, i)) +
                           SQR(coords.Dxc<3>(k, j, i)))) *
                std::abs(facedivb) / B0 * coords.CellVolume(k, j, i);
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  // gives us ctmhd ucthlldmhd or glmmhd
  const auto fluid = pkg->Param<Fluid>("fluid");
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  if (fluid == Fluid::glmmhd){
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                      RelDivBHst, "UserRelDivB"));
    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
    }
  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd){
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    FaceDivBHst, "UserFaceDivB"));
    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
  } 
}

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (!pin->GetOrAddBoolean("problem/smooth_mhd_vortex", "compute_error", false)) return;

  constexpr int NMHD = 8;

  // Initialize errors to zero
  Real l1_err[NMHD]{}, max_err[NMHD]{};

  for (auto &pmb : mesh->block_list) {
    const auto fluid = pmb->packages.Get("Hydro")->Param<Fluid>("fluid");

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
    // Even for MHD, there are only cell-centered mesh variables
    int ncells4 = NMHD;
    // Save analytic solution of conserved variables in 4D scratch array on host
    Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace> cons_(
        "cons scratch", ncells4, pmb->cellbounds.ncellsk(IndexDomain::entire),
        pmb->cellbounds.ncellsj(IndexDomain::entire),
        pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real x1size =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  Real x2size =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);

  const bool two_d = pmb->pmy_mesh->ndim < 3;
  // for 2D sim set x3size to zero so that v_z is 0 below
  Real x3size =
      two_d ? 0
            : pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);


  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // initial parameters
  Real A0_amp = 1.0/(2.0*M_PI);
  B0_ = A0_amp;

  auto &mbd = pmb->meshblock_data.Get();

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd){
    // fills u_cons() with the cell-averaged b values from
    // the face centered values made via the discrete
    // curl of the vector potential
    auto &u_dev_face = mbd->Get("Bface").data;
    auto Bface = u_dev_face.GetHostMirrorAndCopy();

    Bface_Fill_Cons(pmb.get(), cons_, Bface); 
  }

  Real vx0 = 1.0;
  Real vy0 = 1.0; 
  Real p0  = 1.0;
  // now good to fill up the u vector
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e; i++) {
        // vortex coords
        Real X = coords.Xc<1>(i);
        Real Y = coords.Xc<2>(j);
        Real R2 = SQR(X) + SQR(Y);

        Real g = std::exp(0.5*(1.0-R2));

        // velocity perturbation
        Real dvx = -(1.0/(2.0*M_PI)) * Y * g;
        Real dvy =  (1.0/(2.0*M_PI)) * X * g;

        // fill density and mom
        cons_(IDN, k, j, i) = 1.0;
        cons_(IM1, k, j, i) = cons_(IDN, k, j, i) * (vx0 + dvx);
        cons_(IM2, k, j, i) = cons_(IDN, k, j, i) * (vy0 + dvy);
        cons_(IM3, k, j, i) = 0.0;

        // pressure perturbation
        Real dp = (((1.0 - R2) - 1.0) / (8.0 * SQR(M_PI))) * std::exp(1.0-R2);
        Real p = p0 + dp;

      
        if (fluid == Fluid::glmmhd){
          cons_(IB1, k, j, i) = -1.0/(2*M_PI) * Y * g;
          cons_(IB2, k, j, i) =  1.0/(2*M_PI) * X * g;
          cons_(IB3, k, j, i) = 0.0; 
        }

        // will be correct with either ct or glm
        cons_(IEN, k, j, i) =
            p / gm1 +
            0.5 * (SQR(cons_(IB1, k, j, i)) + SQR(cons_(IB2, k, j, i)) + SQR(cons_(IB3, k, j, i))) +
            0.5 * (SQR(cons_(IM1, k, j, i)) + SQR(cons_(IM2, k, j, i)) + SQR(cons_(IM3, k, j, i))) /
                cons_(IDN, k, j, i);        
        
      }
    }
  }

    
    auto u = mbd->Get("cons").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          // Load cell-averaged <U>, either midpoint approx. or fourth-order approx
          Real d1 = cons_(IDN, k, j, i);
          Real m1 = cons_(IM1, k, j, i);
          Real m2 = cons_(IM2, k, j, i);
          Real m3 = cons_(IM3, k, j, i);
          // Weight l1 error by cell volume
          Real vol = pmb->coords.CellVolume(k, j, i);

          l1_err[IDN] += std::abs(d1 - u(IDN, k, j, i)) * vol;
          max_err[IDN] =
              std::max(static_cast<Real>(std::abs(d1 - u(IDN, k, j, i))), max_err[IDN]);
          l1_err[IM1] += std::abs(m1 - u(IM1, k, j, i)) * vol;
          l1_err[IM2] += std::abs(m2 - u(IM2, k, j, i)) * vol;
          l1_err[IM3] += std::abs(m3 - u(IM3, k, j, i)) * vol;
          max_err[IM1] =
              std::max(static_cast<Real>(std::abs(m1 - u(IM1, k, j, i))), max_err[IM1]);
          max_err[IM2] =
              std::max(static_cast<Real>(std::abs(m2 - u(IM2, k, j, i))), max_err[IM2]);
          max_err[IM3] =
              std::max(static_cast<Real>(std::abs(m3 - u(IM3, k, j, i))), max_err[IM3]);

          Real e0 = cons_(IEN, k, j, i);
          l1_err[IEN] += std::abs(e0 - u(IEN, k, j, i)) * vol;
          max_err[IEN] =
              std::max(static_cast<Real>(std::abs(e0 - u(IEN, k, j, i))), max_err[IEN]);

          Real b1 = cons_(IB1, k, j, i);
          Real b2 = cons_(IB2, k, j, i);
          Real b3 = cons_(IB3, k, j, i);
          Real db1 = std::abs(b1 - u(IB1, k, j, i));
          Real db2 = std::abs(b2 - u(IB2, k, j, i));
          Real db3 = std::abs(b3 - u(IB3, k, j, i));

          l1_err[IB1] += db1 * vol;
          l1_err[IB2] += db2 * vol;
          l1_err[IB3] += db3 * vol;
          max_err[IB1] = std::max(db1, max_err[IB1]);
          max_err[IB2] = std::max(db2, max_err[IB2]);
          max_err[IB3] = std::max(db3, max_err[IB3]);
        }
      }
    }
  }
  Real rms_err = 0.0, max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, (NMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, (NMHD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, (NMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, (NMHD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (parthenon::Globals::my_rank == 0) {
    // normalize errors by number of cells
    const auto mesh_size = mesh->mesh_size;
    const auto vol = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR)) *
                     (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR)) *
                     (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
    for (int i = 0; i < (NMHD); ++i)
      l1_err[i] = l1_err[i] / vol;
    // compute rms error
    for (int i = 0; i < (NMHD); ++i) {
      rms_err += SQR(l1_err[i]);
      if (l1_err[i] > 0.0){
      max_max_over_l1 = std::max(max_max_over_l1, (max_err[i] / l1_err[i]));
      }
    }
    rms_err = std::sqrt(rms_err);

    // open output file and write out errors
    std::string fname;
    fname.assign("smoothVortexMHD-errors.dat");
    std::stringstream msg;
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg);
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
      std::fprintf(pfile, "RMS-L1-Error  d_L1  M1_L1  M2_L1  M3_L1  E_L1 ");
      std::fprintf(pfile, "  B1c_L1  B2c_L1  B3c_L1");
      std::fprintf(pfile, "  Largest-Max/L1  d_max  M1_max  M2_max  M3_max  E_max ");
      std::fprintf(pfile, "  B1c_max  B2c_max  B3c_max");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx(X1DIR), mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %d  %d", mesh_size.nx(X3DIR), tm.ncycle);
    std::fprintf(pfile, "  %e  %e", rms_err, l1_err[IDN]);
    std::fprintf(pfile, "  %e  %e  %e", l1_err[IM1], l1_err[IM2], l1_err[IM3]);
    std::fprintf(pfile, "  %e", l1_err[IEN]);
    std::fprintf(pfile, "  %e", l1_err[IB1]);
    std::fprintf(pfile, "  %e", l1_err[IB2]);
    std::fprintf(pfile, "  %e", l1_err[IB3]);
    std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err[IDN]);
    std::fprintf(pfile, "%e  %e  %e", max_err[IM1], max_err[IM2], max_err[IM3]);
    std::fprintf(pfile, "  %e", max_err[IEN]);
    std::fprintf(pfile, "  %e", max_err[IB1]);
    std::fprintf(pfile, "  %e", max_err[IB2]);
    std::fprintf(pfile, "  %e", max_err[IB3]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}

  //========================================================================================
  //! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
  //! \brief smooth mhd vortex problem generator for 2D CT convergence testing.
  //========================================================================================

  void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  
  const auto fluid = pmb->packages.Get("Hydro")->Param<Fluid>("fluid");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;



  Real x1size =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  Real x2size =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);

  const bool two_d = pmb->pmy_mesh->ndim < 3;
  // for 2D sim set x3size to zero so that v_z is 0 below
  Real x3size =
      two_d ? 0
            : pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);


  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // initial parameters
  Real A0_amp = 1.0/(2.0*M_PI);
  B0_ = A0_amp;

  // Initialize density and momenta

  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd){
    // fills u_cons() with the cell-averaged b values from
    // the face centered values made via the discrete
    // curl of the vector potential
    // Also deep copy the Bface vector for later evolution
    auto &u_dev_face = mbd->Get("Bface").data;
    auto Bface = u_dev_face.GetHostMirrorAndCopy();

    Bface_Fill_Cons(pmb, u, Bface); 
    u_dev_face.DeepCopy(Bface);
  }

  Real vx0 = 1.0;
  Real vy0 = 1.0; 
  Real p0  = 1.0;
  // now good to fill up the u vector
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e; i++) {
        // vortex coords
        Real X = coords.Xc<1>(i);
        Real Y = coords.Xc<2>(j);
        Real R2 = SQR(X) + SQR(Y);

        Real g = std::exp(0.5*(1.0-R2));

        // velocity perturbation
        Real dvx = -(1.0/(2.0*M_PI)) * Y * g;
        Real dvy =  (1.0/(2.0*M_PI)) * X * g;

        // fill density and mom
        u(IDN, k, j, i) = 1.0;
        u(IM1, k, j, i) = u(IDN, k, j, i) * (vx0 + dvx);
        u(IM2, k, j, i) = u(IDN, k, j, i) * (vy0 + dvy);
        u(IM3, k, j, i) = 0.0;

        // pressure perturbation
        Real dp = (((1.0 - R2) - 1.0) / (8.0 * SQR(M_PI))) * std::exp(1.0-R2);
        Real p = p0 + dp;

        if (fluid == Fluid::glmmhd){
          u(IB1, k, j, i) = -1.0/(2*M_PI) * Y * g;
          u(IB2, k, j, i) =  1.0/(2*M_PI) * X * g;
          u(IB3, k, j, i) = 0.0; 
          u(IPS, k, j ,i) = 0.0;
        }
        // now energy calculation uses correct, averaged cell-centered B
        u(IEN, k, j, i) =
            p / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) +
            0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                u(IDN, k, j, i);        
        
      }
    }
  }
  u_dev.DeepCopy(u);
}

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface) {

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
 
  // always needed
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real A0_amp = 1.0/(2.0*M_PI);

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // create corner valued vector potential
  int kl, ku;
  if (two_d) {
    kl = kb.s;
    ku = kb.e;
  } else {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }
  for (int k = kl; k <= ku; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        // define az at the cell-corners or nodes
        Real x_node = coords.X<1, parthenon::TopologicalElement::NN>(k, j, i);
        Real y_node = coords.X<2, parthenon::TopologicalElement::NN>(k, j, i);
        Real z_node = coords.X<3, parthenon::TopologicalElement::NN>(k, j, i);
        Real R2 = SQR(x_node) + SQR(y_node);
        az(k, j, i) =
            A0_amp * std::exp(0.5*(1-R2));
        }
      }
  }

  // Initialize density and momenta

  auto &mbd = pmb->meshblock_data.Get();
  // initializing on host
  auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto By_face = Bface.Get(IBF2, 0, 0, 0);
  auto Bz_face = Bface.Get(IBF3, 0, 0, 0);

  // fill Bface first (cell faces so +1 on the bounds)

  // x-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e+1; i++) { // +1 here
        // create face-fields with cell-corner defined az
        Bx_face(k, j, i) =
            (az(k, j + 1, i) - az(k, j, i)) / coords.Dxc<2>(j);
      }
    }
  }
  // y-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e+1; j++) { // +1 here
      for (int i = ib.s; i <= ib.e; i++) { 
        // create face-fields with cell-corner defined az
        By_face(k, j, i) =
        - (az(k, j, i + 1) - az(k, j, i)) / coords.Dxc<1>(i);
      }
    }
  }
  // z-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e; i++) { 
        Bz_face(k, j, i) = 0.0;
      }
    }
  }
  // now good to fill up the Bx/By cons vector
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e; i++) {
        // create cell-centered B from face-centered average
        u(IB1, k, j, i) =
            0.5 * (Bx_face(k, j, i) + Bx_face(k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (By_face(k, j, i) + By_face(k, j + 1, i));
        u(IB3, k, j, i) = 0.0; // for 2D testing, fill this outside of here    
      }
    }
  }
}

} // namespace smooth_mhd_vortex
