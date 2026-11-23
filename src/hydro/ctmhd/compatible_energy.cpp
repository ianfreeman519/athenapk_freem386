// Ideal-only CT energy correction and shared-EMF resistive energy transport.
#include "compatible_energy.hpp"

#include "../../main.hpp"
#include "../diffusion/diffusion.hpp"
#include "../prolongation/custom_ops.hpp"
#include "../reservoir_mask.hpp"
#include "outputs/outputs.hpp"

namespace Hydro::CTMHD {
using TE = parthenon::TopologicalElement;

namespace {
template <int component>
Real ExchangeHistory(MeshData<Real> *md) {
  const auto q = md->PackVariables(std::vector<std::string>{"ct_energy_exchange"});
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  Real total = 0.0;
  Kokkos::parallel_reduce(
      "CT energy exchange history",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {q.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(int b, int k, int j, int i, Real &sum) {
        sum += q(b, component, k, j, i) * q.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}
} // namespace

void InitializeCompatibleEnergy(ParameterInput *pin, StateDescriptor *pkg) {
  if (!pkg->Param<bool>("compatible_ct_energy")) return;
  const bool interface_diagnostics = pin->GetOrAddBoolean(
      "hydro", "ct_interface_diagnostics", false);
  pkg->AddParam("ct_interface_diagnostics", interface_diagnostics);
  if (interface_diagnostics) {
    pkg->AddField("ct_interface_diagnostics",
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
          std::vector<int>{26}, std::vector<std::string>{
            "current_before_sync", "current_after_sync", "eta_j2",
            "ohmic_ez_before_sync", "ohmic_ez_after_sync",
            "update_total", "update_magnetic", "update_kinetic", "update_thermal",
            "boundary_thermal", "boundary_face_magnetic", "center_thermal",
            "pressure_work_rate", "resistive_transport_rate",
            "baseline_total", "baseline_magnetic", "baseline_kinetic",
            "baseline_face_magnetic", "rkl_thermal_increment", "rkl_balance_residual",
            "probe_active", "rkl_transport_increment", "rkl_previous_transport",
            "rkl_initial_transport_rate", "current_before_boundary",
            "current_after_boundary"}));
  }
  const auto integrator = pin->GetString("parthenon/time", "integrator");
  PARTHENON_REQUIRE_THROWS(
      IsUCTFluid(pkg->Param<Fluid>("fluid")) &&
          pin->GetInteger("parthenon/mesh", "nx2") > 1 &&
          pin->GetOrAddInteger("parthenon/mesh", "nx3", 1) == 1 &&
          (integrator == "rk1" || integrator == "rk2" || integrator == "rk3"),
      "compatible CT energy requires 2D UCT and rk1/rk2/rk3.");
  PARTHENON_REQUIRE_THROWS(
      !GetReservoirMask(*pkg).enabled,
      "compatible CT energy does not yet support the reservoir mask.");
  PARTHENON_REQUIRE_THROWS(
      pkg->Param<Conduction>("conduction") == Conduction::none &&
          pkg->Param<Viscosity>("viscosity") == Viscosity::none &&
          (pkg->Param<Resistivity>("resistivity") == Resistivity::none ||
           (pkg->Param<DiffInt>("diffint") == DiffInt::unsplit ||
            pkg->Param<DiffInt>("diffint") == DiffInt::rkl2)),
      "compatible CT energy supports no diffusion or unsplit/RKL2 Ohmic "
      "resistivity, without conduction or viscosity.");

  auto face = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                        Metadata::WithFluxes, Metadata::FillGhost});
  face.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                             parthenon::refinement_ops::RestrictAverage,
                             parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
  pkg->AddField("ct_ideal_reference", face);
  pkg->AddField("ct_ohmic_reference", face);
  // Component 0 is the resistive energy flux; component 1 is the out-of-plane
  // magnetic flux (Bz remains cell-centered in 2D). Their cell values are unused
  // scratch; the generic WithFluxes updater may overwrite those values.
  pkg->AddField("ct_resistive_flux",
                Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                          Metadata::WithFluxes, Metadata::FillGhost},
                         std::vector<int>{2}));
  auto exchange = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Restart,
                            Metadata::Conserved, Metadata::FillGhost},
                           std::vector<int>{2},
                           std::vector<std::string>{"ideal_energy", "eos_energy"});
  exchange.RegisterRefinementOps<Hydro::refinement_ops::ProlongateCellMinModMultiD,
                                 parthenon::refinement_ops::RestrictAverage>();
  pkg->AddField("ct_energy_exchange", exchange);
  if (pkg->Param<bool>("ct_energy_diagnostics")) {
    pkg->AddField("ct_energy_diagnostics",
                  Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                           std::vector<int>{5},
                           std::vector<std::string>{
                               "ideal_correction", "resistive_magnetic_change",
                               "resistive_energy_transport", "raw_thermal_energy",
                               "eos_energy_change"}));
  }
  auto history = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  history.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum, ExchangeHistory<0>, "ct_ideal_energy"));
  history.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::sum, ExchangeHistory<1>, "ct_eos_energy"));
  pkg->UpdateParam(parthenon::hist_param_key, history);
}

TaskStatus InterfaceEnergyProbe(MeshData<Real> *md, int action) {
  const auto pkg=md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (!pkg->Param<bool>("compatible_ct_energy") ||
      !pkg->Param<bool>("ct_interface_diagnostics")) return TaskStatus::complete;
  const auto u=md->PackVariables(std::vector<std::string>{"cons"});
  const auto p=md->PackVariables(std::vector<std::string>{"prim"});
  const auto f=md->PackVariables(std::vector<std::string>{"Bface"});
  const auto e=md->PackVariablesAndFluxes(std::vector<std::string>{"ct_ohmic_reference"});
  const auto flux=md->PackVariablesAndFluxes(std::vector<std::string>{"ct_resistive_flux"});
  const auto d=md->PackVariables(std::vector<std::string>{"ct_interface_diagnostics"});
  const bool resistive=pkg->Param<Resistivity>("resistivity") != Resistivity::none;
  const auto eta=!resistive
      ? OhmicDiffusivity(Resistivity::none,ResistivityCoeff::none,0.,0.,0.,0.,-1.)
      : pkg->Param<OhmicDiffusivity>("ohm_diff");
  const auto ib=md->GetBoundsI(IndexDomain::interior);
  const auto jb=md->GetBoundsJ(IndexDomain::interior);
  const auto kb=md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(DEFAULT_LOOP_PATTERN,"Interface energy probe",DevExecSpace(),
    0,u.GetDim(5)-1,kb.s,kb.e,jb.s,jb.e,ib.s,ib.e,
    KOKKOS_LAMBDA(int b,int k,int j,int i) {
      const auto &c=u.GetCoords(b);
      const Real total=u(b,IEN,k,j,i);
      const Real kinetic=.5*(SQR(u(b,IM1,k,j,i))+SQR(u(b,IM2,k,j,i))+
                            SQR(u(b,IM3,k,j,i)))/u(b,IDN,k,j,i);
      const Real magnetic=.5*(SQR(u(b,IB1,k,j,i))+SQR(u(b,IB2,k,j,i))+
                             SQR(u(b,IB3,k,j,i)));
      const Real face=.5*(SQR(.5*(f(b,TE::F1,0,k,j,i)+f(b,TE::F1,0,k,j,i+1)))+
                           SQR(.5*(f(b,TE::F2,0,k,j,i)+f(b,TE::F2,0,k,j+1,i)))+
                           SQR(u(b,IB3,k,j,i)));
      if (action==4 || action==5) {
        Real current=0.,electric=0.;
        for(int dj=0;dj<2;++dj) for(int di=0;di<2;++di) {
          current+=.25*((f(b,TE::F2,0,k,j+dj,i+di)-f(b,TE::F2,0,k,j+dj,i+di-1))/c.Dxc<1>(k,j,i)-
                        (f(b,TE::F1,0,k,j+dj,i+di)-f(b,TE::F1,0,k,j+dj-1,i+di))/c.Dxc<2>(k,j,i));
          electric+=.25*e(b).template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j+dj,i+di);
        }
        d(b,action==4?0:1,k,j,i)=current;
        d(b,action==4?3:4,k,j,i)=electric;
        const Real jx=(p(b,IB3,k,j+1,i)-p(b,IB3,k,j-1,i))/(2*c.Dxc<2>(k,j,i));
        const Real jy=-(p(b,IB3,k,j,i+1)-p(b,IB3,k,j,i-1))/(2*c.Dxc<1>(k,j,i));
        d(b,2,k,j,i)=resistive ?
            eta.Get(p(b,IPR,k,j,i),p(b,IDN,k,j,i))*(SQR(current)+SQR(jx)+SQR(jy)) : 0.;
      } else if (action==6) {
        d(b,13,k,j,i)=-(flux(b).flux(X1DIR,0,k,j,i+1)-flux(b).flux(X1DIR,0,k,j,i))/c.Dxc<1>(k,j,i)-
                      (flux(b).flux(X2DIR,0,k,j+1,i)-flux(b).flux(X2DIR,0,k,j,i))/c.Dxc<2>(k,j,i);
        d(b,12,k,j,i)=-p(b,IPR,k,j,i)*(
            (p(b,IV1,k,j,i+1)-p(b,IV1,k,j,i-1))/(2*c.Dxc<1>(k,j,i))+
            (p(b,IV2,k,j+1,i)-p(b,IV2,k,j-1,i))/(2*c.Dxc<2>(k,j,i)));
      } else {
        if (action==1 || action==2) {
          Real current=0.;
          for(int dj=0;dj<2;++dj) for(int di=0;di<2;++di)
            current+=.25*((f(b,TE::F2,0,k,j+dj,i+di)-f(b,TE::F2,0,k,j+dj,i+di-1))/c.Dxc<1>(k,j,i)-
                          (f(b,TE::F1,0,k,j+dj,i+di)-f(b,TE::F1,0,k,j+dj-1,i+di))/c.Dxc<2>(k,j,i));
          d(b,action==1?24:25,k,j,i)=current;
        }
        if(action==0) {
          for(int n=5;n<=11;++n) d(b,n,k,j,i)=0.;
          d(b,20,k,j,i)=1.;
        } else if (d(b,20,k,j,i)==1.) {
          const Real de=total-d(b,14,k,j,i),dm=magnetic-d(b,15,k,j,i),
                     dk=kinetic-d(b,16,k,j,i);
          if(action==1) {
            d(b,5,k,j,i)=de;d(b,6,k,j,i)=dm;d(b,7,k,j,i)=dk;d(b,8,k,j,i)=de-dm-dk;
          } else if(action==2) {
            d(b,9,k,j,i)+=de-dm-dk;
            d(b,10,k,j,i)+=face-d(b,17,k,j,i);
          } else if(action==3) d(b,11,k,j,i)+=de-dm-dk;
        }
        d(b,14,k,j,i)=total;d(b,15,k,j,i)=magnetic;d(b,16,k,j,i)=kinetic;d(b,17,k,j,i)=face;
      }
    });
  return TaskStatus::complete;
}

TaskStatus PrepareCompatibleEnergy(MeshData<Real> *md, MeshData<Real> *initial,
                                   Real gam0, Real gam1, bool diffusion_only) {
  const auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool resistive = pkg->Param<Resistivity>("resistivity") != Resistivity::none &&
      (diffusion_only || pkg->Param<DiffInt>("diffint") == DiffInt::unsplit);
  const auto eta = resistive ? pkg->Param<OhmicDiffusivity>("ohm_diff")
                            : OhmicDiffusivity(Resistivity::none, ResistivityCoeff::none,
                                              0., 0., 0., 0., -1.);
  const auto f = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  const auto f1 = initial->PackVariables(std::vector<std::string>{"Bface"});
  const auto ideal = md->PackVariablesAndFluxes(
      std::vector<std::string>{"ct_ideal_reference"});
  const auto ohmic = md->PackVariablesAndFluxes(
      std::vector<std::string>{"ct_ohmic_reference"});
  const auto flux = md->PackVariablesAndFluxes(
      std::vector<std::string>{"ct_resistive_flux"});
  const auto p = md->PackVariables(std::vector<std::string>{"prim"});
  const auto u = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  const auto q = md->PackVariables(std::vector<std::string>{"ct_energy_exchange"});
  const auto q1 = initial->PackVariables(std::vector<std::string>{"ct_energy_exchange"});
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Prepare compatible CT stage", DevExecSpace(), 0,
      f.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto &c = f.GetCoords(b);
        ideal(b).template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i) =
            f(b).template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
        ohmic(b).template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i) = 0.;
        if (j <= jb.e) {
          ideal(b, TE::F1, 0, k, j, i) =
              gam0 * f(b, TE::F1, 0, k, j, i) + gam1 * f1(b, TE::F1, 0, k, j, i);
          Real bzflux = 0.;
          if (resistive) {
            const Real coeff = .5 * (eta.Get(p(b, IPR, k, j, i-1), p(b, IDN, k, j, i-1)) +
                                     eta.Get(p(b, IPR, k, j, i), p(b, IDN, k, j, i)));
            bzflux = -coeff * (p(b, IB3, k, j, i) - p(b, IB3, k, j, i-1)) /
                     c.Dxc<1>(k, j, i);
          }
          flux(b).flux(X1DIR, 0, k, j, i) = 0.;
          flux(b).flux(X1DIR, 1, k, j, i) = bzflux;
          u(b).flux(X1DIR, IB3, k, j, i) += bzflux;
        }
        if (i <= ib.e) {
          ideal(b, TE::F2, 0, k, j, i) =
              gam0 * f(b, TE::F2, 0, k, j, i) + gam1 * f1(b, TE::F2, 0, k, j, i);
          Real bzflux = 0.;
          if (resistive) {
            const Real coeff = .5 * (eta.Get(p(b, IPR, k, j-1, i), p(b, IDN, k, j-1, i)) +
                                     eta.Get(p(b, IPR, k, j, i), p(b, IDN, k, j, i)));
            bzflux = -coeff * (p(b, IB3, k, j, i) - p(b, IB3, k, j-1, i)) /
                     c.Dxc<2>(k, j, i);
          }
          flux(b).flux(X2DIR, 0, k, j, i) = 0.;
          flux(b).flux(X2DIR, 1, k, j, i) = bzflux;
          u(b).flux(X2DIR, IB3, k, j, i) += bzflux;
        }
        if (!diffusion_only && i <= ib.e && j <= jb.e) {
          for (int n = 0; n < 2; ++n) {
            // The first RK stage saves the original accumulator just as the state
            // updater saves u1. Derived budget fields are not updated by that task.
            if (gam0 == 0.) q1(b, n, k, j, i) = q(b, n, k, j, i);
            q(b, n, k, j, i) = gam0*q(b, n, k, j, i) + gam1*q1(b, n, k, j, i);
          }
        }
      });
  return TaskStatus::complete;
}

TaskStatus CompatibleResistiveEnergyFlux(MeshData<Real> *md) {
  const auto f = md->PackVariables(std::vector<std::string>{"Bface"});
  const auto p = md->PackVariables(std::vector<std::string>{"prim"});
  const auto ohmic = md->PackVariablesAndFluxes(
      std::vector<std::string>{"ct_ohmic_reference"});
  const auto flux = md->PackVariablesAndFluxes(
      std::vector<std::string>{"ct_resistive_flux"});
  const auto u = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Compatible resistive Poynting flux", DevExecSpace(), 0,
      f.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e+1,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto &e = ohmic(b);
        const Real ez = e.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j,i);
        if (j <= jb.e) {
          const Real ezt = e.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j+1,i);
          const Real byb = .5*(f(b,TE::F2,0,k,j,i-1)+f(b,TE::F2,0,k,j,i));
          const Real byt = .5*(f(b,TE::F2,0,k,j+1,i-1)+f(b,TE::F2,0,k,j+1,i));
          const Real sx = -.5*(ez*byb+ezt*byt) +
              .5*(p(b,IB3,k,j,i-1)+p(b,IB3,k,j,i))*flux(b).flux(X1DIR,1,k,j,i);
          flux(b).flux(X1DIR,0,k,j,i) = sx;
          u(b).flux(X1DIR,IEN,k,j,i) += sx;
        }
        if (i <= ib.e) {
          const Real ezr = e.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j,i+1);
          const Real bxl = .5*(f(b,TE::F1,0,k,j-1,i)+f(b,TE::F1,0,k,j,i));
          const Real bxr = .5*(f(b,TE::F1,0,k,j-1,i+1)+f(b,TE::F1,0,k,j,i+1));
          const Real sy = .5*(ez*bxl+ezr*bxr) +
              .5*(p(b,IB3,k,j-1,i)+p(b,IB3,k,j,i))*flux(b).flux(X2DIR,1,k,j,i);
          flux(b).flux(X2DIR,0,k,j,i) = sy;
          u(b).flux(X2DIR,IEN,k,j,i) += sy;
        }
      });
  return TaskStatus::complete;
}

TaskStatus CorrectCompatibleEnergy(MeshData<Real> *md, Real beta_dt) {
  const auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool diagnostics = pkg->Param<bool>("ct_energy_diagnostics");
  const auto u = md->PackVariables(std::vector<std::string>{"cons"});
  const auto f = md->PackVariables(std::vector<std::string>{"Bface"});
  const auto ref = md->PackVariablesAndFluxes(std::vector<std::string>{"ct_ideal_reference"});
  const auto flux = md->PackVariablesAndFluxes(std::vector<std::string>{"ct_resistive_flux"});
  const auto q = md->PackVariables(std::vector<std::string>{"ct_energy_exchange"});
  const auto diag = md->PackVariables(std::vector<std::string>{
      diagnostics ? "ct_energy_diagnostics" : "ct_energy_exchange"});
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Ideal-only CT energy correction", DevExecSpace(), 0,
      u.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto &c = u.GetCoords(b);
        const auto &r = ref(b);
        const Real e00 = r.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j,i);
        const Real e10 = r.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j,i+1);
        const Real e01 = r.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j+1,i);
        const Real e11 = r.template flux<parthenon::TopologicalType::Edge>(X3DIR,0,k,j+1,i+1);
        // Match UpdateWithFaceMagDivergence face by face before centering:
        // dBx/dt = -dEz/dy, dBy/dt = +dEz/dx. In particular, the By
        // increment has the opposite sign to the Bx increment. Keeping the
        // same operation order also avoids unnecessary ideal-limit roundoff.
        const Real bxleft = r(TE::F1,0,k,j,i) +
            beta_dt*(e00-e01)/c.Dxc<2>(k,j,i);
        const Real bxright = r(TE::F1,0,k,j,i+1) +
            beta_dt*(e10-e11)/c.Dxc<2>(k,j,i+1);
        const Real bybottom = r(TE::F2,0,k,j,i) -
            beta_dt*(e00-e10)/c.Dxc<1>(k,j,i);
        const Real bytop = r(TE::F2,0,k,j+1,i) -
            beta_dt*(e01-e11)/c.Dxc<1>(k,j+1,i);
        const Real bxideal = .5*(bxleft+bxright);
        const Real byideal = .5*(bybottom+bytop);
        const Real bx = .5*(f(b,TE::F1,0,k,j,i)+f(b,TE::F1,0,k,j,i+1));
        const Real by = .5*(f(b,TE::F2,0,k,j,i)+f(b,TE::F2,0,k,j+1,i));
        const Real correction = .5*(SQR(bxideal)+SQR(byideal) -
                                     SQR(u(b,IB1,k,j,i))-SQR(u(b,IB2,k,j,i)));
        u(b,IEN,k,j,i) += correction;
        q(b,0,k,j,i) += correction;
        if (diagnostics) {
          const Real dbz = -beta_dt * (
              (flux(b).flux(X1DIR,1,k,j,i+1)-flux(b).flux(X1DIR,1,k,j,i))/c.Dxc<1>(k,j,i) +
              (flux(b).flux(X2DIR,1,k,j+1,i)-flux(b).flux(X2DIR,1,k,j,i))/c.Dxc<2>(k,j,i));
          diag(b,0,k,j,i) = correction;
          diag(b,1,k,j,i) = .5*(SQR(bx)+SQR(by)-SQR(bxideal)-SQR(byideal) +
              SQR(u(b,IB3,k,j,i))-SQR(u(b,IB3,k,j,i)-dbz));
          diag(b,2,k,j,i) = -beta_dt * (
              (flux(b).flux(X1DIR,0,k,j,i+1)-flux(b).flux(X1DIR,0,k,j,i))/c.Dxc<1>(k,j,i) +
              (flux(b).flux(X2DIR,0,k,j+1,i)-flux(b).flux(X2DIR,0,k,j,i))/c.Dxc<2>(k,j,i));
        }
        u(b,IB1,k,j,i) = bx;
        u(b,IB2,k,j,i) = by;
      });
  return TaskStatus::complete;
}
} // namespace Hydro::CTMHD
