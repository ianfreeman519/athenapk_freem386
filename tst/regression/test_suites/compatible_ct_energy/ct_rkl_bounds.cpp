// Focused tests of the production compatible-RKL stencil, state guard, retry
// policy, and collective reductions. No simulation or EOS table is required.
#include "../../../../src/hydro/diffusion/diffusion.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool pass, const char *message) {
  if (!pass) throw std::runtime_error(message);
}
struct TestPack {
  Kokkos::View<Real***> data;
  KOKKOS_INLINE_FUNCTION Real operator()(int, int n, int, int j, int i) const {
    return data(n,j,i);
  }
};
struct TestEOS {
  struct State { Real pressure,temperature,zbar,sound_speed; };
  KOKKOS_INLINE_FUNCTION Real GetDensityFloor() const { return 0.; }
  KOKKOS_INLINE_FUNCTION Real GetInternalEFloor() const { return 0.; }
  KOKKOS_INLINE_FUNCTION Real GetInternalECeiling() const { return 1.e30; }
  KOKKOS_INLINE_FUNCTION Real GetVelocityCeiling() const { return 0.; }
  KOKKOS_INLINE_FUNCTION Real GetPressureFloor() const { return 0.; }
  KOKKOS_INLINE_FUNCTION Real GetTemperatureFloor() const { return 0.; }
  KOKKOS_INLINE_FUNCTION Real GetTemperatureCeiling() const { return 1.e30; }
  KOKKOS_INLINE_FUNCTION bool IsTabulated() const { return false; }
  KOKKOS_INLINE_FUNCTION bool ContainsDensityEnergy(Real r, Real e) const {
    return r > 0. && e > 0.;
  }
  KOKKOS_INLINE_FUNCTION State StateFromDensityInternalEnergy(Real r,Real e) const {
    return {2.*r*e/3.,NAN,NAN,1.};
  }
};
void CheckStencil() {
  const IndexRange ib{2,9},jb{2,9},kb{0,0};
  // i=1 and corner (1,1) enter edge-EMF coefficients; i=0 never does.
  for (int probe=0;probe<3;++probe) {
    Real dt=std::numeric_limits<Real>::max();
    Kokkos::parallel_reduce("Test CT coefficient halo",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExecSpace(),{0,0},{12,12}),
      KOKKOS_LAMBDA(int j,int i,Real &result) {
        if (!InCompatibleOhmicStencil(i,j,0,ib,jb,kb)) return;
        const bool high=(probe==0 && i==1 && j==4) ||
                        (probe==1 && i==1 && j==1) ||
                        (probe==2 && i==0 && j==4);
        const Real local=CompatibleOhmicCellTimestep(high ? .2 : .02,.4,1./32.);
        result=fmin(result,local);
      },Kokkos::Min<Real>(dt));
    const Real expected=.25*.4/(32.*32.*(probe==2 ? .02 : .2));
    Require(std::abs(dt-expected)<1.e-15,"incorrect coefficient halo coverage");
  }
  Require(CompatibleOhmicCellTimestep(0.,.4,1.)==std::numeric_limits<Real>::max(),
          "zero resistivity must not constrain timestep");
  Require(CompatibleOhmicCellTimestep(-1.,.4,1.)==0.,"negative eta accepted");
  Require(CompatibleOhmicCellTimestep(NAN,.4,1.)==0.,"NaN eta accepted");
  TestPack u{Kokkos::View<Real***>("test state",8,12,12)};
  Kokkos::deep_copy(u.data,0.);
  Kokkos::parallel_for("Initialize valid state",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExecSpace(),{0,0},{12,12}),
    KOKKOS_LAMBDA(int j,int i) { u.data(IDN,j,i)=1.;u.data(IEN,j,i)=1.5; });
  for (int invalid=0;invalid<2;++invalid) {
    if (invalid)
      Kokkos::parallel_for("Invalidate unused second ghost",Kokkos::RangePolicy<DevExecSpace>(0,1),
        KOKKOS_LAMBDA(int) { u.data(IEN,4,0)=-1.; });
    int failures=0;
    Kokkos::parallel_reduce("Test full state halo",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(DevExecSpace(),{0,0},{12,12}),
      KOKKOS_LAMBDA(int j,int i,int &result) {
        Real pressure;
        result+=CheckCompatibleOhmicState(u,TestEOS{},0,0,j,i,pressure)!=0;
      },failures);
    Require(failures==invalid,"outer ghost state validation was lost");
  }
}
void CheckRetry(int rank) {
  int failures=rank==0 ? 4 : 0;
  Real observed=rank==0 ? .25 : 1.;
  ReduceCompatibleRKLStage(failures,observed);
  Require(failures==4 && observed==.25,"one-rank bound failure was not collective");
  CompatibleRKLRetry retry;
  Real step=1.;
  Require(retry.Reject(failures,observed,step),"first retry refused");
  Require(step==.5 && retry.halvings==1 && retry.Bound(1.)==.225,
          "retry did not halve and learn tighter bound");
  retry.Accept();
  Require(retry.halvings==0 && retry.Bound(1.)==.225,
          "accept must reset retry count while retaining learned bound");
  for (int n=0;n<8;++n)
    Require(retry.Reject(1,.001,step),"fewer than eight halvings allowed");
  const Real old=step;
  Require(!retry.Reject(4,.1,step) && step==old,"ninth halving allowed");
  Require(retry.Bound(1.)==.225,"invalid-state failure changed learned bound");
  failures=rank==0 ? 1 : 0;observed=1.;
  ReduceCompatibleRKLStage(failures,observed);
  Require(failures==1,"one-rank state failure was not collective");
  failures=rank==0 ? 2 : 0;
  ReduceCompatibleRKLStage(failures,observed);
  Require(failures==2,"one-rank invalid coefficient was not collective");
}
} // namespace
int main(int argc,char **argv) {
  int rank=0;
#ifdef MPI_PARALLEL
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
  Kokkos::initialize(argc,argv);
  int status=0;
  try { CheckStencil(); }
  catch (const std::exception &e) { std::cerr<<e.what()<<std::endl;status=1; }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&status,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (status==0) {
    try { CheckRetry(rank); }
    catch (const std::exception &e) { std::cerr<<e.what()<<std::endl;status=1; }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&status,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (rank==0 && status==0) std::cout<<"Compatible CT RKL bounds/retry checks passed\n";
  Kokkos::finalize();
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
  return status;
}
