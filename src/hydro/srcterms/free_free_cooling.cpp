#include "free_free_cooling.hpp"

namespace cooling {

FreeFreeCooling::FreeFreeCooling(ParameterInput *pin,
                                 std::shared_ptr<StateDescriptor> hydro_pkg) {
  PARTHENON_REQUIRE_THROWS(hydro_pkg->Param<std::string>("eos_type") == "tabulated",
                           "Free-free cooling requires hydro/eos=tabulated.");
  PARTHENON_REQUIRE_THROWS(hydro_pkg->AllParams().hasKey("units"),
                           "Free-free cooling requires a physical <units> block.");
  const auto units = hydro_pkg->Param<Units>("units");
  const auto eos = hydro_pkg->Param<TabulatedEOS>("tabulated_eos");
  const Real gaunt = pin->GetReal("cooling", "free_free_gaunt_factor");
  PARTHENON_REQUIRE_THROWS(std::isfinite(gaunt) && gaunt > 0.0,
                           "cooling/free_free_gaunt_factor must be finite and positive.");
  const Real floor = pin->GetOrAddReal("cooling", "free_free_temperature_floor", 1.e3);
  const Real hydro_floor = pin->GetOrAddReal("hydro", "Tfloor", -1.0);
  PARTHENON_REQUIRE_THROWS(std::isfinite(floor) && floor > 0.0 &&
                               std::isfinite(hydro_floor),
                           "Free-free temperature floors must be finite, and the "
                           "free-free floor must be positive.");
  const auto integrator = pin->GetOrAddString("cooling", "integrator", "rk45");
  PARTHENON_REQUIRE_THROWS(integrator == "rk12" || integrator == "rk45",
                           "Free-free cooling supports only rk12 and rk45; Townsend "
                           "requires a temperature-only rate and fixed ionization.");
  integrator_ = integrator == "rk12" ? CoolIntegrator::rk12 : CoolIntegrator::rk45;
  const int max_iter = pin->GetOrAddInteger("cooling", "max_iter", 100);
  const Real tolerance = pin->GetOrAddReal("cooling", "d_e_tol", 1.e-8);
  const Real cfl = pin->GetOrAddReal("cooling", "cfl", 0.1);
  PARTHENON_REQUIRE_THROWS(max_iter > 0 && std::isfinite(tolerance) && tolerance >= 0.0 &&
                               std::isfinite(cfl),
                           "Free-free cooling requires max_iter>0, finite d_e_tol>=0, "
                           "and finite cfl (nonpositive disables timestep limiting).");
  // C has units erg cm^3 s^-1 K^-1/2, with temperature evaluated in kelvin.
  rate_ = {eos, 1.426e-27 * gaunt * units.erg() * std::pow(units.cm(), 3) / units.s(),
           eos.MeanIonMass() * units.atomic_mass_unit()};
  settings_ = {true, eos, std::max(floor, hydro_floor),
               static_cast<unsigned int>(max_iter), tolerance, cfl};
}

void FreeFreeCooling::SrcTerm(MeshData<Real> *md, const Real dt) const {
  if (integrator_ == CoolIntegrator::rk12) {
    IntegrateCooling(md, dt, RK12Stepper(), rate_, settings_);
  } else {
    IntegrateCooling(md, dt, RK45Stepper(), rate_, settings_);
  }
}

Real FreeFreeCooling::EstimateTimeStep(MeshData<Real> *md) const {
  return EstimateCoolingTimeStep(md, rate_, settings_);
}

} // namespace cooling
