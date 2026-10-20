"""Black-box free-free tests using synthetic and production three-chart EOS files.

Run directly with --driver /path/to/athenaPK; all simulations use a temporary directory.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools"))
from generate_free_free_cooling import ATOMIC_MASS_UNIT_G, free_free_emissivity
import phdf

EV_TO_K = 11604.51812155008
CV = 1e3  # J kg^-1 K^-1
GM1 = 2 / 3


def make_eos(path, charge):
    """Exact constant-cv thermodynamic charts; constant or varying composition."""
    lr = np.linspace(-6, 6, 65)
    le = np.linspace(5, 12, 65)
    lp = np.linspace(-3, 24, 129)
    lt = np.linspace(-4, 7, 129)  # eV
    rr, ee = np.meshgrid(lr, le, indexing="ij")
    with h5py.File(path, "w") as f:
        f.attrs.update(
            format_version=1, log_base=10., inverse_margin_decades=1.,
            sound_speed_definition="adiabatic sound speed sqrt((dP/drho)_s)",
            array_index_order="quantity[density_index, thermodynamic_coordinate_index]",
        )
        def put(name, data, units=None):
            ds = f.create_dataset(name, data=data)
            if units is not None:
                ds.attrs["linear_units"] = units
        put("eos_rho_e/axes/log10_density", lr, "kg/m^3")
        put("eos_rho_e/axes/log10_specific_internal_energy", le, "J/kg")
        put("eos_rho_e/log10_pressure", rr + ee + np.log10(GM1), "Pa")
        put("eos_rho_e/log10_temperature", ee - np.log10(CV * EV_TO_K), "eV")
        put("eos_rho_e/log10_sound_speed", .5 * (ee + np.log10((1+GM1)*GM1)), "m/s")
        z = 2 + .5 * (ee - 5) + .1 * rr if charge is None else np.full_like(ee, charge)
        put("eos_rho_e/Zbar", z)
        put("eos_rho_p/axes/log10_density", lr, "kg/m^3")
        put("eos_rho_p/axes/log10_pressure", lp, "Pa")
        put("eos_rho_p/log10_specific_internal_energy", lp[None, :] - lr[:, None] - np.log10(GM1), "J/kg")
        put("eos_rho_T/axes/log10_density", lr, "kg/m^3")
        put("eos_rho_T/axes/log10_temperature", lt, "eV")
        put("eos_rho_T/log10_specific_internal_energy", np.broadcast_to(lt + np.log10(CV*EV_TO_K), (len(lr), len(lt))), "J/kg")


def eos_functions(path):
    with h5py.File(path) as f:
        g = f["eos_rho_e"]
        axes = (g["axes/log10_density"][...], g["axes/log10_specific_internal_energy"][...])
        fields = [RegularGridInterpolator(axes, g[name][...], bounds_error=False, fill_value=None)
                  for name in ("log10_temperature", "Zbar")]
    def state(rho, eps):
        x = np.array([[np.log10(rho), np.log10(eps)]])
        return 10**fields[0](x)[0] * EV_TO_K, fields[1](x)[0]
    return state


def run_case(driver, work, table, name, *, rho=1., temperature=1e6, pressure=None,
             gaunt=1.2, duration=1e-7, integrator="rk45", floor=1e3, cfl=0.,
             nlim=10000, units=(100., 1000., 1.), extra=(), expect_error=None):
    out = work / name
    out.mkdir()
    length, mass, time = units  # cgs base units
    # Multipliers from SI table values to code values.
    rf = .001 / (mass / length**3)
    ef = 1e4 / (length / time)**2
    pf = 10. / (mass / length / time**2)
    p = GM1*rho*CV*temperature if pressure is None else pressure
    template = (ROOT / "inputs/tabulated_sod.in").read_text()
    input_file = out / "input.in"
    input_file.write_text(template)
    args = [
        f"tabulated_eos/filename={table}",
        f"tabulated_eos/density_to_code={rf}",
        f"tabulated_eos/specific_internal_energy_to_code={ef}",
        f"tabulated_eos/pressure_to_code={pf}",
        f"tabulated_eos/sound_speed_to_code={100.*time/length}",
        f"units/code_length_cgs={length}", f"units/code_mass_cgs={mass}",
        f"units/code_time_cgs={time}",
        f"problem/sod/rho_l={rho*rf}", f"problem/sod/rho_r={rho*rf}",
        f"problem/sod/pres_l={p*pf}", f"problem/sod/pres_r={p*pf}",
        "parthenon/mesh/x1max=1e6",  # uniform box: keep hydro dt finite and nonlimiting
        "parthenon/mesh/nx1=8", "parthenon/meshblock/nx1=8",
        "parthenon/mesh/ix1_bc=periodic", "parthenon/mesh/ox1_bc=periodic",
        f"parthenon/time/tlim={duration/time}", f"parthenon/time/nlim={nlim}",
        f"parthenon/output0/dt={duration/time}",
        "cooling/enable_cooling=free_free", f"cooling/free_free_gaunt_factor={gaunt}",
        f"cooling/free_free_temperature_floor={floor}",
        f"cooling/integrator={integrator}", f"cooling/cfl={cfl}",
        "cooling/d_e_tol=1e-9", "cooling/max_iter=1000",
    ]
    if integrator is None:
        args = [arg for arg in args if not arg.startswith("cooling/integrator=")]
    if gaunt is None:
        args = [arg for arg in args if not arg.startswith("cooling/free_free_gaunt_factor=")]
    result = subprocess.run([str(driver), "-i", str(input_file), *args, *extra], cwd=out,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    (out / "run.log").write_text(result.stdout)
    if expect_error:
        assert result.returncode != 0 and expect_error in result.stdout, result.stdout
        print(f"PASS {name}: rejected invalid configuration")
        return
    assert result.returncode == 0, result.stdout
    files = sorted(out.glob("*.phdf"))
    assert len(files) >= 2, result.stdout
    def read(file):
        data = phdf.phdf(str(file))
        components = data.GetComponents(data.Info["ComponentNames"], flatten=False)
        return data.Time*time, components
    t0, start = read(files[0])
    t1, end = read(files[-1])
    for field in ("cons_density", "cons_momentum_density_1", "cons_momentum_density_2", "cons_momentum_density_3"):
        np.testing.assert_allclose(end[field], start[field], rtol=1e-12, atol=1e-12)
    # Uniform stationary boxes have no kinetic/magnetic contribution.
    eps0 = np.mean(start["cons_total_energy_density"] / start["cons_density"]) / ef
    eps1 = np.mean(end["cons_total_energy_density"] / end["cons_density"]) / ef
    assert np.isfinite(eps1) and eps1 > 0
    print(f"PASS {name}: t={t1:.6e} s, eps={eps1:.9e} J/kg")
    return t1-t0, eps0, eps1, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    args = parser.parse_args()
    driver = args.driver.resolve()
    # Independent numerical values and broadcasting checks for the reference API.
    rho = np.array([1e-4, 1e-3, 1e-2])
    temp = np.array([1e3, 1e6, 1e8])
    q = free_free_emissivity(rho, temp, 2., mean_ion_mass=27., gaunt_factor=1.5)
    expected = 1.426e-27*1.5*8*(rho/(27*ATOMIC_MASS_UNIT_G))**2*np.sqrt(temp)
    np.testing.assert_allclose(q, expected, rtol=2e-15)
    assert np.all(free_free_emissivity(rho, temp, 0., mean_ion_mass=27., gaunt_factor=1.5) == 0)
    with tempfile.TemporaryDirectory(prefix="athenapk-free-free-") as tmp:
        work = Path(tmp)
        # Constant-charge analytic solution sqrt(T)=sqrt(T0)-a*t/(2*cv).
        for z, g, rho_si, temp0 in ((0, 1.2, 1., 1e6), (1, 1.2, .1, 1e5),
                                    (2, 1.2, 1., 1e6), (4, 1.2, 1., 1e6),
                                    (2, 2.4, 1., 1e6), (2, 1.2, 1., 1e8)):
            table = work / f"constant-{z}.h5"
            make_eos(table, z)
            q0 = free_free_emissivity(rho_si*.001, temp0, z, mean_ion_mass=27., gaunt_factor=g)
            loss0 = q0/(rho_si*.001)*1e-4  # J/kg/s
            duration = .3*CV*temp0/loss0 if z else 1e-7
            elapsed, eps0, eps1, _ = run_case(driver, work, table, f"analytic-z{z}-g{g}-T{temp0}",
                                            rho=rho_si, temperature=temp0, gaunt=g, duration=duration,
                                            cfl=.05 if z == 0 else 0.)
            expected_eps = CV*(np.sqrt(eps0/CV) - loss0/np.sqrt(temp0)*elapsed/(2*CV))**2
            np.testing.assert_allclose(eps1, expected_eps, rtol=2e-6)
        table = work / "constant-2.h5"
        baseline = run_case(driver, work, table, "units-baseline", duration=1e-8, integrator=None)
        scaled = run_case(driver, work, table, "units-scaled", duration=1e-8, units=(1., .001, 1e-6))
        np.testing.assert_allclose(scaled[2], baseline[2], rtol=2e-9)
        rk12 = run_case(driver, work, table, "rk12", duration=1e-8, integrator="rk12")
        np.testing.assert_allclose(rk12[2], baseline[2], rtol=2e-6)
        stiff = run_case(driver, work, table, "stiff-floor", duration=1e-4,
                         extra=("cooling/max_iter=4",), nlim=1)
        np.testing.assert_allclose(stiff[3]["eos_temperature"], 1e3, rtol=2e-6)
        under = run_case(driver, work, table, "under-floor", temperature=500., duration=1e-10)
        np.testing.assert_allclose(under[3]["eos_temperature"], 1e3, rtol=2e-6)
        hydro_floor = run_case(driver, work, table, "hydro-floor", temperature=500., duration=1e-10,
                               extra=("hydro/Tfloor=2000",))
        np.testing.assert_allclose(hydro_floor[3]["eos_temperature"], 2000., rtol=2e-6)
        limited = run_case(driver, work, table, "timestep", duration=1., cfl=.05, nlim=1)
        q0 = free_free_emissivity(.001, 1e6, 2., mean_ion_mass=27., gaunt_factor=1.2)
        np.testing.assert_allclose(limited[0], .05*limited[1]/(q0/.001*1e-4), rtol=2e-8)
        run_case(driver, work, table, "missing-gaunt", gaunt=None,
                 expect_error="free_free_gaunt_factor")
        for name, overrides, message in (
            ("bad-gaunt", ("cooling/free_free_gaunt_factor=0",), "must be finite and positive"),
            ("bad-floor", ("cooling/free_free_temperature_floor=0",), "free-free floor must be positive"),
            ("bad-maxiter", ("cooling/max_iter=0",), "requires max_iter>0"),
            ("bad-integrator", ("cooling/integrator=townsend",), "supports only rk12 and rk45"),
            ("bad-eos", ("hydro/eos=adiabatic", "hydro/gamma=1.666666666666667", "hydro/He_mass_fraction=0.25"), "requires hydro/eos=tabulated"),
        ):
            run_case(driver, work, table, name, extra=overrides, expect_error=message)
        # Changing charge and density: high-accuracy independent ODE integration.
        variable = work / "variable.h5"
        make_eos(variable, None)
        production = ROOT / "inputs/tables/tabulated_eos.h5"
        for name, eos_path, rho_si, pressure, duration in (
            ("varying-charge", variable, 1., None, 5e-8),
            ("varying-charge-density", variable, .1, None, 5e-7),
            ("aluminum-eos", production, 1., 1e9, 1e-8),
        ):
            if not eos_path.exists():
                raise FileNotFoundError(f"Required EOS fixture is missing: {eos_path}")
            elapsed, eps0, eps1, fields = run_case(driver, work, eos_path, name,
                                                 rho=rho_si, pressure=pressure, duration=duration)
            state = eos_functions(eos_path)
            def rhs(t, y):
                temperature, z = state(rho_si, y[0])
                if temperature <= 1e3:
                    return [0.]
                q = free_free_emissivity(rho_si*.001, temperature, z,
                                         mean_ion_mass=27., gaunt_factor=1.2)
                return [-q/(rho_si*.001)*1e-4]
            reference = solve_ivp(rhs, (0., elapsed), [eps0], method="DOP853", rtol=1e-11, atol=1e-6)
            assert reference.success
            np.testing.assert_allclose(eps1, reference.y[0, -1], rtol=2e-5)
            if eos_path == variable:
                assert abs(state(rho_si, eps0)[1] - state(rho_si, eps1)[1]) > .01
        # Short UCT-HLLE integration using the existing tabulated reconnection setup.
        smoke = work / "uct-smoke"
        smoke.mkdir()
        result = subprocess.run([
            str(driver), "-i", str(ROOT / "inputs/ct_glm_compatible/pulsed_reconnection_tabulated_smoke.in"),
            f"tabulated_eos/filename={production}", "cooling/enable_cooling=free_free",
            "cooling/free_free_gaunt_factor=1.2", "parthenon/time/tlim=1e-4",
            "parthenon/time/nlim=2", "problem/pulsed_reconnection/drive_enable=false",
        ], cwd=smoke, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        assert result.returncode == 0, result.stdout
        data = phdf.phdf(str(sorted(smoke.glob("*.phdf"))[-1]))
        fields = data.GetComponents(data.Info["ComponentNames"], flatten=False)
        assert data.Time > 0
        for name, values in fields.items():
            assert np.isfinite(values).all(), name
        assert fields["eos_temperature"].min() >= 2000*(1-1e-5)
        assert fields["prim_pressure"].min() > 0
        assert any("magnetic" in name for name in fields)
        print("PASS UCT-HLLE free-free smoke")
    print("All free-free cooling checks passed")


if __name__ == "__main__":
    main()
