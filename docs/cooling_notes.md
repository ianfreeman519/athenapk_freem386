---
title: Cooling function conventions
author: Ben Wibking
date: February 12, 2024
---

# Summary

Several different conventions exist for the definition of a collisional ionization equilibrium (CIE) cooling curve. They are explained in detail below, with references.

# Conventions

## $n^2 \Lambda$ convention

This convention uses the *total number density* $n$ to define the volumetric cooling rate. The total particle number density is defined as the sum of the number of ions, neutrals, and electrons divided by the volume.

The total number density $n$ is related to the dimensionless mean molecular weight $\mu$ by:

$$n = \frac{\rho}{\mu m_H} \, ,$$ {#eq:number_density}

where $\rho$ is the mass density of the fluid, $m_H$ is the mass of the hydrogen atom, and $\mu$ is the dimensionless mean molecular weight. This may be taken as the definition of $\mu$.

In this convention, the volumetric cooling rate is given by

$$S_{cool} = n^2 \Lambda(T) \, ,$$ {#eq:cooling_rate_nsqLambda}

where $S_{cool}$ is the volumetric cooling source term, $n$ is the total number density and $\Lambda(T)$ is the CIE cooling curve in this convention.

This convention is used by the PLUTO code (see the [PLUTO User Guide](http://plutocode.ph.unito.it/userguide.pdf), section 9.2, equation 9.5).

## $n_i n_e \Lambda$ convention

In this convention, the volumetric cooling rate is given by

$$S_{cool} = n_i n_e \Lambda(T) \, ,$$ {#eq:cooling_rate_nineLambda}

where $S_{cool}$ is the volumetric cooling source term, $n_i$ is the number density of ions (*usually* including neutrals by convention, but this difference should be negligible when CIE is valid), $n_e$ is the number density of electrons, and $\Lambda(T)$ is the CIE cooling curve in this convention.

This convention is adopted by Sutherland and Dopita (1993) in their tabulation of CIE cooling curves (their section 5.2; see also their equations 51-58).

Sutherland and Dopita additionally comment: *"The normalization factor for the cooling function $\Lambda_N$ is taken as $n_t n_e$ where $n_t$ is the total number density of ions and $n_e$ is the total number density of electrons. This definition is generally valid and independent of composition. It differs from the commonly used factor $n_H n_e$ or even $n_e^2$ commonly used in hydrogen-rich plasmas."*


## $n_H n_e \Lambda$ convention

In this convention, the volumetric cooling rate is given by

$$S_{cool} = n_H n_e \Lambda(T) \, ,$$ {#eq:cooling_rate_nHneLambda}

where $S_{cool}$ is the volumetric cooling source term, $n_H$ is the number density of hydrogen *nuclei* (not ions), $n_e$ is the number density of electrons, and $\Lambda(T)$ is the CIE cooling curve in this convention.

This convention is adopted by Schure et al. (2009) in their definition of $\Lambda_N$. However, this is **not** the same $\Lambda_N$ as used by Sutherland and Dopita (1993)!

This convention is adopted by Gnat and Sternberg (2007) in their definition of $\Lambda$.

The following image illustrates a comparison between different cooling curves in this notation:
![nHne cooling](./cooling/cooling_curves_nHneLambda.png)

## $n_H^2 \Lambda$ convention

In this convention, the volumetric cooling rate is given by

$$S_{cool} = n_H^2 \Lambda(T) \, ,$$ {#eq:cooling_rate_nHsqLambda}

where $S_{cool}$ is the volumetric cooling source term, $n_H$ is the number density of hydrogen *nuclei* (not ions), and $\Lambda(T)$ is the CIE cooling curve in this convention.

This convention is adopted by Dalgarno and McCray (1972) (their equation 24), and Schure et al. (2009) in their definition of $\Lambda_{hd}$ (their equation 2).

### AthenaPK implementation

This is the convention adopted by AthenaPK, except $n_H$ is computed assuming zero metals in the mass budget:

$$ x_H = 1 - Y \, , $$

$$ n_H = X_H \rho / m_H  = (1 - Y) \rho / m_H \, .$$

The difference in neglecting metals in the mass budget in computing $n_H$ is only a $\sim 2$ percent effect on $n_H$ (since metals are 2 percent by mass for solar metallicity) and a $\sim 4$ percent effect on the source term.

The Schure table in the GitHub repository uses the $n_H n_e$ convention, rather than the $n_H^2$ convention. This causes an error in the cooling rate of $n_e / n_H$, or about 20 percent for temperatures above $10^{5.5}$ K.

Above $\sim 10^4$ K, there is a percent-level error in the mean molecular weight due to neglecting the metal contribution to the electron fraction. *Users should be aware that at low absolute electron fraction, metals contribute a substantial proportion of the electrons, so this approximation is no longer valid below $\sim 10^4$ K.* The temperature is computed as:

$$ T = \left( \frac{\rho}{\mu m_H} \right)^{-1} \left( \frac{P}{k_B} \right) \, ,$$
where the dimensionless mean molecular weight $\mu$ is computed assuming a zero metallicity gas:

$$ \mu = \left( \frac{3}{4} Y + 2(1 - Y) \right)^{-1} = \left( 2 - \frac{5}{4} Y \right)^{-1} \, ,$$

where $Y$ is the Helium mass fraction.

For a more precise calculation of $\mu$ assuming a scaled-solar composition of metals, we have:

$$ \mu = \left( 1 - x_e \right) \left( X + \frac{Y}{4} + \frac{Z}{\bar A} \right)^{-1} \, , $$

where $\bar A$ is the mass-weighted mean atomic weight of metals:

$$ \bar A = 16 \, .$$

Additionally, for full ionization for all species, we have:

$$ x_e = \frac{X + 2 Y (m_H/m_{He}) + (\bar A/2) Z (m_H/m_{\bar Z})}{2 X + 3 Y (m_H/m_{He}) + (1 + \bar A/2) 
Z (m_H/m_{\bar Z})} \approx \frac{X + Y/2 + Z/2}{2 X + 3 Y/4 + (1/{\bar A} + 1/2) Z}$$


For $Y = 0.24$ and $Z = 0.02$, the mean molecular weight for complete ionization is $\mu \approx 0.60$ (if $Z = 0$, then this yields $\mu \approx 0.59$).

## $n_e^2 \Lambda$ convention

In this convention, the volumetric cooling rate is given by

$$S_{cool} = n_e^2 \Lambda(T) \, ,$$ {#eq:cooling_rate_nHsqLambda}

where $S_{cool}$ is the volumetric cooling source term, $n_e$ is the number density of electrons, and $\Lambda(T)$ is the CIE cooling curve in this convention.

For the case of a hydrogen-only fully-ionized plasma *only*, this is equivalent to the $n_H^2 \Lambda$ convention, since each hydrogen atom contributes exactly one electron and there are no other sources of electrons.

I am not aware of any common tables that use this convention. However, it is noted as one of the possible conventions by Sutherland and Dopita (1993).

## EOS-aware free-free cooling

For a single-component plasma using `hydro/eos = tabulated`, select the analytic
free-free source instead of a temperature-only cooling table:

```ini
<cooling>
enable_cooling = free_free
free_free_gaunt_factor = 1.2  # illustrative; choose explicitly for your model
free_free_temperature_floor = 1e3  # K; default 1e3
integrator = rk45           # default; rk12 also supported
cfl = 0.1                  # cooling timestep safety factor; <=0 disables limit
max_iter = 100              # maximum adaptive subcycles
d_e_tol = 1e-8              # relative specific-energy integration tolerance
```

The finite, positive, frequency-integrated Gaunt factor is required. No table file,
`lambda_units_cgs`, or density normalization parameter is used by this mode.
`none`, `tabular`, and `free_free` are mutually exclusive values of
`cooling/enable_cooling`. Townsend integration is not supported for free-free.

At **every Runge–Kutta stage**, the EOS supplies temperature $T$ in kelvin and
mean charge $\bar Z$ at the current density and specific internal energy. With
$A$ from `tabulated_eos/mean_ion_mass_hydrogen` (the EOS's ion mass in atomic mass
units), the source computes

$$
n_i=\rho/(A m_u),\quad n_e=\bar Z n_i,\quad
Q_{\rm ff}=1.426\times10^{-27}g_B\bar Z^2 n_e n_i\sqrt{T}
\quad[\mathrm{erg\,cm^{-3}\,s^{-1}}].
$$

The integrated specific-energy rate is $d\epsilon/dt=-Q_{\rm ff}/\rho$.
The implementation converts the coefficient and ion mass to code units through
`Units`; the numerical temperature remains in kelvin. Exactly neutral material
has zero emission. At fixed ion density and temperature, the rate scales as
$\bar Z^3$, equivalently $n_e^2\bar Z\sqrt T$.

The effective temperature floor is the larger of `hydro/Tfloor` and
`free_free_temperature_floor`. As with existing adaptive cooling, an initial state
below this floor is repaired to it. Cooling changes total energy and updates EOS
pressure, preserving density, momentum, and magnetic fields. There is no artificial
upper temperature cutoff; the tabulated EOS's own bounds and diagnostics still
apply. The cooling timestep estimate is `cfl * epsilon / abs(deps_dt)`, or infinity
when no cooling occurs. Existing subcycle controls retain their behavior, including
floor saturation for unresolved stiff cooling; `d_e_tol` is not a guarantee of
accuracy once the subcycle budget is exhausted.

This is an optically thin, nonrelativistic, single-temperature **mean-charge
approximation**: $\bar Z^2$ replaces the charge-resolved second moment. It includes
neither charge-state populations nor charge-dependent Gaunt factors, radiation
transport, electron-electron bremsstrahlung, or separate electron/ion temperatures.
A constant $g_B$ must be selected with these accuracy limits in mind.

`scripts/generate_free_free_cooling.py` is now a reference rate calculator rather
than a temperature-only table generator. Supply CSV rows `rho[g/cm^3], T[K], Zbar`
and explicit `--mean-ion-mass` and `--gaunt-factor`; it writes the states,
volumetric emissivity, and specific-energy loss as CSV. It imposes no fixed charge.
