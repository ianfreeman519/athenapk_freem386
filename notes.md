# RunSimplifiedSDCThermalCoupling: Mathematical Notes

Let the stage begin at time $t^n$ and end at time

$$
t^{n+1} = t^n + \Delta t,
$$

where $\Delta t$ is the stage timestep passed into
`RunSimplifiedSDCThermalCoupling()`.

The coupled thermal solve evolves the specific internal energy $e(t)$ over
that stage using a lagged hydro contribution, tabular cooling, and ohmic
heating.

## Governing ODE

During one outer thermal iteration $m$, the solve can be written as

$$
\frac{de}{dt} = A_e + Q_{\Omega}^{(m)} + Q_{\mathrm{cool}}(e),
\qquad t \in [t^n, t^{n+1}]
$$

with initial condition

$$
e(t^n) = e^n.
$$

Here:

- $e^n$ is the accepted stage-start specific internal energy.
- $A_e$ is the hydro-only stage-average internal-energy rate, frozen during
  the thermal solve.
- $Q_{\Omega}^{(m)}$ is the ohmic heating rate per unit mass, rebuilt from
  the current outer iterate.
- $Q_{\mathrm{cool}}(e)$ is the tabulated cooling rate per unit mass,
  evaluated continuously as a function of the evolving $e$.

The hydro contribution is approximated by

$$
A_e \approx \frac{e_{\mathrm{adv}} - e^n}{\Delta t},
$$

where $e_{\mathrm{adv}}$ is the specific internal energy after the hydro
stage update but before the thermal correction.

## Volumetric Source Form

The code constructs the non-cooling source in volumetric form:

$$
S_{\mathrm{ext}}^{(m)} = \rho A_e + S_{\Omega}^{(m)}.
$$

The ODE actually passed to the thermal subcycler is

$$
\frac{de}{dt}
=
Q_{\mathrm{cool}}(e)
+
\frac{S_{\mathrm{ext}}^{(m)}}{\rho}.
$$

Equivalently,

$$
\frac{de}{dt}
=
Q_{\mathrm{cool}}(e)
+
A_e
+
Q_{\Omega}^{(m)}.
$$

## Outer Thermal Iteration

Let

$$
e^{(0)} = e^n.
$$

Then for outer iteration $m = 0,1,\dots,M-1$:

1. Rebuild ohmic heating from the current iterate:

$$
S_{\Omega}^{(m)} = \mathcal{O}(e^{(m)}).
$$

2. Freeze the non-cooling source over the whole stage:

$$
S_{\mathrm{ext}}^{(m)} = \rho A_e + S_{\Omega}^{(m)}.
$$

3. Solve the stage ODE over the full interval:

$$
e^{(m+1)}
=
\Phi_{\Delta t}
\left(
e^n;\, A_e,\; S_{\Omega}^{(m)},\; Q_{\mathrm{cool}}
\right).
$$

After the last iteration, the accepted thermal state is

$$
e^{(*)} = e^{(M)}.
$$

The code then commits only the thermal correction relative to the hydro-only
stage result:

$$
\Delta e_{\mathrm{thermal}} = e^{(*)} - e_{\mathrm{adv}},
$$

$$
E \leftarrow E + \rho\,\Delta e_{\mathrm{thermal}}.
$$

So the thermal solve does not replace the hydro stage; it corrects only the
internal-energy part of that stage update.

## What "Solve Over the Full $\Delta t$" Means

The operator $\Phi_{\Delta t}$ is not one explicit step. It is approximated
by adaptive subcycling across the whole stage interval.

Write the interval as

$$
t_0 = t^n,\quad
t_1 = t_0 + h_0,\quad
t_2 = t_1 + h_1,\quad
\dots,\quad
t_K = t^n + \Delta t,
$$

with

$$
\sum_{k=0}^{K-1} h_k = \Delta t.
$$

On each subinterval $[t_k, t_{k+1}]$, the code advances

$$
\frac{de}{dt} = f^{(m)}(e),
$$

where

$$
f^{(m)}(e) = Q_{\mathrm{cool}}(e) + \frac{S_{\mathrm{ext}}^{(m)}}{\rho}.
$$

During a single outer iteration $m$, the non-cooling part
$\frac{S_{\mathrm{ext}}^{(m)}}{\rho}$ is frozen. Only the cooling term varies
continuously with $e$ inside the subcycled solve.

## RK12 Subcycling

For one substep of size $h_k$, starting from $e_k$, the code uses an
embedded first/second-order Runge-Kutta pair.

Low-order Euler estimate:

$$
e_{k+1}^{[1]} = e_k + h_k f^{(m)}(e_k)
$$

High-order Heun estimate:

$$
e_{k+1}^{[2]}
=
e_k + \frac{h_k}{2}
\left[
f^{(m)}(e_k) + f^{(m)}\!\left(e_{k+1}^{[1]}\right)
\right]
$$

The accepted substep value is the higher-order estimate:

$$
e_{k+1} = e_{k+1}^{[2]}.
$$

The local relative error estimate is

$$
\mathrm{err}_k
=
\left|
\frac{e_{k+1}^{[2]} - e_{k+1}^{[1]}}{e_{k+1}^{[2]}}
\right|.
$$

If $\mathrm{err}_k$ is too large, the code rejects the substep and retries
with a smaller $h_k$. If it is acceptable, the substep is accepted and the
next size is proposed using

$$
h_{k+1}
=
0.95\, h_k
\left(\frac{\mathrm{tol}}{\mathrm{err}_k}\right)^2,
$$

subject to bounds, including the minimum substep

$$
h_{\min} = \frac{\Delta t}{N_{\max}},
$$

where $N_{\max}$ is `max_iter`.

So "subcycled over the full $\Delta t$" means

$$
e^{(m+1)}
\approx
\left(
\phi_{h_{K-1}} \circ \cdots \circ \phi_{h_1} \circ \phi_{h_0}
\right)(e^n),
$$

with adaptive substeps $h_k$ whose sum is $\Delta t$.

## Role of Each Term

The ODE can be interpreted as

$$
\frac{de}{dt}
=
\underbrace{A_e}_{\text{hydro advection/compression, lagged}}
+
\underbrace{Q_{\Omega}^{(m)}}_{\text{ohmic heating, rebuilt per outer iterate}}
+
\underbrace{Q_{\mathrm{cool}}(e)}_{\text{radiative cooling, evaluated during subcycling}}.
$$

- $A_e$ carries the hydro stage's effect on internal energy into the thermal
  solve.
- $Q_{\Omega}^{(m)}$ adds resistive heating, but is only updated between
  outer iterations, not within each RK substep.
- $Q_{\mathrm{cool}}(e)$ is the only term updated continuously inside the
  subcycled ODE solve.

## Code-to-Math Dictionary

- `dt` in `RunSimplifiedSDCThermalCoupling` $\rightarrow \Delta t$
- `eint_stage_start` $\rightarrow e^n$
- `eint_adv` $\rightarrow e_{\mathrm{adv}}$
- `thermal_ae` $\rightarrow A_e \approx (e_{\mathrm{adv}} - e^n)/\Delta t$
- `eint_sdc` $\rightarrow e^{(m)}$, the current outer iterate
- `eint_next` $\rightarrow e^{(m+1)}$, the next outer iterate
- `thermal_src_ohmic` $\rightarrow S_{\Omega}^{(m)}$, volumetric ohmic source
- `thermal_src_total` inside the iteration loop
  $\rightarrow S_{\mathrm{ext}}^{(m)} = \rho A_e + S_{\Omega}^{(m)}$
- `heating_rate = thermal_src / rho`
  $\rightarrow A_e + Q_{\Omega}^{(m)}$
- `cooling_table_obj.DeDt(e,\rho)` $\rightarrow Q_{\mathrm{cool}}(e)$
- `sub_dt` $\rightarrow h_k$, adaptive substep size
- `sub_t` $\rightarrow t_k - t^n$, elapsed stage time
- `internal_e` inside the subcycler $\rightarrow e_k$, current subcycled state
- `internal_e_next_l` $\rightarrow e_{k+1}^{[1]}$, Euler estimate
- `internal_e_next_h` $\rightarrow e_{k+1}^{[2]}$, Heun estimate
- `d_e_err` $\rightarrow \mathrm{err}_k$
- `d_e_tol` $\rightarrow \mathrm{tol}$
- `min_sub_dt = dt / max_iter` $\rightarrow h_{\min} = \Delta t / N_{\max}$

## Key Approximation

The main approximation is

$$
Q_{\Omega}(e(t)) \approx Q_{\Omega}(e^{(m)})
\quad \text{during one outer iteration.}
$$

So the code is not solving the fully coupled nonlinear problem continuously in
time. It solves a sequence of lagged-source ODEs over $\Delta t$, where
cooling responds within the subcycles but ohmic heating is only refreshed
between outer iterations.

## RKL2 Diffusion Integrator: Mathematical Notes

This section describes the operator-split `rkl2` diffusion integrator in the
same equation-first style.

## Governing Diffusion Problem

Let $U(t)$ denote the vector of evolved cell-centered conserved variables acted
on by the diffusive operators. Abstractly, the diffusion update is written as

$$
\frac{dU}{dt} = \mathcal{D}(U),
$$

where $\mathcal{D}$ is the discrete diffusive operator assembled from
conduction, viscosity, and resistivity flux divergences.

For a single explicit parabolic step, stability would require

$$
\Delta t \lesssim \Delta t_{\mathrm{diff}},
$$

where $\Delta t_{\mathrm{diff}}$ is the strict diffusive timestep estimate.

The purpose of RKL2 super-time-stepping is to advance the diffusion equation
over a much larger interval

$$
\tau \gg \Delta t_{\mathrm{diff}}
$$

using a sequence of stabilized explicit stages.

## Operator-Split Placement in the Hydro Driver

The diffusion update is Strang split around the main hyperbolic stage. For a
full hydro timestep $\Delta t_{\mathrm{hydro}}$, the driver applies:

1. a pre-hydro diffusion half-step of size

$$
\tau = \frac{1}{2}\Delta t_{\mathrm{hydro}},
$$

2. the main hydro update,
3. a post-hydro diffusion half-step of the same size $\tau$.

So the diffusion operator is approximated as

$$
U^{n+1}
\approx
\exp\!\left(\frac{\Delta t_{\mathrm{hydro}}}{2}\mathcal{D}\right)
\,
\mathcal{H}_{\Delta t_{\mathrm{hydro}}}
\,
\exp\!\left(\frac{\Delta t_{\mathrm{hydro}}}{2}\mathcal{D}\right)
U^n,
$$

where $\mathcal{H}_{\Delta t_{\mathrm{hydro}}}$ denotes the hyperbolic stage
update.

## Super-Time Interval and Number of Stages

Within one Strang half-step, the code advances the diffusion problem over the
super-time interval $\tau$ using $s$ RKL2 stages.

The implemented stage count is

$$
s
=
\left\lfloor
\frac{1}{2}\left(\sqrt{9 + 16\,\tau/\Delta t_{\mathrm{diff}}} - 1\right)
\right\rfloor + 1,
$$

followed by forcing $s$ to be odd.

This is the code's practical choice for how many stabilized stages are needed
to cover the interval $\tau$ while remaining explicit.

## Fixed Reference State and Diffusion Operator

At the start of one super-time step, define the initial state

$$
Y_0 = U(t^n).
$$

The code then computes the diffusive residual at that initial state:

$$
M Y_0 := \mathcal{D}(Y_0).
$$

This quantity is stored once and reused in every RKL2 stage.

For later stages, the code also evaluates the current residual

$$
M Y_{j-1} := \mathcal{D}(Y_{j-1}).
$$

So the method uses:

- one residual frozen at the beginning of the super-step, $M Y_0$,
- one residual evaluated at the current stage state, $M Y_{j-1}$.

## First RKL2 Stage

The first stage is

$$
Y_1 = Y_0 + \tilde{\mu}_1 \,\tau\, M Y_0,
$$

with

$$
\tilde{\mu}_1 = \frac{4/3}{s^2 + s - 2}.
$$

The code also stores the two most recent stage states:

$$
Y_{j-1}, \quad Y_{j-2}.
$$

After the first stage these are initialized as

$$
Y_{j-1} \leftarrow Y_1, \qquad Y_{j-2} \leftarrow Y_0.
$$

## Recurrence for Stages $j \ge 2$

For $j = 2, 3, \dots, s$, the code computes the coefficients

$$
b_j = \frac{j^2 + j - 2}{2j(j+1)},
$$

and then

$$
\mu_j = \frac{2j-1}{j}\frac{b_j}{b_{j-1}},
$$

$$
\nu_j = -\frac{j-1}{j}\frac{b_j}{b_{j-2}},
$$

$$
\tilde{\mu}_j = \mu_j w_1,
$$

$$
\tilde{\gamma}_j = -(1-b_{j-1})\tilde{\mu}_j,
$$

with

$$
w_1 = \frac{4}{s^2 + s - 2}.
$$

The implemented stage update is then

$$
Y_j
=
\mu_j Y_{j-1}
+
\nu_j Y_{j-2}
+
\left(1-\mu_j-\nu_j\right)Y_0
+
\tilde{\mu}_j \tau\, M Y_{j-1}
+
\tilde{\gamma}_j \tau\, M Y_0.
$$

After each stage, the recursion is shifted forward:

$$
Y_{j-2} \leftarrow Y_{j-1},
\qquad
Y_{j-1} \leftarrow Y_j.
$$

After the final stage,

$$
U^{n+1/2\text{ diff}} = Y_s
$$

for that Strang half-step.

## What the Integrals Mean

Formally, the exact diffusive evolution over one super-time interval is

$$
U(t^n+\tau) = U(t^n) + \int_{t^n}^{t^n+\tau} \mathcal{D}(U(t))\,dt.
$$

The RKL2 method approximates this integral by a stabilized explicit polynomial
propagator built from repeated evaluations of $\mathcal{D}$.

Unlike the thermal RK12 subcycling, the code does not partition $\tau$ into a
sequence of adaptive physical substeps

$$
\tau = \sum_k h_k.
$$

Instead, it applies a fixed number of algebraic stages

$$
Y_0 \to Y_1 \to \cdots \to Y_s,
$$

where the recurrence is designed so that the resulting explicit stability
polynomial remains stable for parabolic operators over a large interval
$\tau$.

So the "subcycling" in RKL2 is not adaptive timestep subcycling in physical
time. It is a stabilized multi-stage explicit approximation to the integral
over $\tau$.

## What Diffusion Operator is Being Applied

At each stage, the code assembles the diffusive fluxes and then takes their
flux divergence:

$$
\mathcal{D}(U) = -\nabla_h \cdot F_{\mathrm{diff}}(U),
$$

where $F_{\mathrm{diff}}$ may include:

- thermal conduction fluxes,
- viscous fluxes,
- resistive fluxes.

In coupled-ohmic thermal runs there is one important ownership change:

- the STS diffusion update still advances the magnetic resistive part,
- but the resistive contribution to total energy is withheld from the STS
  flux update so the thermal solver remains the sole owner of ohmic heating.

Mathematically, in that coupled-ohmic case the RKL2 operator acts like

$$
\mathcal{D}_{\mathrm{STS}}(U)
=
\mathcal{D}_{\mathrm{cond}}(U)
+
\mathcal{D}_{\mathrm{visc}}(U)
+
\mathcal{D}_{B,\Omega}(U),
$$

without the legacy resistive internal-energy contribution.

## Timestep Logic

The strict diffusive timestep estimate $\Delta t_{\mathrm{diff}}$ is still
computed from the active diffusion operators, but in `rkl2` mode it does not
normally limit the global hydro timestep directly.

Instead:

- the hydro timestep is usually controlled by hyperbolic limits,
- an optional cap `rkl2_max_dt_ratio` can enforce

$$
\frac{\Delta t_{\mathrm{hydro}}}{\Delta t_{\mathrm{diff}}}
\le
\text{max ratio},
$$

- ohmic heating may still impose a separate thermal/heating timestep
  restriction.

So RKL2 removes the strict parabolic stability limit from the outer timestep
selection, replacing it with a stabilized explicit multi-stage solve over the
Strang half-step interval.

## Code-to-Math Dictionary for RKL2

- `tau` $\rightarrow \tau$, one Strang half-step super-time interval
- `dt_diff` $\rightarrow \Delta t_{\mathrm{diff}}$, strict explicit diffusive timestep
- `s_rkl` $\rightarrow s$, number of RKL2 stages
- `Y0` $\rightarrow Y_0$, state at the start of the super-step
- `MY0` $\rightarrow M Y_0 = \mathcal{D}(Y_0)$
- `Yjm1` $\rightarrow Y_{j-1}$, previous stage state
- `Yjm2` $\rightarrow Y_{j-2}$, two-stage-old state
- `MYjm1` $\rightarrow M Y_{j-1} = \mathcal{D}(Y_{j-1})$
- `mu_tilde_1` $\rightarrow \tilde{\mu}_1$
- `b_j` $\rightarrow b_j$
- `mu_j` $\rightarrow \mu_j$
- `nu_j` $\rightarrow \nu_j$
- `mu_tilde_j` $\rightarrow \tilde{\mu}_j$
- `gamma_tilde_j` $\rightarrow \tilde{\gamma}_j$
- `CalcDiffFluxes(...)` $\rightarrow$ assemble $F_{\mathrm{diff}}(U)$
- `FluxDivergence(...)` or `FluxDivHelper(...)` $\rightarrow -\nabla_h \cdot F_{\mathrm{diff}}(U)$

## Summary Formula

One RKL2 Strang half-step approximates

$$
U(t^n+\tau)
=
U(t^n) + \int_{t^n}^{t^n+\tau} \mathcal{D}(U(t))\,dt
$$

by the stabilized stage sequence

$$
Y_0 = U(t^n),
$$

$$
Y_1 = Y_0 + \tilde{\mu}_1 \tau\, \mathcal{D}(Y_0),
$$

and for $j \ge 2$,

$$
Y_j
=
\mu_j Y_{j-1}
+
\nu_j Y_{j-2}
+
\left(1-\mu_j-\nu_j\right)Y_0
+
\tilde{\mu}_j \tau\, \mathcal{D}(Y_{j-1})
+
\tilde{\gamma}_j \tau\, \mathcal{D}(Y_0),
$$

with the final accepted state

$$
U(t^n+\tau) \approx Y_s.
$$

## Coupled Thermal Solver Ordering with RKL2 Diffusion

When both

- `diffusion/integrator = rkl2`, and
- `thermal_source_solver/enabled = true`

are active in the supported coupled-ohmic configuration, the coupled thermal
solve does **not** run in the normal post-hydro unsplit source location.
Instead, it runs once in a special midpoint path during stage 1, after the
pre-hydro RKL2 diffusion half-step and before the main hydro flux update.

## Hydro Ordering

For one full hydro timestep $\Delta t_{\mathrm{hydro}}$, the relevant ordering
is:

1. pre-hydro RKL2 diffusion half-step of size

$$
\tau = \frac{1}{2}\Delta t_{\mathrm{hydro}},
$$

2. midpoint coupled thermal solve over the full hydro timestep
   $\Delta t_{\mathrm{hydro}}$,
3. main hydro stage update,
4. post-hydro RKL2 diffusion half-step of size $\tau$.

So the thermal solve is inserted between the two Strang-split diffusion
half-steps, but before the hyperbolic update.

## Thermodynamic State Used by the Midpoint Thermal Solve

Let $U^{n,\mathrm{pre\text{-}STS}}$ denote the state before the first RKL2
half-step, and let

$$
U^{n,\mathrm{mid}} =
\exp\!\left(\frac{\Delta t_{\mathrm{hydro}}}{2}\mathcal{D}_{\mathrm{STS}}\right)
U^{n,\mathrm{pre\text{-}STS}}
$$

denote the accepted state immediately after that pre-hydro STS half-step.

The midpoint thermal bookkeeping snapshots the specific internal energy from
that accepted state:

$$
e_{\mathrm{stage\_start}} = e\!\left(U^{n,\mathrm{mid}}\right).
$$

The code stores this same snapshot into:

- `eint_stage_start`
- `eint_sdc`
- `eint_adv`

at the beginning of the midpoint thermal bookkeeping.

So the coupled midpoint thermal solve starts from

$$
e(t^n) = e_{\mathrm{stage\_start}}.
$$

At this point, no hyperbolic advection/compression update has occurred yet.
That is why this branch does **not** use the lagged hydro source
$A_e = (e_{\mathrm{adv}} - e^n)/\Delta t$ that appears in the non-`rkl2`
simplified-SDC branch.

## Source Terms Used by the Midpoint Thermal Solve

During the midpoint path, the thermal solve uses:

- cooling rebuilt from the current thermal iterate,
- an ohmic heating source frozen from the accepted pre-hydro STS state.

### Cooling term

At outer thermal iteration $m$, the code rebuilds cooling from the current
thermal iterate:

$$
Q_{\mathrm{cool}}^{(m)}(e) = Q_{\mathrm{cool}}(e^{(m)}),
$$

in the sense that the cooling source field is reconstructed from the current
iterate `eint_sdc` before each outer pass.

Inside the inner RK12 thermal integration, the cooling rate is then evaluated
continuously as a function of the evolving subcycled internal energy.

### Ohmic term

Before the midpoint thermal iterations begin, the code constructs the accepted
pre-hydro ohmic source from the accepted post-STS state:

$$
S_{\Omega,\mathrm{pre}} =
\mathcal{O}\!\left(e_{\mathrm{stage\_start}}\right).
$$

This source is stored in `thermal_src_ohmic_pre_flux`.

During the midpoint thermal solve, that ohmic source is held fixed for the
entire solve:

$$
S_{\Omega}^{(m)} = S_{\Omega,\mathrm{pre}}
\qquad
\text{for all outer iterations } m.
$$

So unlike the non-`rkl2` simplified-SDC branch, the midpoint path does **not**
rebuild ohmic heating from the evolving outer iterate. It uses the frozen
pre-hydro STS snapshot throughout.

## ODE Solved in the Midpoint Thermal Path

Let $e^{(0)} = e_{\mathrm{stage\_start}}$. Then each outer iteration solves
over the full hydro timestep $\Delta t_{\mathrm{hydro}}$:

$$
\frac{de}{dt}
=
Q_{\mathrm{cool}}(e)
+
\frac{S_{\Omega,\mathrm{pre}}}{\rho},
\qquad
t \in [t^n, t^n + \Delta t_{\mathrm{hydro}}],
$$

with initial condition

$$
e(t^n) = e_{\mathrm{stage\_start}}.
$$

So the midpoint path can be viewed as

$$
e^{(m+1)}
=
\Phi_{\Delta t_{\mathrm{hydro}}}
\left(
e_{\mathrm{stage\_start}};\,
S_{\Omega,\mathrm{pre}},\,
Q_{\mathrm{cool}}
\right),
$$

where:

- the integration interval is the full hydro timestep,
- the initial state is the accepted post-STS, pre-hydro state,
- the ohmic source is frozen from that same state,
- the hydro advection/compression source is absent from this branch because the
  hydro stage has not happened yet.

## What Happens Later in the Hydro Step

After the midpoint thermal solve:

1. the main hydro stage update runs,
2. the normal coupled unsplit source hook is reached later,
3. but for this supported `rkl2` configuration the internal-energy solver
   returns immediately instead of running a second thermal solve,
4. the second RKL2 diffusion half-step runs at the end of the hydro timestep,
5. a post-hydro ohmic source may then be reconstructed for bookkeeping only.

So there is exactly one coupled thermal solve per hydro timestep in the
supported `rkl2` midpoint configuration.

## Code-to-Math Dictionary for the Coupled RKL2 Midpoint Path

- pre-hydro STS half-step $\rightarrow$
  $U^{n,\mathrm{mid}} =
  \exp\!\left(\frac{\Delta t_{\mathrm{hydro}}}{2}\mathcal{D}_{\mathrm{STS}}\right)U^n$
- `eint_stage_start` $\rightarrow e_{\mathrm{stage\_start}}$
- `eint_sdc` $\rightarrow e^{(m)}$, current outer midpoint-thermal iterate
- `thermal_src_ohmic_pre_flux` $\rightarrow S_{\Omega,\mathrm{pre}}$
- `thermal_src_total` in this branch $\rightarrow S_{\mathrm{cool}}^{(m)}$
- `thermal_src_lagged` in this branch
  $\rightarrow S_{\mathrm{cool}}^{(m)} + S_{\Omega,\mathrm{pre}}$
- midpoint thermal integration interval $\rightarrow \Delta t_{\mathrm{hydro}}$
- absent `thermal_ae` term $\rightarrow$ no lagged hydro source in this branch

## Mathematical Description of the Current `pulsed_reconnection` Driving

This note describes the current implemented driver in
`src/pgen/pulsed_reconnection.cpp`.

The driver acts only in localized regions around the two wire centers at

$$
(x, y) = \left(0, \pm \frac{d}{2}\right),
$$

where

$$
d = \texttt{array\_separation}.
$$

It does not directly modify momentum. It modifies:

- density $\rho$,
- magnetic field components $B_x, B_y$,
- total energy through the internal-energy part and the magnetic-energy part.

The momentum densities $\rho v_x$, $\rho v_y$, $\rho v_z$ are left unchanged.

### 1. Analytic Wire Target State

At any cell center $(x,y)$, the code defines an analytic two-wire target state

$$
U_{\mathrm{target}}(x,y)
=
\left(
\rho_{\mathrm{target}},
P_{\mathrm{target}},
v_{x,\mathrm{target}},
v_{y,\mathrm{target}},
B_{x,\mathrm{target}},
B_{y,\mathrm{target}}
\right)
$$

through `EvaluateSourceState(...)`.

For each wire $A \in \{-1,+1\}$ with center

$$
y_A = A\frac{d}{2},
$$

define local coordinates

$$
y'_A = y - y_A,
\qquad
r_A^2 = x^2 + y_A'^2,
\qquad
r_A = \sqrt{r_A^2}.
$$

The thermal Gaussian is

$$
G_{\mathrm{th}}(r_A) = \exp\left(-\frac{r_A^2}{w^2}\right),
$$

and the magnetic/current Gaussian uses width $w_B$.

The code constructs the target density as

$$
\rho_{\mathrm{target}}
=
\rho_{\mathrm{bg}}
+
\rho_{\mathrm{wire}}
\sum_A
G_{\mathrm{th}}(r_A)\,
\Pi_{\rho,A},
$$

where $\Pi_{\rho,A}$ is the azimuthal density perturbation factor
$1 + p_\rho \cos(N\theta_A)$.

The target temperature is

$$
T_{\mathrm{target}}
=
T_{\mathrm{bg}}
+
T_{\mathrm{wire}}
\sum_A
G_{\mathrm{th}}(r_A)\,
\Pi_{T,A},
$$

with $\Pi_{T,A} = 1 + p_T \cos(N\theta_A)$.

The target pressure is then

$$
P_{\mathrm{target}}
=
\frac{k_B}{\bar m}\,
\rho_{\mathrm{target}}\,T_{\mathrm{target}}
+
f_{\mathrm{fb}}
\sum_A
P_{\mathrm{fb}}(r_A),
$$

where $f_{\mathrm{fb}} = \texttt{force\_balance}$ and $P_{\mathrm{fb}}$ is the
analytic magnetic force-balance contribution already implemented in the problem
generator.

### 2. Target Current and Measured Current

The user specifies a peak wire current through `current_peak_MA`. This is
converted into the magnetic-field prefactor

$$
C_B = \texttt{current\_field\_prefac}.
$$

For one Gaussian wire, the implemented target peak axial current density is

$$
J_{\mathrm{target}} = \frac{2 C_B}{w_B^2}.
$$

This is what the driver tries to hold near each wire center.

At the start of each drive application, the code measures the numerical
$z$-current using

$$
J_z = (\nabla \times \mathbf{B})_z
=
\frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y},
$$

with centered finite differences on the mesh.

For each wire, it does not use a single cell. Instead it forms a
kernel-weighted average:

$$
\langle J_z \rangle_A
=
\frac{\sum_i W_A(x_i,y_i)\,J_z(x_i,y_i)\,V_i}
     {\sum_i W_A(x_i,y_i)\,V_i},
$$

where $V_i$ is the cell volume and

$$
W_A(x,y)
=
\begin{cases}
\exp\left(-\dfrac{r_A^2}{w_B^2}\right), & r_A \le r_{\mathrm{drive}}, \\
0, & r_A > r_{\mathrm{drive}},
\end{cases}
$$

with

$$
r_{\mathrm{drive}} = \texttt{drive\_radius\_factor}\,w_B.
$$

So the driver compares the measured current
$\langle J_z \rangle_A$ to $J_{\mathrm{target}}$ for each wire separately.

### 3. Magnetic Correction

For each wire, define the current error

$$
\Delta J_A = J_{\mathrm{target}} - \langle J_z \rangle_A.
$$

The code converts this into a correction to the magnetic prefactor using the
Gaussian on-axis relation:

$$
\Delta C_{B,A}^{\mathrm{raw}}
=
\frac{w_B^2}{2}\,\Delta J_A.
$$

This correction is relaxed over the magnetic drive timescale
$\tau_B = \texttt{drive\_tau\_B}$:

$$
f_B = \min\left(\frac{\Delta t}{\tau_B}, 1\right),
$$

so that

$$
\Delta C_{B,A}^{\mathrm{relaxed}}
=
f_B\,\Delta C_{B,A}^{\mathrm{raw}}.
$$

The per-step correction is then capped:

$$
|\Delta C_{B,A}|
\le
f_{B,\max}\,C_B,
$$

where

$$
f_{B,\max}
=
\texttt{drive\_max\_fractional\_B\_change\_per\_step}.
$$

The applied magnetic increment is built from the same analytic single-wire field
shape as the original problem:

$$
B_{\phi,A}(r_A)
=
\frac{C_{B,A}}{r_A}
\left(1 - e^{-r_A^2/w_B^2}\right).
$$

In Cartesian form, the increment from wire $A$ is

$$
\delta B_{x,A}
=
-W_A \left(\frac{C_{B,A}}{r_A^2}\left(1-e^{-r_A^2/w_B^2}\right)\right) y_A',
$$

$$
\delta B_{y,A}
=
W_A \left(\frac{C_{B,A}}{r_A^2}\left(1-e^{-r_A^2/w_B^2}\right)\right) x.
$$

The total magnetic update is the sum over the two wires:

$$
B_x^{n+1} = B_x^n + \sum_A \delta B_{x,A},
\qquad
B_y^{n+1} = B_y^n + \sum_A \delta B_{y,A}.
$$

The driver does not modify $B_z$.

### 4. Density Replenishment

After the magnetic correction is constructed, the code replenishes density
toward the analytic target density from `EvaluateSourceState(...)`.

Let the local support weight be

$$
W(x,y) = \max_A W_A(x,y).
$$

Then the raw density correction is

$$
\Delta \rho^{\mathrm{raw}}
=
\left(\rho_{\mathrm{target}} - \rho\right) W \, f_\rho,
$$

where

$$
f_\rho = \min\left(\frac{\Delta t}{\tau_\rho}, 1\right),
\qquad
\tau_\rho = \texttt{drive\_tau\_rho}.
$$

This is capped as

$$
|\Delta \rho|
\le
f_{\rho,\max}\,\rho_{\mathrm{target}},
$$

where

$$
f_{\rho,\max}
=
\texttt{drive\_max\_fractional\_rho\_change\_per\_step}.
$$

The updated density is

$$
\rho^{n+1}
=
\max\left(\rho_{\mathrm{bg}}, \rho^n + \Delta \rho\right).
$$

So density is explicitly replenished in the wire cores, but momentum is not.

### 5. Pressure / Internal-Energy Relaxation

The code next computes the current internal pressure from the conserved state:

$$
P
=
(\gamma - 1)\left(E - E_{\mathrm{kin}} - E_{\mathrm{mag}}\right).
$$

Because density changes while momentum is held fixed, the kinetic energy used in
this step is

$$
E_{\mathrm{kin}}
=
\frac{1}{2}\frac{M_x^2 + M_y^2 + M_z^2}{\rho},
$$

with $M_i = \rho v_i$ unchanged by the driver.

The pressure is relaxed toward the analytic target pressure
$P_{\mathrm{target}}$ from `EvaluateSourceState(...)`:

$$
\Delta P^{\mathrm{raw}}
=
\left(P_{\mathrm{target}} - P\right) W \, f_P,
$$

where

$$
f_P = \min\left(\frac{\Delta t}{\tau_P}, 1\right),
\qquad
\tau_P = \texttt{drive\_tau\_p}.
$$

The pressure correction is capped:

$$
|\Delta P|
\le
f_{P,\max}\,P_{\mathrm{target}},
$$

where

$$
f_{P,\max}
=
\texttt{drive\_max\_fractional\_p\_change\_per\_step}.
$$

Then the new internal energy is

$$
e_{\mathrm{int}}^{n+1}
=
\frac{P + \Delta P}{\gamma - 1}.
$$

### 6. Total-Energy Update

After the density and magnetic field are updated, the code recomputes kinetic
energy using the new density but the same momentum:

$$
E_{\mathrm{kin}}^{n+1}
=
\frac{1}{2}\frac{M_x^2 + M_y^2 + M_z^2}{\rho^{n+1}}.
$$

It also recomputes magnetic energy from the updated magnetic field:

$$
E_{\mathrm{mag}}^{n+1}
=
\frac{1}{2}\left((B_x^{n+1})^2 + (B_y^{n+1})^2 + (B_z^n)^2\right).
$$

Finally, total energy is overwritten as

$$
E^{n+1}
=
e_{\mathrm{int}}^{n+1}
+
E_{\mathrm{kin}}^{n+1}
+
E_{\mathrm{mag}}^{n+1}.
$$

### 7. What Is and Is Not Being Driven

The current driver does:

- measure kernel-averaged wire-core current,
- correct magnetic field to push that current toward the target,
- replenish density toward the analytic wire-core profile,
- replenish pressure/internal energy toward the analytic wire-core state.

The current driver does not:

- directly modify momentum,
- directly modify the reconnection layer,
- directly impose flow in the sheet.

So mathematically it is a localized relaxation source of the form

$$
\partial_t U \sim \frac{U_{\mathrm{target}} - U}{\tau}
$$

for selected components of $U$, with per-step caps and with the magnetic part
additionally constrained by the measured-current error rather than a direct
state overwrite.
