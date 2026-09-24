# Confined-current magnetic source

`initial_magnetic_profile = confined_current` and
`drive_magnetic_profile = confined_current` use an analytic out-of-plane vector
potential. With `B = curl(A_z zhat)`, the potential is
`A_z(r) = integral_r^r_outer B_phi(s) ds`; equivalently `psi = -A_z` for
`B = zhat cross grad(psi)`. The existing corner-potential/face-curl CT construction
is retained for initialization and time-dependent increments.

Write `S(q) = 6q^5 - 15q^4 + 10q^3`, clamped to zero/one outside `[0,1]`.
The unnormalized field is

```
h(r) = (r_conductor/r)
       * S((r - (r_conductor - width_conductor))/width_conductor)
       * S((r_outer - r)/width_wire).
```

It is zero in the post core and beyond the outer cutoff. Both transitions smoothly
match the intervening `1/r` branch through second radial derivatives (for a
positive-radius post core). The supported limiting case where the conductor ramp
starts at the axis uses a nonsingular polynomial primitive.

The potential uses logarithmic/polynomial primitives, evaluated with a convergent
series where necessary to avoid cancellation. It does not interpolate a radial
potential table. Tables for unrelated profile/pressure-support options remain.

`B_peak_gauss` and `drive_B_peak_gauss` remain the continuum maximum field of each
individual prescribed array profile. They are not imposed maxima of the evolved
solution or guarantees of an exact cell-sampled maximum. The maximum is found in
the conductor ramp, and the whole potential is normalized by it. Superposition,
discretization and subsequent evolution can change the measured maximum.

For conductor radius 1 cm and ramp width 0.5 cm, the maximum occurs at approximately
0.9214333 cm. The current-free coefficient is
`B_phi*r = B_peak * 0.9501544 cm`. Thus the same peak input gives about 5% less
current-free field than the old conductor-surface normalization. If preserving
the old nominal `B_peak*r_conductor` coefficient is the objective, increase the
peak input by approximately `1/0.9501544` for this geometry. No input values are
automatically changed. Startup logging reports `B(rc)/B_peak` for the driven profile.

The wire taper starts at `wire_outer_radius - wire_transition_width`.
`wire_inner_radius` constrains the allowed width and supplies its default; with an
explicit width it does not independently set the taper start.

The magnetic source still adds potential increments following the pulse. It does
not reset the evolved field. Density/temperature masks, replenishment and velocity
injection are unchanged, including any reservoir/annulus gap in existing inputs.
Magnetic force-balance support remains unsupported for this profile.
