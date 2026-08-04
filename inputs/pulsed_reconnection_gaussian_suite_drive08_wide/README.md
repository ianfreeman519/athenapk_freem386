# Pulsed Reconnection Gaussian Suite: drive08 wide

This suite is derived from `inputs/pulsed_reconnection_gaussian_suite`.

Changes relative to the source suite:

- Set `w_drive = 0.8` for every profile.
- Expand both x1 and x2 domains from `[-6, 6]` to `[-12, 12]`.
- Increase `nx1` and `nx2` from `1536` to `3072`, preserving the original uniform cell spacing.

The source timing and output cadence are retained:

- `tlim = 4e-7`
- output `dt = 1e-8`

Problem parameters other than `w_drive` are unchanged. With `array_separation = 6`, the drive centers remain at `y = +/-3`; `w_drive = 0.8` keeps the reconnection layer near `y = 0` outside the intended driven-source region.
