# Vendored LeJEPA SIGReg

Source: [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa),
commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`.
Author: Randall Balestriero. The upstream `LICENSE` is included verbatim.

Copied components:

- `base.py`: upstream `lejepa/univariate/base.py`, unchanged.
- `epps_pulley.py`: upstream `lejepa/univariate/epps_pulley.py`, through
  `EppsPulley`; the unused `DeprecatedEppsPulley` class is omitted.
- `slicing.py`: upstream `lejepa/multivariate/slicing.py`, unchanged.

The local `SIGReg` adapter maps `knots` to `n_points` and `num_proj` to
`num_slices`, keeping the integration range `[0, 3]`, mean reduction, and no
clipping. It also maps the three integration buffers from the previous
JEPA-WM checkpoint layout into `univariate_test`; `global_step` keeps its name.
One instance is shared by global and patch regularization within each model,
preserving the existing projection-counter call order.

Distributed statistics use LeJEPA's averaging of per-rank means and assume
equal sample counts per rank. Inputs use shape `(..., B, D)`, with batch
samples on the second-last axis. No external LeJEPA installation is needed.
