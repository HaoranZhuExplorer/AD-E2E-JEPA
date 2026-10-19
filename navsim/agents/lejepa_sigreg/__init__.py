from .epps_pulley import EppsPulley
from .slicing import SlicingUnivariateTest


class SIGReg(SlicingUnivariateTest):
    """LeJEPA SIGReg with the existing JEPA-WM configuration interface."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__(
            univariate_test=EppsPulley(t_max=3, n_points=knots),
            num_slices=num_proj,
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Previous JEPA-WM checkpoints stored these buffers directly under SIGReg.
        for name in ("t", "phi", "weights"):
            old_key = prefix + name
            if old_key in state_dict:
                state_dict[prefix + "univariate_test." + name] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
