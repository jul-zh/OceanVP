from lib.models import MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL_UV
from .my_baseline_kanhead_spline_residual import MY_BASELINE_KANHEAD_SPLINE_RESIDUAL


class MY_BASELINE_KANHEAD_SPLINE_RESIDUAL_UV(MY_BASELINE_KANHEAD_SPLINE_RESIDUAL):
    def _build_model(self, args):
        return MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL_UV(**args).to(self.device)
