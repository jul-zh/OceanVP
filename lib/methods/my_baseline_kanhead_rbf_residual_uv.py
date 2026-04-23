from lib.models import MY_OCEAN_BASELINE_KANHEAD_RBF_RESIDUAL_UV
from .my_baseline_kanhead_rbf_residual import MY_BASELINE_KANHEAD_RBF_RESIDUAL


class MY_BASELINE_KANHEAD_RBF_RESIDUAL_UV(MY_BASELINE_KANHEAD_RBF_RESIDUAL):
    def _build_model(self, args):
        return MY_OCEAN_BASELINE_KANHEAD_RBF_RESIDUAL_UV(**args).to(self.device)
