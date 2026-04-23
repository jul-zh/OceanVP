from lib.models import MY_OCEAN_BASELINE_UV
from .my_baseline import MY_BASELINE


class MY_BASELINE_UV(MY_BASELINE):
    def _build_model(self, args):
        return MY_OCEAN_BASELINE_UV(**args).to(self.device)
