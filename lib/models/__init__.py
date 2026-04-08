# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

from .cotere_model import COTERE_Model
from .my_ocean_baseline import MY_OCEAN_BASELINE
from .my_ocean_baseline_sdefeat import MY_OCEAN_BASELINE_SDEFEAT
from .my_ocean_baseline_sdelight import MY_OCEAN_BASELINE_SDELIGHT
from .my_ocean_baseline_sdelight_prob import MY_OCEAN_BASELINE_SDELIGHT_PROB

__all__ = [
    'COTERE_Model',
    'MY_OCEAN_BASELINE',
    'MY_OCEAN_BASELINE_SDEFEAT',
    'MY_OCEAN_BASELINE_SDELIGHT',
    'MY_OCEAN_BASELINE_SDELIGHT_PROB'
]
