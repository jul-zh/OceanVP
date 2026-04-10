from .cotere import COTERE
from .my_baseline import MY_BASELINE
from .my_baseline_sdefeat import MY_BASELINE_SDEFEAT
from .my_baseline_sdelight import MY_BASELINE_SDELIGHT
from .my_baseline_sdelight_prob import MY_BASELINE_SDELIGHT_PROB
from .my_baseline_rbfkan_prob import MY_BASELINE_RBFKAN_PROB
from .prob import PROB
from .probkan  import PROBKAN
from .probkan_v2 import PROBKAN_V2

method_maps = {
    'cotere': COTERE,
    'my_baseline': MY_BASELINE,
    'my_baseline_sdefeat': MY_BASELINE_SDEFEAT,
    'my_baseline_sdelight': MY_BASELINE_SDELIGHT,
    'my_baseline_sdelight_prob': MY_BASELINE_SDELIGHT_PROB,
    'my_baseline_rbfkan_prob': MY_BASELINE_RBFKAN_PROB,
    'prob': PROB,
    'probkan': PROBKAN,
    'probkan_v2': PROBKAN_V2,
}

__all__ = [
    'COTERE',
    'MY_BASELINE',
    'MY_BASELINE_SDEFEAT',
    'MY_BASELINE_SDELIGHT',
    'MY_BASELINE_SDELIGHT_PROB',
    'MY_BASELINE_RBFKAN_PROB',
    'PROB',
    'PROBKAN',
    'PROBKAN_V2'
]



