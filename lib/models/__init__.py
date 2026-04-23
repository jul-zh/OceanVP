# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

from .cotere_model import COTERE_Model
from .my_ocean_baseline import MY_OCEAN_BASELINE
from .my_ocean_baseline_sdefeat import MY_OCEAN_BASELINE_SDEFEAT
from .my_ocean_baseline_sdelight import MY_OCEAN_BASELINE_SDELIGHT
from .my_ocean_baseline_sdelight_prob import MY_OCEAN_BASELINE_SDELIGHT_PROB
from .my_ocean_baseline_rbfkan_prob import MY_OCEAN_BASELINE_RBFKAN_PROB
from .prob_model import PROB_Model
from .probkan_model import PROBKAN_Model
from .probkan_v2_model import PROBKANV2_Model
from .kanattn_model import KANATTN_Model
from .my_ocean_baseline_kanattn_rbf import MY_OCEAN_BASELINE_KANATTN_RBF
from .my_ocean_baseline_kanattn_rbf_v2 import MY_OCEAN_BASELINE_KANATTN_RBF_V2
from .my_ocean_baseline_kanhead_rbf import MY_OCEAN_BASELINE_KANHEAD_RBF
from .my_ocean_baseline_kanhead_spline import MY_OCEAN_BASELINE_KANHEAD_SPLINE
from .my_ocean_baseline_kanhead_rbf_residual import MY_OCEAN_BASELINE_KANHEAD_RBF_RESIDUAL
from .my_ocean_baseline_kandecoder_gate_rbf import MY_OCEAN_BASELINE_KANDECODER_GATE_RBF
from .my_ocean_baseline_kanskip_fusion_rbf import MY_OCEAN_BASELINE_KANSKIP_FUSION_RBF
from .my_ocean_baseline_kanhead_spline_residual import MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL
from .my_ocean_baseline_sdelight_kanhead_rbf_residual import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL
from .my_ocean_baseline_strongenc import MY_OCEAN_BASELINE_STRONGENC
from .my_ocean_baseline_strongenc_kanhead_rbf_residual import MY_OCEAN_BASELINE_STRONGENC_KANHEAD_RBF_RESIDUAL
from .my_ocean_baseline_strongenc_sdelight_kanhead_rbf_residual import MY_OCEAN_BASELINE_STRONGENC_SDELIGHT_KANHEAD_RBF_RESIDUAL

from .my_ocean_baseline_sdelight_kanhead_rbf_residual_perstep import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_PERSTEP

from .my_ocean_baseline_sdelight_kanhead_rbf_residual_multikan import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_MULTIKAN
from .my_ocean_baseline_sdelight_kanhead_rbf_residual_gatedkan import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_GATEDKAN
from .my_ocean_baseline_sdelight_kanhead_rbf_residual_tempmix import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_TEMPMIX

from .my_ocean_baseline_sdelight_kanhead_rbf_residual_stemplus import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_STEMPLUS
from .my_ocean_baseline_sdelight_kanhead_rbf_residual_dilated import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_DILATED
from .my_ocean_baseline_sdelight_kanhead_rbf_residual_skipgate import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SKIPGATE

from .my_ocean_baseline_uv import MY_OCEAN_BASELINE_UV
from .my_ocean_baseline_kanhead_rbf_residual_uv import MY_OCEAN_BASELINE_KANHEAD_RBF_RESIDUAL_UV
from .my_ocean_baseline_kanhead_spline_residual_uv import MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL_UV
from .my_ocean_baseline_sdelight_kanhead_rbf_residual_uv import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_UV

__all__ = [
    'COTERE_Model',
    'MY_OCEAN_BASELINE',
    'MY_OCEAN_BASELINE_SDEFEAT',
    'MY_OCEAN_BASELINE_SDELIGHT',
    'MY_OCEAN_BASELINE_SDELIGHT_PROB',
    'MY_OCEAN_BASELINE_RBFKAN_PROB',
    'PROB_Model',
    'PROBKAN_Model',
    'PROBKANV2_Model',
    'KANATTN_Model',
    'MY_OCEAN_BASELINE_KANATTN_RBF',
    'MY_OCEAN_BASELINE_KANATTN_RBF_V2',
    'MY_OCEAN_BASELINE_KANHEAD_RBF',
    'MY_OCEAN_BASELINE_KANHEAD_SPLINE',
    'MY_OCEAN_BASELINE_KANHEAD_RBF_RESIDUAL',
    'MY_OCEAN_BASELINE_KANDECODER_GATE_RBF',
    'MY_OCEAN_BASELINE_KANSKIP_FUSION_RBF',
    'MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL',
    'MY_OCEAN_BASELINE_STRONGENC',
    'MY_OCEAN_BASELINE_STRONGENC_KANHEAD_RBF_RESIDUAL',
    'MY_OCEAN_BASELINE_STRONGENC_SDELIGHT_KANHEAD_RBF_RESIDUAL',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_PERSTEP',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_MULTIKAN',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_GATEDKAN',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_TEMPMIX',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_STEMPLUS',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_DILATED',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SKIPGATE',
    'MY_OCEAN_BASELINE_UV',
    'MY_OCEAN_BASELINE_KANHEAD_RBF_RESIDUAL_UV',
    'MY_OCEAN_BASELINE_KANHEAD_SPLINE_RESIDUAL_UV',
    'MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_UV',
]


