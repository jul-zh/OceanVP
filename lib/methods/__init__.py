from .cotere import COTERE
from .my_baseline import MY_BASELINE
from .my_baseline_sdefeat import MY_BASELINE_SDEFEAT
from .my_baseline_sdelight import MY_BASELINE_SDELIGHT
from .my_baseline_sdelight_prob import MY_BASELINE_SDELIGHT_PROB
from .my_baseline_rbfkan_prob import MY_BASELINE_RBFKAN_PROB
from .prob import PROB
from .probkan  import PROBKAN
from .probkan_v2 import PROBKAN_V2
from .kanattn import KANATTN
from .my_baseline_kanattn_rbf import MY_BASELINE_KANATTN_RBF
from .my_baseline_kanattn_rbf_v2 import MY_BASELINE_KANATTN_RBF_V2
from .my_baseline_kanhead_rbf import MY_BASELINE_KANHEAD_RBF
from .my_baseline_kanhead_spline import MY_BASELINE_KANHEAD_SPLINE
from .my_baseline_kanhead_rbf_residual import MY_BASELINE_KANHEAD_RBF_RESIDUAL
from .my_baseline_kandecoder_gate_rbf import MY_BASELINE_KANDECODER_GATE_RBF
from .my_baseline_kanskip_fusion_rbf import MY_BASELINE_KANSKIP_FUSION_RBF
from .my_baseline_kanhead_spline_residual import MY_BASELINE_KANHEAD_SPLINE_RESIDUAL
from .my_baseline_sdelight_kanhead_rbf_residual import MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL
from .my_baseline_sdelight_kanhead_rbf_residual_sdeloss import MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS

from .my_baseline_sdelight_kanhead_rbf_residual_sdeloss_regab import MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_REGAB
from .my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins import MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS
from .my_baseline_sdelight_kanhead_rbf_residual_sdeenergy_bins import MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_BINS

from .my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins_v2 import MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_V2
from .my_baseline_strongenc import MY_BASELINE_STRONGENC
from .my_baseline_strongenc_kanhead_rbf_residual import MY_BASELINE_STRONGENC_KANHEAD_RBF_RESIDUAL
from .my_baseline_strongenc_sdelight_kanhead_rbf_residual_sdeloss import MY_BASELINE_STRONGENC_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS

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
    'kanattn': KANATTN,
    'my_baseline_kanattn_rbf': MY_BASELINE_KANATTN_RBF,
    'my_baseline_kanattn_rbf_v2': MY_BASELINE_KANATTN_RBF_V2,
    'my_baseline_kanhead_rbf': MY_BASELINE_KANHEAD_RBF,
    'my_baseline_kanhead_spline': MY_BASELINE_KANHEAD_SPLINE,
    'my_baseline_kanhead_rbf_residual': MY_BASELINE_KANHEAD_RBF_RESIDUAL,
    'my_baseline_kandecoder_gate_rbf': MY_BASELINE_KANDECODER_GATE_RBF,
    'my_baseline_kanskip_fusion_rbf': MY_BASELINE_KANSKIP_FUSION_RBF,
    'my_baseline_kanhead_spline_residual': MY_BASELINE_KANHEAD_SPLINE_RESIDUAL,
    'my_baseline_sdelight_kanhead_rbf_residual': MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL,
    'my_baseline_sdelight_kanhead_rbf_residual_sdeloss': MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS,
    'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_regab': MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_REGAB,
    'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins': MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS,
    'my_baseline_sdelight_kanhead_rbf_residual_sdeenergy_bins': MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_BINS,
    'my_baseline_sdelight_kanhead_rbf_residual_sdeloss_bins_v2': MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_V2,
    'my_baseline_strongenc': MY_BASELINE_STRONGENC,
    'my_baseline_strongenc_kanhead_rbf_residual': MY_BASELINE_STRONGENC_KANHEAD_RBF_RESIDUAL,
    'my_baseline_strongenc_sdelight_kanhead_rbf_residual_sdeloss': MY_BASELINE_STRONGENC_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS,
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
    'PROBKAN_V2',
    'KANATTN',
    'MY_BASELINE_KANATTN_RBF',
    'MY_BASELINE_KANATTN_RBF_V2',
    'MY_BASELINE_KANHEAD_RBF',
    'MY_BASELINE_KANHEAD_SPLINE',
    'MY_BASELINE_KANHEAD_RBF_RESIDUAL',
    'MY_BASELINE_KANDECODER_GATE_RBF',
    'MY_BASELINE_KANSKIP_FUSION_RBF',
    'MY_BASELINE_KANHEAD_SPLINE_RESIDUAL',
    'MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL',
    'MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS',
    'MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_REGAB',
    'MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS',
    'MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDEENERGY_BINS',
    'MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_V2',
    'MY_BASELINE_STRONGENC',
    'MY_BASELINE_STRONGENC_KANHEAD_RBF_RESIDUAL',
    'MY_BASELINE_STRONGENC_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS',
]



