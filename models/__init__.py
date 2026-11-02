from .component_query_swin import ComponentQuerySwin
from .de_mambanet import DEMambaNet
from .de_mambanet_lite import DEMambaNetLite
from .de_mambanet_lite_balanced import DEMambaNetLiteBalanced
from .de_mambanet_lite_balanced_plus import DEMambaNetLiteBalancedPlus
from .de_mambanet_lite_pro import DEMambaNetLitePro
from .de_mambanet_lite_pro_ptca import DEMambaNetLiteProPTCA
from .de_mambanet_lite_pro_lsra import DEMambaNetLiteProLSRA
from .de_mambanet_lite_pro_pcr import DEMambaNetLiteProPCR
from .de_mambanet_lite_pro_wide_encoder import DEMambaNetLiteProWideEncoder
from .de_mambanet_lite_pro_ablations import DEMambaNetLiteProNarrow, DEMambaNetLiteProNoS3
from .de_mambanet_lite_pro_segr import DEMambaNetLiteProSEGR
from .de_mambanet_lite_pro_csdr import DEMambaNetLiteProCSDR
from .de_mambanet_lite_pro_fbpv import DEMambaNetLiteProFBPV
from .hrconvnext_fpn import HRConvNeXtFPN
from .hr_topomamba import HRTopoMambaNet


MODEL_REGISTRY = {
    "ComponentQuerySwin": ComponentQuerySwin,
    "DEMambaNet": DEMambaNet,
    "DEMambaNetLite": DEMambaNetLite,
    "DEMambaNetLiteBalanced": DEMambaNetLiteBalanced,
    "DEMambaNetLiteBalancedPlus": DEMambaNetLiteBalancedPlus,
    "DEMambaNetLitePro": DEMambaNetLitePro,
    "DEMambaNetLiteProPTCA": DEMambaNetLiteProPTCA,
    "DEMambaNetLiteProLSRA": DEMambaNetLiteProLSRA,
    "DEMambaNetLiteProPCR": DEMambaNetLiteProPCR,
    "DEMambaNetLiteProWideEncoder": DEMambaNetLiteProWideEncoder,
    "DEMambaNetLiteProNoS3": DEMambaNetLiteProNoS3,
    "DEMambaNetLiteProNarrow": DEMambaNetLiteProNarrow,
    "DEMambaNetLiteProSEGR": DEMambaNetLiteProSEGR,
    "DEMambaNetLiteProCSDR": DEMambaNetLiteProCSDR,
    "DEMambaNetLiteProFBPV": DEMambaNetLiteProFBPV,
    "HRConvNeXtFPN": HRConvNeXtFPN,
    "HRTopoMambaNet": HRTopoMambaNet,
}


def build_model(name: str, **kwargs):
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model.name '{name}'. Available models: {available}.") from error
    return model_class(**kwargs)


__all__ = [
    "ComponentQuerySwin",
    "DEMambaNet",
    "DEMambaNetLite",
    "DEMambaNetLiteBalanced",
    "DEMambaNetLiteBalancedPlus",
    "DEMambaNetLitePro",
    "DEMambaNetLiteProPTCA",
    "DEMambaNetLiteProLSRA",
    "DEMambaNetLiteProPCR",
    "DEMambaNetLiteProWideEncoder",
    "DEMambaNetLiteProNoS3",
    "DEMambaNetLiteProNarrow",
    "DEMambaNetLiteProSEGR",
    "DEMambaNetLiteProCSDR",
    "DEMambaNetLiteProFBPV",
    "HRConvNeXtFPN",
    "HRTopoMambaNet",
    "MODEL_REGISTRY",
    "build_model",
]
