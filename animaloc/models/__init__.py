__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


from .register import MODELS

from .faster_rcnn import *
from .dla import *
from .herdnet import *
from .herdnet_timm_dla import *
from .herdnet_timm_convnext import *
from .herdnet_timm_convnext_camouflaged import *
from .herdnet_timm_convnext_camouflaged_v2 import *
from .herdnet_timm_convnext_camouflaged_v3 import *
from .herdnet_timm_dinoSvin import *
from .herdnet_timm_swin import *
from .herdnet_dino_v2 import *
from .herdnet_timm_dinoV3 import *
from .herdnet_timm_dinoV3_fpn import *
from .dual_head_ensemble import *
from .herdnet_timm_dinoV3_fpn_ida import *
from .herdnet_dino_v3_attn import *

from .herdnet_timm_dino_fpn import *
from .herdnet_hybrid_convnext_transformer import *

from .herdnet_p2p import *
from .utils import *
from .ss_dla import *

__all__ = ['MODELS', *MODELS.registry_names]