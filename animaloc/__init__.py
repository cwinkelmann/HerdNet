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


# Silence two benign, high-frequency UserWarnings that flood training logs:
#   1. sklearn confusion_matrix complains when `labels` argument has a
#      single value — intentional for our binary (iguana-only) detector,
#      see animaloc/eval/metrics.py.
#   2. albumentations warns that the val/test Compose has KeypointParams
#      configured but no keypoint-processing transform. Keypoints pass
#      through Normalize unchanged on purpose.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r"A single label was found in 'y_true' and 'y_pred'\.",
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"Got processor for keypoints, but no transform to process it\.",
    category=UserWarning,
)
del _warnings


from animaloc import data
from animaloc import datasets
from animaloc import eval
from animaloc import models
from animaloc import train
from animaloc import utils
from animaloc import vizual