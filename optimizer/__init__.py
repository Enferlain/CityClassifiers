
from typing import Dict, List, Type
from optimizer.optimizer_utils import OPTIMIZER

from optimizer.adabelief import AdaBelief
from optimizer.adammini import AdamMini
from optimizer.adan import Adan
from optimizer.ademamix import AdEMAMix
from optimizer.adopt import ADOPT
from optimizer.came import CAME
from optimizer.compass import Compass, Compass8BitBNB, CompassPlus, CompassADOPT, CompassADOPTMARS, CompassAO
from optimizer.farmscrop import FARMSCrop, FARMSCropV2
from optimizer.fcompass import FCompass, FCompassPlus, FCompassADOPT, FCompassADOPTMARS
from optimizer.fishmonger import FishMonger, FishMonger8BitBNB
from optimizer.fmarscrop import FMARSCrop, FMARSCropV2, FMARSCropV2ExMachina, FMARSCropV3, FMARSCropV3ExMachina
from optimizer.galore import GaLore
from optimizer.grokfast import GrokFastAdamW
from optimizer.laprop import LaProp
from optimizer.lpfadamw import LPFAdamW
from optimizer.ranger21 import Ranger21
from optimizer.rmsprop import RMSProp, RMSPropADOPT, RMSPropADOPTMARS
from optimizer.schedulefree import (
    ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTEMAMixScheduleFree, ADOPTNesterovScheduleFree, 
    FADOPTScheduleFree, ADOPTMARSScheduleFree, FADOPTMARSScheduleFree, ADOPTAOScheduleFree
    )

from optimizer.clybius_experiments import MomentusCaution
from optimizer.sgd import SGDSaI
from optimizer.shampoo import ScalableShampoo
from optimizer.adam import AdamW8bitAO, AdamW4bitAO, AdamWfp8AO
from .distributed_shampoo.distributed_shampoo import DistributedShampoo
# from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree # missing?

# --- Imports for Schedulers ---
from torch.optim.lr_scheduler import LRScheduler # Base type
from .CosineAnnealingWarmRestarts import CosineAnnealingWarmRestarts
from .RexAnnealingWarmRestarts import RexAnnealingWarmRestarts

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ADOPT,
    ADOPTAOScheduleFree,
    ADOPTEMAMixScheduleFree,
    ADOPTMARSScheduleFree,
    ADOPTNesterovScheduleFree,
    ADOPTScheduleFree,
    AdEMAMix,
    AdaBelief,
    AdamMini,
    Adan,
    AdamW4bitAO,
    AdamW8bitAO,
    AdamWfp8AO,
    CAME,
    Compass,
    CompassAO,
    Compass8BitBNB,
    CompassADOPT,
    CompassADOPTMARS,
    CompassPlus,
    DistributedShampoo,
    FADOPTMARSScheduleFree,
    FADOPTScheduleFree,
    FARMSCrop,
    FARMSCropV2,
    FCompass,
    FCompassADOPT,
    FCompassADOPTMARS,
    FCompassPlus,
    FMARSCrop,
    FMARSCropV2,
    FMARSCropV2ExMachina,
    FMARSCropV3,
    FMARSCropV3ExMachina,
    FishMonger,
    FishMonger8BitBNB,
    GaLore,
    GrokFastAdamW,
    LPFAdamW,
    LaProp,
    MomentusCaution,
    # ProdigyPlusScheduleFree,
    RMSProp,
    RMSPropADOPT,
    RMSPropADOPTMARS,
    Ranger21,
    SGDSaI,
    ScalableShampoo,
    ScheduleFreeWrapper,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}

# --- Scheduler Dictionary ---
SCHEDULER_LIST: List[Type[LRScheduler]] = [
    # Add all CUSTOM scheduler classes here
    CosineAnnealingWarmRestarts,
    RexAnnealingWarmRestarts,
    # ... etc ...
]
SCHEDULERS: Dict[str, Type[LRScheduler]] = {
    scheduler.__name__.lower(): scheduler for scheduler in SCHEDULER_LIST
}
