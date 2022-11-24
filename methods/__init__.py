from .base import BaseModel
from .labelsmoothing import LabelSmoothing
from .mfvi import MFVI
from .sl import SummaryLikelihood
from .slim import SummaryLikelihoodIm
from .pl import PredictionLikelihood
from .labelsmoothing import LabelSmoothing
from .edl import EvidentialDeepLearning
from .sgd import SGDDeterministic, SGDSLDeterministic

# For lookup
mfvi = MFVI
sl = SummaryLikelihood
slim = SummaryLikelihoodIm
pl = PredictionLikelihood
ls = LabelSmoothing
edl = EvidentialDeepLearning
sgd = SGDDeterministic
sgdsl = SGDSLDeterministic
