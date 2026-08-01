"""mavka: an external memory for world models.

Build a Memory (configured via MavkaConfig), feed it experiences through a
WorldModelAdapter (SyntheticWorldModel is a dependency-free reference
implementation for development and tests), then call memory.observe(...) to
write and memory.recall(...) to retrieve.

Optional components a user may pass into Memory (a scorer, a trigger, a
fusion predictor, the lifecycle/maintenance pieces) are deliberately not
exported here, to keep this front door small -- reach them via their own
submodules, e.g. `from mavka.retrieval.scorer import FixedWeightScorer`.
"""

from mavka.adapter import SyntheticWorldModel, WorldModelAdapter
from mavka.config import MavkaConfig
from mavka.memory import Memory

__version__ = "0.0.1"

__all__ = ["Memory", "MavkaConfig", "WorldModelAdapter", "SyntheticWorldModel"]
