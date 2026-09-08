# We keep this small compatibility layer for older notebooks and scripts.
#
# The actual implementation lives in src.models.deep_learning, but this file
# still lets us import from src.deep_learning when older code expects it.
from src.models.deep_learning import *  # noqa: F401,F403
