"""PVFM-facing native-q9 entry points for the external DAG benchmark cores."""

from .dag import Model as DAGNativeQ9
from .gcgnet import Model as GCGNetNativeQ9
from .timexer import Model as TimeXerNativeQ9
from .tide import Model as TiDENativeQ9

__all__ = ["DAGNativeQ9", "GCGNetNativeQ9", "TimeXerNativeQ9", "TiDENativeQ9"]
