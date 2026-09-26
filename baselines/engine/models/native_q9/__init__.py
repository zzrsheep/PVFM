"""Native quantile heads for trainable PVFM baselines.

The modules in this package keep the original deterministic implementations
untouched.  A native-q9 model replaces only the model's final forecast head;
it does not append a second linear layer to a point forecast.
"""

from .common import NativeQuantileMixin, parse_quantile_levels

__all__ = ["NativeQuantileMixin", "parse_quantile_levels"]
