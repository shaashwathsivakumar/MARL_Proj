import warnings

from pettingzoo.mpe.simple.simple import env, parallel_env, raw_env

warnings.warn(
    "The environment `pettingzoo.mpe.simple_v3` has been moved to `mpe2.simple_v3` and will be removed in a future release. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["env", "parallel_env", "raw_env"]
