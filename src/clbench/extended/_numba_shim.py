"""
Minimal stand-in for ``numba`` (not installed in this environment): tigramite's CMIknn decorates two methods
with ``@jit(forceobj=True)`` (object mode, i.e. plain Python semantics). The shim's ``jit`` / ``njit`` return the
function unchanged. Registered as ``sys.modules['numba']`` by ``clbench.extended.nonlinear.install_shims``
only when the real package is absent.
"""

from __future__ import annotations

__version__ = "0.0-shim"


def jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def deco(fn):
        return fn

    return deco


njit = jit
