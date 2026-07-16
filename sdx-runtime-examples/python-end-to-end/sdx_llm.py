# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""Re-export shim — forwards all public names from the canonical ``sdx_llm``
module in the main deeplearning4j repository.

The canonical source of truth is:
    deeplearning4j/libnd4j/include/dsp/runtime/bindings/python/sdx_llm.py

Loader resolution (mirrors end_to_end.py's sdx_runtime pattern):
1. ``$SDX_RUNTIME_HOME/wrappers/python/sdx_llm.py``
2. Sibling deeplearning4j checkout:
   ``../../deeplearning4j/libnd4j/include/dsp/runtime/bindings/python/sdx_llm.py``

NOTE on the shadow-import problem
----------------------------------
A file named ``sdx_llm.py`` cannot do ``import sdx_llm`` or add its own
directory to ``sys.path`` and then import by name, because the module system
would find *this* file again (infinite recursion / the wrong object).

We avoid this by using ``importlib.util.spec_from_file_location`` to load the
canonical file *by absolute path* under the internal name
``_sdx_llm_canonical``.  We then re-export its ``__all__`` members (or every
public attribute when ``__all__`` is absent) into this module's global
namespace so that ``llm_example.py``'s ``from sdx_llm import ...`` import
works unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

_THIS_DIR = pathlib.Path(__file__).resolve().parent

# ── Locate the canonical sdx_llm.py ─────────────────────────────────────────

_CANONICAL_CANDIDATES: list[pathlib.Path] = []

_sdx_runtime_home = os.environ.get("SDX_RUNTIME_HOME", "")
if _sdx_runtime_home:
    _CANONICAL_CANDIDATES.append(
        pathlib.Path(_sdx_runtime_home) / "wrappers" / "python" / "sdx_llm.py"
    )

# ../../deeplearning4j relative to this file (sdx-runtime-examples/python-end-to-end/)
#   -> sdx-runtime-examples/
#   -> deeplearning4j-examples/
#   -> deeplearning4j/  (sibling repo)
_CANONICAL_CANDIDATES.append(
    _THIS_DIR.parents[2]
    / "deeplearning4j"
    / "libnd4j"
    / "include"
    / "dsp"
    / "runtime"
    / "bindings"
    / "python"
    / "sdx_llm.py"
)

_canonical_path: pathlib.Path | None = None
for _cand in _CANONICAL_CANDIDATES:
    if _cand.exists():
        _canonical_path = _cand
        break

if _canonical_path is None:
    raise ImportError(
        "Cannot locate the canonical sdx_llm.py.  "
        "Set SDX_RUNTIME_HOME to the unpacked SDK package root (containing "
        "wrappers/python/sdx_llm.py), or ensure a sibling deeplearning4j "
        "checkout exists at ../../deeplearning4j relative to this examples repo.\n"
        "Searched:\n" + "\n".join(f"  {p}" for p in _CANONICAL_CANDIDATES)
    )

# ── Load by path to avoid the shadow-import problem ──────────────────────────
# Register in sys.modules BEFORE exec_module so that @dataclass (which calls
# sys.modules[cls.__module__]) and similar patterns that look up the module
# by name find a valid entry.

_INTERNAL_NAME = "_sdx_llm_canonical"
_spec = importlib.util.spec_from_file_location(_INTERNAL_NAME, str(_canonical_path))
_canonical_mod: types.ModuleType = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules[_INTERNAL_NAME] = _canonical_mod
_spec.loader.exec_module(_canonical_mod)  # type: ignore[union-attr]

# ── Re-export public API into this module's namespace ────────────────────────

_names_to_export: list[str] = (
    list(_canonical_mod.__all__)
    if hasattr(_canonical_mod, "__all__")
    else [n for n in dir(_canonical_mod) if not n.startswith("_")]
)

for _name in _names_to_export:
    globals()[_name] = getattr(_canonical_mod, _name)

__all__ = _names_to_export  # type: ignore[assignment]
