"""The one and only module in this codebase allowed to import `litellm`.

Why a whole module for an import:

LiteLLM ships a model price map inside the package, but by default it may also
fetch a newer copy over the network at import time. If it does, our cost
figures depend on when the process happened to start and whether it had
internet — which makes the savings report irreproducible and quietly wrong.

`LITELLM_LOCAL_MODEL_COST_MAP` pins it to the packaged copy, and it is only
read **when `litellm` is first imported**. Setting it in an entrypoint works
right up until someone adds an `import litellm` higher in the import graph, at
which point the guarantee silently disappears and nothing fails.

So the guarantee is structural instead: the env var is set here, immediately
above the only `import litellm` in the repo, and `.importlinter` fails the
build if any other module imports `litellm` directly. Adapters import their
handle from here.
"""

from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# This import MUST stay below the line above. That ordering is the guarantee,
# and tests/unit/test_litellm_boundary.py asserts it.
import litellm

__all__ = ["litellm", "local_cost_map_is_pinned"]


def local_cost_map_is_pinned() -> bool:
    """Assertable at startup, so the guarantee is observable and not just documented."""
    return os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
