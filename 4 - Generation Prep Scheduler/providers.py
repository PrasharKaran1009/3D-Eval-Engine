"""
Layer 4 - Generator Providers  (the BYOA "open circuit")
========================================================

The diagram's Layer 4 has two sides: BYOM reads uploaded meshes, BYOA generates
them on demand via each model's API. This module is the seam between them.

Every generator answers ONE question for a (model, test-case) cell:

    resolve(model, prompt_id, prompt_spec) -> str | None
        Return the path to a local GLB for that cell, generating + caching it if
        necessary, or None if it cannot be produced.

The scheduler doesn't care which kind it is — it asks the registry for the right
generator by the model's `source` field and calls resolve(). That is the open
circuit: today the registry is all BYOM, so BYOMGenerator handles every cell.
Register a model with source="BYOA" and BYOAGenerator takes over that cell — no
scheduler change, no new wiring. Drop the box in and current flows.

    get_generator("BYOM") -> BYOMGenerator   (fully implemented)
    get_generator("BYOA") -> BYOAGenerator   (documented stub — the empty socket)
"""

from __future__ import annotations

import os
from typing import Optional, Protocol

import asset_indexer


class Generator(Protocol):
    """The contract every generation path implements. Two methods so the
    scheduler can build the asset index once (for lookups) and then resolve each
    cell — but a generator is free to ignore the manifest (BYOA does)."""

    source: str

    def resolve(self, model: dict, prompt_id: str,
                prompt_spec: Optional[dict] = None) -> Optional[str]:
        ...


# --------------------------------------------------------------------------- #
#  BYOM — Bring Your Own Models  (FULLY IMPLEMENTED)                            #
# --------------------------------------------------------------------------- #
class BYOMGenerator:
    """
    Resolve a cell to an already-uploaded mesh on disk. No generation happens —
    the mesh was provided by the user and canonically named by the upload
    endpoint, then indexed by asset_indexer. This is the path that must be
    complete for the BYOM product, and it is.
    """

    source = "BYOM"

    def __init__(self, manifest: dict):
        # {prompt_id: {model_id: glb_path}} — the registry of what's available.
        self._assets = (manifest or {}).get("assets", {})

    def resolve(self, model: dict, prompt_id: str,
                prompt_spec: Optional[dict] = None) -> Optional[str]:
        path = self._assets.get(prompt_id, {}).get(model["id"])
        return path if path and os.path.exists(path) else None


# --------------------------------------------------------------------------- #
#  BYOA — Bring Your Own API  (THE STUB — this is the socket to fill)           #
# --------------------------------------------------------------------------- #
class BYOAGenerationNotConfigured(NotImplementedError):
    """Raised when a BYOA cell is scheduled but the generation box isn't wired
    yet. Carries a precise, actionable message rather than a bare stack trace."""


class BYOAGenerator:
    """
    Generate a model's mesh on demand from its API, then cache it so each model
    is generated exactly once (the diagram's lazy-generation benefit: n calls,
    not nC2). Everything the real implementation needs is already carried on the
    registry entry and passed in here — this stub just marks the wiring points.

    To energize the circuit, implement `resolve()`:

        1. cache check   — if cache/<prompt_id>/<model_id>/model.glb exists,
                           return it (this is what makes it n, not nC2).
        2. read config   — endpoint = model["endpoint"];
                           key = os.environ[model["api_key_env"]]  (NEVER stored
                           in the registry — only the env var *name* is).
        3. build request — prompt_spec carries {prompt, seed, guidance, refs}
                           for this (model, test-case); map to the provider's
                           schema (mirror vlm_judge.py's provider-agnostic style).
        4. call + poll   — POST, then poll the job until the asset URL is ready;
                           retry transient 429/5xx with backoff.
        5. download+cache — save the GLB to the cache path from step 1, record
                           generation_time / credits into metadata, return path.

    Until then it raises BYOAGenerationNotConfigured with the exact cell, so a
    mixed registry fails loudly and specifically instead of silently skipping.
    """

    source = "BYOA"

    def __init__(self, manifest: dict | None = None, cache_root: str | None = None):
        self.cache_root = cache_root

    def resolve(self, model: dict, prompt_id: str,
                prompt_spec: Optional[dict] = None) -> Optional[str]:
        raise BYOAGenerationNotConfigured(
            f"BYOA generation not implemented for {model.get('id')} "
            f"({model.get('name')}) on {prompt_id}. "
            f"Endpoint={model.get('endpoint') or 'unset'}, "
            f"key env=${model.get('api_key_env') or 'unset'}. "
            f"Implement BYOAGenerator.resolve() in Layer 4/providers.py to wire "
            f"this cell — the scheduler already routes here."
        )


# --------------------------------------------------------------------------- #
#  registry / dispatch                                                          #
# --------------------------------------------------------------------------- #
def get_generator(source: str, manifest: dict | None = None) -> Generator:
    """Return the generator for a model's `source`. This one function is the
    whole dispatch: the scheduler calls it per model and never branches itself."""
    src = (source or "BYOM").upper()
    if src == "BYOM":
        return BYOMGenerator(manifest or {})
    if src == "BYOA":
        return BYOAGenerator(manifest)
    raise ValueError(f"unknown model source {source!r} (expected BYOM or BYOA)")


def build_generators(roster: list, manifest: dict | None = None) -> dict:
    """Pre-build one generator per distinct source present in the roster, so the
    scheduler can resolve any cell by looking up its model's source."""
    sources = {(m.get("source") or "BYOM").upper() for m in roster}
    return {s: get_generator(s, manifest) for s in sources}
