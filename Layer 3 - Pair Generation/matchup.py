"""
The Matchup object — single source of truth for the whole pipeline.
===================================================================

A matchup is the fundamental unit that travels Layer 3 -> 4 -> 5 -> 6A/6B.
Each layer ENRICHES the same object and passes it forward:

    Layer 3 (origin):   identity        prompt_id, model_a/b, names, ids
    Layer 4 (schedule): + availability  glb_a, glb_b, status
    Layer 5 (cache):    + metrics       metrics_a, metrics_b, cache paths

It is a dataclass for structure, but serializes cleanly to/from plain dicts
(asdict / from_dict) so it can be written to JSON between layers and rehydrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Matchup:
    # --- Layer 3: identity (the core {model_a, model_b, prompt_id}) ---------
    matchup_id: int            # globally unique, 0..(pairs*prompts - 1)
    prompt_id: str             # which prompt/test case, e.g. "TC001"
    pair_id: int               # which model-pair within a prompt, 0..nC2-1
    model_a: str               # model id, e.g. "M1"
    model_b: str               # model id, e.g. "M2"
    name_a: str                # display name, e.g. "Trellis 2"
    name_b: str                # display name, e.g. "Tripo 3D-2.5"

    # --- Layer 4: availability / scheduling --------------------------------
    glb_a: Optional[str] = None        # resolved path to model_a's GLB
    glb_b: Optional[str] = None        # resolved path to model_b's GLB
    status: str = "pending"            # pending | ready | missing | cached

    # --- Layer 5: extracted metrics + cache location -----------------------
    metrics_a: Optional[dict] = None
    metrics_b: Optional[dict] = None
    cache_a: Optional[str] = None      # cache dir for model_a
    cache_b: Optional[str] = None      # cache dir for model_b

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Matchup":
        return cls(**d)
