"""
Small shared helpers used by more than one node.
"""
from __future__ import annotations
from typing import Type, TypeVar
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def coerce(model_cls: Type[M], value) -> M:
    """
    Re-validate a state field into `model_cls` instead of assuming it's
    already that type.

    Why this exists: a `ResearchState` field typed as `ResearchReport` (or
    `SubFinding`, `CritiqueResult`, `SubQuestion`) is only guaranteed to
    *start* as that Pydantic type. Whether it comes back out the other side
    of a checkpoint save/restore still a Pydantic instance, or as a plain
    dict, depends on the checkpointer backend and its serializer:

    - `InMemorySaver` never actually serializes anything within a single
      process, so you generally get the exact same object back.
    - `SqliteSaver` (or any backend that persists to disk and can be
      reloaded in a *different* process — e.g. resuming a session after
      restarting the CLI) round-trips state through a real serializer,
      and custom Pydantic types aren't guaranteed to survive that as the
      same Python class rather than a dict.

    Rather than get this right once and risk it being wrong the one time it
    matters (a resumed session, days later), every node that reads a
    structured field back out of state goes through this function.
    """
    if isinstance(value, model_cls):
        return value
    return model_cls.model_validate(value)
