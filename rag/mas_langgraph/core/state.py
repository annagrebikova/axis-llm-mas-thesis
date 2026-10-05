"""Compatibility re-export — MCQ ``CooperativeState`` lives in ``cases.mcq.cooperative_state``.

Importing this module loads MCQ-specific types. Generic backbone code should use
``BaseCooperativeState`` from ``core.state_base`` instead.
"""

from ..cases.mcq.cooperative_state import CooperativeState

__all__ = ["CooperativeState"]
