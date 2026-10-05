"""Deterministic correct-answer display positioning for CandidateMCQ options.

Maps stable question identity (``q_00000001``, …) onto labels A–D so factual,
NOOP, and revision paths share the same display position for the same ``q_id``.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

from .entities import AnswerOption

OPTION_LABELS = ("A", "B", "C", "D")
_QID_NUMERIC = re.compile(r"^q_(\d+)$", re.IGNORECASE)


def target_option_label_for_qid(stable_key: str) -> str:
    """Return A/B/C/D for ``stable_key`` (numeric q_id cycle, else SHA-256)."""
    key = (stable_key or "").strip()
    match = _QID_NUMERIC.match(key)
    if match:
        n = int(match.group(1))
        if n >= 1:
            return OPTION_LABELS[(n - 1) % len(OPTION_LABELS)]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return OPTION_LABELS[int(digest[:8], 16) % len(OPTION_LABELS)]


def _unique_correct_option(
    options: Sequence[AnswerOption],
    correct_option_id: Optional[str] = None,
) -> Optional[AnswerOption]:
    marked = [o for o in options if o.is_correct]
    if len(marked) == 1:
        return marked[0]
    if len(marked) == 0 and correct_option_id:
        matches = [o for o in options if o.option_id == correct_option_id]
        if len(matches) == 1:
            return matches[0]
    return None


def assign_deterministic_option_positions(
    options: Sequence[AnswerOption],
    *,
    stable_key: str,
    correct_option_id: Optional[str] = None,
) -> Tuple[List[AnswerOption], str]:
    """Reorder four options so the semantic correct answer sits at a fixed label.

    Preserves option text, ``is_correct``, explanation, misconception, and
    ``source_refs``. Relabels ``option_id`` to A–D. Returns ``(options, correct_option_id)``.

    If there is not exactly one semantic correct option among exactly four
    options, returns deep-copied inputs unchanged (no invented correct answer).
    """
    opts = list(options)
    if len(opts) != 4:
        cid = correct_option_id or ""
        if not cid:
            marked = [o for o in opts if o.is_correct]
            if len(marked) == 1:
                cid = marked[0].option_id
        return [deepcopy(o) for o in opts], cid

    correct = _unique_correct_option(opts, correct_option_id)
    if correct is None:
        cid = correct_option_id or ""
        return [deepcopy(o) for o in opts], cid

    distractors = [o for o in opts if o is not correct and not o.is_correct]
    # If is_correct was missing but correct_option_id matched, exclude by identity.
    if len(distractors) != 3:
        distractors = [o for o in opts if o is not correct]
    if len(distractors) != 3:
        return [deepcopy(o) for o in opts], correct.option_id

    target = target_option_label_for_qid(stable_key)
    target_idx = OPTION_LABELS.index(target)

    ordered: List[AnswerOption] = []
    d_iter = iter(distractors)
    for i in range(4):
        if i == target_idx:
            src = correct
            is_correct = True
        else:
            src = next(d_iter)
            is_correct = False
        ordered.append(
            AnswerOption(
                option_id=OPTION_LABELS[i],
                text=src.text,
                is_correct=is_correct,
                explanation=src.explanation,
                misconception=src.misconception,
                source_refs=list(src.source_refs),
            )
        )
    return ordered, target
