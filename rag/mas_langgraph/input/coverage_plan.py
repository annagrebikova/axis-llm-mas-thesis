"""Target quiz coverage — edit counts per difficulty and question type."""

from ..core.entities import CoveragePlan

DEFAULT_COVERAGE_PLAN = CoveragePlan(
    default_per_difficulty_type={
        ("beginner", "multiple_choice"): 1,
        ("intermediate", "multiple_choice"): 2,
        ("advanced", "multiple_choice"): 1,
        ("beginner", "open_ended"): 0,
        ("intermediate", "open_ended"): 1,
        ("advanced", "open_ended"): 0,
        ("beginner", "true_false"): 0,
        ("intermediate", "true_false"): 0,
        ("advanced", "true_false"): 0,
    },
    per_topic=None,
    per_lo=None,
)
