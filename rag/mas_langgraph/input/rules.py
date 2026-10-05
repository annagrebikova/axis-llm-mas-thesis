"""Business rules and rubrics for curator, atomizer, verifier — edit for your domain."""

CHUNK_QUALITY_CRITERIA = [
    "Low noise: content is substantive, not boilerplate or off-topic.",
    "No duplication: content does not largely repeat another chunk in the pool.",
    "Source trust: from a credible source (e.g. document section, page) and traceable.",
]

ATOM_EXTRACTION_RULES = [
    "One fact per atom: no multi-fact paragraphs; split into separate atoms.",
    "Concrete and testable: avoid vague statements or non-testable claims.",
    "Types: 'fact' (single factual claim), 'definition' (concept definition), 'procedure' (step or method).",
    "Content: one or two clear sentences; no bullet lists inside one atom.",
]

VERIFICATION_RULES = [
    "Answer supported by evidence: The correct answer must be explicitly or clearly implied by the evidence chunks.",
    "Options consistent: For multiple choice, all options must be distinct and plausible; exactly one must be correct.",
    "No hallucinations: The question must not introduce facts not present in the evidence chunks.",
    "Question clarity: The question must be unambiguous and answerable from the evidence alone.",
]

STRUCTURE_PLACEHOLDER_BLOCKLIST = frozenset({
    "section_id", "title", "start_slide", "end_slide", "weight",
    "importance_weight", "slide_range", "section", "placeholder", "n/a", "tbd", "tba", "",
})

SECTION_ID_PATTERN = r"^sec_\d+$"
WEIGHT_MIN, WEIGHT_MAX = 0.1, 1.0

DIFFICULTY_LEVELS = ("beginner", "intermediate", "advanced")
QUESTION_TYPES = ("multiple_choice", "open_ended", "true_false")
ATOM_TYPES = ("fact", "definition", "procedure")
