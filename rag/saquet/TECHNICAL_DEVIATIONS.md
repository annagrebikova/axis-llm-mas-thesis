# Technical deviations from the supplied SAQUET notebook

Only the following execution adaptations were made. Evaluation methodology is unchanged.

1. OpenAI API key is read from the `OPENAI_API_KEY` environment variable instead of a hardcoded secret string.
2. `absolute_terms_verify` accepts the `question` object (and callers pass `question`), fixing the notebook NameError where the verifier referenced an out-of-scope `question` while receiving only a string.
4. OpenAI chat completions used by `llm_rating` and `more_than_one_correct` now have an explicit request timeout (`SAQUET_LLM_TIMEOUT_S`, default 60s) and a bounded attempt limit (`SAQUET_LLM_MAX_ATTEMPTS`, default 8). Failed API calls and malformed responses both count toward the limit. Rating responses must be a single integer in the prompt range 1–10; `more_than_one_correct` JSON must contain a `correct_answers` list of string maps. After exhaustion the criterion raises `SaquetEvaluationError` so the batch evaluator records NA rather than treating the failure as “flaw” or “no flaw.” Valid-rating aggregation (mode/average) and early stopping on two agreeing high/low ratings are unchanged. Retry sleep after API errors remains 10 seconds (`SAQUET_LLM_RETRY_SLEEP_S`) and is skipped for malformed-but-completed responses.
