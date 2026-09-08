This folder orders eval runs by production time.

Layout:
- `YYYY-MM-DD/HHMMSS_microseconds/main/` for standard evals
- `YYYY-MM-DD/HHMMSS_microseconds/citation_fix/` for citation-focused runs
- `YYYY-MM-DD/HHMMSS_microseconds/failed_subset/` for failed-subset and regression runs

The timestamp is taken from the file's `generated_at` field when available, otherwise from file modified time.
