# Evaluation runs

Grouped by the question set a run was scored against, then by date within the
filename (`YYYY-MM-DD__<original name>.json`), so each group reads
chronologically while staying separated by what it actually measures.

| Group | What it is |
| --- | --- |
| `old-208-full/` | Complete runs of `question_eval_set/2026-07-11/questions_final_208.json` — 160 single-turn plus 48 multi-turn |
| `old-208-partial/` | Targeted runs against that same set, scoped to specific case ids |
| `new-208-full/` | Complete runs of `question_eval_set/2026-08-29/questions_fresh_208.json` |
| `new-208-partial/` | Targeted runs against that set |
| `subsets/` | Failure subsets and regression checks built during debugging |
| `website-live/` | Runs against the deployed site rather than a local instance |
| `adversarial/` | Prompt-injection and cross-source claim probes |
| `other-full-runs/` | Full runs of question files that predate the two 208 sets |

Full and partial are split because a file's question set does not tell you
whether it was a benchmark or a two-case spot check — several `n=1` runs sit
against the 208 file, and averaging those into the headline number would be
meaningless.

The headline results are in the repository README, section 8.
