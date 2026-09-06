#!/usr/bin/env bash
#
# Sync a source branch (default: main) to the Hugging Face Space.
#
# The Space needs two things GitHub should not have:
#   * YAML frontmatter at the top of README.md declaring sdk/app_port —
#     GitHub renders it as a stray table, HF needs it to build.
#   * no Eval_ordered/ or question_eval_set/ — hundreds of MB of run artifacts
#     that belong in the repo but not in the container image.
#
# So hf-deploy is a *derived* branch: this script rebuilds it from the source
# tree every time. Never edit hf-deploy by hand — the next run overwrites it.
#
set -euo pipefail

SRC_BRANCH="${1:-main}"
DEPLOY_BRANCH="hf-deploy"
REMOTE="hf"
HEADER_PATH=".hf/space-header.md"
EXCLUDE=(Eval_ordered question_eval_set)

cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "error: working tree has uncommitted changes. Commit or stash first." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

git rev-parse --verify --quiet "$SRC_BRANCH" >/dev/null \
  || { echo "error: no such branch: $SRC_BRANCH" >&2; exit 1; }

# Read the header out of the source tree, not the worktree, so it is whatever
# the commit we are deploying says it is.
HEADER="$(git show "$SRC_BRANCH:$HEADER_PATH")" \
  || { echo "error: $HEADER_PATH missing on $SRC_BRANCH" >&2; exit 1; }

SRC_SHA="$(git rev-parse --short "$SRC_BRANCH")"
START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
trap 'git checkout --quiet "$START_BRANCH"' EXIT

git checkout --quiet "$DEPLOY_BRANCH"

# Take the source tree wholesale so the deploy branch cannot drift. Files the
# index knows about but the source tree does not are removed from the worktree,
# so refuse to run if anything untracked has been staged by mistake.
if [ -n "$(git diff --cached --name-only)" ]; then
  echo "error: the index is not clean; refusing to reset the worktree." >&2
  git diff --cached --name-only >&2
  exit 1
fi
git read-tree -u --reset "$SRC_BRANCH"

# Eval artifacts stay on GitHub. Drop them from the index and from git's view
# of the worktree, but never `rm -rf` the directories: they also hold untracked
# local run output that git cannot restore afterwards.
for path in "${EXCLUDE[@]}"; do
  git rm -r --quiet --cached --ignore-unmatch -- "$path" >/dev/null
done
{
  printf '\n# Deploy branch: eval artifacts stay on GitHub, not in the Space image.\n'
  printf '%s/\n' "${EXCLUDE[@]}"
} >> .gitignore

# Space config goes back on top of the README.
printf '%s\n\n' "$HEADER" | cat - README.md > README.hf.tmp
mv README.hf.tmp README.md

# Stage only what this script changed. `git add -A` would sweep in whatever
# untracked scratch files happen to be sitting in the worktree.
git add -- .gitignore README.md
if git diff --cached --quiet; then
  echo "hf-deploy already matches $SRC_BRANCH ($SRC_SHA); nothing to push."
  exit 0
fi

git commit --quiet -m "Deploy $SRC_BRANCH $SRC_SHA to the Space"
echo "Built $DEPLOY_BRANCH from $SRC_BRANCH $SRC_SHA:"
git show --stat --oneline HEAD | head -20

if [ "${HF_DEPLOY_DRY_RUN:-}" = "1" ]; then
  echo "HF_DEPLOY_DRY_RUN=1 — built the branch but did not push."
  exit 0
fi

git push "$REMOTE" "$DEPLOY_BRANCH:main"
echo "Pushed to $REMOTE."
