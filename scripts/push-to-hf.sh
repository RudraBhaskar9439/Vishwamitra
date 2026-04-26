#!/usr/bin/env bash
# push-to-hf.sh — Sync the GitHub-clean README.md to the HF Space remote
# with the YAML frontmatter prepended. The frontmatter is required by HF
# Spaces but rendered as an ugly metadata table on GitHub, so we keep
# it out of the canonical README.md and re-attach it only on push.
#
# Usage:
#     ./scripts/push-to-hf.sh
#
# Assumes a remote named `hf` exists. If not:
#     git remote add hf https://huggingface.co/spaces/<user>/<space>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .hf_frontmatter.md ]]; then
    echo "ERROR: .hf_frontmatter.md not found at repo root" >&2
    exit 1
fi

# Backup the canonical README, prepend frontmatter for HF, commit, push.
cp README.md /tmp/README.github.md
cat .hf_frontmatter.md README.md > /tmp/README.hf.md

cp /tmp/README.hf.md README.md
git add README.md
if git diff --cached --quiet; then
    echo "No README changes to commit before HF push."
else
    git commit -m "chore(hf): prepend HF Spaces frontmatter for deployment"
fi

echo "Pushing to hf remote..."
git push hf main

# Restore the GitHub-clean README locally so future commits stay clean.
cp /tmp/README.github.md README.md
git add README.md
if git diff --cached --quiet; then
    echo "Local README already clean."
else
    git commit -m "chore(github): restore frontmatter-free README"
    echo "Pushing the cleanup commit to origin..."
    git push origin main
fi

echo "Done."
