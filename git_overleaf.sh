#!/bin/bash

cd "$(dirname "$0")"

git checkout main
git fetch origin --prune

OVERLEAF_BRANCH=$(git branch -r | grep 'origin/overleaf-' | sed 's|.*origin/||' | sort | tail -n 1 | xargs)

if [ -z "$OVERLEAF_BRANCH" ]; then
  echo "No origin/overleaf-* branch found. Nothing to merge."
  exit 0
fi

echo "Merging origin/$OVERLEAF_BRANCH into main (GitHub side wins on conflicts)..."
git merge -X ours --no-edit "origin/$OVERLEAF_BRANCH"

git push origin main
