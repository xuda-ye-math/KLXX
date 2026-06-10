#!/bin/bash

cd "$(dirname "$0")"

REMOTE_URL="https://github.com/xuda-ye-math/Log-Likelihood-Ratio-Discrepancy.git"

git init
git config user.name "xuda-ye-math"
git config user.email "xuda-ye-math@users.noreply.github.com"

gh auth setup-git

git remote add origin "$REMOTE_URL" 2>/dev/null \
  || git remote set-url origin "$REMOTE_URL"

git branch -M main
