#!/bin/bash

cd "$(dirname "$0")"

REMOTE_URL="https://github.com/xuda-ye-math/Detailed-Balance-Normalizing-Flow.git"

# Wipe local history so force-push fully resets the remote
rm -rf .git

# Configure git identity for this repo
git init
git config user.name "xuda-ye-math"
git config user.email "xuda-ye-math@users.noreply.github.com"

# Use gh for authentication
gh auth setup-git

# Set remote (add or update if already exists)
git remote add origin "$REMOTE_URL" 2>/dev/null \
  || git remote set-url origin "$REMOTE_URL"

# Create .gitignore
cat > .gitignore << 'EOF'
# Hidden files (except .gitignore)
.*
!.gitignore

# Data / model checkpoints (any subfolder)
*.pth
*.pt

# Python bytecode caches
__pycache__/
**/__pycache__/

EOF

git add .gitignore Paper Codes *.sh

git commit -m "Edit main.tex & Update codes"

git branch -M main
git push -u origin main --force
