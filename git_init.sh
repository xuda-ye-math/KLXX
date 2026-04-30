#!/bin/bash

cd "$(dirname "$0")"

REMOTE_URL="https://github.com/xuda-ye-math/Detailed-Balance-Normalizing-Flow.git"

git init
git config user.name "xuda-ye-math"
git config user.email "xuda-ye-math@users.noreply.github.com"

gh auth setup-git

git remote add origin "$REMOTE_URL" 2>/dev/null \
  || git remote set-url origin "$REMOTE_URL"

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

git branch -M main
