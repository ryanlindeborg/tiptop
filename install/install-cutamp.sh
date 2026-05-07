#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-cutamp"
    exit 1
fi

REPO_URL="https://github.com/tiptop-robot/cuTAMP.git"
INSTALL_DIR="cutamp"
BRANCH="will/new-fixes"

echo "==> Installing cuTAMP from branch $BRANCH..."

# Check if directory exists and is a healthy git repo
should_clone=true
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    if git fsck --full &> /dev/null; then
        echo "Updating existing cuTAMP repository to latest $BRANCH..."
        git fetch origin "$BRANCH"
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
        should_clone=false
        cd ..
    else
        echo "✗ cuTAMP repository is corrupted, removing..."
        cd ..
        rm -rf "$INSTALL_DIR"
    fi
fi

# Clone the branch
if [ "$should_clone" = true ]; then
    echo "Cloning cuTAMP branch $BRANCH..."
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

# Install
cd "$INSTALL_DIR"
echo "Installing cuTAMP..."
pip install -e . --no-build-isolation --no-deps
cd ..
echo "✓ cuTAMP ($BRANCH) installed successfully"
