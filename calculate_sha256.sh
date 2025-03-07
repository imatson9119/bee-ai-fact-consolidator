#!/bin/bash

# This script helps calculate the SHA256 hash for your release tarball
# Run this after creating a release on GitHub

if [ -z "$1" ]; then
    echo "Usage: $0 <release_tarball_url>"
    echo "Example: $0 https://github.com/imatson9119/bee-ai-fact-consolidator/archive/refs/tags/v0.1.0.tar.gz"
    exit 1
fi

URL=$1
echo "Calculating SHA256 for $URL..."

# Download and calculate hash
if command -v curl &> /dev/null && command -v shasum &> /dev/null; then
    HASH=$(curl -sL "$URL" | shasum -a 256 | cut -d' ' -f1)
    echo "SHA256: $HASH"
    echo ""
    echo "Update your Homebrew formula with this hash:"
    echo "  sha256 \"$HASH\""
else
    echo "Error: curl and shasum are required for this script."
    exit 1
fi 