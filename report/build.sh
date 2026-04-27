#!/usr/bin/env bash
set -euo pipefail

# LaTeX PDF Build Script
# Usage: ./build.sh [filename]   (default: report)

TEXFILE="${1:-report}"
TEXFILE="${TEXFILE%.tex}"

cd "$(dirname "$0")"

if [[ ! -f "${TEXFILE}.tex" ]]; then
    echo "[ERROR] ${TEXFILE}.tex not found in $(pwd)"
    exit 1
fi

echo "========================================"
echo " Building ${TEXFILE}.tex"
echo "========================================"

# Pass 1
echo "[1/2] First pass..."
xelatex -interaction=nonstopmode "${TEXFILE}.tex" > /dev/null 2>&1 || \
    echo "[WARN] First pass had errors, check ${TEXFILE}.log"

# Pass 2 (resolve cross-references)
echo "[2/2] Second pass (cross-references)..."
xelatex -interaction=nonstopmode "${TEXFILE}.tex" > /dev/null 2>&1 || \
    echo "[WARN] Second pass had errors, check ${TEXFILE}.log"

if [[ -f "${TEXFILE}.pdf" ]]; then
    echo "========================================"
    echo " Done! Output: ${TEXFILE}.pdf"
    echo "========================================"
else
    echo "[ERROR] PDF generation failed. See ${TEXFILE}.log"
    exit 1
fi
