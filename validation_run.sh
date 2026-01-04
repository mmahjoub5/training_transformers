#!/usr/bin/env bash
set -e

if command -v conda >/dev/null 2>&1; then
  echo "✅ Conda already installed"
else
  echo "⚠️ Conda not found — installing Miniconda"

  OS="$(uname -s)"
  ARCH="$(uname -m)"

  if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
  elif [[ "$OS" == "Linux" && "$ARCH" == "aarch64" ]]; then
    INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
  elif [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
  elif [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
  else
    echo "❌ Unsupported OS/architecture: $OS $ARCH"
    exit 1
  fi

  curl -fsSL "https://repo.anaconda.com/miniconda/$INSTALLER" -o "$INSTALLER"

  if [[ -d "$HOME/miniconda3" ]]; then
    echo "⚠️ $HOME/miniconda3 already exists; skipping install"
  else
    echo "📦 Installing Miniconda (non-interactive)"
    bash "$INSTALLER" -b -p "$HOME/miniconda3"
  fi

  rm -f "$INSTALLER"

  # make conda available in THIS script
  source "$HOME/miniconda3/etc/profile.d/conda.sh"

  echo "✅ Conda ready"
fi

###############################################
# Set up ENV
###############################################
REPO_URL="https://github.com/mmahjoub5/training_transformers.git"
REPO_DIR="training_transformers"
ENV_NAME="llm_env"

# Set to "true" to force update even if env exists, "false" to skip update
FORCE_UPDATE="${FORCE_UPDATE:-false}"

# Make conda available in this script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if it doesn't exist, or update if FORCE_UPDATE=true
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [ "$FORCE_UPDATE" = "true" ]; then
    echo "🔄 Updating conda env '$ENV_NAME' from environment.yml (FORCE_UPDATE=true)"
    conda env update -n "$ENV_NAME" -f ./environment.yml --prune -y
  else
    echo "✅ Conda env '$ENV_NAME' already exists — skipping update (set FORCE_UPDATE=true to update)"
  fi
else
  echo "🚀 Creating conda env '$ENV_NAME' from environment.yml"
  conda env create -n "$ENV_NAME" -f ./environment.yml -y
fi

# Activate env
conda activate "$ENV_NAME"

###############################################
# RUN Script
###############################################

python -m  scripts.test.phi2_validation_script  \
  --json data/qa_prompts.jsonl \
  --checkpoint /home/ubuntu/training_transformers/output/1231/phi2/checkpoint-500\
  --baseline microsoft/phi-2 \
  --output results.json \
  --max-tokens 200 \
  --temperature 0.7 \
  --quiet
