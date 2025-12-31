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
# CHECK FOR GIT
###############################################
if command -v git >/dev/null 2>&1; then
  echo "✅ Git already installed"
else
  echo "⚠️ Git not found — installing"

  OS="$(uname -s)"

  if [[ "$OS" == "Linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -y
      sudo apt-get install -y git
    elif command -v yum >/dev/null 2>&1; then
      sudo yum install -y git
    else
      echo "❌ Unsupported Linux package manager"
      exit 1
    fi

  elif [[ "$OS" == "Darwin" ]]; then
    # macOS: git comes with Xcode Command Line Tools
    if ! xcode-select -p >/dev/null 2>&1; then
      echo "📦 Installing Xcode Command Line Tools (git)"
      xcode-select --install
      echo "⚠️ Please rerun the script after installation completes"
      exit 1
    fi
  else
    echo "❌ Unsupported OS: $OS"
    exit 1
  fi

  echo "✅ Git installed"
fi

###############################################
# CLONE REPO
###############################################
REPO_URL="https://github.com/mmahjoub5/training_transformers.git"
REPO_DIR="training_transformers"

if [[ -d "$REPO_DIR" ]]; then
  echo "⚠️ Repo '$REPO_DIR' already exists — skipping clone"
else
  echo "⬇️ Cloning repository"
  git clone "$REPO_URL"
fi

###############################################
# Set up ENV
###############################################
ENV_NAME="llm_env"

cd "$REPO_DIR"

# Make conda available in this script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "✅ Conda env '$ENV_NAME' already exists — updating from environment.yml"
  conda env update -n "$ENV_NAME" -f ./environment.yml --prune
else
  echo "🚀 Creating conda env '$ENV_NAME' from environment.yml"
  conda env create -n "$ENV_NAME" -f ./environment.yml
fi

# Activate env
conda activate "$ENV_NAME"

###############################################
# RUN Script
###############################################
python -m scripts.train_clm_qa_lora --config configs/phi2qa.yaml --proc 4