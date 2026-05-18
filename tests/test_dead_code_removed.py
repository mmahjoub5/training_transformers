"""Tests confirming dead modules from issues #11 and #13 are gone."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_src_data_preprocess_deleted():
    assert not (REPO_ROOT / "src" / "data" / "preprocess.py").exists()


def test_scripts_train_deleted():
    assert not (REPO_ROOT / "scripts" / "train.py").exists()


def test_no_imports_from_dead_preprocess():
    for py in REPO_ROOT.glob("**/*.py"):
        if any(p in py.parts for p in ("__pycache__", ".git", "tests")):
            continue
        text = py.read_text()
        assert "from src.data.preprocess" not in text, f"{py}"
        assert "from scripts.train" not in text, f"{py}"
