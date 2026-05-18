"""Tests confirming dead scripts from issue #13 are gone."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_main_py_deleted():
    assert not (REPO_ROOT / "main.py").exists()


def test_phi35_testing_deleted():
    assert not (REPO_ROOT / "scripts" / "phi35_testing.py").exists()


def test_scripts_train_deleted():
    assert not (REPO_ROOT / "scripts" / "train.py").exists()


def test_no_hardcoded_ubuntu_paths_in_tracked_files():
    sentinel = "/home/" + "ubuntu"
    for py in REPO_ROOT.glob("**/*.py"):
        if any(p in py.parts for p in ("__pycache__", ".git", "1-6-remotate-training", "2-3-checkpoints", "output", "runs", "tests")):
            continue
        text = py.read_text()
        assert sentinel not in text, f"hardcoded path found in {py}"
