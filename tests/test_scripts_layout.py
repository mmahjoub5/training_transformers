"""Tests confirming scripts/test/ has been relocated (issue #16)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scripts_test_directory_removed():
    assert not (REPO_ROOT / "scripts" / "test").exists()


def test_validate_checkpoint_exists():
    assert (REPO_ROOT / "scripts" / "validate_checkpoint.py").exists()


def test_validation_run_sh_references_new_path():
    sh = (REPO_ROOT / "validation_run.sh").read_text()
    assert "scripts.validate_checkpoint" in sh
    assert "scripts.test.phi2_validation_script" not in sh
