"""Tests verifying README.md contains the required entry-point sections (issue #19)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_exists():
    assert (REPO_ROOT / "README.md").exists()


def test_readme_has_required_sections():
    content = (REPO_ROOT / "README.md").read_text()
    for heading in (
        "Quick Start",
        "Config File Schema",
        "Adding a New Preprocessor",
        "Adding a New Behavior Metric",
        "Tests",
    ):
        assert heading in content, f"missing section: {heading}"


def test_readme_has_install_command():
    content = (REPO_ROOT / "README.md").read_text()
    assert "python -m scripts.lora_train" in content
