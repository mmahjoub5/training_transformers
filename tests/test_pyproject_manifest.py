"""Tests verifying pyproject.toml contains required project metadata (issue #18)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject():
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_has_project_metadata(pyproject):
    assert "project" in pyproject
    assert pyproject["project"]["name"]
    assert pyproject["project"]["version"]


def test_required_dependencies_listed(pyproject):
    deps_str = " ".join(pyproject["project"]["dependencies"])
    for required in ("transformers", "peft", "trl", "datasets", "torch", "accelerate", "pyyaml", "scikit-learn"):
        assert required in deps_str, f"missing {required}"


def test_optional_groups(pyproject):
    optional = pyproject["project"].get("optional-dependencies", {})
    assert "deepspeed" in optional
    assert "wandb" in optional


def test_lora_train_script_entry(pyproject):
    scripts = pyproject["project"].get("scripts", {})
    assert scripts.get("lora-train") == "scripts.lora_train:main"
