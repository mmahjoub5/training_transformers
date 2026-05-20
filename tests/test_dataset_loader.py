"""Unit tests for src/data/dataset_loader.py (issue #21)."""

import json

import pytest

pytest.importorskip("datasets", reason="datasets not installed in this environment")

from src.data.dataset_loader import load_dataset_generic  # noqa: E402


@pytest.fixture
def tiny_jsonl(tmp_path):
    rows = [{"text": f"row {i}", "label": i} for i in range(10)]
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_loads_train_split_size(tiny_jsonl):
    cfg = {
        "name": "json",
        "config_name": None,
        "data_file": tiny_jsonl,
        "train_split": "train",
        "shuffle_train": False,
        "seed": 42,
        "max_train_samples": None,
        "val_split": None,
        "max_val_samples": None,
        "test_split": None,
        "max_test_samples": None,
    }
    ds = load_dataset_generic(cfg)
    assert "train" in ds
    assert len(ds["train"]) == 10


def test_respects_max_train_samples(tiny_jsonl):
    cfg = {
        "name": "json",
        "config_name": None,
        "data_file": tiny_jsonl,
        "train_split": "train",
        "shuffle_train": False,
        "seed": 42,
        "max_train_samples": 3,
        "val_split": None,
        "max_val_samples": None,
        "test_split": None,
        "max_test_samples": None,
    }
    ds = load_dataset_generic(cfg)
    assert len(ds["train"]) == 3


def test_shuffle_changes_order_when_enabled(tiny_jsonl):
    base_cfg = {
        "name": "json",
        "config_name": None,
        "data_file": tiny_jsonl,
        "train_split": "train",
        "seed": 42,
        "max_train_samples": None,
        "val_split": None,
        "max_val_samples": None,
        "test_split": None,
        "max_test_samples": None,
    }
    ds_unshuffled = load_dataset_generic({**base_cfg, "shuffle_train": False})
    ds_shuffled = load_dataset_generic({**base_cfg, "shuffle_train": True})

    unshuffled_labels = [ex["label"] for ex in ds_unshuffled["train"]]
    shuffled_labels = [ex["label"] for ex in ds_shuffled["train"]]
    assert sorted(unshuffled_labels) == sorted(shuffled_labels)
    # With 10 deterministic rows and seed=42 shuffle should reorder
    assert shuffled_labels != unshuffled_labels
