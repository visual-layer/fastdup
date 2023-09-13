import pytest
from fastdup.datasets import FastdupHFDataset
from datasets.config import HF_DATASETS_CACHE
import os
import pandas as pd


@pytest.fixture(scope="module")
def dataset():
    return FastdupHFDataset("zh-plus/tiny-imagenet", split="train[:5%]")


def test_load_dataset(dataset):
    assert dataset.info.dataset_name == "tiny-imagenet"
    assert dataset.cache_dir == HF_DATASETS_CACHE
    assert dataset.img_key == "image"
    assert dataset.label_key == "label"
    assert dataset.jpg_save_dir == "jpg_images"


def test_img_dir(dataset):
    assert dataset.img_dir == os.path.join(
        HF_DATASETS_CACHE, "tiny-imagenet", "jpg_images"
    )


def test_dataset_len(dataset):
    assert len(dataset) == 5000


def test_img_dir_structure(dataset):
    expected_img_dir = os.path.join(
        dataset.cache_dir, dataset.hf_dataset.info.dataset_name, dataset.jpg_save_dir
    )
    assert dataset.img_dir == expected_img_dir


def test_img_dir_exists(dataset):
    assert os.path.isdir(dataset.img_dir)


def test_annotations_columns(dataset):
    assert "filename" in dataset.annotations.columns
    assert "label" in dataset.annotations.columns


def test_annotations_type(dataset):
    assert isinstance(dataset.annotations, pd.DataFrame)


def test_annotations_content(dataset):
    sample_row = dataset.annotations.iloc[0]
    assert os.path.exists(sample_row["filename"])
    assert isinstance(sample_row["label"], str)
