import pytest
from fastdup.datasets import FastdupHFDataset
from datasets.config import HF_DATASETS_CACHE
import os
import pandas as pd


@pytest.fixture(scope="module")
def dataset():
    return FastdupHFDataset("zh-plus/tiny-imagenet", split="train[:5%]")


def test_instantiating_dataset_with_valid_dataset_name_and_split(dataset):
    assert isinstance(dataset, FastdupHFDataset)
    assert dataset.img_key == "image"
    assert dataset.label_key == "label"
    assert dataset.jpg_save_dir == "jpg_images"
    assert dataset.cache_dir == HF_DATASETS_CACHE
    assert dataset.info.dataset_name == "tiny-imagenet"
    assert len(dataset) == 5000


def test_img_dir_exists(dataset):
    assert os.path.isdir(dataset.img_dir)
    assert dataset.img_dir == os.path.join(
        HF_DATASETS_CACHE, "tiny-imagenet", "jpg_images"
    )


# Accessing img_dir property after successful instantiation
def test_accessing_img_dir_property(dataset):
    assert isinstance(dataset.img_dir, str)
    assert os.path.exists(dataset.img_dir)


# Accessing annotations property after successful instantiation
def test_accessing_annotations_property_after_successful_instantiation(dataset):
    annotations = dataset.annotations
    assert isinstance(annotations, pd.DataFrame)
    assert "filename" in annotations.columns
    assert "label" in annotations.columns

    sample_row = dataset.annotations.iloc[0]
    assert os.path.exists(sample_row["filename"])
    assert isinstance(sample_row["label"], str)
