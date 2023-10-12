import fastdup
from fastdup.datasets import FastdupHFDataset
import shutil


def test_datasets_example():
    dataset = FastdupHFDataset("zh-plus/tiny-imagenet", split="train[:5%]") # Download 5% of the train set
    fd = fastdup.create(input_dir=dataset.img_dir)
    fd.run(annotations=dataset.annotations, num_images=100)
    # Clean up to remove clutter
    shutil.rmtree("work_dir")