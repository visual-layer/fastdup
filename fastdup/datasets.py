import os
import pandas as pd
from datasets import load_dataset, Dataset
from datasets.config import HF_DATASETS_CACHE
from tqdm.auto import tqdm
import hashlib


class FastdupHFDataset(Dataset):
    """
    FastdupHFDataset is a subclass of the Dataset class from the Hugging Face's `datasets` library.
    It is designed to load datasets from the Hugging Face dataset hub, cache them, and perform image conversion
    if the dataset has changed since the last cached version.

    Attributes:
        dataset_name (Dataset): The dataset object from the Hugging Face dataset library.
        img_key (str): The key used to access image data in the dataset.
        label_key (str): The key used to access label data in the dataset.
        cache_dir (str): The directory where the dataset cache will be stored.
        split (str): The dataset split.
        reconvert (bool): Whether to run the reconvertion of the dataset into jpg locally.
        

    Methods:
        _generate_cache_dir_hash(): Generates a SHA-256 hash of the current dataset directory.
        _cache_metadata(str): Caches the current dataset directory hash to a file.
        _retrieve_cached_metadata(): Retrieves the previously cached dataset directory hash.
        _save_as_image_files(): Converts the dataset items to .jpg image files and saves them in the cache directory.

    Properties:
        images_dir (str) : Returns the path where the dataset is downloaded.
        annotations (pd.DataFrame): Returns a DataFrame containing filenames and their corresponding labels.

    Example Usage:
        >>> dataset = FastdupHFDataset(dataset_name='your_dataset_name', split='train')
        >>> annotations_df = dataset.annotations 
        >>> dataset.images_dir
    """

    def __init__(
        self,
        dataset_name,
        split="train",
        cache_dir=None,
        img_key="image",
        label_key="label",
        reconvert=False,
        **kwargs,
    ):
        self.img_key = img_key
        self.label_key = label_key
        self.reconvert_jpg = reconvert

        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = HF_DATASETS_CACHE  # default location for storing cache

        self.hf_dataset = load_dataset(
            dataset_name, split=split, cache_dir=self.cache_dir, **kwargs
        )

        super().__init__(
            self.hf_dataset.data, self.hf_dataset.info, self.hf_dataset.split
        )

        # Get hash for the current dataset downloaded in the cache_dir
        current_hash = self._generate_cache_dir_hash()

        # Retrieve the hash of the previously processed dataset, if exists
        previous_hash = self._retrieve_cached_metadata()

        # Compare hashes
        if (current_hash != previous_hash) or reconvert:
            print("Running image conversion.")
            self._save_as_image_files()

            # Update the cache with the new hash
            self._cache_metadata(current_hash)
        else:
            print("No changes in dataset. Skipping image conversion.")

    @property
    def images_dir(self):
        return os.path.join(self.cache_dir, self.hf_dataset.info.dataset_name)

    def _generate_cache_dir_hash(self):
        files = []

        def scan_dir(directory):
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        files.append(entry.path)
                    elif entry.is_dir():
                        scan_dir(entry.path)

        scan_dir(self.cache_dir)
        files.sort()  # Ensure consistent order
        data = "".join(files)
        return hashlib.sha256(data.encode()).hexdigest()

    def _cache_metadata(self, cache_hash):
        cache_file = os.path.join(self.cache_dir, "dataset_dir_hash.txt")
        with open(cache_file, "w") as f:
            f.write(cache_hash)

    def _retrieve_cached_metadata(self):
        cache_file = os.path.join(self.cache_dir, "dataset_dir_hash.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return f.read()
        return None

    def _save_as_image_files(self):
        for idx, item in tqdm(
            enumerate(self.hf_dataset),
            total=len(self.hf_dataset),
            desc="Converting to .jpg images",
        ):
            # extract the image and label
            image = item[self.img_key]
            label = item[self.label_key]

            # create a directory for the class if it doesn't exist
            label_dir = os.path.join(
                os.path.join(
                    f"{self.cache_dir}",
                    f"{self.hf_dataset.info.dataset_name}",
                    "images",
                ),
                str(label),
            )
            os.makedirs(label_dir, exist_ok=True)

            # save the image to the appropriate directory
            image.save(os.path.join(label_dir, f"{idx}.jpg"))

    @property
    def annotations(self):
        """Returns a Pandas DataFrame with filename, label column"""
        path = os.path.join(self.images_dir, "images")
        filenames = []
        labels = []

        for entry in os.scandir(path):
            if entry.is_dir():
                label = entry.name
                for subentry in os.scandir(entry.path):
                    if subentry.is_file():
                        filenames.append(subentry.path)
                        labels.append(label)

        df = pd.DataFrame({"filename": filenames, "label": labels})
        return df
