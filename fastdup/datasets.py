import os
import pandas as pd
from datasets import load_dataset, Dataset
from datasets.config import HF_DATASETS_CACHE
from tqdm.auto import tqdm
import hashlib
import logging
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(level=logging.INFO)


class FastdupHFDataset(Dataset):
    """
    FastdupHFDataset is a subclass of Hugging Face's Dataset, tailored for usage in fastdup.

    Attributes:
        img_key (str): Key to access image data.
        label_key (str): Key to access label data.
        cache_dir (str): Directory for caching datasets.
        reconvert (bool): Flag to force reconversion of images from .parquet to .jpg.

    Methods:
        _generate_cache_dir_hash(): Creates a hash of the dataset directory.
        _cache_metadata(): Caches the hash.
        _retrieve_cached_metadata(): Retrieves the cached hash.
        _save_as_image_files(): Converts and saves images in .jpg format.
        _save_single_image(): Saves a single image (internal use).

    Properties:
        img_dir (str): Directory where images are stored.
        annotations (pd.DataFrame): Dataframe of filenames and labels.

    Example:
        >>> from fastdup.datasets import FastdupHFDataset
        >>> dataset = FastdupHFDataset('your_dataset_name', split='train')
        >>> import fastdup
        >>> fd = fastdup.create(input_dir=dataset.img_dir)
        >>> fd.run(annotations=dataset.annotations)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        img_key: str = "image",
        label_key: str = "label",
        reconvert_jpg: bool = False,
        jpg_save_dir: str = "jpg_images",
        **kwargs: Any,
    ) -> None:
        self.img_key: str = img_key
        self.label_key: str = label_key
        self.jpg_save_dir = jpg_save_dir

        if cache_dir:
            self.cache_dir: str = cache_dir
        else:
            self.cache_dir: str = HF_DATASETS_CACHE

        try:
            self.hf_dataset = load_dataset(
                dataset_name, split=split, cache_dir=self.cache_dir, **kwargs
            )
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return

        super().__init__(
            self.hf_dataset.data, self.hf_dataset.info, self.hf_dataset.split
        )

        # If jpg folder does not exist, run conversion and cache the folder
        jpg_img_folder = os.path.join(self.cache_dir, self.hf_dataset.info.dataset_name, self.jpg_save_dir)
        if not os.path.exists(jpg_img_folder):
            logging.info(f"Running image conversion. Destination: {jpg_img_folder}")
            self._save_as_image_files()
            current_hash: str = self._generate_img_folder_hash()
            self._cache_metadata(current_hash)
            return

        try:
            current_hash: str = self._generate_img_folder_hash()
            previous_hash: Optional[str] = self._retrieve_cached_metadata()
        except Exception as e:
            logging.error(f"Error generating or retrieving hash: {e}")
            return

        if (current_hash != previous_hash) or reconvert_jpg:
            logging.info(f"Running image conversion. Destination: {jpg_img_folder}")
            self._save_as_image_files()
            self._cache_metadata(current_hash)
        else:
            logging.info(f"No changes in dataset folder: {jpg_img_folder}. Skipping image conversion.")

    @property
    def img_dir(self) -> str:
        return os.path.join(self.cache_dir, self.hf_dataset.info.dataset_name, self.jpg_save_dir)

    def _generate_img_folder_hash(self) -> str:
        files = []

        def scan_dir(directory: str) -> None:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        files.append(entry.path)
                    elif entry.is_dir():
                        scan_dir(entry.path)

        scan_dir(os.path.join(self.cache_dir, self.hf_dataset.info.dataset_name, self.jpg_save_dir))
        data: str = "".join(files)
        return hashlib.sha256(data.encode()).hexdigest()

    def _cache_metadata(self, cache_hash: str) -> None:
        cache_file: str = os.path.join(self.cache_dir, f"dataset_dir_hash_{self.jpg_save_dir}.txt")
        try:
            with open(cache_file, "w") as f:
                f.write(cache_hash)
        except Exception as e:
            logging.error(f"Error caching metadata: {e}")

    def _retrieve_cached_metadata(self) -> Optional[str]:
        cache_file: str = os.path.join(self.cache_dir, f"dataset_dir_hash_{self.jpg_save_dir}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Error reading cached metadata: {e}")
        return None

    def _save_single_image(self, idx: int, item: dict, pbar) -> None:
        try:
            image = item[self.img_key]
            label = item[self.label_key]
            label_dir: str = os.path.join(
                os.path.join(
                    f"{self.cache_dir}",
                    f"{self.hf_dataset.info.dataset_name}",
                    f"{self.jpg_save_dir}",
                ),
                str(label),
            )
            os.makedirs(label_dir, exist_ok=True)
            image.save(os.path.join(label_dir, f"{idx}.jpg"))
            pbar.update(1)
        except Exception as e:
            logging.error(f"Error in saving image at index {idx}: {e}")

    def _save_as_image_files(self) -> None:
        with tqdm(total=len(self.hf_dataset), desc="Converting to .jpg images") as pbar:
            with ThreadPoolExecutor() as executor:
                executor.map(
                    self._save_single_image,
                    range(len(self.hf_dataset)),
                    self.hf_dataset,
                    [pbar] * len(self.hf_dataset),
                )

    @property
    def annotations(self) -> pd.DataFrame:
        path: str = os.path.join(self.img_dir)
        filenames: list[str] = []
        labels: list[str] = []

        try:
            for entry in os.scandir(path):
                if entry.is_dir():
                    label: str = entry.name
                    for subentry in os.scandir(entry.path):
                        if subentry.is_file():
                            filenames.append(subentry.path)
                            labels.append(label)
        except Exception as e:
            logging.error(f"Error in generating annotations: {e}")
            return pd.DataFrame()

        df: pd.DataFrame = pd.DataFrame({"filename": filenames, "label": labels})
        return df


