import os
import cv2
import numpy as np
import logging
from PIL import Image
from tqdm.auto import tqdm
from fastdup.sentry import fastdup_capture_exception
from fastdup.image import fastdup_imread
from fastdup.utils import get_images_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.embeddings.timm")

try:
    import torch
except ImportError as e:
    fastdup_capture_exception("embeddings_missing_pytorch_install", e, True)
    logger.error(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )

try:
    import timm
except ImportError as e:
    fastdup_capture_exception("embeddings_missing_timm_install", e, True)
    logger.error(
        "The `timm` package is not installed. Please run `pip install timm` or equivalent."
    )


class TimmEncoder:
    """
    A wrapper class for TIMM (PyTorch Image Models) to simplify model initialization and
    feature extraction for image datasets.

    Attributes:
        model_name (str): The name of the model architecture to use.
        num_classes (int): The number of classes for the model. Use num_features=0 to exclude the last layer.
        pretrained (bool): Whether to load pretrained weights.
        device (str): Which device to load the model on. Choices: "cuda" or "cpu".
        torch_compile (bool): Whether to use torch.compile to optimize model.

        embeddings (np.ndarray): The computed embeddings for the images.
        file_paths (list): The file paths corresponding to the computed embeddings.
        img_folder (str): The folder path containing images for which embeddings are computed.

    Methods:
        __init__(model_name, num_classes=0, pretrained=True, **kwargs): Initialize the wrapper.
        _initialize_model(**kwargs): Internal method to initialize the TIMM model.
        compute_embeddings(image_folder_path, save_dir="."): Compute and save embeddings in a local folder.

    Example:
        >>> wrapper = TimmEncoder(model_name='resnet18')
        >>> wrapper.compute_embeddings('path/to/image/folder')
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 0,
        pretrained: bool = True,
        device: str = None,
        torch_compile: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.torch_compile = torch_compile

        # Pick available device if not specified.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self._initialize_model(**kwargs)
        self.embeddings = None
        self.file_paths = None
        self.img_folder = None

    def _initialize_model(self, **kwargs):
        logger.info(f"Initializing model - {self.model_name}.")
        self.model = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            **kwargs,
        )

        if self.torch_compile:
            logger.info("Running torch.compile.")
            self.model = torch.compile(self.model, mode="max-autotune")

        self.model.eval()
        self.model = self.model.to(self.device)

        logger.info(f"Model loaded on device - {self.device}")

    def compute_embeddings(self, image_folder_path, save_dir="saved_embeddings"):
        self.img_folder = image_folder_path

        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        embeddings_list = []
        file_paths = []

        # Get images with extensions supported in fastdup
        total_images = len(get_images_from_path(image_folder_path))

        for image_file in tqdm(
            os.listdir(image_folder_path),
            desc="Computing embeddings",
            total=total_images,
            unit=" images",
        ):
            img_path = os.path.join(image_folder_path, image_file)

            try:
                img = fastdup_imread(img_path, input_dir=None, kwargs=None)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = Image.fromarray(img)
                img_tensor = transforms(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model.forward_features(img_tensor)
                    output = self.model.forward_head(output, pre_logits=True)

                embeddings = output.cpu().numpy()
                embeddings_list.append(embeddings)
                file_paths.append(os.path.abspath(img_path))

            except Exception as e:
                logger.error(f"Skipping {img_path} due to error: {e}")

        self.embeddings = np.vstack(embeddings_list)
        self.file_paths = file_paths

        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Saving embeddings in directory - {save_dir} .")

        np.save(
            os.path.join(save_dir, f"{self.model_name.split('/')[-1]}_embeddings.npy"),
            self.embeddings,
        )

        with open(
            os.path.join(save_dir, f"{self.model_name.split('/')[-1]}_file_paths.txt"),
            "w",
        ) as f:
            for path in self.file_paths:
                f.write(f"{path}\n")
