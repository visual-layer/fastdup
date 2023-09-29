import os
import numpy as np
import logging
from PIL import Image
from tqdm.auto import tqdm

try:
    import torch
except ImportError:
    raise ImportError(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )

try:
    import timm
except ImportError:
    raise ImportError(
        "The `timm` package is not installed. Please run `pip install timm`."
    )

logging.basicConfig(level=logging.INFO)


class FastdupTimmModel:
    """
    A wrapper class for TIMM (PyTorch Image Models) to simplify model initialization and
    feature extraction for image datasets.

    Attributes:
        model_name (str): The name of the model architecture to use.
        num_classes (int): The number of classes for the model. Use num_features=0 to exclude the last layer. 
        pretrained (bool): Whether to load pretrained weights.
        embeddings (np.ndarray): The computed embeddings for the images.
        file_paths (list): The file paths corresponding to the computed embeddings.
        img_folder (str): The folder path containing images for which embeddings are computed.

    Methods:
        __init__(model_name, num_classes=0, pretrained=True, **kwargs): Initialize the wrapper.
        _initialize_model(**kwargs): Internal method to initialize the TIMM model.
        compute_embeddings(image_folder_path, save_dir="."): Compute and save embeddings in a local folder.

    Example:
        >>> wrapper = FastdupTimmWrapper(model_name='resnet18')
        >>> wrapper.compute_embeddings('path/to/image/folder')
    """

    def __init__(self, model_name, num_classes=0, pretrained=True, **kwargs):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._initialize_model(**kwargs)
        self.embeddings = None
        self.file_paths = None
        self.img_folder = None

    def _initialize_model(self, **kwargs):
        self.model = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            **kwargs,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def compute_embeddings(self, image_folder_path, save_dir="."):
        self.img_folder = image_folder_path
        self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        embeddings_list = []
        file_paths = []
        img_extensions = (".jpg", ".png", ".jpeg")
        total_images = sum(
            1 for f in os.listdir(image_folder_path) if f.endswith(img_extensions)
        )

        for image_file in tqdm(
            os.listdir(image_folder_path),
            desc="Computing embeddings",
            total=total_images,
            unit=" images",
        ):
            if image_file.endswith(img_extensions):
                img_path = os.path.join(image_folder_path, image_file)

                img = Image.open(img_path)

                try:
                    img_tensor = transforms(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.model.forward_features(img_tensor)
                        output = self.model.forward_head(output, pre_logits=True)

                    embeddings = output.cpu().numpy()
                    embeddings_list.append(embeddings)
                    file_paths.append(os.path.abspath(img_path))

                except Exception as e:
                    logging.error(f"Skipping {img_path} due to error: {e}")

        self.embeddings = np.vstack(embeddings_list)
        self.file_paths = file_paths

        np.save(
            os.path.join(save_dir, f"{self.model_name}_embeddings.npy"), self.embeddings
        )

        with open(
            os.path.join(save_dir, f"{self.model_name}_file_paths.txt"), "w"
        ) as f:
            for path in self.file_paths:
                f.write(f"{path}\n")
