import logging
from PIL import Image
import contextlib
import io
import os
import subprocess

try:
    import torch
except ImportError:
    raise ImportError(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )

try:
    from ram.models import ram
    from ram import inference_ram
    from ram import get_transform
except ImportError:
    raise ImportError(
        "The `recognize-anything` package is not installed. Please run `!pip install --upgrade setuptools && pip install -U git+https://github.com/xinyu1205/recognize-anything.git`."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.models.ram")


class RecognizeAnythingModel:
    def __init__(
        self, model_path: str = "ram_swin_large_14m.pth", device: str = "cuda"
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if not os.path.exists(self.model_path):
            download_url = "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth"
            logger.info(
                f"Model file not found. Downloading model from {download_url}..."
            )
            subprocess.run(
                ["wget", "-O", self.model_path, download_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            logger.info("Model downloaded.")

        logger.info(f"Loading model checkpoint - {self.model_path}")

        # Hide the printing during model loading
        with contextlib.redirect_stdout(io.StringIO()):
            model = ram(pretrained=self.model_path, image_size=384, vit="swin_l")
            model.eval()
            model = model.to(self.device)
        logger.info(f"Model loaded to device - {self.device}")
        return model

    def run_inference(self, image_path: str) -> str:
        transform = get_transform(image_size=384)
        image = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        results = inference_ram(image, self.model)
        tags = results[0].replace(" | ", " . ")
        return tags
