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
    from ram.models import tag2text
    from ram import inference_tag2text
    from ram import get_transform
except ImportError:
    raise ImportError(
        "The `recognize-anything` package is not installed. Please run `pip install --upgrade setuptools && pip install -U git+https://github.com/xinyu1205/recognize-anything.git`."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.model.tag2text")


class Tag2TextModel:
    def __init__(
        self, model_path: str = "tag2text_swin_14m.pth", device: str = "cuda"
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if not os.path.exists(self.model_path):
            download_url = "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/tag2text_swin_14m.pth"
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
            # delete some tags that may disturb captioning
            # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
            delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

            model = tag2text(
                pretrained=self.model_path,
                image_size=384,
                vit="swin_b",
                delete_tag_index=delete_tag_index,
            )
            model.threshold = 0.68
            model.eval()
            model = model.to(self.device)
        logger.info(f"Model loaded to device - {self.device}")
        return model

    def run_inference(self, image_path: str, user_tags="None") -> tuple:
        transform = get_transform(image_size=384)
        image = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        results = inference_tag2text(image, self.model, user_tags)
        return results
