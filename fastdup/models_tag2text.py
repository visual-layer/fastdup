import logging
import cv2
from fastdup.utils import find_model
from fastdup.image import fastdup_imread
from fastdup.sentry import fastdup_capture_exception
from PIL import Image
import contextlib
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.model.tag2text")

try:
    import torch
except ImportError as e:
    fastdup_capture_exception("enrichment_missing_dependencies", e, True)
    logger.error(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )

try:
    from ram.models import tag2text
    from ram import inference_tag2text
    from ram import get_transform
except ImportError as e:
    fastdup_capture_exception("enrichment_missing_dependencies", e, True)
    logger.error(
        f"The `recognize-anything` package is not installed. Please run `pip install -U git+https://github.com/xinyu1205/recognize-anything.git`."
    )

WEIGHTS_DOWNLOAD_URL = "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/tag2text_swin_14m.pth"


class Tag2TextModel:
    def __init__(self, model_weights: str = None, device: str = None) -> None:
        # If no config/weights provided, search on local path, if not found, download from URL
        self.model_weights = (
            model_weights
            if model_weights is not None
            else find_model("tag2text_swin_14m.pth", WEIGHTS_DOWNLOAD_URL)
        )

        # Pick available device if not specified.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        try:
            self.model = self._load_model()
        except Exception as e:
            fastdup_capture_exception("model_checkpoint_corrupted", e, True)
            logger.error(
                f"Failed to load model checkpoint from {self.model_weights}. Verify the checkpoint file is not corrupted or re-download the model checkpoint."
            )

    def _load_model(self) -> torch.nn.Module:
        logger.info(f"Loading model checkpoint from - {self.model_weights}")

        # Hide the printing during model loading
        with contextlib.redirect_stdout(io.StringIO()):
            # delete some tags that may disturb captioning
            # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
            delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

            model = tag2text(
                pretrained=self.model_weights,
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
        img = fastdup_imread(image_path, input_dir=None, kwargs=None)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = Image.fromarray(img)
        transform = get_transform(image_size=384)
        image = transform(image).unsqueeze(0).to(self.device)
        results = inference_tag2text(image, self.model, user_tags)
        return results

    def unload_model(self):
        # Move the model to CPU
        self.model.cpu()

        # Remove model references
        del self.model

        # Explicitly clear CUDA cache
        torch.cuda.empty_cache()