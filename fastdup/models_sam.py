import logging
import logging
import cv2
from fastdup.utilities import find_model
from fastdup.image import fastdup_imread
from fastdup.sentry import fastdup_capture_exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.model.sam")

try:
    import torch
except ImportError as e:
    fastdup_capture_exception("enrichment_missing_dependencies", e, True)
    logger.error(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )
    raise

try:
    from segment_anything import SamPredictor, sam_model_registry
except (ImportError, ModuleNotFoundError) as e:
    fastdup_capture_exception("enrichment_missing_dependencies", e, True)
    logger.error(
        f"The `segment-anything` package is not installed. Please run `pip install git+https://github.com/facebookresearch/segment-anything.git`."
    )
    raise

WEIGHTS_DOWNLOAD_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)


class SegmentAnythingModel:
    def __init__(
        self, model_weights: str = None, model_type: str = None, device: str = None
    ) -> None:
        # If no config/weights provided, search on local path, if not found, download from URL
        self.model_weights = (
            model_weights
            if model_weights is not None
            else find_model("sam_vit_h_4b8939.pth", WEIGHTS_DOWNLOAD_URL)
        )

        self.model_type = model_type if model_type is not None else "vit_h"

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

    def _load_model(self):
        logger.info(f"Loading model checkpoint from - {self.model_weights}")

        build_sam = sam_model_registry[self.model_type]
        model = SamPredictor(build_sam(checkpoint=self.model_weights))
        model.mode = model.model.to(self.device)
        return model

    def run_inference(self, image_path, bboxes):
        img = fastdup_imread(image_path, input_dir=None, kwargs=None)
        assert img is not None, f"Failed to read image {image_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        assert img is not None, f"Failed to color image {image_path}"

        self.model.set_image(img)
        transformed_boxes = self.model.transform.apply_boxes_torch(
            bboxes, img.shape[:2]
        )
        transformed_boxes = transformed_boxes.to(self.model.model.device)

        masks, _, _ = self.model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return masks

    def unload_model(self):
        # Move the model to CPU
        self.model.model.cpu()

        # Remove model references
        del self.model

        # Explicitly clear CUDA cache
        torch.cuda.empty_cache()