try:
    import torch
except ImportError:
    raise ImportError(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    raise ImportError(
        "The `segment-anything` package is not installed. Please run `pip install git+https://github.com/facebookresearch/segment-anything.git`."
    )


import os
import logging
import subprocess
import cv2
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.model.tag2text")


class SegmentAnythingModel:
    def __init__(
        self, vit_type="vit_h", weights="sam_vit_h_4b8939.pth", device="cuda"
    ) -> None:
        self.vit_type = vit_type
        self.weights = weights
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.weights):
            download_url = (
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )
            logger.info(
                f"Model file not found. Downloading model from {download_url}..."
            )
            subprocess.run(["wget", "-O", self.weights, download_url])
            logger.info("Model downloaded.")
        build_sam = sam_model_registry[self.vit_type]
        sam_model = SamPredictor(build_sam(checkpoint=self.weights))
        sam_model.mode = sam_model.model.to(self.device)
        return sam_model

    def run_inference(self, image_path, bboxes):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.model.set_image(image)
        transformed_boxes = self.model.transform.apply_boxes_torch(
            bboxes, image.shape[:2]
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
