import logging
import cv2
from fastdup.utilities import find_model
from fastdup.image import fastdup_imread
from fastdup.sentry import fastdup_capture_exception
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastdup.models.grounding_dino")

try:
    import torch
except ImportError as e:
    fastdup_capture_exception("enrichment_missing_dependencies", e, True)
    logger.error(
        "The `torch` package is not installed. Please run `pip install torch` or equivalent."
    )

try:
    from mmengine.config import Config
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

    grounding_dino_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
except ImportError as e:
    fastdup_capture_exception("enrichment_missing_dependencies", e, True)
    installation = (
        "`pip install mmengine "
        "groundingdino-py "
        "git+https://github.com/open-mmlab/mmdetection.git` "
    )
    logger.error(
        f"Your system is missing some dependencies. Please install with {installation}."
    )


# Download URLs
CONFIG_DOWNLOAD_URL = "https://raw.githubusercontent.com/open-mmlab/playground/main/mmdet_sam/configs/GroundingDINO_SwinT_OGC.py"
WEIGHTS_DOWNLOAD_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"


class GroundingDINO:
    def __init__(
        self,
        model_config: str = None,
        model_weights: str = None,
        device: str = None,
    ) -> None:
        # If no config/weights provided, search on local path, if not found, download from URL
        self.model_config = (
            model_config
            if model_config is not None
            else find_model("GroundingDINO_SwinT_OGC.py", CONFIG_DOWNLOAD_URL)
        )
        self.model_weights = (
            model_weights
            if model_weights is not None
            else find_model("groundingdino_swint_ogc.pth", WEIGHTS_DOWNLOAD_URL)
        )

        # Pick available device if not specified.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self._load_model()

    def _load_model(self):
        logger.info(f"Loading model checkpoint from - {self.model_weights}")
        gdino_args = Config.fromfile(self.model_config)
        model = build_model(gdino_args)

        checkpoint = torch.load(self.model_weights, map_location=self.device)

        # Ensure the checkpoint contains the necessary "model" key
        if "model" not in checkpoint:
            logger.error(
                "The model checkpoint does not contain the expected 'model' key."
            )
            raise ValueError("Invalid checkpoint format")

        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

        model.eval()
        model.to(self.device)
        logger.info(f"Model loaded on device - {self.device}")

        return model

    def run_inference(
        self,
        image_path,
        text_prompt,
        box_threshold=0.3,
        text_threshold=0.25,
        apply_original_groundingdino=False,
    ):
        pred_dict = {}
        img = fastdup_imread(image_path, input_dir=None, kwargs=None)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image_pil = Image.fromarray(img)
        image_pil = self._apply_exif_orientation(image_pil)
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        text_prompt = text_prompt.lower().strip()
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        # Original GroundingDINO use text-thr to get class name,
        # the result will always result in categories that we don't want,
        # so we provide a category-restricted approach to address this
        if not apply_original_groundingdino:
            custom_vocabulary = text_prompt[:-1].split(".")
            label_name = [c.strip() for c in custom_vocabulary]
            tokens_positive = []
            start_i = 0
            separation_tokens = " . "
            for _index, label in enumerate(label_name):
                end_i = start_i + len(label)
                tokens_positive.append([(start_i, end_i)])
                if _index != len(label_name) - 1:
                    start_i = end_i + len(separation_tokens)
            tokenizer = get_tokenlizer.get_tokenlizer("bert-base-uncased")
            tokenized = tokenizer(text_prompt, padding="longest", return_tensors="pt")
            positive_map_label_to_token = self._create_positive_dict(
                tokenized, tokens_positive, list(range(len(label_name)))
            )

        image = image.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model(image[None], captions=[text_prompt])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        if not apply_original_groundingdino:
            logits = self._convert_grounding_to_od_logits(
                logits, len(label_name), positive_map_label_to_token
            )

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        if apply_original_groundingdino:
            tokenlizer = self.model.tokenizer
            tokenized = tokenlizer(text_prompt)
            pred_labels = []
            pred_scores = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(
                    logit > text_threshold, tokenized, tokenlizer
                )
                pred_labels.append(pred_phrase)
                pred_scores.append(round(float(logit.max().item()), 4))

        else:
            scores, pred_phrase_idxs = logits_filt.max(1)
            pred_labels = []
            pred_scores = []
            for score, pred_phrase_idx in zip(scores, pred_phrase_idxs):
                pred_labels.append(label_name[pred_phrase_idx])
                pred_scores.append(round(float(score.item()), 4))

        pred_dict["labels"] = pred_labels
        pred_dict["scores"] = pred_scores

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        pred_dict["boxes"] = [
            tuple(round(coord, 2) for coord in box) for box in boxes_filt.tolist()
        ]

        return pred_dict

    def _convert_grounding_to_od_logits(
        self, logits, num_classes, positive_map, score_agg="MEAN"
    ):
        """
        logits: (num_query, max_seq_len)
        num_classes: 80 for COCO
        """
        assert logits.ndim == 2
        assert positive_map is not None
        scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)
        # 256 -> 80, average for each class
        # score aggregation method
        if score_agg == "MEAN":  # True
            for label_j in positive_map:
                scores[:, label_j] = logits[
                    :, torch.LongTensor(positive_map[label_j])
                ].mean(-1)
        else:
            raise NotImplementedError
        return scores

    def _apply_exif_orientation(self, image):
        """Applies the exif orientation correctly.

        This code exists per the bug:
        https://github.com/python-pillow/Pillow/issues/3973
        with the function `ImageOps.exif_transpose`.
        The Pillow source raises errors with
        various methods, especially `tobytes`
        Function based on:
        https://github.com/facebookresearch/detectron2/\
        blob/78d5b4f335005091fe0364ce4775d711ec93566e/\
        detectron2/data/detection_utils.py#L119
        Args:
            image (PIL.Image): a PIL image
        Returns:
            (PIL.Image): the PIL image with exif orientation applied, if applicable
        """
        _EXIF_ORIENT = 274
        if not hasattr(image, "getexif"):
            return image

        try:
            exif = image.getexif()
        except Exception:
            # https://github.com/facebookresearch/detectron2/issues/1885
            exif = None

        if exif is None:
            return image

        orientation = exif.get(_EXIF_ORIENT)

        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            return image.transpose(method)
        return image

    def _create_positive_dict(self, tokenized, tokens_positive, labels):
        """construct a dictionary such that positive_map[i] = j,
        if token i is mapped to j label"""

        positive_map_label_to_token = {}

        for j, tok_list in enumerate(tokens_positive):
            for beg, end in tok_list:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)

                assert beg_pos is not None and end_pos is not None
                positive_map_label_to_token[labels[j]] = []
                for i in range(beg_pos, end_pos + 1):
                    positive_map_label_to_token[labels[j]].append(i)

        return positive_map_label_to_token

    def unload_model(self):
        # Move the model to CPU
        self.model.cpu()

        # Remove model references
        del self.model

        # Explicitly clear CUDA cache
        torch.cuda.empty_cache()
