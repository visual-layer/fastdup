import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from PIL import Image
import numpy as np


def convert_to_coco_format(df, bbox_col, label_col, json_filename):
    # Initialize COCO formatted dictionary
    coco_format = defaultdict(list)

    # Initialize counters for unique IDs
    image_id = 0
    annotation_id = 0

    # Initialize a category set to keep track of unique categories
    category_set = set()

    # Iterate through each row in the DataFrame to populate the COCO dictionary
    for _, row in df.iterrows():
        # Add image information
        image_id += 1

        with Image.open(row["filename"]) as img:
            width, height = img.size

        image_info = {
            "id": image_id,
            "file_name": row["filename"],
            "width": width,
            "height": height,
        }
        coco_format["images"].append(image_info)

        # Parse bounding boxes and labels
        bboxes = row[bbox_col]
        labels = row[label_col]

        for bbox, label in zip(bboxes, labels):
            # Update category set
            category_set.add(label)

            # Add annotation information
            annotation_id += 1
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label,  # Set to label name instead of None
                "bbox": [x1, y1, width, height],
                "area": area,
                "iscrowd": 0,
            }
            coco_format["annotations"].append(annotation_info)

    # Add categories to COCO format
    for category_id, category_name in enumerate(sorted(list(category_set)), start=1):
        category_info = {"id": category_id, "name": category_name}
        coco_format["categories"].append(category_info)

    # Update category IDs in annotations
    category_map = {
        name: idx for idx, name in enumerate(sorted(list(category_set)), start=1)
    }
    for annotation in coco_format["annotations"]:
        annotation["category_id"] = category_map[annotation["category_id"]]

    with open(json_filename, 'w') as f:
        json.dump(coco_format, f)


def plot_annotations(df, image_col='filename', bbox_col=None, labels_col=None, scores_col=None, tags_col=None,
                     masks_col=None, num_rows=5):
    df = df.head(num_rows)

    if tags_col:
        unique_tags = {tag for labels in df[tags_col] for tag in labels.replace(" ", "").split('.')}
        cmap = plt.cm.get_cmap('hsv', len(unique_tags))
        label_color_map = {label: cmap(idx)[:3] for idx, label in enumerate(sorted(unique_tags))}
    else:
        label_color_map = {}

    num_subplots = 1  # always show original
    if bbox_col:
        num_subplots += 1
    if masks_col:
        num_subplots += 1

    num_rows = len(df)
    fig, axes = plt.subplots(num_rows, num_subplots, figsize=(6 * num_subplots, 6 * num_rows))

    if num_rows == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(df.iterrows()):
        # Read image
        try:
            image = cv2.imread(row[image_col])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {row[image_col]}: {e}")
            continue

        # Original image
        axes[idx][0].imshow(image_rgb)
        axes[idx][0].set_title("Original Image")
        axes[idx][0].axis('off')

        subplot_idx = 1

        # Bounding boxes
        if bbox_col:
            axes[idx][subplot_idx].imshow(image_rgb)
            axes[idx][subplot_idx].set_title("Annotated Boxes")
            axes[idx][subplot_idx].axis('off')
            for bbox, label, score in zip(row[bbox_col], row[labels_col], row[scores_col]):
                x_min, y_min, x_max, y_max = bbox
                edge_color = label_color_map.get(label, (1, 1, 1))
                axes[idx][subplot_idx].add_patch(
                    plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=edge_color,
                                  facecolor='none'))
                axes[idx][subplot_idx].text(x_min, y_min - 5, f'{label} | {score:.2f}', color='white',
                                            bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.5'))
            subplot_idx += 1

        # Masks
        if masks_col:
            axes[idx][subplot_idx].imshow(image_rgb)
            axes[idx][subplot_idx].set_title("Annotated Masks")
            axes[idx][subplot_idx].axis('off')
            masks = row[masks_col].cpu().numpy()
            for mask in masks:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                axes[idx][subplot_idx].imshow(mask_image, alpha=0.9)

    plt.tight_layout()
    plt.show()