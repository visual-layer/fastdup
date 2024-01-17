import cv2
from collections import defaultdict
import json
from PIL import Image
import numpy as np



def export_to_coco(df, bbox_col, label_col, json_filename):
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

    with open(json_filename, "w") as f:
        json.dump(coco_format, f)


def annotate_image(image_path: str, annotations: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except:
        print("matplotlib dependency is needed please install using pip3 install matplotlib")
        raise

    # Read the image
    img = plt.imread(image_path)

    # Create a new figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    label_to_color = generate_colormap(annotations["labels"])

    # Iterate over the labels, scores, and boxes to draw them on the image
    for label, score, box in zip(
        annotations["labels"], annotations["scores"], annotations["boxes"]
    ):
        x1, y1, x2, y2 = box
        color = label_to_color.get(label, (0, 0, 0, 1))  # Fallback color is black
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1,
            f"{label} | {score:.2f}",
            fontsize=10,
            color="white",
            bbox=dict(facecolor=color, edgecolor=color, boxstyle="round,pad=0.5"),
        )

    # Show the image with annotations
    plt.show()


def plot_annotations(
    df,
    image_col="filename",
    bbox_col=None,
    labels_col=None,
    scores_col=None,
    tags_col=None,
    masks_col=None,
    num_rows=5,
):
    try:
        import matplotlib.pyplot as plt
    except:
        print("matplotlib dependency is needed please install using pip3 install matplotlib")
        raise

    df = df.head(num_rows)

    num_subplots = 1  # always show original
    if bbox_col:
        num_subplots += 1
    if masks_col:
        num_subplots += 1

    num_rows = len(df)
    fig, axes = plt.subplots(
        num_rows, num_subplots, figsize=(6 * num_subplots, 6 * num_rows)
    )

    if num_rows == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            image = cv2.imread(row[image_col])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {row[image_col]}: {e}")
            continue

        # Original image
        axes[idx][0].imshow(image_rgb)
        axes[idx][0].set_title("Original Image")
        axes[idx][0].axis("off")

        subplot_idx = 1

        # Bounding boxes
        if bbox_col:
            axes[idx][subplot_idx].imshow(image_rgb)
            axes[idx][subplot_idx].set_title("Annotated Boxes")
            axes[idx][subplot_idx].axis("off")
            for bbox, label, score in zip(
                row[bbox_col], row[labels_col], row[scores_col]
            ):
                x_min, y_min, x_max, y_max = bbox
                label_to_color = generate_colormap(row[labels_col])
                color = label_to_color.get(
                    label, (0, 0, 0, 1)
                )  # Fallback color is black
                axes[idx][subplot_idx].add_patch(
                    plt.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                )
                axes[idx][subplot_idx].text(
                    x_min,
                    y_min - 5,
                    f"{label} | {score:.2f}",
                    color="white",
                    bbox=dict(
                        facecolor=color, edgecolor=color, boxstyle="round,pad=0.5"
                    ),
                )
            subplot_idx += 1

        # Masks
        if masks_col:
            axes[idx][subplot_idx].imshow(image_rgb)
            axes[idx][subplot_idx].set_title("Annotated Masks")
            axes[idx][subplot_idx].axis("off")
            masks = row[masks_col].cpu().numpy()
            for mask in masks:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                axes[idx][subplot_idx].imshow(mask_image, alpha=0.9)

    plt.tight_layout()
    plt.show()


def generate_colormap(labels, hue_start=0.1, hue_end=0.9, saturation=0.9, value=0.8):
    """
    Generate a colormap for a set of unique labels while avoiding bright colors.

    Parameters:
        labels (iterable): An iterable object containing labels.
        hue_start (float): The start value of the hue range in HSV space.
        hue_end (float): The end value of the hue range in HSV space.
        saturation (float): Saturation level to set for the colors.
        value (float): Brightness level to set for the colors.

    Returns:
        dict: A dictionary mapping labels to colors in RGB format.
    """
    try:
        from matplotlib.colors import hsv_to_rgb
    except:
        print("matplotlib dependency is needed please install using pip3 install matplotlib")
        raise

    unique_labels = set(labels)
    n_labels = len(unique_labels)

    # Define ranges for Hue
    hue_range = np.linspace(hue_start, hue_end, n_labels)

    # Create colormap in HSV and then convert to RGB
    colormap_hsv = np.zeros((n_labels, 3))
    colormap_hsv[:, 0] = hue_range
    colormap_hsv[:, 1] = saturation
    colormap_hsv[:, 2] = value
    colormap = [hsv_to_rgb(color) for color in colormap_hsv]

    # Create a label to color mapping
    label_to_color = {label: colormap[i] for i, label in enumerate(unique_labels)}

    return label_to_color
