import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
from pathlib import Path

from fastdup.fastdup_controller import FastdupController
from webcolors import rgb_to_name
import fastdup.fastup_constants as FD


def create_square(color, size=(256, 256)):
    img = np.ones((*size, len(color)))
    for i, c in enumerate(color):
        img[:, :, i] = img[:, :, i] * c
    return img


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % tuple(rgb)


def get_label(color):
    return ['red', 'green', 'blue'][np.argmax(color)]


def get_df_dict(color, suffix=''):
    split = np.random.choice(['test', 'train', 'val'], p=[0.1, 0.8, 0.1])
    return {FD.ANNOT_FILENAME: f'{rgb_to_hex(color)}{suffix}.png',
            'label': get_label(color),
            FD.ANNOT_SPLIT: split}


def save_color_im(output_dir, color, suffix=''):
    color_hex_name = rgb_to_hex(color)
    img = create_square(color)
    im = Image.fromarray(img.astype(np.uint8))
    im.save(os.path.join(output_dir, f'{color_hex_name}{suffix}.png'))
    return f'{color_hex_name}.png'


def create_corrupted_image(output_dir, color):
    """Create a corrupted image that will return error when loaded by PIL or cv2
    and save it to the output directory."""
    img = pd.DataFrame([{rgb_to_hex(color): 1}])
    img.to_pickle(os.path.join(output_dir, f'{rgb_to_hex(color)}_corrupted.png'))
    return f'{rgb_to_hex(color)}_corrupted.png'


def gen_data(output_dir, n_valid, n_corrupted, n_duplicated, n_no_annotation, n_no_image):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    colors = []
    color_idx = 0
    total_samples = n_valid + n_corrupted + n_duplicated + n_no_annotation + n_no_image
    valid_colors, corrupted_colors, duplicated_colors, no_annotation_colors, no_image_colors = [], [], [], [], []

    while len(colors) < total_samples:
        color = tuple(np.random.choice(range(256), size=3))
        if color not in colors:
            colors.append(color)

    # create valid images
    for color in colors[color_idx:n_valid]:
        valid_colors.append(get_df_dict(color))
        save_color_im(output_dir, color)
    color_idx += n_valid

    # create images with no annotation
    for color in colors[color_idx:color_idx + n_no_annotation]:
        no_annotation_colors.append(get_df_dict(color, f'_not_in_annot'))
        save_color_im(output_dir, color, f'_not_in_annot')
    color_idx += n_no_annotation

    # create corrupted images
    for color in colors[color_idx:color_idx + n_corrupted]:
        corrupted_colors.append(get_df_dict(color, f'_corrupted'))
        create_corrupted_image(output_dir, color)
    color_idx += n_corrupted

    # create duplicated images
    for color in colors[color_idx:color_idx + n_duplicated]:
        for i in range(2):
            duplicated_colors.append(get_df_dict(color, f'_{i}_duplicated'))
            save_color_im(output_dir, color, f'_{i}_duplicated')
    color_idx += n_duplicated

    # create images with no image
    for color in colors[color_idx:color_idx + n_no_image]:
        no_image_colors.append(get_df_dict(color, f'_no_image'))
    color_idx += n_no_image

    return valid_colors, corrupted_colors, duplicated_colors, no_annotation_colors, no_image_colors


def create_synthetic_data(target_dir, n_valid=100, n_corrupted=21, n_duplicated=22, n_no_annotation=23, n_no_image=24):

    valid_colors, corrupted_colors, duplicated_colors, no_annotation_colors, no_image_colors = \
        gen_data(target_dir, n_valid, n_corrupted, n_duplicated, n_no_annotation, n_no_image)

    df_annot = pd.DataFrame(valid_colors + corrupted_colors + duplicated_colors + no_image_colors)

    df_valid = pd.DataFrame(valid_colors)
    df_corrupted = pd.DataFrame(corrupted_colors)
    df_not_in_annot = pd.DataFrame(no_annotation_colors)
    df_duplicated = pd.DataFrame(duplicated_colors)
    df_no_image = pd.DataFrame(no_image_colors)
    return df_annot, df_valid, df_corrupted, df_not_in_annot, df_duplicated, df_no_image


