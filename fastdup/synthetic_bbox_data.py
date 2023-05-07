import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
from pathlib import Path
import cv2
from fastdup.fastdup_controller import FastdupController
from webcolors import rgb_to_name
import fastdup.definitions as FD
import matplotlib.pyplot as plt


def create_bboxes(bg_color, colors, bbox_x, bbox_y, bbox_h, bbox_w, size=(256, 256)):
    img = np.ones((*size, len(bg_color)))
    for i, c in enumerate(bg_color):
        img[:, :, i] = img[:, :, i] * c

    for x, y, h, w, color in zip(bbox_x, bbox_y, bbox_h, bbox_w, colors):
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (int(color[0]), int(color[1]), int(color[2])), -1)

    return img


def rgb_to_hex(rgb_colors: list):
    return '_'.join(['%02x%02x%02x' % tuple(c) for c in rgb_colors])


def get_label(color):
    return ['red', 'green', 'blue'][np.argmax(color)]


def get_df_dict(colors, label_color, bbox_x, bbox_y, bbox_h, bbox_w, suffix=''):
    split = np.random.choice(['test', 'train', 'val'], p=[0.1, 0.8, 0.1])
    return {FD.ANNOT_FILENAME: f'{rgb_to_hex(colors)}{suffix}.png',
            'label': get_label(label_color),
            'img_label': get_label(np.sum(colors, axis=0)),
            FD.ANNOT_SPLIT: split,
            FD.ANNOT_BBOX_X: bbox_x,
            FD.ANNOT_BBOX_Y: bbox_y,
            FD.ANNOT_BBOX_H: bbox_h,
            FD.ANNOT_BBOX_W: bbox_w,
            FD.ANNOT_IMG_H: 256,
            FD.ANNOT_IMG_W: 256}

def save_bbox_im(output_dir, bg_color, colors, bbox_x, bbox_y, bbox_h, bbox_w, suffix=''):
    color_hex_name = rgb_to_hex([bg_color] + list(colors))
    img = create_bboxes(bg_color, colors, bbox_x, bbox_y, bbox_h, bbox_w)
    im = Image.fromarray(img.astype(np.uint8))
    im.save(os.path.join(output_dir, f'{color_hex_name}{suffix}.png'))
    return f'{color_hex_name}.png'


def create_corrupted_image(output_dir, color):
    """Create a corrupted image that will return error when loaded by PIL or cv2
    and save it to the output directory."""
    img = pd.DataFrame([{rgb_to_hex(color): 1}])
    img.to_pickle(os.path.join(output_dir, f'{rgb_to_hex(color)}_corrupted.png'))
    return f'{rgb_to_hex(color)}_corrupted.png'


def gen_data(output_dir, n_valid_single_bbox, n_valid_double_bbox, n_duplicated_bbox,
             n_corrupted_image, n_no_image):
    colors = []
    color_idx = 0
    total_samples = n_valid_single_bbox + n_valid_double_bbox + n_duplicated_bbox + \
                    n_corrupted_image + n_no_image

    valid_colors_single_bbox, valid_colors_double_bbox, corrupted_bbox, duplicated_bbox, no_image_bbox = \
        [], [], [], [], []

    while len(colors) < total_samples * 3:
        color = tuple(np.random.choice(range(256), size=3))
        if color not in colors:
            colors.append(color)

    # create single bbox valid images
    for bg_color, bbox_color in zip(colors[color_idx:color_idx + n_valid_single_bbox],
                                    colors[color_idx + n_valid_single_bbox:color_idx + 2*n_valid_single_bbox]):
        bbox_x, bbox_y, bbox_h, bbox_w = np.random.randint(0, 180), np.random.randint(0, 180), \
                                         np.random.randint(10, 60), np.random.randint(10, 60)
        save_bbox_im(output_dir, bg_color, [bbox_color], [bbox_x], [bbox_y], [bbox_h], [bbox_w])
        valid_colors_single_bbox.append(get_df_dict([bg_color, bbox_color], bbox_color, bbox_x, bbox_y, bbox_h, bbox_w))
    color_idx += 2*n_valid_single_bbox

    # create double bbox images
    for bg_color, bbox_color1, bbox_color2 in zip(colors[color_idx:color_idx + n_valid_single_bbox],
                                    colors[color_idx + n_valid_single_bbox:color_idx + 2*n_valid_single_bbox],
                                    colors[color_idx + 2*n_valid_single_bbox:color_idx + 3*n_valid_single_bbox]):
        bbox_x_1, bbox_y_1, bbox_h_1, bbox_w_1 = np.random.randint(0, 50), np.random.randint(0, 50), \
                                                 np.random.randint(10, 60), np.random.randint(10, 60)
        bbox_x_2, bbox_y_2, bbox_h_2, bbox_w_2 = np.random.randint(120, 180), np.random.randint(120, 180), \
                                                 np.random.randint(10, 60), np.random.randint(10, 60)
        save_bbox_im(output_dir, bg_color, [bbox_color1, bbox_color2], [bbox_x_1, bbox_x_2],
                     [bbox_y_1, bbox_y_2], [bbox_h_1, bbox_h_2], [bbox_w_1, bbox_w_2])
        valid_colors_double_bbox.append(get_df_dict([bg_color, bbox_color1, bbox_color2], bbox_color1, bbox_x_1, bbox_y_1, bbox_h_1, bbox_w_1))
        valid_colors_double_bbox.append(get_df_dict([bg_color, bbox_color1, bbox_color2], bbox_color2, bbox_x_2, bbox_y_2, bbox_h_2, bbox_w_2))
    color_idx += 3*n_valid_single_bbox

    # create duplicated bbox images
    for bg_color1, bg_color2, bbox_color in zip(colors[color_idx:color_idx + n_valid_single_bbox],
                                                  colors[color_idx + n_valid_single_bbox:color_idx + 2 * n_valid_single_bbox],
                                                  colors[color_idx + 2 * n_valid_single_bbox:color_idx + 3 * n_valid_single_bbox]):
        bbox_x1, bbox_y1 = np.random.randint(0, 180), np.random.randint(0, 180)
        bbox_x2, bbox_y2 = np.random.randint(0, 180), np.random.randint(0, 180)
        bbox_h, bbox_w = np.random.randint(10, 60), np.random.randint(10, 60)
        save_bbox_im(output_dir, bg_color1, [bbox_color], [bbox_x1], [bbox_y1], [bbox_h], [bbox_w], suffix='_1_duplicate')
        save_bbox_im(output_dir, bg_color2, [bbox_color], [bbox_x2], [bbox_y2], [bbox_h], [bbox_w], suffix='_2_duplicate')
        duplicated_bbox.append(get_df_dict([bg_color1, bbox_color], bbox_color, bbox_x1, bbox_y1, bbox_h, bbox_w, suffix='_1_duplicate'))
        duplicated_bbox.append(get_df_dict([bg_color2, bbox_color], bbox_color, bbox_x2, bbox_y2, bbox_h, bbox_w, suffix='_2_duplicate'))
    color_idx += 3*n_valid_single_bbox

    # create corrupted images
    for color in colors[color_idx:color_idx + n_corrupted_image]:
        bbox_x, bbox_y, bbox_h, bbox_w = np.random.randint(0, 180), np.random.randint(0, 180), \
                                         np.random.randint(10, 60), np.random.randint(10, 60)
        create_corrupted_image(output_dir, [color])
        corrupted_bbox.append(get_df_dict([color], color, bbox_x, bbox_y, bbox_h, bbox_w, suffix='_corrupted'))

    color_idx += n_corrupted_image

    # create annot with no image
    for color in colors[color_idx:color_idx + n_no_image]:
        bbox_x, bbox_y, bbox_h, bbox_w = np.random.randint(0, 180), np.random.randint(0, 180), \
            np.random.randint(10, 60), np.random.randint(10, 60)
        no_image_bbox.append(get_df_dict([color], color, bbox_x, bbox_y, bbox_h, bbox_w, suffix='_no_image'))
    color_idx += n_no_image

    return pd.DataFrame(valid_colors_single_bbox), pd.DataFrame(valid_colors_double_bbox), \
        pd.DataFrame(duplicated_bbox), pd.DataFrame(corrupted_bbox), pd.DataFrame(no_image_bbox)


def create_invalid_bbox_ims(output_dir):
    invalid_bbox = []
    color = tuple(np.random.choice(range(256), size=3))

    # list of invalid bbox x, y, h, w. h,
    # w < 10, or w < 10 or x >= 256 or y >= 256 or x < 0 or y < 0 or x + w > 256 or y + h > 256
    invalid_bbox_args = [
        (-1, -1, -1, -1),
        (0, 0, 0, 0),
        (0, 0, 0, 9),
        (0, 0, 9, 0),
        (0, 0, 9, 9),
        (0, 0, 9, 10),
        (0, 0, 10, 9),
        (70, 70, 10, 9),
        (70, 70, 9, 10),
        (56, 56, 201, 90),
        (56, 56, 90, 201),
        (-1, 56, 90, 90),
        (56, -1, 90, 90),
        (256, 56, 90, 90),
        (56, 256, 90, 90),
    ]

    for i, (bbox_x, bbox_y, bbox_h, bbox_w) in enumerate(invalid_bbox_args):
        save_bbox_im(output_dir, color, [color], [0], [0], [10], [10], suffix=f'_{i}_invalid_bbox')
        invalid_bbox.append(get_df_dict([color, color], color, bbox_x, bbox_y, bbox_h, bbox_w, suffix=f'_{i}_invalid_bbox'))
    return pd.DataFrame(invalid_bbox)


def create_synthetic_data(target_dir, n_valid_single_bbox=50, n_valid_double_bbox=50, n_duplicated_bbox=21,
                          n_corrupted_image=22, n_no_image=23):
    #shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    df_single, df_double, df_duplicate, df_corrupted, df_no_image = \
        gen_data(target_dir, n_valid_single_bbox, n_valid_double_bbox, n_duplicated_bbox, n_corrupted_image, n_no_image)
    df_invalid_bbox = create_invalid_bbox_ims(target_dir)
    df_annot = pd.concat([df_single, df_double, df_duplicate, df_corrupted, df_no_image, df_invalid_bbox])

    return df_annot, df_invalid_bbox, df_single, df_double, df_duplicate, df_corrupted, df_no_image

