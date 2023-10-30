import fastdup
import pandas as pd
from fastdup.utils import get_images_from_path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import os
from glob import glob
import time
import base64
from fastdup.models_utils import generate_colormap
import json
import shutil
from tqdm import tqdm


import torch
from PIL import Image
import sys
sys.path.append('/Users/achiyajerbi/projects/misc/')
from typing import List, Optional

from recognize_anything.ram.models import ram, ram_plus
from recognize_anything.ram import inference_ram as inference
from recognize_anything.ram import get_transform


IMGS_WIDTH = 1000
IMGS_PATH = '/Users/achiyajerbi/projects/matrix_imgs_batch_0'
OUTPUT_DIR = '/Users/achiyajerbi/projects/tags_vis_dir'


def ram_inference(img_path: str, image_size: int = 384):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=image_size)

    #######load model
    model = ram(pretrained='/Users/achiyajerbi/ram_swin_large_14m.pth',
                             image_size=image_size,
                             vit='swin_l')
    model.eval()
    model = model.to(device)
    image = transform(Image.open(img_path)).unsqueeze(0).to(device)
    res = inference(image, model)
    print("Image Tags: ", res[0])
    return res[0].replace(" | ", " . ")


def ram_plus_inference(img_path: str, image_size: int = 384):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=image_size)

    #######load model
    model = ram_plus(pretrained='/Users/achiyajerbi/Downloads/ram_plus_swin_large_14m.pth',
                             image_size=image_size,
                             vit='swin_l')
    model.eval()
    model = model.to(device)
    image = transform(Image.open(img_path)).unsqueeze(0).to(device)
    res = inference(image, model)
    print("Image Tags: ", res[0])
    return res[0].replace(" | ", " . ")


def resize_bboxes(bboxes, ratio):
    return [tuple(x * ratio for x in bbox) for bbox in bboxes]


def plot_bboxes(ax, bboxes, labels, scores):
    for bbox, label, score in zip(
        bboxes, labels, scores
    ):
        x_min, y_min, x_max, y_max = bbox
        label_to_color = generate_colormap(labels)
        color = label_to_color.get(
            label, (0, 0, 0, 1)
        )  # Fallback color is black
        ax.add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x_min,
            y_min - 5,
            f"{label} | {score:.2f}",
            color="white",
            bbox=dict(
                facecolor=color, edgecolor=color, boxstyle="round,pad=0.5"
            ),
        )

    return ax


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_html(image_files, output_file_path, orig_tags_names: Optional[str] = None):
    # Start building the HTML content
    html_content = "<html><body>   <style> .green-text {color: green;} .red-text {color: red;} </style>"

    # Iterate over each image
    for image_file in image_files:
        data_dict = json.load(open(image_file.replace('.jpg', '.json')))
        assert orig_tags_names is None or orig_tags_names in data_dict, f"expected {orig_tags_names} key in data dict"
        orig_tags = set(data_dict[orig_tags_names].split(' . '))

        # Open a container div for each image and table with vertical centering
        html_content += "<div style='display: flex; align-items: center;'>"

        # Add image to the HTML content
        html_content += "<img src='data:image/png;base64," + image_to_base64(image_file) +  "' alt='Image' style='max-height: 2000px; width: 2000px;'>"

        # Add table to the right of the image
        html_content += "<table border='1' style='margin-left: 10px; height: 100px;'>"
        for key, value in data_dict.items():            
            html_content += f"<tr style='height: 50%;'><td style='font-size: 30px;' >{key}</td><td style='font-size: 30px;'>{value}</td></tr>"
            if orig_tags_names is not None and key.endswith(' Tags') and key != orig_tags_names:
                curr_tags = value.split(' . ')
                additions = [('+' + x) for x in list(set(curr_tags) - orig_tags)]
                deletions = [('-' + x) for x in list(orig_tags - set(curr_tags))]
                html_content += f"<tr style='height: 50%;'><td style='font-size: 30px;' >{key} additions:</td> <td class='green-text' style='font-size: 30px;'>{', '.join(additions)}</td></tr>"
                html_content += f"<tr style='height: 50%;'><td style='font-size: 30px;' >{key} deletions:</td> <td class='red-text' style='font-size: 30px;'>{', '.join(deletions)}</td></tr>"
        html_content += "</table>"

        # Close the container div and add horizontal line
        html_content += "</div> <style> hr { border: none; border-top: 5px solid #000; margin: 0; width: 100%; } </style> <hr>"

    # Close the HTML content
    html_content += "</body></html>"

    # Write the HTML content to a file
    with open(output_file_path, 'w') as html_file:
        html_file.write(html_content.encode('ascii', 'ignore').decode('ascii'))


def run_comparison(device: str, num_imgs: int):
    output_dir = os.path.join(OUTPUT_DIR, f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_{num_imgs}_imgs')
    cap_start = time.perf_counter()
    fd = fastdup.create(input_dir=IMGS_PATH)
    fd.run()
    filenames = get_images_from_path(fd.input_dir)
    np.random.seed(123)
    chosen_filenames = np.random.choice(filenames, num_imgs, replace=False).tolist()
    df = pd.DataFrame({'filename': chosen_filenames})
    # df = fd.caption(model_name='blip2', device = 'gpu' if device == 'cuda' else 'cpu', batch_size=8, subset=chosen_filenames)
    
    # df = fd.enrich(task='zero-shot-classification', model='recognize-anything-model', device=device, input_df=df, input_col='filename')

    df['ram_plus_tags'] = 'N/A'
    df['ram_tags'] = 'N/A'

    ram_start = time.perf_counter()
    for i, row in tqdm(df.iterrows(), desc='Running RAM inference', total=len(df)):
        df.iloc[i]['ram_tags'] = ram_inference(row['filename'])
    ram_inference_time = time.perf_counter() - ram_start
    
    ram_plus_start = time.perf_counter()
    for i, row in tqdm(df.iterrows(), desc='Running RAM++ inference', total=len(df)):
        df.iloc[i]['ram_plus_tags'] = ram_plus_inference(row['filename'])
    ram_plus_inference_time = time.perf_counter() - ram_plus_start

    df = fd.enrich(task='zero-shot-detection', 
               model='grounding-dino', 
               input_df=df, 
               input_col='ram_tags'
     )
    df.rename(columns={
        'grounding_dino_bboxes': 'grounding_dino_bboxes_ram', 
        'grounding_dino_scores': 'grounding_dino_scores_ram',
        'grounding_dino_labels': 'grounding_dino_labels_ram'}, inplace=True)

    df = fd.enrich(task='zero-shot-detection', 
               model='grounding-dino', 
               input_df=df, 
               input_col='ram_plus_tags'
     )
    df.rename(columns={
        'grounding_dino_bboxes': 'grounding_dino_bboxes_ram_plus', 
        'grounding_dino_scores': 'grounding_dino_scores_ram_plus',
        'grounding_dino_labels': 'grounding_dino_labels_ram_plus'}, inplace=True)

    cap_end = time.perf_counter()
    print(f'Captioning on {device} took {cap_end - cap_start} seconds')

    plots_output_path = os.path.join(output_dir, 'images')
    os.makedirs(plots_output_path, exist_ok=True)

    for _, row in df.iterrows():
        filename = row['filename']
        ram_labels = row['ram_tags']
        ram_plus_labels = row['ram_plus_tags']
        nice_filename = '_'.join(filename.split('/')[-2:])
        img_output_path = os.path.join(plots_output_path, nice_filename)

        # Read the image using PIL
        image = Image.open(filename)
        resize_ratio = float(IMGS_WIDTH / image.size[1])
        new_height = int(resize_ratio * image.size[0])
        resized_img = image.resize((new_height, IMGS_WIDTH))
        
        _, axes = plt.subplots(1, 3, figsize=(30, 15))

        axes[0].imshow(resized_img)
        axes[0].set_title('Original image')
        axes[0].axis('off')

        axes[1].imshow(resized_img)
        axes[1].axis('off')
        plot_bboxes(
            ax=axes[1], 
            bboxes=resize_bboxes(row['grounding_dino_bboxes_ram'], resize_ratio),
            scores=row['grounding_dino_scores_ram'],
            labels=row['grounding_dino_labels_ram'],
            )
        axes[1].set_title('Grounding DINO + RAM tags')
       
        axes[2].imshow(resized_img)
        axes[2].axis('off')
        plot_bboxes(
            ax=axes[2], 
            bboxes=resize_bboxes(row['grounding_dino_bboxes_ram_plus'], resize_ratio),
            scores=row['grounding_dino_scores_ram_plus'],
            labels=row['grounding_dino_labels_ram_plus'],
            )
        axes[2].set_title('Grounding DINO + RAM++ tags')

        plt.savefig(img_output_path)
        plt.close()

        data_dict = {
            'Filename': nice_filename,
            'RAM Tags': ram_labels,
            'RAM++ Tags': ram_plus_labels,
            # 'BLIP2 Caption': row['caption']
        }
        json.dump(data_dict, open(img_output_path.replace('.jpg', '.json'), 'w'))

    html_path = os.path.join(output_dir, 'all_imgs.html')
    imgs_path = glob(os.path.join(plots_output_path, "*.jpg"))    
    generate_html(imgs_path, output_file_path=html_path, orig_tags_names='RAM Tags') 
    print(f'Entire pipeline took {time.perf_counter() - cap_start} seconds')
    print(f'Output HTML is under: {html_path}')
    print(f'RAM inference on {len(df)} imgs took avg of {(ram_inference_time) / len(df)} seconds per image')
    print(f'RAM++ inference on {len(df)} imgs took avg of {(ram_plus_inference_time) / len(df)} seconds per image')

    shutil.rmtree(plots_output_path)  # we can delete the plots path since the data is already in the HTML


if __name__ == '__main__':
    device = 'cpu'
    num_imgs = 50
    run_comparison(device, num_imgs)
