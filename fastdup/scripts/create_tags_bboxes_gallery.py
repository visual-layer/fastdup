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


IMGS_WIDTH = 1000
IMGS_PATH = '/Users/achiyajerbi/projects/matrix_imgs_batch_0'
OUTPUT_DIR = f'/Users/achiyajerbi/projects/tags_vis_dir/'


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


def generate_html(image_files, output_file_path):
    # Start building the HTML content
    html_content = "<html><body>"

    # Iterate over each image
    for image_file in image_files:
        data_dict = json.load(open(image_file.replace('.jpg', '.json')))

        # Open a container div for each image and table with vertical centering
        html_content += "<div style='display: flex; align-items: center;'>"

        # Add image to the HTML content
        html_content += "<img src='data:image/png;base64," + image_to_base64(image_file) +  "' alt='Image' style='max-height: 2000px; width: 2000px;'>"

        # Add table to the right of the image
        html_content += "<table border='1' style='margin-left: 10px; height: 100px;'>"
        for key, value in data_dict.items():
            html_content += f"<tr style='height: 50%;'><td style='font-size: 30px;'>{key}</td><td style='font-size: 30px;'>{value}</td></tr>"
        html_content += "</table>"

        # Close the container div and add horizontal line
        html_content += "</div> <style> hr { border: none; border-top: 5px solid #000; margin: 0; width: 100%; } </style> <hr>"

    # Close the HTML content
    html_content += "</body></html>"

    # Write the HTML content to a file
    with open(output_file_path, 'w') as html_file:
        html_file.write(html_content)


def run_comparison(device: str, num_imgs: int):
    output_dir = os.path.join(OUTPUT_DIR, f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_{num_imgs}_imgs')
    cap_start = time.perf_counter()
    fd = fastdup.create(input_dir=IMGS_PATH)
    fd.run()
    filenames = get_images_from_path(fd.input_dir)
    np.random.seed(123)
    chosen_filenames = np.random.choice(filenames, num_imgs, replace=False).tolist()
    df = pd.DataFrame({'filename': chosen_filenames})
    df = fd.caption(model_name='blip2', device = device, batch_size=8, subset=chosen_filenames)
    
    df = fd.enrich(task='zero-shot-classification', model='recognize-anything-model', device=device, input_df=df, input_col='filename')
    df = fd.enrich(task='zero-shot-classification', model='tag2text', device=device, input_df=df, input_col='filename')

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
               input_col='tag2text_tags'
     )

    df.rename(columns={
        'grounding_dino_bboxes': 'grounding_dino_bboxes_tag2text', 
        'grounding_dino_scores': 'grounding_dino_scores_tag2text',
        'grounding_dino_labels': 'grounding_dino_labels_tag2text'}, inplace=True)

    cap_end = time.perf_counter()
    print(f'Captioning on {device} took {cap_end - cap_start} seconds')

    plots_output_path = os.path.join(output_dir, 'images')
    os.makedirs(plots_output_path, exist_ok=True)

    for _, row in df.iterrows():
        filename = row['filename']
        nice_filename = '_'.join(filename.split('/')[-2:])
        ram_labels = row['ram_tags']
        tag2text_labels = row['tag2text_tags']
        tag2text_caption = row['tag2text_caption']

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
            bboxes=resize_bboxes(row['grounding_dino_bboxes_tag2text'], resize_ratio),
            scores=row['grounding_dino_scores_tag2text'],
            labels=row['grounding_dino_labels_tag2text'],
            )
        axes[2].set_title('Grounding DINO + Tag2Text tags')
        
        img_output_path = os.path.join(plots_output_path, nice_filename)
        plt.savefig(img_output_path)
        plt.close()

        data_dict = {
            'Filename': nice_filename,
            'RAM Tags': ram_labels,
            'Tag2Text Tags': tag2text_labels,
            'Tag2Text Caption': tag2text_caption,
            'BLIP2 Caption': row['caption']
        }
        json.dump(data_dict, open(img_output_path.replace('.jpg', '.json'), 'w'))

    html_path = os.path.join(output_dir, 'all_imgs.html')
    imgs_path = glob(os.path.join(plots_output_path, "*.jpg"))    
    generate_html(imgs_path, output_file_path=html_path) 
    print(f'Entire pipeline took {time.perf_counter() - cap_start} seconds')
    print(f'Output HTML is under: {html_path}')
    shutil.rmtree(plots_output_path)  # we can delete the plots path since the data is already in the HTML


if __name__ == '__main__':
    device = 'cpu'
    num_imgs = 3
    run_comparison(device, num_imgs)
