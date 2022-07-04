
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector
from tqdm import tqdm
from PIL import Image
IMAGE_SIZE = 100



def register_embedding(embedding_tensor_name, meta_data_fname, log_dir, sprite_path, with_images=True):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = meta_data_fname
    if (with_images):
        embedding.sprite.image_path = os.path.basename(sprite_path)
        embedding.sprite.single_image_dim.extend([IMAGE_SIZE, IMAGE_SIZE])
    projector.visualize_embeddings(log_dir, config)


def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label))

def generate_sprite_image(img_path, sample_size, log_dir, get_label_func = None, h = 0, w = 0):
    # Generate sprite image
    images_pil = []

    labels = []
    H = IMAGE_SIZE if h == 0 else h
    W = IMAGE_SIZE if w == 0 else w

    for i  in tqdm(img_path[:sample_size], total=min(len(img_path),sample_size)):
        # Save both tf image for prediction and PIL image for sprite
        if isinstance(i, str):
            img_pil = Image.open(i).resize((W,H))
        else:
            img_pil = i.resize((W,H))
        images_pil.append(img_pil)
        # Assuming your output data is directly the label
        if callable(get_label_func):
            label = get_label_func(i)
        else:
            label = "N/A"
        labels.append(label)

    NUM_IMAGES_WIDTH = int(np.ceil(np.sqrt(min(sample_size, len(img_path)))))

    # Create a sprite image
    spriteimage = Image.new(
        mode='RGB',
        size=(W*NUM_IMAGES_WIDTH, H*NUM_IMAGES_WIDTH),
        color=(0,0,0) # fully transparent
    )
    for count, image in enumerate(images_pil):
        h_loc = count // NUM_IMAGES_WIDTH
        w_loc = count % NUM_IMAGES_WIDTH
        spriteimage.paste(image, (w_loc*W, h_loc*H))

    if isinstance(img_path[0], str):
    	SPRITE_PATH= f'{log_dir}/sprite.png'
    	spriteimage.convert('RGB').save(SPRITE_PATH)
    	return SPRITE_PATH, labels
    else:
        return spriteimage.convert('RGB'), labels




def export_to_tensorboard_projector_inner(imglist, features, log_dir, sample_size,
                                          sample_method='random', with_images=True,
                                          get_label_func=None):


    try:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    except:
        print('Failed to create log_dir', log_dir)
        return 1
    df = pd.DataFrame({'filenames':imglist})

    if (sample_method == 'random'):
        sample = df.sample(min(sample_size, len(df))).reset_index()
    else:
        print('sample method', sample_method, 'is not supported')
        return

    EMBEDDINGS_TENSOR_NAME = 'embeddings'
    EMBEDDINGS_FPATH = os.path.join(log_dir, EMBEDDINGS_TENSOR_NAME + '.ckpt')
    STEP = 0
    
    img_path = list(sample['filenames'].values)
    SPRITE_PATH, labels = generate_sprite_image(img_path, sample_size, log_dir, get_label_func)

    META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
    register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, log_dir, SPRITE_PATH, with_images)
    save_labels_tsv(labels, META_DATA_FNAME, log_dir)

    ids = sample['index'].values
    assert len(ids)
    assert features.shape[1] == 576

    tensor_embeddings = tf.Variable(features[ids,:], name=EMBEDDINGS_TENSOR_NAME)
    saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
    saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)

    print('Finish exporting to tensorboard projector, now run')
    if 'JPY_PARENT_PID' in os.environ:
        print('%load_ext tensorboard')
        print(f'%tensorboard --logdir={log_dir}')
    else:
        print(f'tensorboard --logdir={log_dir}')
