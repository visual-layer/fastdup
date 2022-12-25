
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import cv2
IMAGE_SIZE = 100
from fastdup.image import my_resize


def register_embedding(embedding_tensor_name, meta_data_fname, log_dir, sprite_path, with_images=True):
    from tensorboard.plugins import projector
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

def generate_sprite_image(img_path, sample_size, log_dir, get_label_func = None, h = 0, w = 0, alternative_filename = None, alternative_width=None, max_width=None):
    # Generate sprite image
    images_pil = []

    labels = []
    H = IMAGE_SIZE if h == 0 else h
    W = IMAGE_SIZE if w == 0 else w

    if max_width is not None and h != 0 and w != 0:
        if W > max_width:
            scale = 1.0*W/max_width
            H = int(1.0*H/scale)
            W = int(1.0*w/scale)
    else:
        if W > 320:
            scale = 1.0*W/320
            H = int(1.0*H/scale)
            W = int(1.0*W/scale)

    if alternative_width is not None:
        NUM_IMAGES_WIDTH = alternative_width
        if (alternative_width < sample_size):
            sample_size = alternative_width
        height = 1
    else:
        NUM_IMAGES_WIDTH = int(1.4*np.ceil(np.sqrt(min(sample_size, len(img_path)))))
        divs = int(np.ceil(min(sample_size,len(img_path)) / NUM_IMAGES_WIDTH))
        height = min(divs, NUM_IMAGES_WIDTH)

    for i  in img_path[:sample_size]:
        # Save both tf image for prediction and PIL image for sprite
        if isinstance(i, str):
            try:
                assert os.path.exists(i)
                img_pil = cv2.imread(i)
                assert img_pil is not None, f"Failed to read image from {i}"
                img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
                img_pil = cv2.resize(img_pil, (W, H))
            except Exception as ex:
                print("Failed to load image" + i)
                continue
        else:
            img_pil = cv2.resize(i, (W, H))
            img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
        images_pil.append(Image.fromarray(img_pil))

        # Assuming your output data is directly the label
        if callable(get_label_func):
            label = get_label_func(i)
        else:
            label = "N/A"
        labels.append(label)


    # Create a sprite imagei

    spriteimage = Image.new(
        mode='RGB',
        size=(W*NUM_IMAGES_WIDTH, H*height),
        color=(255,255,255)
    )
    for count, image in enumerate(images_pil):
        h_loc = count // NUM_IMAGES_WIDTH
        w_loc = count % NUM_IMAGES_WIDTH
        spriteimage.paste(image, (w_loc*W, h_loc*H))


    if max_width is not None:
        factor = max_width / spriteimage.width
        spriteimage = spriteimage.resize((int(spriteimage.width * factor), int(spriteimage.height * factor)))

    if isinstance(img_path[0], str):
        if alternative_filename is not None:
            SPRITE_PATH =alternative_filename
        else:
            SPRITE_PATH= f'{log_dir}/sprite.png'
        spriteimage.convert('RGB').save(SPRITE_PATH)
        return SPRITE_PATH, labels
    else:
        return np.array(spriteimage.convert('RGB')), labels




def export_to_tensorboard_projector_inner(imglist, features, log_dir, sample_size,
                                          sample_method='random', with_images=True,
                                          get_label_func=None,d = 576):

    
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
    assert features.shape[1] == d, "Wrong share for the feature vectors exected {} got {}".format(d, features.shape[1])

    import tensorflow as tf
    tensor_embeddings = tf.Variable(features[ids,:], name=EMBEDDINGS_TENSOR_NAME)
    saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
    saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)

    print('Finish exporting to tensorboard projector, now run')
    if 'JPY_PARENT_PID' in os.environ:
        print('%load_ext tensorboard')
        print(f'%tensorboard --logdir={log_dir}')
    else:
        print(f'tensorboard --logdir={log_dir}')
