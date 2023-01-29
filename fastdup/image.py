
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import cv2
import numpy as np
import base64
import io
from fastdup.definitions import *
from fastdup.sentry import fastdup_capture_exception
import tarfile

def calc_image_path(lazy_load, save_path, filename):
    if lazy_load:
        imgpath = os.path.join(save_path, "images", os.path.basename(filename).replace('/',''))
    else:
        imgpath = os.path.join(save_path, filename.replace('/',''))

    p, ext = os.path.splitext(imgpath)
    if ext is not None and ext != '' and ext.lower() not in ['png','tiff','tif','jpeg','jpg','gif']:
        imgpath += ".jpg"
    return imgpath


def pad_image(image, target_width, target_height):
    # Get the width and height of the original image
    (height, width) = image.shape[:2]

    # Calculate the padding sizes
    pad_left = max((target_width - width) // 2, 0)
    pad_right = max((target_width - width) // 2, 0) + (target_width - width) % 2
    pad_top = max((target_height - height) // 2, 0)
    pad_bottom = max((target_height - height) // 2, 0) + (target_height - height) % 2

    # Create a padded image with the same type as the original image
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    return padded_image


def clean_images(lazy_load, img_paths, section):
    if not lazy_load:
        for i in img_paths:
            try:
                if i is not None:
                    os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)
                fastdup_capture_exception(section, ex)

def download_minio(path, save_path):
    ret = os.system(f"mc cp {path} {save_path}")
    assert ret == 0, f"Failed to download from minio {path} to {save_path}"

def download_s3(path, save_path):
    ret = os.system(f"aws s3 cp {path} {save_path}")
    assert ret == 0, f"Failed to download from s3 {path} to {save_path}"

def truncate_folder_name(path):
    pos = path.find(S3_TEMP_FOLDER)
    if pos != -1:
        return path[pos+len(S3_TEMP_FOLDER)+1:]
    pos = path.find(S3_TEST_TEMP_FOLDER)
    if pos != -1:
        return path[pos+len(S3_TEST_TEMP_FOLDER)+1:]
    return None

def fastdup_imread(img1_path, input_dir, kwargs):
    """
    Read an image from local file, or from a tar file, or from s3/minio path using minio client mc
    Parameters:
    img1_path (str): path to the image
    input_dir (str): optional directory path in case the image is found on a webdataset in another path or found in s3

    Returns:
        img1 (np.array): the image
    """

    is_minio_or_s3 = False
    if input_dir is not None:
        if not input_dir.startswith("s3://") and not input_dir.startswith("minio://"):
            assert os.path.exists(input_dir), "Failed to find input_dir: " + input_dir
        else:
            is_minio_or_s3 = True

    #print('Got fastdup_imread', img1_path, input_dir)

    if os.path.exists(img1_path):
        return cv2.imread(img1_path)
    elif ('/' +S3_TEMP_FOLDER + '/' in img1_path or '/' + S3_TEST_TEMP_FOLDER + '/' in img1_path) and \
         '.tar/' in img1_path:
        assert os.path.exists(input_dir), "Failed to find input dir " + input_dir
        pos = os.path.dirname(img1_path).find(input_dir.replace('/',''))
        tar_file = os.path.dirname(img1_path)[pos+len(input_dir.replace('/','')):]
        tar_file = os.path.join(input_dir, tar_file)
        if kwargs is not None and "reformat_tar_name" in kwargs and callable(kwargs['reformat_tar_name']):
            tar_file = kwargs["reformat_tar_name"](tar_file)

        print('Found tar file', tar_file)
        img_name = os.path.basename(img1_path)
        try:
            with tarfile.open(tar_file, "r") as tar:
                f = tar.extractfile(img_name)
                return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as ex:
            fastdup_capture_exception("fastdup_imread", ex)
            print("Error reading from tar file: ", tar_file, ex)
            return None
    elif is_minio_or_s3:
        if input_dir.startswith("minio://"):
            local_dir_no_temp = truncate_folder_name(os.path.dirname(img1_path))
            minio_prefix = "/".join(input_dir.replace("minio://", "").split('/')[:2])
            #print('minio_prefix', minio_prefix)
            download_minio(minio_prefix + '/' + local_dir_no_temp + '/' + os.path.basename(img1_path), S3_TEMP_FOLDER)
            ret =  cv2.imread(os.path.join(S3_TEMP_FOLDER, os.path.basename(img1_path)))
            assert ret is not None, f"Failed to read image {os.path.join(S3_TEMP_FOLDER, os.path.basename(img1_path))}"
            return ret
        elif input_dir.startswith("s3://"):
            local_dir_no_temp = truncate_folder_name(os.path.dirname(img1_path))
            s3_prefix = 's3://' + "/".join(input_dir.replace("s3://", "").split('/')[:1])
            #print('s3_prefix', s3_prefix)
            download_s3(s3_prefix + '/' + local_dir_no_temp + '/' + os.path.basename(img1_path), S3_TEMP_FOLDER)
            ret = cv2.imread(os.path.join(S3_TEMP_FOLDER, os.path.basename(img1_path)))
            assert ret is not None, f"Failed to read image {os.path.join(S3_TEMP_FOLDER, os.path.basename(img1_path))}"
            return ret

    print('Failed to read image from img_path', img1_path)
    return None


def get_type(str):
    if 'train' in str:
        return 'train'
    if 'test' in str:
        return 'test'
    if 'val' in str:
        return 'val'
    return 'unknown'

def image_base64(im):
    if im is None:
        return "None"
    if isinstance(im, str):
        im = cv2.imread(im)
    is_success, buffer = cv2.imencode(".jpg", np.array(im))
    io_buf = io.BytesIO(buffer)
    return base64.b64encode(io_buf.getvalue()).decode()

def imageformatter(im, max_width=None):
    if im is None:
        return ""
    if max_width is not None:
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}" width="{max_width}">'
    else:
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def get_optimal_font_scale(text, width):

    for scale in reversed(range(0, 60, 1)):
        text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=scale/100, thickness=1)
        new_width = text_size[0]
        if (new_width <= width):
            return scale/100, text_size
    return 1, text_size

def my_resize(img, max_width):
    h, w, c = img.shape
    w1 = 320
    if max_width is not None and w > max_width:
        w1 = max_width
    aspect = h/w
    if h > w1 or w > w1:
        img = cv2.resize(img, (int(w1/aspect), w1))
    return img

def plot_bounding_box(img, get_bounding_box_func, filename):
    bbox_list = []
    if callable(get_bounding_box_func):
        bbox_list = get_bounding_box_func(filename)
    elif isinstance(get_bounding_box_func, dict):
        bbox_list = get_bounding_box_func.get(filename, [])
    for i in bbox_list:
        cur_bbox = i
        cur_bbox = [int(x) for x in cur_bbox]
        img = cv2.rectangle(img, (cur_bbox[0], cur_bbox[1]), (cur_bbox[2], cur_bbox[3]), (0, 255, 0), 3)
    return img


def draw_text(img, text,
          font= cv2.FONT_HERSHEY_TRIPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=0,
          text_color=(0, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    font_scale = 3*(img.shape[1]//6)
    font_scale, text_size  = get_optimal_font_scale(text, font_scale)
    #cv2.putText(img, text, org, font, font_size, color, thickness, cv2.LINE_AA)

    x, y = pos
    #text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    org = (x, int(y + text_h + font_scale - 1))
    cv2.rectangle(img, (int(x - text_w*0.01), int(y - text_h*0.4)), (int(x + text_w*1.02), int(y + text_h*1.4)), text_color_bg, -1)
    cv2.putText(img=img, text=text, org=org, 
                fontFace=font, fontScale=font_scale, color=text_color, thickness=font_thickness)

    return text_size, img

def create_triplet_img(img1_path, img2_path, ptype, distance, save_path, get_bounding_box_func=None, input_dir=None, kwargs=None):

    img1 = fastdup_imread(img1_path, input_dir, kwargs)
    img2 = fastdup_imread(img2_path, input_dir, kwargs)

    assert img1 is not None, "Failed to read image1 " + img1_path
    assert img2 is not None, "Failed to read image2 " + img2_path

    img1 = plot_bounding_box(img1, get_bounding_box_func, img1_path)
    img2 = plot_bounding_box(img2, get_bounding_box_func, img2_path)

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    assert h1 > 0 and h2 > 0 and w1 > 0 and w2 > 0

    w = 320
    rimg1 = cv2.resize(img1, (w, int(h1*w/h1)))
    rimg2 = cv2.resize(img2, (w, int(h1*w/h1)))
    assert rimg1.shape[0] > 0 and rimg2.shape[0] > 0

    alpha = 0.5
    cimage = cv2.addWeighted(rimg1,alpha,rimg2,1-alpha,0)

    hierarchical_run = kwargs is not None and 'hierarchical_run' in kwargs and kwargs['hierarchical_run']
    text1 = os.path.splitext(os.path.basename(img1_path))[0]
    text2 = os.path.splitext(os.path.basename(img2_path))[0]
    if hierarchical_run:
        text1 = text1.split('_')[2]
        text2 = text2.split('_')[2]

    (w, h),nimg1 = draw_text(rimg1, text1, font_scale=1, pos=(10, 10))
    (w, h),nimg2 = draw_text(rimg2, text2, font_scale=1, pos=(10, 10))
    (w, h),cimage = draw_text(cimage, 'blended image', font_scale=1, pos=(10, 10))

    assert cimage.shape[0] > 0 and cimage.shape[1] > 0

    if hierarchical_run:
        hcon_img = hconcat_resize_min([nimg1, nimg2])
    else:
        hcon_img = hconcat_resize_min([nimg1, nimg2, cimage])

    summary_txt = 'type: {0}, distance: {1:.2f}'.format(ptype, distance)
    
    y = int(hcon_img.shape[0]*0.9)
    x = int(hcon_img.shape[1]/3)
    if not hierarchical_run:
        (w, h),hcon_img = draw_text(hcon_img, summary_txt, font_scale=1, pos=(10, y))

    name1 = os.path.splitext(os.path.basename(img1_path))[0]
    name2 = os.path.splitext(os.path.basename(img2_path))[0]
    pid = '{0}_{1}'.format(name1,name2)
    lazy_load = 'lazy_load' in kwargs and kwargs['lazy_load']
    if lazy_load:
        hcon_img_path = f'{save_path}/images/{pid}.jpg'
    else:
        hcon_img_path = f'{save_path}/{pid}.jpg'
    cv2.imwrite(hcon_img_path, hcon_img)
    assert os.path.exists(hcon_img_path)

    return hcon_img, hcon_img_path

