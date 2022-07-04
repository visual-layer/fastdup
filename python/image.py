
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import cv2
import numpy as np
import base64
import io


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

def imageformatter(im):
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

def create_triplet_img(img1_path, img2_path, ptype, distance, save_path):
    assert os.path.exists(img1_path) and os.path.exists(img2_path)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    assert h1 > 0 and h2 > 0 and w1 > 0 and w2 > 0

    w = 320
    rimg1 = cv2.resize(img1, (w, int(h1*w/h1)))
    rimg2 = cv2.resize(img2, (w, int(h1*w/h1)))
    assert rimg1.shape[0] > 0 and rimg2.shape[0] > 0

    alpha = 0.5
    cimage = cv2.addWeighted(rimg1,alpha,rimg2,1-alpha,0)
    
    (w, h),nimg1 = draw_text(rimg1, os.path.splitext(os.path.basename(img1_path))[0], font_scale=1, pos=(10, 10))
    (w, h),nimg2 = draw_text(rimg2, os.path.splitext(os.path.basename(img2_path))[0], font_scale=1, pos=(10, 10))
    (w, h),cimage = draw_text(cimage, 'blended image', font_scale=1, pos=(10, 10))

    assert cimage.shape[0] > 0 and cimage.shape[1] > 0

    hcon_img = hconcat_resize_min([nimg1, nimg2, cimage])
    summary_txt = 'type: {0}, distance: {1:.2f}'.format(ptype, distance)
    
    y = int(hcon_img.shape[0]*0.9)
    x = int(hcon_img.shape[1]/3)    
    (w, h),hcon_img = draw_text(hcon_img, summary_txt, font_scale=1, pos=(10, y))

    name1 = os.path.splitext(os.path.basename(img1_path))[0]
    name2 = os.path.splitext(os.path.basename(img2_path))[0]
    pid = '{0}_{1}'.format(name1,name2)
    hcon_img_path = '{0}/{1}.jpg'.format(save_path, pid)
    cv2.imwrite(hcon_img_path, hcon_img)
    assert os.path.exists(hcon_img_path)

    return hcon_img, hcon_img_path

