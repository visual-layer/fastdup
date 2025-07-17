
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import pandas as pd
import cv2
import time
import numpy as np
import traceback
import shutil
import pathlib
from fastdup.image import plot_bounding_box, my_resize, get_type, imageformatter, create_triplet_img, fastdup_imread, calc_image_path, clean_images, pad_image, enhance_image, fastdup_imwrite
from fastdup.definitions import *
import re
from multiprocessing import Pool
from fastdup.sentry import *
from fastdup.utilities import load_filenames, merge_with_filenames, get_bounding_box_func_helper, load_stats, load_labels, sample_from_components, calc_save_dir, convert_v1_to_v02

try:
    from tqdm.auto import tqdm
except:
    tqdm = (lambda x, total=None, desc=None: x)


# def get_label(filename, get_label_func):
#     ret = filename
#     try:
#         if isinstance(get_label_func, dict):
#             ret += "<br>" + "Label: " + get_label_func.get(filename, MISSING_LABEL)
#         elif callable(get_label_func):
#             ret += "<br>" + "Label: " + get_label_func(filename)
#         else:
#             assert False, f"Failed to understand get_label_func type {type(get_label_func)}"
#     except Exception as ex:
#         ret += "<br>Failed to get label for " + filename + " with error " + ex
#     return ret

def print_success_msg(report_name, out_file, lazy_load):
    print(f"Stored {report_name} visual view in ", out_file)
    if lazy_load:
        print("Note: when using lazy_load=True, the images are relative to the location of the html file. When sharing the report please make"
              " sure to include also subolders images & assets.")



def shorten_image(x, save_path):
    #print('IMAGES WHERE', x, save_path)
    if save_path is None:
        return str(x)
    else:
        save_path = str(save_path)
        if save_path.endswith('/'):
            save_path = save_path[:-1]
        x = str(x)
        if x.startswith(save_path + '/'):
            x = x.replace(save_path + '/', '')
        return x

def format_image_html_string(img_paths, lazy_load, max_width, save_path=None):
    if not lazy_load:
        return [imageformatter(x, max_width) for x in img_paths]
    else:
        return ["<img src=\"" + shorten_image(x, save_path) + "\" loading=\"lazy\">" for x in img_paths]


def swap_dataframe(subdf, cols):
    cols_no_images = [x for x in cols if (x.lower() != 'image' and not x.startswith('info') and  x.lower()!= 'similar')]
    new_rows = []
    for i,row in subdf[cols_no_images].iterrows():
        dfrow = pd.DataFrame(row)
        new_rows.append(dfrow)
    return new_rows




def find_label(get_label_func, df, in_col, out_col, vqa_prompt: str = None, kwargs=None):


    if (get_label_func is not None):
        if isinstance(get_label_func, str):
            if os.path.exists(get_label_func):
                df_labels = load_labels(get_label_func, kwargs)
                assert len(df_labels) == len(df), f"Error: wrong length of labels file {get_label_func} expected {len(df)} got {len(df_labels)}"
                df[out_col] = df_labels['label']
            elif get_label_func in df.columns:
                df[out_col] = df['label']
            elif get_label_func in CAPTION_MODEL_NAMES:
                from fastdup.captions import generate_labels
                df[out_col] = generate_labels(df[in_col], get_label_func, device='cpu')
            elif get_label_func == VQA_MODEL1_NAME:
                from fastdup.captions import generate_vqa_labels
                df[out_col] = generate_vqa_labels(df[in_col], vqa_prompt, kwargs)
            elif get_label_func == AGE_LABEL1_NAME:
                from fastdup.captions import generate_age_labels
                df[out_col] = generate_age_labels(df[in_col], kwargs)
            else:
                assert False, f"Found str label {get_label_func} but it is neither a file nor a column name in the dataframe {df.columns}"
        elif isinstance(get_label_func, dict):
            df[out_col] = df[in_col].apply(lambda x: get_label_func.get(x, MISSING_LABEL))
        elif callable(get_label_func):
            assert len(df), "Empty dataframe"
            assert in_col in df.columns, f"Missing column {in_col}"
            df[out_col] = df[in_col].apply(lambda x: get_label_func(x))
        else:
            assert False, f"Failed to understand get_label_func type {type(get_label_func)}"

    if kwargs is not None and 'debug_labels' in kwargs:
        print(df.head())
    return df

def split_str(x):
    return re.split('[?.,:;&^%$#@!()]', x)


def slice_df(df, slice, colname, kwargs=None):
    assert len(df), "Df has no rows"

    split_sentence_to_label_list = kwargs is not None and 'split_sentence_to_label_list' in kwargs and kwargs['split_sentence_to_label_list']
    debug_labels = kwargs is not None and 'debug_labels' in kwargs and kwargs['debug_labels']
    grouped = kwargs is not None and 'grouped' in kwargs and kwargs['grouped']

    if slice is not None:
        if isinstance(slice, str):
            # cover the case labels are string or lists of strings
            if split_sentence_to_label_list:
                labels = df[colname].astype(str).apply(lambda x: split_str(x.lower())).values
                if debug_labels:
                    print('Label with split sentence', labels[:10])
            else:
                labels = df[colname].astype(str).values
                if debug_labels:
                    print('label without split sentence', labels[:10])

            is_list = isinstance(labels[0], list)
            if grouped:
                df = df[df[colname].apply(lambda x: slice in x)]
                assert len(df), f"Failed to find any labels with value={slice}"
            elif is_list:
                labels = [item for sublist in labels for item in sublist]
                if debug_labels:
                    print('labels after merging sublists', labels[:10])
                df = df[df[colname].apply(lambda x: slice in [y.lower() for y in x])]
            else:
                df2 = df[df[colname] == slice]
                if len(df2) == 0:
                    df2 = df[df[colname].apply(lambda x: slice in str(x))]
                df = df2

        elif isinstance(slice, list):
            if isinstance(df[colname].values[0], list):
                df = df[df[colname].apply(lambda x: len(set(x)&set(slice)) > 0)]
            else:
                df = df[df[colname].isin(slice)]
            assert len(df), f"Failed to find any labels with {slice}"
        else:
            assert False, "slice must be a string or a list of strings"

    return df

def slice_two_labels(df, slice):
    if isinstance(slice, str):
        if slice == "diff":
            df = df[df['label'] != df['label2']]
        elif slice == "same":
            df = df[df['label'] == df['label2']]

    return df

def lookup_filename(filename, work_dir):
    assert isinstance(filename, str), f"Wrong for type {filename} {type(filename)}"

    if os.path.exists(filename):
        return filename
    if filename.startswith(S3_TEMP_FOLDER + get_sep())  or filename.startswith(S3_TEST_TEMP_FOLDER + get_sep()):
        assert work_dir is not None, f"Failed to find work_dir on remote_fs: filename was {filename}"
        filename = os.path.join(work_dir, filename)
    return filename


def extract_filenames(row, work_dir, save_path, kwargs):
    debug_hierarchical = 'debug_hierarchical' in kwargs and kwargs['debug_hierarchical']
    hierarchical_run =  'hierarchical_run' in kwargs and kwargs['hierarchical_run']
    draw_orig_image = 'draw_orig_image' in kwargs and kwargs['draw_orig_image']

    if hierarchical_run and not draw_orig_image:
        assert 'cluster_from' in row, "Failed to find cluster_from in row " + str(row)
        impath1 = save_path + f"/images/component_{row['counter_from']}_{row['cluster_from']}.jpg"
        impath2 = save_path + f"/images/component_{row['counter_to']}_{row['cluster_to']}.jpg"
        if debug_hierarchical:
            print('was in extract_filenames', impath1, impath2)
    else:
        assert not pd.isnull(row['to']) and not pd.isnull(row['from']), f"Found nan in row {row}"
        impath1 = lookup_filename(row['from'], work_dir)
        impath2 = lookup_filename(row['to'], work_dir)

    dist = row['distance']
    if ~impath1.startswith(S3_TEMP_FOLDER) and ~impath1.startswith(S3_TEST_TEMP_FOLDER):
        os.path.exists(impath1), "Failed to find image file " + impath1
    if ~impath2.startswith(S3_TEMP_FOLDER) and ~impath2.startswith(S3_TEST_TEMP_FOLDER):
        os.path.exists(impath2), "Failed to find image file " + impath2

    if 'label' in row:
        type1 = row['label']
    else:
        type1 = get_type(impath1)
    if 'label2' in row:
        type2 = row['label2']
    else:
        type2 = get_type(impath2)

    if type1 == "unknown" and type2 == "unknown":
        ptype = ""
    else:
        ptype = '{0}_{1}'.format(type1, type2)
    return impath1, impath2, dist, ptype




def prepare_hierarchy(df, work_dir, save_path, debug_hierarchical, kwargs):
    # from,to,cluster_from,cluster_do,distance
    # /mnt/data/sku110k/val_245.jpg,/mnt/data/sku110k/train_953.jpg,4,0,0.876736
    # /mnt/data/sku110k/train_6339.jpg,/mnt/data/sku110k/train_953.jpg,19,0,0.891410
    # /mnt/data/sku110k/train_6339.jpg,/mnt/data/sku110k/val_245.jpg,19,4,0.947931
    assert(work_dir is not None), "work_dir must be specified when running hierarchical_run"
    assert os.path.exists(os.path.join(save_path, 'images')), "Failed to find images folder in save_path, run fastdup.create_components_gallery(..., lazy_load=True) first"
    draw_orig_image = 'draw_orig_image' in kwargs and kwargs['draw_orig_image']

    comp_images = os.listdir(os.path.join(save_path, 'images'))
    comp_images = [x for x in comp_images if 'component_' in x]
    comp_map = {}
    assert len(comp_images), "Failed to find any component images in save_path, run fastdup.create_components_gallery(..., lazy_load=True) first"

    for i in comp_images:
        counter = int(os.path.basename(i).split('_')[1])
        assert (counter >= 0), "Failed to parse component counter from index " + i
        comp_id = int(os.path.basename(i).split('_')[2].replace('.jpg', ''))
        assert( comp_id >= 0), "Failed to parse component id from index " + i
        comp_map[comp_id] = counter
    if (debug_hierarchical):
        print('comp_map', comp_map)

    assert len(comp_map), "Failed to find any component images in save_path, run fastdup.create_components_gallery(..., lazy_load=True) first"
    comp_map_set = set(comp_map.keys())
    assert 'cluster_from' in df.columns and 'cluster_to' in df.columns, "Failed to find cluster_from and cluster_to columns in similarity file, run fastdup.create_components_gallery(..., lazy_load=True) first"

    df['counter_from'] = df['cluster_from'].apply(lambda x: comp_map.get(x,-1))
    df['counter_to'] = df['cluster_to'].apply(lambda x: comp_map.get(x,-1))
    if (debug_hierarchical):
        print('df', df.head())
        print('len df sim orig', len(df))
        print('Going to filter by set', comp_map_set)

    if draw_orig_image:
        df = df[df['cluster_from'].isin(comp_map_set) & df['cluster_to'].isin(comp_map_set)]
    assert len(df), "Failed to find any rows with custers in top component set < " + str(len(comp_images)) + " Try to run create_components_gallery with larger number of components, using num_images=XX"
    if debug_hierarchical:
        print('df after removed set', df.head())
        print('len df sim after removed set', len(df))
    df = df.sort_values('distance', ascending=False)
    if (debug_hierarchical):
        print('sorted df', df.head())
    return df

def do_create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True,
                              lazy_load=False, get_label_func=None, slice=None, max_width=None,
                                 get_bounding_box_func=None, get_reformat_filename_func=None,
                                 get_extra_col_func=None, input_dir=None, work_dir=None, threshold=None, **kwargs):
    '''

    Function to create and display a gallery of images computed by the similarity metrics

    Parameters:
        similarity_file (str): csv file with the computed similarities by the fastdup tool, alternatively it can be a pandas dataframe with the computed similarities.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the gallery.

        input_dir (str): Optional parameter to allow reading images from a different path, or from webdataset tar files which are found on a different path

        work_dir (str): Optional parameter to specify fastdup work_dir when the similarity file is a pd.DataFrame

        threshold (float): Optional parameter to allow filtering out images with a similarity score above a certain threshold (allowed values 0 -> 1)

        save_artifacts (boolean): Optional parameter to allow saving the intermediate artifacts (raw images, csv with results) to the output folder

    Returns:
        ret (int): 0 if success, 1 if failed

    '''


    img_paths = []
    #v1 = 'id_to_filename_func' in kwargs
    kwargs['lazy_load'] = lazy_load
    hierarchical_run = kwargs is not None and 'hierarchical_run' in kwargs and kwargs['hierarchical_run']
    draw_orig_image = 'draw_orig_image' in kwargs and kwargs['draw_orig_image']
    blur_threshold = None
    if 'blur_threshold' in kwargs:
        blur_threshold = kwargs['blur_threshold']
    save_dir = calc_save_dir(save_path)

    df = similarity_file
    df = convert_v1_to_v02(df)

    if df['from'].dtype in [int, np.int64]:
        assert df['to'].dtype in [int, np.int64], "Wrong types, expect both str or both int"
        filenames = load_filenames(work_dir, kwargs)
        filenames = filenames[["index","filename"]]
        df = merge_with_filenames(df, filenames)

    get_bounding_box_func = get_bounding_box_func_helper(get_bounding_box_func)

    if blur_threshold is not None:
        stats = load_stats(work_dir, None, kwargs)
        df['blur'] = stats['blur']
        orig_len = len(df)
        df = df[df['blur'] > blur_threshold]
        print(f"Filtered {orig_len-len(df)} blurry images, remained with {len(df)}")
        assert len(df), f"Failed to find images above blur threshold {blur_threshold}"

    if threshold is not None:
        df = df[df['distance'] >= threshold]
        assert len(df), f"Failed to find any duplicates images with similarity score >= {threshold}"

    if 'external_df' not in kwargs: # external_df contains sorting by the user
        df = df.sort_values('distance', ascending=not descending)
        if 'crop_filename_from' not in df.columns:
            df = df.drop_duplicates(subset=['from', 'to'])

    if get_label_func is not None and slice is not None:
        df = find_label(get_label_func, df, 'from', 'label', kwargs)
        df = slice_df(df, slice, 'label', kwargs)
        if slice in ["diff","same"]:
            df = find_label(get_label_func, df, 'to', 'label2', kwargs)
            df = slice_two_labels(df, slice)

    debug_hierarchical= kwargs is not None and 'debug_hierarchical' in kwargs and kwargs['debug_hierarchical']
    if 'hierarchical_run' in kwargs and kwargs['hierarchical_run']:
        df = prepare_hierarchy(df, work_dir, save_dir, debug_hierarchical, kwargs)


    sets = {}

    if 'is_video' in kwargs:
        filenames = load_filenames(work_dir, kwargs)
        filenames['dirname'] = filenames['filename'].apply(os.path.dirname)
        frames = filenames.groupby(['dirname']).size().reset_index(name='num_frames')
        df = similarity_file.merge(frames, how='left', left_on=['subfolder1'], right_on=['dirname'])

    subdf = df.head(num_images)
    # lazy eval of labels as this may be slow
    if get_label_func is not None and slice is None and 'label' not in subdf.columns and 'label2' not in subdf.columns:
        subdf = find_label(get_label_func, subdf, 'from', 'label', kwargs)
        subdf = find_label(get_label_func, subdf, 'to', 'label2', kwargs)

    subdf = subdf.reset_index()

    if 'is_video' in kwargs:
        subdf['ratio'] = subdf['counts'].astype(float) / subdf['num_frames'].astype(float)
        subdf['ratio'] = subdf['ratio'].apply(lambda x: round(x,3))

    indexes = []
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf)), desc="Generating gallery"):
        if 'crop_filename_from' in row:
            im1, im2 = str(row['crop_filename_from']), str(row['crop_filename_to'])
        else:
            im1, im2 = str(row['from']), str(row['to'])

        if im1 + '_' + im2 in sets:
            continue
        try:
            img, imgpath = create_triplet_img(i, row, work_dir, save_dir, extract_filenames, get_bounding_box_func,
                                              input_dir, kwargs)
            sets[im1 +'_' + im2] = True
            sets[im2 +'_' + im1] = True
            indexes.append(i)
            img_paths.append(imgpath)

        except Exception as ex:
            fastdup_capture_exception("triplet image", ex)
            print("Failed to generate viz for images", im1, im2, ex)
            #img_paths.append(None)

    subdf = subdf.iloc[indexes]
    import fastdup.html_writer

    html_img = format_image_html_string(img_paths, lazy_load, None, save_dir)
    subdf.insert(0, 'Image', html_img)

    if str(save_path).endswith(".html"):
        out_file = save_path
    else:
        out_file = os.path.join(save_path, FILENAME_DUPLICATES_HTML) if not hierarchical_run else os.path.join(save_path, 'similarity_hierarchical.html')


    subdf = subdf.rename(columns={'from':'From', 'to':'To'}, inplace=False)
    subdf = subdf.rename(columns={'distance':'Distance'}, inplace=False)
    fields = ['Image', 'Distance', 'From', 'To']
    if get_label_func is not None or ('label' in subdf.columns and 'label2' in subdf.columns):
        subdf = subdf.rename(columns={'label':'From_Label','label2':'To_Label'}, inplace=False)
        fields.extend(['From_Label', 'To_Label'])

    # for video, show duplicate counts between frames
    if 'ratio' in subdf.columns:
        fields = ['ratio'] + fields


    if callable(get_extra_col_func):
        subdf['extra'] = subdf['From'].apply(lambda x: get_extra_col_func(x))
        subdf['extra2'] = subdf['To'].apply(lambda x: get_extra_col_func(x))
        fields.append('extra')
        fields.append('extra2')

    if get_reformat_filename_func is not None and callable(get_reformat_filename_func):
        subdf['From'] = subdf['From'].apply(lambda x: get_reformat_filename_func(x))
        subdf['To'] = subdf['To'].apply(lambda x: get_reformat_filename_func(x))

    title = 'Duplicates Report'
    subtitle = "Showing duplicate"
    if 'is_video' in kwargs:
        title = 'Video Duplicates Report'
        subtitle += " video pairs"
    else:
        subtitle += " image pairs"

    if hierarchical_run:
        title = 'Hierarchical Duplicates Report'
        subtitle = "Showing hierarchical images pairs"

    if slice is not None:
        if slice == "diff":
            title += ", of different classes"
        else:
            title += ", for label " + str(slice)

    assert len(subdf), "Error: failed to find any duplicates, try to run() with lower threshold"

    if 'get_display_filename_func' in kwargs:
        subdf['From'] = subdf['From'].apply(kwargs['get_display_filename_func'])
        subdf['To'] = subdf['To'].apply(kwargs['get_display_filename_func'])
    #elif 'id_to_filename_func' in kwargs:
    #    subdf['From'] = subdf['From'].apply(kwargs['id_to_filename_func'])
    #    subdf['To'] = subdf['To'].apply(kwargs['id_to_filename_func'])
    subdf['info'] = swap_dataframe(subdf, fields)
    if max_width is None:
        max_width = 600
    fastdup.html_writer.write_to_html_file(subdf[['Image','info']], title, out_file, None, None, max_width,
                                           jupyter_html=kwargs.get('jupyter_html', False))
    assert os.path.exists(out_file), "Failed to generate out file " + out_file
    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']
    if save_artifacts:
        save_artifacts_file = os.path.join(save_dir, 'similarity_artifacts.csv')
        subdf[list(set(fields)-set(['Image']))].to_csv(save_artifacts_file, index=False)
        print("Stored similarity artifacts in ", save_artifacts_file)

    print_success_msg('similarity', out_file, lazy_load)
    clean_images(lazy_load or save_artifacts, img_paths, "create_duplicates_gallery")
    return 0



def load_one_image_for_outliers(args):
    row, work_dir, input_dir, get_bounding_box_func, max_width, save_path, kwargs = args
    impath1, impath2, dist, ptype = extract_filenames(row, work_dir, save_path, kwargs)

    try:
        img = fastdup_imread(impath1, input_dir, kwargs)
        assert img is not None, f"Failed to read image from {impath1} {input_dir}"
        #find the index to retreive the bounding box.
        if 'crop_filename_outlier' in row:
          outlier_id = row['crop_filename_outlier']
        else:
          outlier_id = row['from']
        img = plot_bounding_box(img, get_bounding_box_func, outlier_id)
        assert img is not None, f"Failed to plot bb from {impath1} {input_dir}"
        img = my_resize(img, max_width=max_width)
        assert img is not None, f"Failed to resize image from {impath1} {input_dir}"
        if 'enhance_image' in kwargs and kwargs['enhance_image']:
            img = enhance_image(img)

        #consider saving second image as well!
        #make sure image file is unique, so add also folder name into the imagefile
        lazy_load = 'lazy_load' in kwargs and kwargs['lazy_load']
        imgpath = calc_image_path(lazy_load, save_path, impath1, "")
        imgpath = fastdup_imwrite(imgpath, img)

    except Exception as ex:
        fastdup_capture_exception(f"load_one_image_for_outliers", ex)
        imgpath = None

    return imgpath




def do_create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None, descending=True, max_width=None, get_bounding_box_func=None, get_reformat_filename_func=None,
                               get_extra_col_func=None, input_dir= None, work_dir = None, **kwargs):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        outliers_file (str): csv file with the computed outliers by the fastdup tool. Altenriously, this can be a pandas dataframe with the computed outliers.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        how (str): Optional outlier selection method. one = take the image that is far away from any one image (but may have other images close to it).
                                                      all = take the image that is far away from all other images. Default is one.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label.

        descending (boolean): Optional parameter to set the order of the components. Default is True namely list components from largest to smallest.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the gallery.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        work_dir (str): Optional parameter to specify the working directory in case of giving an hourlier file which is a dataframe.

    Returns:
        ret (int): 0 if successful, 1 otherwise
    '''

    nrows = None
    if 'nrows' in kwargs:
        nrows = kwargs['nrows']

    hierarchical_run = 'hierarchical_run' in kwargs and kwargs['hierarchical_run']
    debug_hierarchical = 'debug_hierarchical' in kwargs and kwargs['debug_hierarchical']
    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']
    save_dir = calc_save_dir(save_path)


    img_paths = []
    kwargs['lazy_load'] = lazy_load

    df = outliers_file
    df = convert_v1_to_v02(df)
    if df['from'].dtype in [int, np.int64]:
        filenames = load_filenames(work_dir, kwargs)
        df = merge_with_filenames(df, filenames)

    get_bounding_box_func = get_bounding_box_func_helper(get_bounding_box_func)

    if (how == 'all'):
        if isinstance(outliers_file, pd.DataFrame):
            assert work_dir is not None and isinstance(work_dir, str) and os.path.isdir(work_dir), "Failed to find fastdup work_dir folder, please rerun with work_dir pointing to fastdup run"
        dups_file = os.path.join(work_dir, FILENAME_SIMILARITY)
        assert os.path.exists(dups_file), f'Failed to find input file {dups_file} which is needed for computing how=all similarities, . Please run using fastdup.run(..) to generate this file.'

        dups = pd.read_csv(dups_file, nrows=nrows)
        assert len(dups), "Error: Failed to locate similarity file file " + dups_file
        dups = dups[dups['distance'] >= dups['distance'].astype(float).mean()]
        assert len(dups), f"Did not find any images with similarity more than the mean {dups['distance'].mean()}"
        if dups['from'].dtype in [int, np.int64]:
            filenames = load_filenames(work_dir, kwargs)
            dups = merge_with_filenames(dups, filenames)

        joined = df.merge(dups, on='from', how='left')
        joined = joined[pd.isnull(joined['distance_y'])]

        assert len(joined), 'Failed to find outlier images that are not included in the duplicates similarity files, run with how="one".'

        df = joined.rename(columns={"distance_x": "distance", "to_x": "to"})

    comp_images = []
    comp_map = {}
    if hierarchical_run:
        if debug_hierarchical:
            pd.set_option('display.max_rows', 50)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
        df, comp_images, comp_map = prepare_hierarchy(df, work_dir, save_dir, debug_hierarchical, kwargs)

    if get_label_func is not None and slice is not None:
        df = find_label(get_label_func, df, 'from', 'label', kwargs)
        df = slice_df(df, slice, 'label')
        assert df is not None, f"Failed to find any labels with {slice} value"

    df = df.drop_duplicates(subset='from')
    if 'external_df' not in kwargs:
        df = df.sort_values(by='distance', ascending=not descending)
    df = df.head(num_images)

    if get_label_func is not None and slice is None and 'label' not in df.columns:
        df = find_label(get_label_func, df, 'from', 'label', kwargs)

    all_args = []
    for i, row in tqdm(df.iterrows(), total=min(num_images, len(df)), desc="Generating gallery"):
        args = row, work_dir, input_dir, get_bounding_box_func, max_width, save_dir, kwargs
        all_args.append(args)

    # trying to deal with the following runtime error:
    #An attempt has been made to start a new process before the
    #current process has finished its bootstrapping phase.
    parallel_run = 'parallel_run' in kwargs and kwargs['parallel_run']
    if parallel_run:
        try:
            with Pool() as pool:
                img_paths = pool.map(load_one_image_for_outliers, all_args)
        except RuntimeError as e:
            fastdup_capture_exception("create_outliers_gallery_pool", e)
    else:
        for i in all_args:
            img_paths.append(load_one_image_for_outliers(i))

    import fastdup.html_writer
    img_html = format_image_html_string(img_paths, lazy_load, max_width, save_dir)
    df.insert(0, 'Image', img_html)

    df = df.rename(columns={'distance':'Distance','from':'Path'}, inplace=False)

    out_file = os.path.join(save_path, 'outliers.html') if not str(save_path).endswith(".html") else save_path
    title = 'Outliers Report'
    subtitle = "Showing image outliers, one per row"
    if slice is not None:
        title += ", " + str(slice)

    cols = ['Image','Distance','Path']
    if callable(get_extra_col_func):
        df['extra'] = df['Path'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')

    # if get_reformat_filename_func is not None and callable(get_reformat_filename_func):
    #     df['Path'] = df['Path'].apply(lambda x: get_reformat_filename_func(x))

    if 'label' in df.columns:
        cols.append('label')

    reformat_disp_path = kwargs.get('get_display_filename_func', lambda x: x)
    df['Path'] = df['Path'].apply(lambda x: reformat_disp_path(x))
    df['info'] = swap_dataframe(df, cols)
    fastdup.html_writer.write_to_html_file(df[['Image','info']], title, out_file, subtitle=subtitle, jupyter_html=kwargs.get('jupyter_html', False))

    if save_artifacts:
        df[list(set(cols)-set(['Image']))].to_csv(f'{save_path}/outliers_report.csv')
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    if hierarchical_run:
        print("Stored outliers hierarchical view in ", os.path.join(out_file))
    else:
        print_success_msg("outliers", out_file, lazy_load)
    clean_images(lazy_load or save_artifacts, img_paths, "create_outliers_gallery")
    return 0


def load_one_image(args):
    try:
        f, fid, input_dir, get_bounding_box_func, kwargs = args
        assert not pd.isnull(f), f"Got None image name {fid} {input_dir} {kwargs}"
        img = fastdup_imread(f, input_dir, kwargs)
        assert img is not None, f"Failed to read image {f} {input_dir} {kwargs}"
        img = plot_bounding_box(img, get_bounding_box_func, fid)
        assert img is not None, f"Failed to read image {f} {input_dir} {kwargs}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None, f"Failed to read image {f} {input_dir} {kwargs}"
        if 'enhance_image' in kwargs and kwargs['enhance_image']:
            img = enhance_image(img)
            assert img is not None, f"Failed to enchance image {f} {input_dir} {kwargs}"
        return img, img.shape[1], img.shape[0]
    except Exception as ex:
        print("Warning: Failed to load image ", f, "skipping image due to error", ex)
        fastdup_capture_exception("load_one_image", ex)
        return None,None,None




def visualize_top_components(work_dir, save_path, num_components, get_label_func=None, group_by='visual', slice=None,
                             get_bounding_box_func=None, max_width=None, threshold=None, metric=None, descending=True,
                             max_items = None, min_items=None, keyword=None, comp_type="component",
                             input_dir=None, kwargs=None):
    '''
    Visualize the top connected components

    Args:
        work_dir (str): directory with the output of fastdup run or a dataframe with the content of connected_components.csv

        save_path (str): directory to save the output to

        num_components (int): number of top components to plot

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): 'visual' or 'label'

        slice (str): slice the dataframe based on the label or a list of labels

        get_bounding_box_func (callable): option function to get bounding box for each image given image filename

        max_width (int): optional maximum width of the image

        threshold (float): optional threshold to filter out components with similarity less than this value

        metric (str): optional metric to use for sorting the components

        descending (bool): optional sort in descending order

        max_items (int): optional max number of items to include in the component, namely show only components with less than max_items items

        min_items (int): optional min number of items to include in the component, namely show only components with at least this many items

        keyword (str): optional keyword to filter out components with labels that do not contain this keyword

        return_stats (bool): optional return the stats of the components namely statistics about component sizes

        component_type (str): comp type, should be one of component|cluster

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'


    Returns:
        ret (pd.DataFrame): with the top components
        img_list (list): of the top components images
        stats (pd.DataFrame): optional return value of the stats of the components namely statistics about component sizes
    '''

    try:
        from fastdup.tensorboard_projector import generate_sprite_image
        import traceback
    except Exception as ex:
        print(ex)
        fastdup_capture_exception("visualize_top_components", ex)
        return None, None

    assert num_components > 0, "Number of components should be larger than zero"
    MAX_IMAGES_IN_GRID = 54
    #v1 = 'id_to_filename_func' in kwargs

    if isinstance(work_dir, pd.DataFrame):
        if 'distance' in work_dir.columns and 'component_id' in work_dir.columns \
                and 'files' in work_dir.columns and 'len' in work_dir.columns and 'files_ids' in work_dir.columns and len(
            work_dir):
            if slice is not None:
                assert 'label' in work_dir.columns, "Failed to find 'label' in dataframe, when using slice string need to provide label column"
            kwargs['grouped'] = True
            top_components = slice_df(work_dir, slice, 'label', kwargs)

        else:
            assert False, f"Got dataframe with the columns: {work_dir.columns} while expecting to get the columns: \
               ['component_id', 'distance', 'files', 'files_ids', 'len'] and optionally label and or crop_filename. \
               component_id is integer index of cluster, files, files_ids, label, crop_filename are lists of files in the component. files include the filenames, files_ids are integer unique indexe for the files\
               label is an optional list of labels per ima, crop_filename are optional list of crops. "
    else:
        top_components = do_find_top_components(work_dir=work_dir, get_label_func=get_label_func, group_by=group_by,
                                                slice=slice, threshold=threshold, metric=metric, descending=descending,
                                                max_items=max_items,  min_items=min_items, keyword=keyword, save_path=save_path,
                                                input_dir=input_dir,
                                                comp_type=comp_type, kwargs=kwargs)


    assert top_components is not None, f"Failed to find components with more than {min_items} images. Try to run fastdup.run() with turi_param='ccthreshold=0.9' namely to lower the threshold for grouping components"
    top_components = top_components.head(num_components)
    if 'debug_cc' in kwargs:
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(top_components.head())
    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']
    keep_aspect_ratio = 'keep_aspect_ratio' in kwargs and kwargs['keep_aspect_ratio']

    assert top_components is not None and len(top_components), f"Failed to find components with more than {min_items} images. Try to run fastdup.run() with turi_param='ccthreshold=0.9' namely to lower the threshold for grouping components"
    comp_col = "component_id" if comp_type == "component" else "cluster"

    save_dir = calc_save_dir(save_path)
    save_dir = os.path.join(save_dir, "images")
    lazy_load = kwargs.get('lazy_load', False)
    subfolder = "" if not lazy_load else "images/"
    os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)

    # iterate over the top components
    img_paths = []
    counter = 0
    #filname_transform_func = kwargs.get('id_to_filename_func', lambda x: x)
    all_labels = []
    for i,row in tqdm(top_components.iterrows(), total = len(top_components), desc="Generating gallery"):
        try:
            # find the component id
            component_id = row[comp_col]
            # find all the image filenames linked to this `id`
            if save_artifacts:
                if not os.path.exists(os.path.join(save_path, 'images')):
                    os.mkdir(os.path.join(save_path, 'images'))
                pd.DataFrame({'filename':row['files']}).to_csv(os.path.join(save_path, 'images', f'component_{counter}_files.csv'))
            files_ids = sample_from_components(row, metric, kwargs, MAX_IMAGES_IN_GRID)
            if (len(files_ids) == 0):
                print(f"Failed to find any files for component_id {component_id}");
                break

            files, files_ids = zip(*files_ids)
            # if v1 and isinstance(files[0], str):
            #     files = [os.path.join(input_dir, x) for x in files]

            if save_artifacts:
                if not os.path.exists(os.path.join(save_path , "images", f"raw_images_{counter}")):
                    os.mkdir(os.path.join(save_path, "images", f"raw_images_{counter}"))
                for f in files:
                    shutil.copy(f, os.path.join(save_path, "images", f"raw_images_{counter}"))

            tmp_images = []
            w,h = [], []
            val_array = []
            for f, fid in zip(files, files_ids):
                assert not pd.isnull(f), f"Found None image name on {fid} {input_dir} {row}"
                #if v1:
                #    assert isinstance(fid, (int,np.uint32)), f"found a wrong file_id {fid} {type(fid)}"
                val_array.append([f, fid, input_dir, get_bounding_box_func, kwargs])

            # trying to deal with the following runtime error:
            #An attempt has been made to start a new process before the
            #current process has finished its bootstrapping phase.

            parallel_run = 'parallel_run' in kwargs and kwargs['parallel_run']
            if parallel_run:
                try:
                    with Pool() as pool:
                        result = pool.map(load_one_image, val_array)
                except RuntimeError as e:
                    fastdup_capture_exception("visualize_top_components", e)
            else:
                result = []
                for i in val_array:
                    img = load_one_image(i)
                    if img[0] is not None:
                        result.append(img)

            for t,x in enumerate(result):
                if x[0] is not None:
                    if save_artifacts:
                        if not os.path.exists(f'{save_path}/images/comp_{counter}/'):
                            os.mkdir(f'{save_path}/images/comp_{counter}')
                        cv2.imwrite(f'{save_path}/images/comp_{counter}/{os.path.basename(files[t])}', x[0])
                    tmp_images.append(x[0])
                    w.append(x[1])
                    h.append(x[2])
                

            assert len(tmp_images),"Failed to read all images"

            avg_h = int(np.mean(h))
            avg_w = int(np.mean(w))
            max_h = int(np.max(h))
            max_w = int(np.max(w))
            if keep_aspect_ratio:
                avg_h = max_h
                avg_w = max_w

            images = []
            for f in tmp_images:
                assert f is not None, "Failed to read image"
                if not keep_aspect_ratio:
                    f = cv2.resize(f, (avg_w,avg_h))
                else:
                    f = pad_image(f, avg_w, avg_h)
                images.append(f)

            labels = row['label'] if 'label' in row else None

            if len(images) <= 3:
                img, labels = generate_sprite_image(images,  len(images), '', labels, h=avg_h, w=avg_w, alternative_width=len(images), max_width=max_width)
            else:
                img, labels = generate_sprite_image(images,  len(images), '', labels, h=avg_h, w=avg_w, max_width=max_width)

            all_labels.append(labels)
            #all_files.append(files)

            if group_by == "label":
                local_file = os.path.join(save_dir, f'{subfolder}component_{counter}_{row["label"]}.jpg')
            else:
                local_file = os.path.join(save_dir, f'{subfolder}component_{counter}_{component_id}.jpg')
            local_file = fastdup_imwrite(local_file, img)
            img_paths.append(local_file)
            counter+=1


        except ModuleNotFoundError as ex:
            print('Your system is missing some dependencies please install then with pip install:')
            fastdup_capture_exception("visualize_top_components", ex)

        except Exception as ex:
            print('Failed on component', i, ex)
            fastdup_capture_exception("visualize_top_components", ex)

    print(f'Finished OK. Components are stored as image files {save_path}/components_[index].jpg')
    if 'label' in top_components:
        top_components['label'] = top_components['label'].apply(lambda x: x[:MAX_IMAGES_IN_GRID])
    #top_components['files'] = all_files

    return top_components.head(num_components), img_paths


def read_clusters_from_file(work_dir, get_label_func, kwargs):
    nrows = None
    if 'nrows' in kwargs:
        nrows = kwargs['nrows']

    if isinstance(work_dir, str):
        if os.path.isdir(work_dir):
            work_dir = os.path.join(work_dir, FILENAME_KMEANS_ASSIGNMENTS)
        assert os.path.exists(work_dir), f'Failed to find work_dir {work_dir}'
        df = pd.read_csv(work_dir, nrows=nrows)
        assert len(df), f"Failed to read dataframe from {work_dir} or empty dataframe"
    elif isinstance(work_dir, pd.DataFrame):
        df = work_dir
        assert "filename" in work_dir.columns, "Failed to find filename in dataframe columns"
        assert "cluster" in work_dir.columns
        assert "distance" in work_dir.columns
        assert len(df), f"Empty dataframe"
        if nrows is not None:
            df = df.head(nrows)

    if get_label_func is not None:
        df = find_label(get_label_func, df, 'filename', 'label', kwargs)

    return df


def read_components_from_file(work_dir, get_label_func, kwargs):
    if isinstance(work_dir, pd.DataFrame):
        assert len(work_dir), "Empty dataframe"
        assert 'input_dir' in kwargs and os.path.exists(kwargs['input_dir']), "Failed to find fastdup inut_dir, since input given was a dataframe. Please rim with input_dir='XXXX' parameter to point to the input directory where fastdup output is found. "

    nrows = None
    if len(kwargs) and 'nrows' in kwargs:
        nrows = kwargs['nrows']
    load_crops = (kwargs is not None) and ('load_crops' in kwargs) and kwargs['load_crops']
    draw_bbox = (kwargs is not None) and ('draw_bbox' in kwargs) and kwargs['draw_bbox']

    debug_cc = (kwargs is not None) and ("debug_cc" in kwargs) and kwargs["debug_cc"]

    # read fastdup connected components, for each image id we get component id
    if isinstance(work_dir, pathlib.Path):
        work_dir = str(work_dir)
    if isinstance(work_dir, str):
        # read a specific given file
        if str(work_dir).endswith('.csv'):
            components = pd.read_csv(work_dir, nrows=nrows)
        else: # read the default component file
            assert os.path.exists(os.path.join(work_dir, FILENAME_CONNECTED_COMPONENTS)), "Failed to find fastdup output file. Please run using fastdup.run(..) to generate this file. " + work_dir
            components = pd.read_csv(os.path.join(work_dir, FILENAME_CONNECTED_COMPONENTS), nrows=nrows)
    elif isinstance(work_dir, pd.DataFrame):
        components = work_dir
        if nrows is not None:
            components = components.head(nrows)

    if 'min_distance' not in components.columns:
        components["min_distance"] = 0

    if debug_cc:
        print(components.head())


    local_dir = work_dir if isinstance(work_dir, str) else kwargs['input_dir']
    if local_dir.endswith(".csv"):
        local_dir = os.path.dirname(os.path.abspath(local_dir))


    filenames = load_filenames(local_dir, kwargs)
    if (len(components) != len(filenames)) or load_crops:
        components = components.merge(filenames, left_on="__id", right_on="index", how="left")
    # now join the two tables to get both id and image name
    else:
        components['filename'] = filenames['filename']
    if load_crops and not draw_bbox: # crop filename contains both crop filename and original image filename, if we want to load_crop then we need to pick the later
        components['filename'] = filenames["crop_filename"]
    
    if debug_cc:
        print(components.head())

    if 'is_video' in kwargs:
        components['dirname'] = components['filename'].apply(os.path.dirname)
        sizes = components.groupby(['dirname']).size().reset_index(name='num_frames')
        components = components.merge(sizes, how='left', left_on=['dirname'], right_on=['dirname'])


    components = find_label(get_label_func, components, 'filename', 'label', kwargs)
    if debug_cc:
        print(components.head())
    return components


def remove_frames_from_overlapping_videos(comps):
    short_files = []
    short_videos = []
    assert 'video' in comps.columns

    comps['orig_len'] = comps['len']
    for i,row in comps.iterrows():
        row_files = row['files']
        row_videos = row['video']
        assert len(row_files) == len(row_videos), str(row_files) + str(row_videos)
        temp = pd.DataFrame({'files':row_files, 'video':row_videos})
        temp = temp.drop_duplicates(subset='video')
        short_files.append(temp['files'].values)
        short_videos.append(temp['video'].values)
    comps['files'] = short_files
    comps['video'] = short_videos
    comps['len'] = comps['video'].apply(lambda x:len(set(x)))
    comps = comps[comps['len'] > 1]
    assert len(comps), "Failed to find duplicate videos"

    return comps


def load_and_merge_stats(components, metric, work_dir, kwargs):

    if metric is not None:
        cols_to_use = ['index', metric]
        if metric == 'size':
            cols_to_use = ['index', 'width', 'height']
        stats = load_stats(work_dir,  None, kwargs, usecols=cols_to_use)
        assert len(stats), f"Failed to load stats {work_dir}"

        if metric == 'size':
            stats['size'] = stats.apply(lambda x: x['width']*x['height'], axis=1)

        if len(stats) != len(components):
            col_key = '__id' if '__id' in components.columns else 'index'
            if col_key == 'index':
                del stats['filename']
            assert col_key in components.columns, f"Failed to find key columns {col_key} in df {components.head()}"
            components = components.merge(stats, left_on=col_key, right_on='index', how='left')
        else:
            components[metric] = stats[metric]
            del stats
        assert metric in components.columns, "Failed to find metric"
    return components

def do_find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None, threshold=None, metric=None,
                           descending=True, min_items=None, max_items = None, keyword=None, save_path=None,
                           comp_type="component", input_dir=None, kwargs=None):
    '''
    Function to find the largest components of duplicate images

    Args:
        work_dir (str): working directory where fastdup.run was run, alternatively a pd.DataFrame with the output of connected_components.csv

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): 'visual' or 'label'

        slice (str): optional label names or list of label names to slice the dataframe. Supported slices could be str or list of str.
            Two reservied keyworks are: "diff" for only showing components with different labels. "same" for only showing components with same label.

        threshold (float): optional threshold to filter the dataframe

        metric (str): optional metric to use for ordering components by metric. Allowed values are 'blur','size','mean','min','max','unique','stdv'.

        descending (bool): optional flag to order components by metric in descending order.

        min_items (int): optional minimum number of items to consider a component.

        max_items (int): optional maximum number of items to consider a component.

        return_stats (bool): optional flag to return the statistics about sizes of the components

        save_path (str): optional path to save the top components statistics

        comp_type (str): component type should be coponent | cluster.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

    	Returns:
        ret (pd.DataFrame): of top components. The column component_id includes the component name.
        	The column files includes a list of all image files in this component.

        stats (str): HTML statistics about the sizes of the components (only when return_stats=True)


    '''
    kwargs['input_dir'] = input_dir
    if comp_type == "component":
        components = read_components_from_file(work_dir, get_label_func, kwargs)
        comp_col = "component_id"
        distance_col = "min_distance"
        index_col = "__id"
    elif comp_type == "cluster":
        components = read_clusters_from_file(work_dir, get_label_func, kwargs)
        comp_col = "cluster"
        distance_col = "distance"
        index_col = "index"
    else:
        assert False, f"Wrong component type {comp_type}"

    assert components is not None and len(components), f"Failed to read components file {work_dir} or found no images to cluster to components. Try to run fastdup again with lower ccthreshold." \
                                                       f"Value ccthreshold values are 0 to 1, where 1 no images are cluster together and zero means all images are clustered together."


    components = load_and_merge_stats(components, metric, work_dir, kwargs)

    if (get_label_func is not None):
        assert 'label' in components.columns, "Failed to find label column in components dataframe"
        if slice is not None:
            if slice in ["diff","same"]:
                pass
            else:
                components = slice_df(components, slice, 'label', kwargs)

        if 'path' in group_by:
            components['path'] = components['filename'].apply(lambda x: os.path.dirname(x))

        if group_by == 'visual':
            top_labels = components.groupby(comp_col)['label'].apply(list)
            top_files = components.groupby(comp_col)['filename'].apply(list)
            files_ids = components.groupby(comp_col)[index_col].apply(list)

            dict_cols = {'files':top_files, 'label':top_labels, 'files_ids':files_ids}

            if kwargs and 'is_video' in kwargs:
                top_dirs = components.groupby(comp_col)['dirname'].apply(list)
                dict_cols['video'] = top_dirs

            #if threshold is not None or metric is not None
            if distance_col in components.columns:
                distance = components.groupby(comp_col)[distance_col].apply(np.min)
                dict_cols['distance'] = distance

            if metric is not None:
                top_metric = components.groupby(comp_col)[metric].apply(list)
                dict_cols[metric] = top_metric

            comps = pd.DataFrame(dict_cols).reset_index()
            if slice is not None:
                assert 'label' in comps.columns, "Failed to find label column in components dataframe"
                if 'debug_cc' in kwargs and kwargs['debug_cc']:
                    print('labels before slicing', comps['label'].values[:10])
                if slice == "diff":
                    comps = comps[comps['label'].apply(lambda x: len(set(x)) > 1)]
                elif slice == "same":
                    comps = comps[comps['label'].apply(lambda x: len(set(x)) == 1)]
                assert(len(comps)), "Failed to find any components with different labels" if slice == "diff" else "Failed to find any components with same labels"

        elif group_by == 'label':
            is_list = isinstance(components['label'].values[0], list)
            if is_list:
                 components = components.explode(column='label', ignore_index=True).reset_index()

            top_files = components.groupby('label')['filename'].apply(list)
            top_components = components.groupby('label')[comp_col].apply(list)
            files_ids = components.groupby('label')[index_col].apply(list)

            dict_cols = {'files':top_files, comp_col:top_components, 'files_ids':files_ids}

            if kwargs and 'is_video' in kwargs:
                top_dirs = components.groupby('label')['dirname'].apply(list)
                dict_cols['video'] = top_dirs

            #if threshold is not None or metric is not None:
            if distance_col in components.columns:
                distance = components.groupby('label')[distance_col].apply(np.min)
                dict_cols['distance'] = distance
            if metric is not None:
                top_metric = components.groupby('label')[metric].apply(list)
                dict_cols[metric] = top_metric
            comps = pd.DataFrame(dict_cols).reset_index()
        else:
            assert(False), "group_by should be visual or label, got " + group_by

    else:
        top_components = components.groupby(comp_col)['filename'].apply(list)
        files_ids = components.groupby(comp_col)[index_col].apply(list)

        if 'debug_cc' in kwargs:
            print(top_components.head())

        dict_cols = {'files':top_components, 'files_ids':files_ids}
        #if threshold is not None or metric is not None:

        if distance_col in components.columns:
            distance = components.groupby(comp_col)[distance_col].apply(np.min)
            dict_cols['distance'] = distance
        if metric is not None:
            top_metric = components.groupby(comp_col)[metric].apply(list)
            dict_cols[metric] = top_metric

        if kwargs and 'is_video' in kwargs:
            top_dirs = components.groupby(comp_col)['dirname'].apply(list)
            dict_cols['video'] = top_dirs

        comps = pd.DataFrame(dict_cols).reset_index()



    assert len(comps), "No components found"

    comps['len'] = comps['files'].apply(lambda x: len(x))
    comps = comps[comps['len'] > 1]
    assert len(comps), "No components found with more than one image/video"

    # keep a single frame from each video
    if kwargs and 'is_video' in kwargs:
        comps = remove_frames_from_overlapping_videos(comps)
        if comps is None:
            return None

        # in case labels are list of lists, namely list of attributes per image, flatten the list
    if 'label' in comps.columns:
        try:
            print(comps['label'].values[0][0])
            if isinstance(comps['label'].values[0][0], list):
                comps['label'] = comps['label'].apply(lambda x: [item for sublist in x for item in sublist])
        except Exception as ex:
            print('Failed to flatten labels', ex)
            fastdup_capture_exception("find_top_components", ex)
            pass

    if slice == "diff":
        comps = comps.sort_values('distance', ascending=not descending)
    elif metric is None:
        if 'sort_by' in kwargs and kwargs['sort_by'] != 'comp_size':
            if kwargs['sort_by'] in comps.columns:
                comps = comps.sort_values(kwargs['sort_by'] , ascending=not descending)
            else:
                print(f"Warning: asked to sort by column {kwargs['sort_by']} but this column is missing. Available columns are {comps.columns}")
        elif 'external_df' not in kwargs:
            comps = comps.sort_values('len', ascending=not descending)
    else:
        comps['avg_metric'] = comps[metric].apply(lambda x: np.mean(x))
        comps = comps.sort_values('avg_metric', ascending=not descending)

    if threshold is not None:
        if metric is None:
            comps = comps[comps['distance'] > threshold]
        else:
            comps = comps[np.mean(comps[metric]) > threshold]
            assert len(comps) > 0, "Error: Failed to find any components with metric " + metric + " greater than threshold " + str(threshold)

    if keyword is not None:
        assert get_label_func is not None, "keyword can only be used with a callable get_label_func"
        assert group_by == 'visual', "keyword can only be used with group_by=visual"
        comps = comps[comps['label'].apply(lambda x: sum([1 if keyword in y else 0 for y in x]) > 0)]
        assert len(comps), "Error: Failed to find any components with label keyword " + keyword

    if min_items is not None:
        assert min_items > 1, "min_items should be a positive integer larger than 1"
        comps = comps[comps['len'] >= min_items]
        assert len(comps), f"Error: Failed to find any components with {min_items} or more items, try lowering the min_items threshold"

    if max_items is not None:
        assert max_items > 1, "min_items should be a positive integer larger than 1"
        comps = comps[comps['len'] <= max_items]
        assert len(comps), f"Failed to find any components with {max_items} or less items, try lowering the max_items threshold"
        comps = comps[comps['len'] > 1] # remove any singleton components

    if threshold is not None or metric is not None or keyword is not None:
        if 'save_artifacts' in kwargs and kwargs['save_artifacts']:
            if comp_type == "component":
                comps.to_pickle(f'{save_path}/{FILENAME_TOP_COMPONENTS}')
            else:
                comps.to_pickle(f'{save_path}/{FILENAME_TOP_CLUSTERS}')


    return comps



def do_create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 group_by='visual', slice=None, max_width=None, max_items=None, min_items=None,
                                 get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None,
                                 threshold=None ,metric=None, descending=True, keyword=None, comp_type="component", input_dir=None,
                                 **kwargs):
    '''

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir. Alternatively (for advanced users):
        * pd.DataFrame containing the content of connected_components.csv file. The file columns should contain: __id,component_id,min_distance.
        * or pd.DataFrame containing the top components. The df should include the fields: component_id,files,distance,len. Where component_id is integer, files is a list of files
        in this component, files is a list of absoluate image filenames in the component, distance is float in the range 0..1, len the files len.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

        slice(str): optional label to draw only a subset of the components conforming to this label. Or a list of labels.

        max_width (int): optional parameter to control resulting html width. Default is None

        max_items (int): optional parameter to control th number of items displayed in statistics: top max_items labels (for group_by='visual')
            or top max_items components (for group_by='label'). Default is None namely show all items.

        min_items (int): optional parameter to select only components with at least min_items items. Default is None.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists (callable): optional function to get bounding box of an image and add them to the report

        get_reformat_filename_func (callable): optional function to reformat the filename to be displayed in the report

        get_extra_col_func (callable): optional function to get extra column to be displayed in the report

        threshold (float): optional parameter to filter out components with distance below threshold. Default is None.

        metric (str): optional parameter to specify the metric used to chose the components. Default is None.

        descending (boolean): optional parameter to specify the order of the components. Default is True namely components are given from largest to smallest.

        keyword (str): optional parameter to select only components with a keyword as a substring in the label. Default is None.

        comp_type (str): optional parameter, default is "component" (for visualizing connected components) other option is "cluster" (for visualizing kmeans)

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        kwargs (dict): Optional parameter to pass additional parameters to the function.
            split_sentence_to_label_list (boolean): Optional parameter to split the label into a list of labels. Default is False.
            limit_labels_printed (int): Optional parameter to limit the number of labels printed in the html report. Default is max_items.
            nrows (int): limit the number of read rows for debugging purposes of the report
     '''

    start = time.time()
    #v1 = 'id_to_filename_func' in kwargs

    if num_images > 1000 and not lazy_load:
        print("Warning: When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    assert num_images >= 1, "Please select one or more images"
    assert group_by == 'label' or group_by == 'visual', "Allowed values for group_by=[visual|label], got " + group_by
    if group_by == 'label':
        assert get_label_func is not None, "missing get_label_func, when grouping by labels need to set get_label_func"
    assert comp_type in ['component','cluster']
    num_items_title = 'num_images' if 'is_video' not in kwargs else 'num_videos'
    if isinstance(work_dir, pd.DataFrame):
        run_hierarchical = False
    else:
        run_hierarchical = (work_dir.endswith("csv") and "hierarchical" in work_dir) or \
                           (kwargs.get('run_hierarchical', False))

    get_bounding_box_func = get_bounding_box_func_helper(get_bounding_box_func)

    kwargs['lazy_load'] = lazy_load
    kwargs['run_hierarchical'] = run_hierarchical
    if 'selection_strategy' not in kwargs:
        kwargs['selection_strategy'] = SELECTION_STRATEGY_FIRST
    else:
        assert isinstance(kwargs['selection_strategy'],int) and kwargs['selection_strategy'] >= 0 and kwargs['selection_strategy'] <= 2

    save_dir = calc_save_dir(save_path)

    ret = visualize_top_components(work_dir, save_dir, num_images,
                                                get_label_func, group_by, slice,
                                                get_bounding_box_func, max_width, threshold, metric,
                                                descending, max_items, min_items, keyword,
                                                comp_type=comp_type, input_dir=input_dir, kwargs=kwargs)
    if ret is None:
        return None
    subdf, img_paths = ret
    if subdf is None or len(img_paths) == 0:
        return None

    assert len(subdf) == len(img_paths), "Number of components and number of images do not match"
    import fastdup.html_writer
    save_artifacts= 'save_artifacts' in kwargs and kwargs['save_artifacts']
    if save_artifacts:
        subdf.to_csv(f'{save_dir}/components.csv')

    comp_col = "component_id" if comp_type == "component" else "cluster"

    cols_dict = {comp_col:subdf[comp_col].values,'files':subdf['files'].values,
                 num_items_title:subdf['len'].apply(lambda x: "{:,}".format(x)).values}
    if 'distance' in subdf.columns:
        cols_dict['distance'] = subdf['distance'].values
    if 'label' in subdf.columns:
        cols_dict['label'] = subdf['label'].values
    elif 'is_video' in kwargs:
        cols_dict['num_images'] = subdf['orig_len'].apply(lambda x: "{:,}".format(x)).values
        subdf['label'] = subdf['video']

    if metric in subdf.columns:
        cols_dict[metric] = subdf[metric].apply(lambda x: round(np.mean(x),2)).values

    ret2 = pd.DataFrame(cols_dict)
 
    info_list = []
    counter =0

    for i,row in ret2.iterrows():
        if group_by == 'visual':
            comp = row[comp_col]
            num = row[num_items_title]
            dict_rows = {'component':[comp], num_items_title :[num]}
            if 'distance' in row:
                dist = row['distance']
                dict_rows['mean_distance'] = [np.mean(dist)]
            if metric is not None:
                dict_rows[metric] = [row[metric]]
            if kwargs and 'is_video' in kwargs:
                dict_rows['num_images'] = row['num_images']

            info_df = pd.DataFrame(dict_rows).T
            info_list.append(info_df)
        elif group_by == 'label':
            label = row['label']
            num = row[num_items_title]
            dict_rows = {'label':[label], num_items_title :[num]}

            if 'distance' in row:
                dist = row['distance']
                dict_rows['mean_distance'] = [np.mean(dist)]
            if metric is not None:
                dict_rows[metric] = [row[metric]]

            info_df = pd.DataFrame(dict_rows).T
            info_list.append(info_df)
        #if save_artifacts:
        #    info_df.to_csv(f'{save_dir}/component_{counter}_df.csv')
        counter += 1

    ret = pd.DataFrame({'info': info_list})

    if 'label' in subdf.columns:
        if group_by == 'visual':
            labels_table = []
            counter = 0
            for i,row in subdf.iterrows():
                labels = list(row['label'])
                #if save_artifacts:
                #    pd.DataFrame({'label':labels,'files':list(row['files'])}).to_csv(os.path.join(save_dir, f"component_{counter}_labels.csv", index=False))
                if callable(get_reformat_filename_func) and 'is_video' in kwargs:
                    labels = [get_reformat_filename_func(x) for x in labels]

                unique, counts = np.unique(np.array(labels), return_counts=True)
                lencount = len(counts)
                if max_items is not None and max_items < lencount:
                    lencount = max_items
                if 'limit_labels_printed' in kwargs:
                    lencount = kwargs['limit_labels_printed']
                counts_df = pd.DataFrame({"counts":counts}, index=unique).sort_values('counts', ascending=False)
                if save_artifacts:
                    counts_df.to_csv(f'{save_dir}/counts_{counter}.csv')

                counts_df = counts_df.head(lencount)#.reset_index().rename({'index': 'label'}, axis=1)
                counts_df.index.names = ['label']
                # counts_df = counts_df
                labels_table.append(counts_df)
                counter+=1
            ret.insert(0, 'label', labels_table)
        else:
            comp_table = []
            counter = 0
            for i,row in subdf.iterrows():
                unique, counts = np.unique(np.array(row[comp_col]), return_counts=True)
                lencount = len(counts)
                if max_items is not None and max_items < lencount:
                    lencount = max_items;
                if kwargs is not None and 'limit_labels_printed' in kwargs:
                    lencount = kwargs['limit_labels_printed']
                counts_df = pd.DataFrame({"counts":counts}, index=unique).sort_values('counts', ascending=False)
                #if save_artifacts:
                #    counts_df.to_csv(f'{save_dir}/counts_{counter}.csv')
                counts_df = counts_df.head(lencount)
                comp_table.append(counts_df)
                counter+=1
            ret.insert(0, 'components', comp_table)

    img_html = format_image_html_string(img_paths, lazy_load, max_width, save_path)
    ret.insert(0, 'image', img_html)

    if str(save_path).endswith('.html'):
        out_file = save_path
    else:
        out_file = os.path.join(save_dir, "components_hierarchical.html") if run_hierarchical else os.path.join(save_dir, 'components.html')
    columns = ['info','image']
    if 'label' in subdf.columns:
        if group_by == 'visual':
            columns.append('label')
        elif group_by == 'label':
            columns.append('components')

    if comp_type == "component":
        if 'is_video' in kwargs:
            title = 'Video Components Report'
            subtitle = "Showing groups of similar videos"
        elif run_hierarchical:
            title = 'Hierarchical Components Report'
            subtitle = "Showing hierarchical groups of similar images"

        else:
            title = 'Components Report'
            subtitle = "Showing groups of similar images"
    else:
        title = "KMeans Cluster Report"
        subtitle = "Showing groups of similar images"


    if slice is not None:
        if slice == "diff":
            subtitle += ", from different classes"
        elif slice =="same":
            subtitle += ", from the same class"
        else:
            subtitle += ", for label: " + str(slice)
    if metric is not None:
        subtitle = ", Sorted by " + metric + " descending" if descending else "Sorted by " + metric + " ascending"

    ret = ret[['image','info', 'label']] if 'label' in ret.columns else ret[['image','info']]
    if callable(get_extra_col_func):
        ret['files'] = subdf['files'].values#.apply(lambda x: [get_extra_col_func(y) for y in x])
    fastdup.html_writer.write_to_html_file(ret, title, out_file, None, subtitle, max_width,
                                           jupyter_html=kwargs.get('jupyter_html', False))
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    if comp_type == "component":
        print_success_msg('components', out_file, lazy_load)
    else:
        print_success_msg("kmeans clusters", out_file, lazy_load)

    clean_images(lazy_load or save_artifacts or (threshold is not None), img_paths, "create_components_gallery")
    print('Execution time in seconds', round(time.time() - start, 1))
    return 0


def do_create_stats_gallery(stats_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            metric='blur', slice=None, max_width=None, descending=False, get_bounding_box_func=None,
                            get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None, work_dir=None,
                            **kwargs):
    '''

    Function to create and display a gallery of images computed by the outliers metrics.
    Note that fastdup generates a histogram of all the encountred valued for the specific metric. The red dashed line on this plot resulting in the number of images displayed in the report.
    For example, assume you have unique image values between 30-250, and the report displays 100 images with values betwewen 30-50. We plot a red line on the value 50.

    Parameters:
        stats_file (str): csv file with the computed image statistics by the fastdup tool. alternatively, a pandas dataframe can be passed in directly with the stats computed by fastdup.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        metric (str): Optional metric selection. One of blur, size, mean, min, max, unique, stdv. Default is blur.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the image file name. This is a function the user implements that gets the full file path and returns a new file name.

        get_extra_col_func (callable): Optional parameter to allow adding extra column to the report.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'


     '''


    img_paths = []
    get_bounding_box_func = get_bounding_box_func_helper(get_bounding_box_func)

    df = stats_file

    if metric is not None and metric == 'size':
        df['size'] = df['width'] * df['height']

    assert metric in df.columns, "Failed to find metric " + metric + " in " + str(df.columns)

    if metric in ['unique', 'width', 'height', 'size']:
        df = df[df[metric] > DEFUALT_METRIC_ZERO]
    elif metric in ['blur', 'mean', 'min', 'max', 'stdv']:
        df = df[df[metric] != DEFAULT_METRIC_MINUS_ONE]

    if slice is not None:
        subdf = find_label(get_label_func, df, 'filename', 'label', kwargs)
        subdf = slice_df(subdf, slice, 'label', kwargs)
        subdf = subdf.sort_values(metric, ascending=not descending).head(num_images)

    else:
        if 'external_df' not in kwargs:
            subdf = df.sort_values(metric, ascending=not descending).head(num_images)
        else:
            subdf = df.head(num_images)
        assert len(subdf), "Encountered an empty stats data frame"
        subdf = find_label(get_label_func, subdf, 'filename', 'label', kwargs)

    save_dir = calc_save_dir(save_path)
    stat_info = ""
    filename = "N/A"
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf)), desc="Generating gallery"):
        try:
            assert row['filename'] is not None, f"Failed with empty filename {subdf.head(2)}"
            filename = lookup_filename(row['filename'], work_dir)
            img = fastdup_imread(filename, None, None)
            assert img is not None, "Failed to read image " + filename + " orig filename " + row['filename']
            img = plot_bounding_box(img, get_bounding_box_func, filename)
            img = my_resize(img, max_width)

            imgpath = calc_image_path(lazy_load, save_dir, filename)
            imgpath = fastdup_imwrite(imgpath, img)

        except Exception as ex:
            fastdup_capture_exception("do_create_stats_gallery", ex)
            traceback.print_exc()
            print("Failed to generate viz for images", filename, ex)
            imgpath = None
        img_paths.append(imgpath)

    import fastdup.html_writer
    img_html = format_image_html_string(img_paths, lazy_load, max_width, save_dir)
    subdf.insert(0, 'Image', img_html)

    cols = [metric,'Image','filename']

    if callable(get_extra_col_func):
        subdf['extra'] = subdf['filename'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')

    if callable(get_reformat_filename_func):
        subdf['filename'] = subdf['filename'].apply(lambda x: get_reformat_filename_func(x))

    out_file = os.path.join(save_path, metric + '.html') if not str(save_path).endswith(".html") else save_path
    title = metric + ' Image Report'
    if metric == "mean" and descending:
        title = "Bright Image Report"
    elif metric == "mean":
        title = "Dark Image Report"
    elif metric == "size" and descending:
        title = "Largest Image Report"
    elif metric == "size":
        title = "Smallest Image Report"
    elif metric == "blur" and not descending:
        title = "Blurry Image Report"
    elif metric == "blur":
        title = "Sharpest Image Report"

    subtitle = "Showing example images, sort by "
    subtitle += "descending" if descending else "ascending"
    subtitle += " order"

    if slice is not None:
        subtitle += ", " + str(slice)

    if metric == 'size':
        cols.append('width')
        cols.append('height')

    if 'label' in subdf.columns:
        cols.append('label')

    subdf['info'] = swap_dataframe(subdf, cols)
    fastdup.html_writer.write_to_html_file(subdf[['Image','info']], title, out_file, stat_info, subtitle, 
                                           jupyter_html=kwargs.get('jupyter_html', False))
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print_success_msg(metric, out_file, lazy_load)
    clean_images(lazy_load, img_paths, "create_stats_gallery")
    return 0

def do_create_similarity_gallery(similarity_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 slice=None, max_width=None, descending=False, get_bounding_box_func =None,
                                 get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None, work_dir = None, min_items=2,
                                 max_items=None, **kwargs):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        similarity_file (str): csv file with the computed image statistics by the fastdup tool, alternatively a pandas dataframe can be passed in directly.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        metric (str): Optional metric selection. One of blur, size, mean, min, max, width, height, unique.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels. A special value is 'label_score' which is used for comparing both images and labels of the nearest neighbors.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the filename before displaying it in the report. This is a function the user implements that gets the full file path and returns a string with the reformatted filename.

        get_extra_col_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        work_dir (str): Optional parameter to fastdup work_dir. Needed when similarity_file is a pd.DataFrame.

        min_items (int): Minimal number of items in the similarity group (optional).

        max_items (int): Maximal number of items in the similarity group (optional).


    Returns:
        ret (pd.DataFrame): Dataframe with the image statistics
    '''


    from fastdup import generate_sprite_image
    img_paths2 = []
    from_paths = []
    info0 = []
    info = []
    label_score = []
    lengths = []
    debug_sim = False

    #v1 = 'id_to_filename_func' in kwargs
    df = similarity_file
    if debug_sim:
        print("sim df", df.head())
    get_bounding_box_func = get_bounding_box_func_helper(get_bounding_box_func)
    reformat_disp_path = kwargs.get('get_display_filename_func', lambda x: x)
    load_crops = kwargs.get('load_crops', False)

    save_dir = calc_save_dir(save_path)
    subdir = os.path.join(save_dir, "images")
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    if 'from_filename' not in df.columns and 'to_filename' not in df.columns:
        if load_crops:
            assert "filename" not in df.columns
            filenames = load_filenames(work_dir, kwargs)
            assert filenames is not None and not filenames.empty, f"Failed to read crop files from {work_dir}"
            assert "index" in filenames.columns and "filename" in filenames.columns
            df = merge_with_filenames(df, filenames[["index","filename"]])
            if debug_sim:
                print("after merge", df.head())
        else:
            df = similarity_file
            if df['from'].dtype in [int, np.int64]:
                assert df['to'].dtype in [int, np.int64], "Wrong types, expect both str or both int"
                filenames = load_filenames(work_dir, kwargs)
                filenames = filenames[["index", "filename"]]
                df = merge_with_filenames(df, filenames)
                if debug_sim:
                    print("after merge", df.head())
    else:
        df = convert_v1_to_v02(df)

    if get_label_func is not None and ('label' not in df.columns or 'label2' not in df.columns):
        df = find_label(get_label_func, df, 'from', 'label', kwargs)
        df = find_label(get_label_func, df, 'to', 'label2', kwargs)

        if slice != 'label_score':
            df = slice_df(df, slice, 'label')
            if df is None:
                return 1
    else:
        print("Warning: you are running create_similarity_gallery() without providing get_label_func so similarities are not computed between different classes. "
              "It is recommended to run this report with labels. Without labels this report output is similar to create_duplicate_gallery()")


    df = df.sort_values(['from','distance'], ascending= not descending)
    if 'label' in df.columns and 'label2' in df.columns:
        top_labels_to = df.groupby('from')['label2'].apply(list)
        top_labels_from = df.groupby('from')['label'].apply(list)

    tos = df.groupby('from')['to'].apply(list)
    distances = df.groupby('from')['distance'].apply(list)
    assert len(tos), "Empty list"

    if 'label' in df.columns:
        subdf = pd.DataFrame({'to':tos, 'label':top_labels_from, 'label2':top_labels_to, 'distance':distances}).reset_index()
    else:
        subdf = pd.DataFrame({'to':tos, 'distance':distances}).reset_index()
    if debug_sim:
        print("subdf", subdf.head())

    info_df = None


    if slice is None or slice != 'label_score':
        subdf = subdf.sort_values(['distance'], ascending=not descending)
        assert len(subdf), "Empty dataframe"
        df2 = subdf.copy()
        subdf = subdf.head(num_images)
        assert len(subdf), "Empty dataframe"
        stat_info = None
    else:
        assert len(subdf), "Empty dataframe"
        for i, row in tqdm(subdf.iterrows(), total=len(subdf), desc="Generating gallery"):
            filename = str(row["from"])
            filename = lookup_filename(filename, work_dir)

            from_label = row['label'][0]
            to_label = row['label2']
            similar = [x==from_label for x in list(to_label)]
            similar = 100.0*sum(similar)/(1.0*len(to_label))
            lengths.append(len(row['label']))
            label_score.append(similar)
        subdf['score'] = label_score
        subdf['length'] = lengths

        subdf = subdf[subdf['length'] >= min_items]
        if max_items is not None:
            subdf = subdf[subdf['length'] <= max_items]
        subdf = subdf.sort_values(['score','length'], ascending=not descending)
        df2 = subdf.copy()
        subdf = subdf.head(num_images)

    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf)), desc="Generating gallery"):

        info_df = None
        info0_df = None
        try:
            label = None
            filename = row["from"]
            filename = lookup_filename(filename, work_dir)
            if 'label' in row:
                label = row['label']
                if isinstance(label, list):
                  label = label[0]

            disp_filename = reformat_disp_path(filename)
            if callable(get_reformat_filename_func):
                new_filename = get_reformat_filename_func(filename)
            else:
                new_filename = disp_filename

            if label is not None:
                info0_df = pd.DataFrame({'label':[label],'from':[new_filename]}).T
            else:
                info0_df = pd.DataFrame({'from':[new_filename]}).T


            img = fastdup_imread(filename, input_dir=input_dir, kwargs=kwargs)
            assert img is not None, f"Failed to read image {str(filename)} {input_dir}"
            img = plot_bounding_box(img, get_bounding_box_func, filename)
            img = my_resize(img, max_width)
            if 'enhance_image' in kwargs and kwargs['enhance_image']:
                img = enhance_image(img)

            image_suffix = ''

            imgpath = calc_image_path(lazy_load, subdir, filename, filename_suffix=image_suffix)
            imgpath = fastdup_imwrite(imgpath, img)

            MAX_IMAGES = 10
            to_impaths_ = row["to"][:MAX_IMAGES]
            assert len(to_impaths_), "Empty image path list"
            #else:
            imgs = [plot_bounding_box(fastdup_imread(im, input_dir=input_dir, kwargs=kwargs),get_bounding_box_func,im) for im  in to_impaths_]
            assert len(imgs), "Empty image  list"

            keep_aspect_ratio = True
            if kwargs is not None and 'keep_aspect_ratio' in kwargs and not kwargs['keep_aspect_ratio']:
                keep_aspect_ratio = False
            h = []
            w = []
            for im in imgs:
                if im is not None:
                    h.append(im.shape[0])
                    w.append(im.shape[1])

            assert len(h), f"Failed to read all images from {input_dir}"

            avg_h = int(np.mean(h))
            avg_w = int(np.mean(w))
            max_h = int(np.max(h))
            max_w = int(np.max(w))
            if keep_aspect_ratio:
                avg_h = max_h
                avg_w = max_w

            img2 = []
            for f in imgs:
                if not keep_aspect_ratio:
                    f = cv2.resize(f, (avg_w,avg_h))
                else:
                    f = pad_image(f, avg_w, avg_h)
                img2.append(f)

            to_impaths = []
            for im, imgpath2 in zip(img2, to_impaths_):
                assert imgpath2 != imgpath, f"Found duplicate image {imgpath} {imgpath2}"
                image_suffix = ''
                imgpath2 = calc_image_path(lazy_load, save_dir, imgpath2, filename_suffix=image_suffix)
                if 'enhance_image' in kwargs and kwargs['enhance_image']:
                    im = enhance_image(im)
                imgpath2 = fastdup_imwrite(imgpath2, im)
                to_impaths.append(imgpath2)

            distances = row['distance'][:MAX_IMAGES]
            imgpath2 = f"{subdir}/to_image_{i}.jpg"
            info_df = pd.DataFrame({'distance':distances, 'to':[lookup_filename(im, work_dir) for im in to_impaths]})

            info_df['to'] = [reformat_disp_path(fid) for fid in to_impaths_]
            if callable(get_reformat_filename_func):
                info_df['to'] = info_df['to'].apply(lambda x: get_reformat_filename_func(x))

            if 'label2' in subdf.columns:
                info_df['label2'] = row['label2'][:MAX_IMAGES]
            info_df = info_df.sort_values('distance',ascending=False)
            info_df = info_df.set_index('distance')

            h = max_width if max_width is not None else 0
            w = h
            if keep_aspect_ratio:
                h = avg_h
                w = avg_w
            to_labels = None
            if 'label2' in info_df.columns:
                to_labels = info_df['label2'].values
            sample_size=  min(len(imgs), MAX_IMAGES)
            to_impaths = to_impaths[:sample_size]
            to_impaths.reverse()
            generate_sprite_image(to_impaths, min(len(imgs), MAX_IMAGES), save_dir, to_labels, h, w, imgpath2, min(len(imgs),MAX_IMAGES), max_width=max_width)

            assert os.path.exists(imgpath2), "Failed to generate sprite image " + imgpath2

            # This addition should be last before exception otherwise lengths do not match in case of exception



        except Exception as ex:
            fastdup_capture_exception("create_similarity_gallery", ex)
            print("Failed to generate viz for images", filename, ex)
            imgpath = None
            imgpath2 = None
            info_df = None
            info0_df = None

        if imgpath2 is not None and imgpath is not None and info_df is not None and info0_df is not None:
            img_paths2.append(imgpath2)
            from_paths.append(imgpath)
            info.append(info_df)
            info0.append(info0_df)


    import fastdup.html_writer
    img_html1 = format_image_html_string(from_paths, lazy_load, max_width, save_dir)
    img_html2 = format_image_html_string(img_paths2, lazy_load, None, save_dir)
    subdf.insert(0, 'Query Image', img_html1)
    subdf.insert(0, 'Similar', img_html2)
    subdf['info_to'] = info
    subdf['info_from'] = info0

    if not str(save_path).endswith('.html'):
        out_file = os.path.join(save_path, 'similarity.html')
    else:
        out_file = save_path
    title = 'Similarity Report'
    if slice is not None:
        title += ", " + str(slice)

    cols = ['info_from','info_to', 'Query Image','Similar']
    #if slice is not None and slice == 'label_score':
    #    cols = ['score'] + cols
    if callable(get_extra_col_func):
        subdf['extra'] = subdf['from'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')

    subdf['info'] = swap_dataframe(subdf, cols)
    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file, "", max_width,
                                           jupyter_html=kwargs.get('jupyter_html', False))
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print_success_msg('similar images', out_file, lazy_load)
    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']
    clean_images(lazy_load or save_artifacts, set(img_paths2).union(set(from_paths)), "create_similarity_gallery")

    return df2



if __name__ == "__main__":
    import pandas as pd
    import fastdup
    import os
    if False:
        os.chdir('/Users/dannybickson/Downloads/mafat')
        df = pd.read_csv('mafat.csv')
        # df['img_filename'] = df['filename']
        # del df['filename']
        import shutil
        #shutil.rmtree('output2')
        fd = fastdup.create(input_dir='.', work_dir='output2')

        fd.run(annotations=df, overwrite=True, license=os.environ["LICENSE"], bounding_box='rotated', augmentation_additive_margin=15,
               verbose=True, ccthreshold=0.8, num_images=200)    #ret = fd.vis.similarity_gallery(load_crops=True,slice='label_score',ascending=True,num_images=100,enhance_image=True,keep_aspect_ratio=True)
        #ret = fd.vis.component_gallery(load_crops=True,slice='diff',ascending=True,num_images=30,enhance_image=True,keep_aspect_ratio=True,get_extra_col_func=lambda x: x)
        #ret = fd.vis.outliers_gallery(load_crops=False)
        ret= fd.vis.stats_gallery(load_crops=False, metric='bright')
        #ret = fd.vis.similarity_gallery(load_crops=True,slice='label_score',ascending=True,num_images=30,enhance_image=True,keep_aspect_ratio=True)
        #ret = fd.vis.similarity_gallery(load_crops=True,slice='label_score')
        #fd.vis.component_gallery(load_crops=True,enhance_image=True,keep_aspect_ratio=True, save_artifacts=True,slice='same')
        #fd.vis.similarity_gallery(load_crops=True, enhance_image=True, keep_aspect_ratio=True, save_artifacts=True,slice="label_score",ascending=True)
    elif False:
        import shutil
        os.chdir('/Users/dannybickson/visual_database/cxx/unittests')
        import fastdup;
        fd = fastdup.create(input_dir='.', work_dir='mafat_out4')
        ret = fd.run()
        fd.vis.component_gallery(load_crops=False)
    elif False:
        import shutil
        os.chdir('/Users/dannybickson/Downloads/facedemo/')
        import fastdup;
        shutil.rmtree('frames_out')
        fd = fastdup.create(input_dir='frames', work_dir='frames_out')
        fd.run(bounding_box='face', license=os.environ["LICENSE"],threshold=0.85)
        fd.vis.component_gallery()

        fd.vis.component_gallery(load_crops=False, draw_bbox=True)
        fd.vis.duplicates_gallery(load_crops=True, draw_bbox=True)
        fd.vis.outliers_gallery(load_crops=True, draw_bbox=False)
        fd.vis.outliers_gallery(load_crops=True, draw_bbox=True)
        fd.vis.outliers_gallery(load_crops=False, draw_bbox=False)
        fd.vis.duplicates_gallery(load_crops=True, draw_bbox=False)
        fd.vis.duplicates_gallery(load_crops=True, draw_bbox=True)
        fd.vis.stats_gallery() # no load_crops, should load
        fd.vis.stats_gallery(load_crops=True, draw_bbox=True)
        fd.vis.stats_gallery(load_crops=False, draw_bbox=False)
    elif False:
        import fastdup
        if os.path.exists("tmp_out"):
            shutil.rmtree('tmp_out')
        fd = fastdup.create(input_dir='/mnt/data/sku110k', work_dir='tmp_out')
        fd.run(threshold=0.85, ccthreshold=0.85, num_images=100)
        #fd.vis.similarity_gallery(load_crops=False)
        df = fastdup.find_top_components('tmp_out')
        df['label'] = df['files'].apply(lambda x: [os.path.basename(y)[:2] for y in x])
        print(df.head())
        fd.vis.component_gallery(load_crops=False, sort_by='comp_size', external_df=df, label_col='label', ascending=True)
        files = fd.annotations()
        files['label'] = files['filename'].apply(lambda x: os.path.basename(x)[:2])
        sim = fd.similarity()
        df = merge_with_filenames(sim, files)
        df['label'] = df['from'].apply(lambda x: os.path.basename(x)[:2])
        df['label2'] = df['to'].apply(lambda x: os.path.basename(x)[:2])
        fd.vis.duplicates_gallery(load_rops=False, label_col='label', external_df=df)
        fd.vis.outliers_gallery(load_crops=False, label_col='label', external_df=df)
        stats = fd.img_stats()
        stats['label'] = stats['filename'].apply(lambda x: os.path.basename(x)[:2])
        fd.vis.stats_gallery(load_crops=False, label_col='label', external_df=stats)
    elif False:
        if os.path.exists("tmp_out"):
            shutil.rmtree('tmp_out')
        fd = fastdup.create(input_dir='/mnt/data/sku110k', work_dir='tmp_out')
        fd.run(threshold=0.85, ccthreshold=0.85, num_images=100)
        fastdup.create_duplicates_gallery('tmp_out', 'saved_duplicates.html')
        #fastdup.create_stats_gallery('tmp_out', 'saved_stats.html')
    elif False:
        if os.path.exists("tmp_out"):
            shutil.rmtree('tmp_out')
        fd = fastdup.create(input_dir='/Users/dannybickson/visual_database/cxx/unittests/tiktok_faceframe', work_dir='tmp_out')
        fd.run(threshold=0.85, ccthreshold=0.85, num_images=100, bounding_box='face', license=os.environ["LICENSE"])
        fd.vis.component_gallery(load_crops=True)
    elif False:
        df = pd.read_csv("/mnt/data/image_captioning/captions.csv")
        df['from'] = df['from'].apply(lambda x: "/mnt/data/image_captioning/" + x)
        import fastdup
        fastdup.create_outliers_gallery(df, '.', get_label_func='label', num_images=10)
    elif False:
        import shutil
        os.chdir('/mnt/data/image_captioning/')
        if os.path.exists("tmp_out"):
            shutil.rmtree('tmp_out')
        import fastdup
        fd= fastdup.create(input_dir='../fight-frames', work_dir='tmp_out')
        fd.run()
        fd.vis.outliers_gallery(label_col=VQA_MODEL1_NAME, ascending=False)
        #fastdup.run('../fight-frames', work_dir='tmp_out')
        #fastdup.create_outliers_gallery('tmp_out', '.', get_label_func='automatic')
        #fd.run(num_images=15,ccthreshold=0.7)
    elif False:
        import fastdup
        os.chdir('/Users/dannybickson/visual_database/cxx/unittests/')
        fd = fastdup.create(input_dir='two_images', work_dir='roi_bug2')
        fd.run(overwrite=True)
        #import shutil
        #shutil.rmtree('roi_bug2')
        #fastdup.run(input_dir='two_images', work_dir='roi_bug2')
        fastdup.init_search(2, 'roi_bug2', license=os.environ['LICENSE'])
        df = fastdup.search('one_image/test_1234.jpg', None, verbose=1)
        assert df is not None and len(df) == 1
        print(df.head())
        fastdup.create_duplicates_gallery(df, ".",input_dir='/Users/dannybickson/visual_database/cxx/unittests',work_dir='roi_bug')
    elif False:
        os.chdir('/Users/dannybickson/Downloads/uveye_bug/uveye_subset')
        import fastdup
        #fd = fastdup.create(input_dir='cycle_8-9_vl_reformat_nolabel.csv', work_dir='uv_out')
        anot = pd.read_csv('cycle_8-9_vl_reformat_nolabel.csv')
        anot = anot.rename(columns={'width':'bbox_w','height':'bbox_h', 'row_y':'bbox_y', 'col_x':'bbox_x'})
        fd = fastdup.create(input_dir='.', work_dir='uv_out')

        fd.run(overwrite=True,  annotations=anot)
        fd.vis.component_gallery()

    elif False:
        import fastdup
        import pandas as pd
        df = pd.read_csv('../unittests/rfile.txt')
        fd = fastdup.create(input_dir='.', work_dir='tmp_out')
        x = fd.run(license=os.environ["LICENSE"], annotations=df, overwrite=True, cc_threshold=0.3)
        fd.vis.component_gallery()
    elif False:
        import fastdup
        import pandas as pd
        files = sorted(os.listdir("/mnt/data/sku110k"))[:10]
        files = ["/mnt/data/sku110k/" +f for f in files]
        df = pd.DataFrame({'filename':files})
        df["label"] = df['filename'].apply(lambda x: os.path.basename(x)[:10])
        #df["index"] = range(0,10)
        fd = fastdup.create(input_dir = "/mnt/data/sku110k", work_dir='tmp_out')
        fd.run( overwrite=True, num_images=10, annotations=df)
    elif False:
        import fastdup
        os.chdir("../unittests")
        fd = fastdup.create(input_dir='two_images', work_dir='tmp_out')
        fd.run(model_path='dinov2b', overwrite=True, verbose=True)
        fd.vis.component_gallery()
    elif False:
        os.chdir("/Users/dannybickson/Downloads/mafat")
        import fastdup
        fastdup.run('mafat.csv', 'new_output', license=os.environ['LICENSE'], bounding_box='rotated',
                    turi_param='augmentation_additive_margin=25,ccthreshold=0.9',
                    verbose=False, model_path='dinov2s')
    elif False:
        import os
        files = os.listdir('/mnt/data/sku110k')
        files = [os.path.join('/mnt/data/sku110k/',f) for f in files]
        # create a fastdup with the input files and run it
        fd = fastdup.create(work_dir="/tmp/fastdub_workdir", input_dir=files[:10])
        fd.run(ccthreshold=0.9)
    elif False:
        import fastdup
        os.chdir("../unittests")
        fd = fastdup.create(input_dir='hilitu', work_dir='out1111')
        fd.run(overwrite=True)
        fd.vis.component_gallery()
        im = fastdup_imread('hilitu/#real or #fake _🙃_test.png', 'hilitu', {})
        assert im is not None
    elif False:
        import fastdup
        import shutil
        try:
            shutil.rmtree("tmp1234/tmp")
        except:
            pass
        fd = fastdup.create(input_dir='s3://visualdb/sku110k/', work_dir='tmp1234')
        fd.run(sync_s3_to_local=True, verbose=True, overwrite=True, num_images=100)
        fd.vis.outliers_gallery()
        #fastdup.run('s3://visualdb/sku110k/', 'tmp1234', turi_param='sync_s3_to_local=1', num_images=100, verbose=1)
    elif False:
        import fastdup
        os.chdir("../unittests")

        files = os.listdir("two_images")
        files = ['two_images/' + f for f in files]
        import pandas as pd
        df = pd.DataFrame({'filename':files, 'label':['A','N']})
        fd = fastdup.create(input_dir='two_images', work_dir='tmp11111')
        fd.run(overwrite=True, annotations=df,ccthreshold=0.3,threshold=0.3)
        #fd.vis.outliers_gallery()
        fd.vis.similarity_gallery(slice='A')
    elif False:
        import fastdup
        os.chdir("../unittests")
        fd = fastdup.create(input_dir='hilitu', work_dir='out1111')
        #fd.run(overwrite=True, verbose=True)
        fd.run(overwrite=True, model_path='dinov2s', verbose=False,num_threads=1,print_summary=False)
        fd.run(overwrite=True, verbose=True, num_threads=1,print_summary=False,d=576)
    elif False:
        import fastdup
        os.chdir("../unittests")
        flist = ["two_images/test_1234.jpg", 'two_images/train_1274.jpg']
        import numpy as np
        matrix = np.random.rand(2, 576).astype('float32')
        fd2 = fastdup.create(input_dir='two_images/', work_dir='out3')
        fd2.run(annotations=flist, embeddings=matrix, print_summary=False, overwrite=True, verbose=True)
    elif False:
        import fastdup
        os.chdir("../unittests")
        flist = ["two_images/test_1234.jpg", 'two_images/train_1274.jpg']
        import numpy as np
        matrix = np.random.rand(2, 576).astype('float32')
        import shutil
        shutil.rmtree('output')
        os.mkdir('output')
        fastdup.save_binary_feature('output', flist, matrix)
        fastdup.run(input_dir='two_images', work_dir='output', run_mode=2, verbose=True)
    elif False:
        coco_csv = '/mnt/data/coco_minitrain_25k/annotations/coco_minitrain2017.csv'
        coco_annotations = pd.read_csv(coco_csv, header=None, names=['filename', 'col_x', 'row_y',
                                                                     'width', 'height', 'label', 'ext'])
        #coco_annotations['split'] = 'train'  # Only train files were loaded
        coco_annotations['filename'] = coco_annotations['filename'].apply(
            lambda x: '/mnt/data/coco_minitrain_25k/images/train2017/' + x)
        coco_annotations = coco_annotations.drop_duplicates()
        coco_annotations.reset_index().head(1000)[['index', 'filename', 'col_x', 'row_y',
                                        'width', 'height']].to_csv('coco.csv', index=False)
        input_dir = '.'
        work_dir = 'fastdup_minicoco'
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        import fastdup
        fd = fastdup.create(work_dir=work_dir, input_dir=input_dir)
        ret = fd.run(annotations=coco_annotations.head(1000), overwrite=True, num_images=1000, cc_threshold=0.97, threshold=0.97, print_summary=False, verbose=0, num_threads=6, save_thumbnails=1)
        assert ret == 0
        ret = fd.vis.component_gallery(draw_bbox=True, load_crops=False, slice='airplane', label_col='label',debug_labels=True, group_by='label')
        #sys.exit(0)
        assert ret == 0
        df = fd.connected_components_grouped(ascending=False)
        assert len(df) == 7
        #assert df['len'].max() == 3
        assert df['len'].min() == 2
        assert set(df[df['distance'].max() == df['distance']]['label'].values[0]) == set(['car','truck'])
        ret = fd.vis.component_gallery(draw_bbox=False, load_crops=True)


        assert os.path.exists(os.path.join(work_dir, 'galleries', 'components.html'))
        ret = fd.vis.component_gallery(metric='size', slice='diff',draw_bbox=False)
        assert ret == 0
        assert os.path.exists(os.path.join(work_dir, 'galleries', 'components.html'))

        fd.vis.duplicates_gallery(draw_bbox=True)
        ret = fd.vis.outliers_gallery(draw_bbox=True)
        assert ret == 0
        ret = fd.vis.outliers_gallery(draw_bbox=False)
        assert ret == 0
        a,b = fd.embeddings()
        assert isinstance(a, list)
        assert isinstance(b, np.ndarray)
        assert b.shape[0] == len(a)
    elif False:
        coco_csv = '/mnt/data/coco_minitrain_25k/annotations/coco_minitrain2017.csv'
        coco_annotations = pd.read_csv(coco_csv, header=None, names=['filename', 'col_x', 'row_y',
                                                                     'width', 'height', 'label', 'ext'])
        #coco_annotations['split'] = 'train'  # Only train files were loaded
        coco_annotations['filename'] = coco_annotations['filename'].apply(
            lambda x: '/mnt/data/coco_minitrain_25k/images/train2017/' + x)
        coco_annotations = coco_annotations.drop_duplicates()
        coco_annotations.reset_index()[['index','filename','col_x','row_y',
        'width','height']].head(1000).to_csv('coco2.csv',index=False)
        work_dir = 'fastdup_minicoco2'
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        import fastdup
        ret = fastdup.run(input_dir='coco2.csv', work_dir=work_dir, num_images=1000, turi_param='ccthreshold=0.95,store_int=1' ,threshold=0.95, verbose=0, num_threads=6, bounding_box='xywh_bbox')
        fastdup.create_components_gallery(work_dir, work_dir, draw_bbox=True, input_dir='/mnt/data/coco_minitrain_25k/images/train2017/')

    elif False:
        coco_csv = '/mnt/data/coco_minitrain_25k/annotations/coco_minitrain2017.csv'
        coco_annotations = pd.read_csv(coco_csv, header=None, names=['filename', 'col_x', 'row_y',
                                                                     'width', 'height', 'label', 'ext'])
        coco_annotations['split'] = 'train'  # Only train files were loaded
        coco_annotations['filename'] = coco_annotations['filename'].apply(
            lambda x: 'train2017/' + x)
        coco_annotations = coco_annotations.drop_duplicates()
        input_dir = '/mnt/data/coco_minitrain_25k/images/'
        work_dir = 'fastdup_minicoco'
        import fastdup
        fd = fastdup.create(work_dir=work_dir, input_dir=input_dir)
        fd.run(annotations=coco_annotations, overwrite=True, num_images=1000, cc_threshold=0.96, threshold=0.96)
        ret = fd.vis.component_gallery(metric='size', slice='diff')
        ret = fd.vis.component_gallery(metric='size', slice='diff')
        ret = fd.vis.outliers_gallery(draw_bbox=True)
        assert ret == 0
        ret = fd.vis.outliers_gallery(draw_bbox=False)
        assert ret == 0
    elif False:
        import fastdup
        os.chdir("../unittests")
        input_dir = 'omer_test'
        work_dir = 'fastdup_workdir'
        #
        df_annot = pd.DataFrame([
            {'filename': 'omer_test/000000001.jpg', 'label': 'dup1'},
            {'filename': 'omer_test/test_1234.jpg', 'label': 'dup2'},
            {'filename': 'omer_test/test_1234a.jpg', 'label': 'dup3'},
            {'filename': 'omer_test/test_1234b.jpg', 'label': 'dup2'},
            {'filename': 'omer_test/train_1274.jpg', 'label': 'foo'},
        ])
        #
        fd = fastdup.create(work_dir=work_dir, input_dir=input_dir)
        ret = fd.run(threshold=0.8, overwrite=True, annotations=df_annot, verbose=True)
        assert ret == 0
        print(fd.invalid_instances())
        df = fd.annotations(valid_only=False)
        print(df)
        assert len(df) == 5
        assert df['error_code'].nunique() == 4
    elif False:
        # create a fastdup object, the input dir is "." since we added the folder name into the filename before.
        import fastdup
        fd = fastdup.create(work_dir='tiny-coco3', input_dir='/mnt/data/tiny-coco/small_coco/train_2017_small/')
        fd.run(annotations='/mnt/data/tiny-coco/small_coco/instances_train2017_small.json')
    elif False:
        import fastdup
        fd = fastdup.create(work_dir='outtest', input_dir='/Users/dannybickson/visual_database/cxx/unittests/two_images')
        fd.run(model_path='clip', verbose=True, overwrite=True)
    elif False:
        import fastdup
        os.chdir("../unittests")
        df = pd.read_csv('tom_apostrophe/test_annot.csv')
        print(df)
        fd = fastdup.create(input_dir='tom_apostrophe', work_dir='tom_out')
        ret = fd.run( overwrite=True, annotations=df, threshold=0.2, ccthreshold=0.2)
        assert ret == 0
        assert len(fd.annotations()) == 2
        ret = fd.vis.duplicates_gallery()
        fd.vis.stats_gallery()
        assert ret == 0
    elif False:
        import pathlib
        dir = pathlib.Path('~/')
        from fastdup.utilities import shorten_path
        dir = shorten_path(dir)
        assert os.path.exists(dir)
        dir = pathlib.Path('~')
        dir = shorten_path(dir)
        assert os.path.exists(dir)
    elif False:
        import os
        if os.path.exists('sprite123.png'):
            os.unlink("sprite123.png")
        files = os.listdir('/mnt/data/sku110k')[:25]
        files = [os.path.join('/mnt/data/sku110k',f) for f in files]
        kwargs = {}
        kwargs['force_width'] = 5
        kwargs['force_height'] = 5
        if not os.path.exists('tmplog'):
            os.makedirs('tmplog')
        fastdup.generate_sprite_image(files[:23], 23, 'tmplog',force_width=5, force_height=5, alternative_filename='sprite123.png')
        assert os.path.exists('sprite123.png')
    elif False:
        os.chdir('/Users/dannybickson/Downloads/elbit2')
        import fastdup
        fd = fastdup.create(input_dir="21425910_not_rotated", work_dir="output")
        fd.run(annotations="21425910_not_rotated/annotations_910/instances_default_rotated_fixed.json",
               overwrite=True, license=os.environ["LICENSE"])
        os.chdir('/tmp')
        import fastdup
        fd = fastdup.create(input_dir="/Users/dannybickson/Downloads/elbit2/21425910_not_rotated", work_dir="output")
        fd.run(annotations="/Users/dannybickson/Downloads/elbit2/21425910_not_rotated/annotations_910/instances_default_rotated_fixed.json",
               overwrite=True, license=os.environ["LICENSE"])
    elif False:
        import fastdup
        fd = fastdup.create(input_dir='../unittests/two_images')
        fd.run(overwrite=True,verbose=True,run_advanced_stats=1)
        stats = fd.img_stats()
        for i in stats.columns:
            print(f"assert stats['{i}'].values[0] == {stats[i].values[0]}")

        import math
        assert len(stats) == 2
        assert stats['index'].values[0] == 0
        assert stats['img_w'].values[0] == 2448
        assert stats['img_h'].values[0] == 3264
        assert stats['unique'].values[0] == 256
        assert stats['blur'].values[0] == 5328.6621
        assert stats['mean'].values[0] == 91.513
        assert stats['min'].values[0] == 0.0
        assert stats['max'].values[0] == 255.0
        assert stats['stdv'].values[0] == 60.5834
        assert stats['file_size'].values[0] == 894163
        assert stats['rms_contrast'].values[0] == 0.6619
        assert stats['mean_rel_intensity_r'].values[0] == 1.0
        assert stats['mean_rel_intensity_b'].values[0] == 1.2026
        assert stats['mean_rel_intensity_g'].values[0] == 1.0323
        assert stats['contrast'].values[0] == 1.0
        assert stats['mean_hue'].values[0] == 69.3986
        assert stats['mean_saturation'].values[0] == 106.3207
        assert stats['mean_val'].values[0] == 116.1325
        assert stats['edge_density'].values[0] == 0.2416
        assert stats['mean_r'].values[0] == 84.868
        assert stats['mean_g'].values[0] == 102.0607
        assert stats['mean_b'].values[0] == 87.6103
        assert stats['filename'].values[0] == '../unittests/two_images/test_1234.jpg'
        assert stats['error_code'].values[0] == 'VALID'
        assert stats['is_valid'].values[0] == True
        assert stats['fd_index'].values[0] == 0

    elif False:
        os.chdir('/Users/dannybickson/Downloads/tiktok_yael/frames/tmp')
        import fastdup
        import pandas as pd
        df = pd.read_csv('ocr.csv')
        fd = fastdup.create(input_dir='.', work_dir='../work')
        fd.run(annotations=df, license=os.environ['LICENSE'])

    elif False:
        import fastdup
        from fastdup.image import inner_read
        img = inner_read('/Users/dannybickson/visual_database/cxx/unittests/heic/colors-no-alpha.heic')
    elif False:
        import fastdup
        from fastdup.image import fastdup_imread
        fastdup_imread('~/visual_database/cxx/unittests/two_images/test_1234.jpg',
                            '~/visual_database/cxx/unittests/two_images/', {})
    elif False:
        import fastdup
        import shutil
        os.chdir('/Users/dannybickson/Downloads/tiktok_videos')
        if os.path.exists('ocr_out'):
            shutil.rmtree('ocr_out')

        fd = fastdup.create(input_dir='frames/datatiktokdownload.online_1672753738821.mp4', work_dir='ocr_out')
        fd.run(bounding_box='ocr', license=os.environ['LICENSE'], verbose=True, overwrite=True)
        fd.vis.component_gallery()
    elif False:
        import fastdup
        fd = fastdup.create(input_dir=".", work_dir='out')
        ret = fd.feature_vector('../unittests/two_images/test_1234.jpg')
        assert len(ret) == 2
        ret = ret[0]
        assert ret.shape[1] == 576
        assert ret.shape[0] == 1

        ret = fd.feature_vectors(['../unittests/two_images/test_1234.jpg','../unittests/two_images/train_1274.jpg'])
        assert len(ret) == 2
        ret = ret[0]
        assert ret.shape[1] == 576
        assert ret.shape[0] == 2
    elif False:
        import os
        os.chdir('/Users/dannybickson/Downloads/Kitti_bug')
        import fastdup
        import pandas as pd

        data = pd.read_csv('kitti_annotations.csv')
        #data['index'] = range(len(data))
        #data = data[data['width'] > 20]
        print(data)
        fd = fastdup.create(input_dir='.')
        fd.run(annotations=data, overwrite=True, augmentation_additive_margin=15, verbose=1, num_threads=1, num_images=20)
        print(fd.invalid_instances())
        #fd.vis.duplicates_gallery(load_crops=True)
        fd.vis.duplicates_gallery(load_crops=False, draw_bbox=True)
        #fd.vis.stats_gallery(load_crops=True)
        #fd.vis.stats_gallery(load_crops=False)
        #fd.vis.outliers_gallery(load_crops=True)

    elif False:
        import fastdup
        os.chdir('/Users/dannybickson/Downloads/stuttgart/')
        if os.path.exists('out/atrain_croos.csv'):
            os.unlink('out/atrain_crops.csv')
        fd = fastdup.create(input_dir='frames', work_dir='/Users/dannybickson/Downloads/stuttgart/out2')
        fd.run(bounding_box='ocr', verbose=True, overwrite=True, num_images=3, threshold=0.3)
        fd.vis.duplicates_gallery()
    elif False:
        import fastdup
        fastdup.remove_duplicates(input_dir="../unittests/two_images", distance=0.2)



