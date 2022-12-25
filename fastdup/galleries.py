
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import pandas as pd
import cv2
import time
import numpy as np
import traceback
from fastdup.image import plot_bounding_box, my_resize, get_type, imageformatter, create_triplet_img, fastdup_imread, calc_image_path, clean_images
from fastdup.definitions import *
import re
from multiprocessing import Pool
from fastdup.sentry import *

try:
    from tqdm import tqdm
except:
    tqdm = (lambda x: x)


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

def find_label(get_label_func, df, in_col, out_col, kwargs=None):
    assert len(df), "Df has now rows"

    if (get_label_func is not None):
        if isinstance(get_label_func, str):
            assert os.path.exists(get_label_func), f"Failed to find get_label_func file {get_label_func}"
            import csv
            nrows = None
            if len(kwargs) and 'nrows' in kwargs:
                nrows = kwargs['nrows'] 
            # use quoting=csv.QUOTE_NONE for LAOIN
            quoting = 0
            if len(kwargs) and 'quoting' in kwargs:
                quoting = kwargs['quoting']
            df_labels = pd.read_csv(get_label_func, nrows=nrows, quoting=quoting)
            if len(df_labels) != len(df):
                print(f"Error: wrong length of labels file {get_label_func} expected {len(df)} got {len(df_labels)}")
                return None
            if 'label' not in df_labels.columns:
                print(f"Error: wrong columns in labels file {get_label_func} expected 'label' column")
                return None
            df[out_col] = df_labels['label']
        elif isinstance(get_label_func, dict):
            df[out_col] = df[in_col].apply(lambda x: get_label_func.get(x, MISSING_LABEL))
        elif callable(get_label_func):
            df[out_col] = df[in_col].apply(lambda x: get_label_func(x))
        else:
            assert False, f"Failed to understand get_label_func type {type(get_label_func)}"

    if kwargs is not None and 'debug_labels' in kwargs:
        print(df.head())
    return df

def split_str(x):
    return re.split('[?.,:;&^%$#@!()]', x)


def slice_df(df, slice, kwargs=None):
    split_sentence_to_label_list = kwargs is not None and 'split_sentence_to_label_list' in kwargs and kwargs['split_sentence_to_label_list']
    debug_labels = kwargs is not None and 'debug_labels' in kwargs and kwargs['debug_labels']

    if slice is not None:
        if isinstance(slice, str):
            # cover the case labels are string or lists of strings
            if split_sentence_to_label_list:
                labels = df['label'].astype(str).apply(lambda x: split_str(x.lower())).values
                if debug_labels:
                    print('Label with split sentence', labels[:10])
            else:
                labels = df['label'].astype(str).values
                if debug_labels:
                    print('label without split sentence', labels[:10])
            is_list = isinstance(labels[0], list)
            if is_list:
                labels = [item for sublist in labels for item in sublist]
                if debug_labels:
                    print('labels after merging sublists', labels[:10])

            if not is_list:
                df2 = df[df['label'] == slice]
                if len(df2) == 0:
                    df2 = df[df['label'].apply(lambda x: slice in str(x))]
                df = df2
            else:
                df = df[df['label'].apply(lambda x: slice in [y.lower() for y in x])]
            assert len(df), f"Failed to find any labels with str {slice}  {is_list}"
        elif isinstance(slice, list):
            if isinstance(df['label'].values[0], list):
                df = df[df['label'].apply(lambda x: len(set(x)&set(slice)) > 0)]
            else:
                df = df[df['label'].isin(slice)]
            assert len(df), f"Failed to find any labels with {slice}"
        else:
            assert False, "slice must be a string or a list of strings"

    return df


def lookup_filename(filename, work_dir):
    if filename.startswith(S3_TEMP_FOLDER + '/')  or filename.startswith(S3_TEST_TEMP_FOLDER + '/'):
        assert work_dir is not None, "Failed to find work_dir on remote_fs"
        filename = os.path.join(work_dir, filename)
    return filename


def extract_filenames(row, work_dir = None):

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
    ptype = '{0}_{1}'.format(type1, type2)
    return impath1, impath2, dist, ptype







def do_create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True,
                              lazy_load=False, get_label_func=None, slice=None, max_width=None,
                                 get_bounding_box_func=None, get_reformat_filename_func=None,
                                 get_extra_col_func=None, input_dir=None, work_dir=None, threshold=None, kwargs=None):
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
            Each bounding box should be 4 integers: x1, y1, x2, y2. An example list is [[100,100,200,200]] which contains a single bounding box.

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
    nrows = None
    if 'nrows' in kwargs:
        nrows = kwargs['nrows']
    kwargs['lazy_load'] = lazy_load

    if isinstance(similarity_file, pd.DataFrame):
        df = similarity_file
    else:
        assert os.path.exists(similarity_file), f"Failed to find fastdup input file {similarity_file}. Please run using fastdup.run(..) to generate this file."
        df = pd.read_csv(similarity_file, nrows=nrows)
        assert len(df), "Failed to read similarity file " + similarity_file

    if threshold is not None:
        df = df[df['distance'] >= threshold]
        if len(df) == 0:
            print("Failed to find any duplicates images with similarity score >= ", threshold)
            return 1

    if slice is not None and get_label_func is not None:
        df = find_label(get_label_func, df, 'from', 'label', kwargs)
        if isinstance(slice, str):
            if slice == "diff":
                df = find_label(get_label_func, df, 'to', 'label2', kwargs)
                df = df[df['label'] != df['label2']]
            elif slice == "same":
                df = find_label(get_label_func, df, 'to', 'label2', kwargs)
                df = df[df['label'] == df['label2']]
            else:
                if slice not in df['label'].unique():
                    print(f"Slice label {slice} not found in the similarity file")
                    return None
                df = df[df['label'] == slice]
                assert len(df), "Failed to find slice " + slice
        elif isinstance(slice, list):
            df = df[df['label'].isin(slice)]
            assert len(df), "Failed to find any rows with label values " + str(slice)

    sets = {}

    if 'is_video' in kwargs:
        assert os.path.exists(work_dir), f"Failed to find fastdup work dir {work_dir}"
        filenames = pd.read_csv(f'{work_dir}/atrain_{FILENAME_IMAGE_LIST}')
        filenames['dirname'] = filenames['filename'].apply(os.path.dirname)
        frames = filenames.groupby(['dirname']).size().reset_index(name='num_frames')
        df = similarity_file.merge(frames, how='left', left_on=['subfolder1'], right_on=['dirname'])

    subdf = df.head(num_images) if descending else df.tail(num_images)
    subdf = subdf.reset_index()

    if 'is_video' in kwargs:
        subdf['ratio'] = subdf['counts'].astype(float) / subdf['num_frames'].astype(float)
        subdf['ratio'] = subdf['ratio'].apply(lambda x: round(x,3))

    indexes = []
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        impath1, impath2, dist, ptype = extract_filenames(row, work_dir)
        if impath1 + '_' + impath2 in sets:
            continue
        try:
            img, imgpath = create_triplet_img(impath1, impath2, ptype, dist, save_path, get_bounding_box_func, input_dir, kwargs)
            sets[impath1 +'_' + impath2] = True
            sets[impath2 +'_' + impath1] = True
            indexes.append(i)
            img_paths.append(imgpath)

        except Exception as ex:
            fastdup_capture_exception("triplet image", ex)
            print("Failed to generate viz for images", impath1, impath2, ex)
            img_paths.append(None)

    subdf = subdf.iloc[indexes]
    import fastdup.html_writer

    if not lazy_load:
        subdf.insert(0, 'Image', [imageformatter(x, max_width) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + str(x) + "\" loading=\"lazy\">" for x in img_paths]
        subdf.insert(0, 'Image', img_paths2)
    out_file = os.path.join(save_path, 'similarity.html')
    if get_label_func is not None:
        #subdf.insert(2, 'From', subdf['from'].apply(lambda x: get_label(x, get_label_func)))
        subdf = find_label(get_label_func, subdf, 'from', 'From')
        #subdf.insert(3, 'To', subdf['to'].apply(lambda x: get_label(x, get_label_func)))
        subdf = find_label(get_label_func, subdf, 'to', 'To')
    else:
        subdf = subdf.rename(columns={'from':'From', 'to':'To'}, inplace=False)
    subdf = subdf.rename(columns={'distance':'Distance'}, inplace=False)
    fields = ['Image', 'Distance', 'From', 'To']

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

    title = 'Fastdup Tool - Similarity Report'
    if 'is_video' in kwargs:
        title = 'Fastdup Tool - Video Similarity Report'

    if slice is not None:
        if slice == "diff":
            title += ", of different classes"
        else:
            title += ", for label " + str(slice)

    assert len(subdf), "Error: failed to find any duplicates, try to run() with lower threshold"

    fastdup.html_writer.write_to_html_file(subdf[fields], title, out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file
    print("Stored similarity visual view in ", out_file)

    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']
    if save_artifacts:
        save_artifacts_file = os.path.join(save_path, 'similarity_artifacts.csv')
        subdf[list(set(fields)-set(['Image']))].to_csv(save_artifacts_file, index=False)
        print("Stored similarity artifacts in ", save_artifacts_file)

    clean_images(lazy_load or save_artifacts, img_paths, "create_duplicates_gallery")
    return 0



def load_one_image_for_outliers(args):
    row, work_dir, input_dir, get_bounding_box_func, max_width, save_path, kwargs = args
    impath1, impath2, dist, ptype = extract_filenames(row, work_dir)
    try:
        img = fastdup_imread(impath1, input_dir, kwargs)
        img = plot_bounding_box(img, get_bounding_box_func, impath1)
        img = my_resize(img, max_width=max_width)

        #consider saving second image as well!
        #make sure image file is unique, so add also folder name into the imagefile
        lazy_load = 'lazy_load' in kwargs and kwargs['lazy_load']
        imgpath = calc_image_path(lazy_load, save_path, impath1)
        cv2.imwrite(imgpath, img)
        assert os.path.exists(imgpath), "Failed to save img to " + imgpath

    except Exception as ex:
        fastdup_capture_exception(f"load_one_image_for_outliers", ex)
        imgpath = None

    return imgpath




def do_create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None, max_width=None, get_bounding_box_func=None, get_reformat_filename_func=None,
                               get_extra_col_func=None, input_dir= None, **kwargs):
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

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. An example list is [[100,100,200,200]] which contains a single bounding box.

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the gallery.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

    Returns:
        ret (int): 0 if successful, 1 otherwise
    '''



    img_paths = []
    work_dir = None
    nrows = None
    if nrows in kwargs:
        nrows = kwargs['nrows']
    kwargs['lazy_load'] = lazy_load

    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']


    if isinstance(outliers_file, pd.DataFrame):
        df = outliers_file
    else:
        assert os.path.exists(outliers_file), f"Failed to find fastdup input file {outliers_file}. Please run using fastdup.run(..) to generate this file."
        df = pd.read_csv(outliers_file, nrows=nrows)
        print('read outliers', len(df))
        work_dir = os.path.dirname(outliers_file)
    assert len(df), "Failed to read outliers file " + outliers_file

    if (how == 'all'):
        dups_file = os.path.join(os.path.dirname(outliers_file), FILENAME_SIMILARITY)
        assert os.path.exists(dups_file), f'Failed to find input file {dups_file} which is needed for computing how=all similarities, . Please run using fastdup.run(..) to generate this file.'

        dups = pd.read_csv(dups_file, nrows=nrows)
        assert len(dups), "Error: Failed to locate similarity file file " + dups_file
        dups = dups[dups['distance'] >= dups['distance'].astype(float).mean()]
        assert len(dups), f"Did not find any images with similarity more than the mean {dups['distance'].mean()}"

        joined = df.merge(dups, on='from', how='left')
        joined = joined[pd.isnull(joined['distance_y'])]

        assert len(joined), 'Failed to find outlier images that are not included in the duplicates similarity files, run with how="one".'

        subdf = joined.rename(columns={"distance_x": "distance", "to_x": "to"}).sort_values('distance', ascending=True)
    else:
        subdf = df.sort_values(by='distance', ascending=True)

    if get_label_func is not None:
        if isinstance(get_label_func, str):
            subdf = find_label(get_label_func, subdf, 'from', 'label',serial=False)
        else:
            subdf = find_label(get_label_func, subdf, 'from', 'label')
        subdf = slice_df(subdf, slice)
        if subdf is None:
            return 1

    subdf = subdf.drop_duplicates(subset='from').sort_values(by='distance', ascending=True).head(num_images)
    all_args = []
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):

        args = row, work_dir, input_dir, get_bounding_box_func, max_width, save_path, kwargs
        all_args.append(args)

    # trying to deal with the following runtime error:
    #An attempt has been made to start a new process before the
    #current process has finished its bootstrapping phase.
    try:
        with Pool() as pool:
            img_paths = pool.map(load_one_image_for_outliers, all_args)
    except RuntimeError as e:
        fastdup_capture_exception("create_outliers_gallery_pool", e)
        img_paths = []
        for i in all_args:
            img_paths.append(load_one_image_for_outliers(i))

    import fastdup.html_writer
    if not lazy_load:
        subdf.insert(0, 'Image', [imageformatter(x, max_width) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        subdf.insert(0, 'Image', img_paths2)

    subdf = subdf.rename(columns={'distance':'Distance','from':'Path'}, inplace=False)

    out_file = os.path.join(save_path, 'outliers.html')
    title = 'Fastdup Tool - Outliers Report'
    if slice is not None:
        title += ", " + str(slice)

    cols = ['Image','Distance','Path']
    if callable(get_extra_col_func):
        subdf['extra'] = subdf['Path'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')

    if get_reformat_filename_func is not None and callable(get_reformat_filename_func):
        subdf['Path'] = subdf['Path'].apply(lambda x: get_reformat_filename_func(x))

    if 'label' in subdf.columns:
        cols.append('label')

    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file)

    if save_artifacts:
        subdf[list(set(cols)-set(['Image']))].to_csv(f'{save_path}/outliers_report.csv')
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored outliers visual view in ", os.path.join(out_file))
    clean_images(lazy_load or save_artifacts, img_paths, "create_outliers_gallery")
    return 0


def load_one_image(args):
    try:
        f, input_dir, get_bounding_box_func, kwargs = args
        img = fastdup_imread(f, input_dir, kwargs)
        img = plot_bounding_box(img, get_bounding_box_func, f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img.shape[1], img.shape[0]
    except Exception as ex:
        print("Warning: Failed to load image ", f, "skipping image due to error", ex)
        fastdup_capture_exception("load_one_image", ex)
        return None,None,None




def visualize_top_components(work_dir, save_path, num_components, get_label_func=None, group_by='visual', slice=None,
                             get_bounding_box_func=None, max_width=None, threshold=None, metric=None, descending=True,
                             max_items = None, min_items=None, keyword=None, return_stats=False, comp_type="component",
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
        print('Your system is missing some dependencies, please pip install matplotlib matplotlib-inline torchvision')
        print(ex)
        fastdup_capture_exception("visualize_top_components", ex)
        return None, None

    assert num_components > 0, "Number of components should be larger than zero"

    MAX_IMAGES_IN_GRID = 54

    if isinstance(work_dir, pd.DataFrame) and 'distance' in work_dir.columns and 'component_id' in work_dir.columns \
        and 'files' in work_dir.columns and 'len' in work_dir.columns and len(work_dir):
        top_components = work_dir
    else:
        top_components = do_find_top_components(work_dir=work_dir, get_label_func=get_label_func, group_by=group_by,
                                                slice=slice, threshold=threshold, metric=metric, descending=descending,
                                                max_items=max_items,  min_items=min_items, keyword=keyword, save_path=save_path,
                                                return_stats=False, input_dir=input_dir,
                                                comp_type=comp_type, kwargs=kwargs)

    assert top_components is not None, f"Failed to find components with more than {min_items} images. Try to run fastdup.run() with turi_param='ccthreshold=0.9' namely to lower the threshold for grouping components"
    top_components = top_components.head(num_components)
    if 'debug_cc' in kwargs:
        print(top_components.head())
    save_artifacts = 'save_artifacts' in kwargs and kwargs['save_artifacts']

    assert top_components is not None and len(top_components), f"Failed to find components with more than {min_items} images. Try to run fastdup.run() with turi_param='ccthreshold=0.9' namely to lower the threshold for grouping components"
    comp_col = "component_id" if comp_type == "component" else "cluster"

    # iterate over the top components
    img_paths = []
    counter = 0
    for i,row in tqdm(top_components.iterrows(), total = len(top_components)):
        try:
            # find the component id
            component_id = row[comp_col]
            # find all the image filenames linked to this id
            if save_artifacts:
                pd.DataFrame({'filename':row['files']}).to_csv(os.path.join(save_path, f'component_{counter}_files.csv'))
            files = row['files'][:MAX_IMAGES_IN_GRID]
            if (len(files) == 0):
                print(f"Failed to find any files for component_id {component_id}");
                break

            tmp_images = []
            w,h = [], []
            val_array = []
            for f in files:
                #t,w1,h1 = load_one_image(f, input_dir, get_bounding_box_func)
                val_array.append([f, input_dir, get_bounding_box_func, kwargs])

            # trying to deal with the following runtime error:
            #An attempt has been made to start a new process before the
            #current process has finished its bootstrapping phase.
            try:
                with Pool() as pool:
                    result = pool.map(load_one_image, val_array)
            except RuntimeError as e:
                fastdup_capture_exception("visualize_top_components", e)
                result = []
                for i in val_array:
                    result.append(load_one_image(i))

            for t,x in enumerate(result):
                if x[0] is not None:
                    if save_artifacts:
                        if not os.path.exists(f'{save_path}/comp_{counter}/'):
                            os.mkdir(f'{save_path}/comp_{counter}')
                        cv2.imwrite(f'{save_path}/comp_{counter}/{os.path.basename(files[t])}', x[0])
                    tmp_images.append(x[0])
                    w.append(x[1])
                    h.append(x[2])
                

            assert len(tmp_images),"Failed to read all images"

            avg_h = int(np.mean(h))
            avg_w = int(np.mean(w))
            images = []
            for f in tmp_images:
                f = cv2.resize(f, (avg_w,avg_h))
                images.append(f)

            if len(images) <= 3:
                img, labels = generate_sprite_image(images,  len(images), '', None, h=avg_h, w=avg_w, alternative_width=len(images), max_width=max_width)
            else:
                img, labels = generate_sprite_image(images,  len(images), '', None, h=avg_h, w=avg_w, max_width=max_width)

            lazy_load = 'lazy_load' in kwargs and kwargs['lazy_load']
            subfolder = "" if not lazy_load else "images/"
            local_file = os.path.join(save_path, f'{subfolder}component_{counter}.jpg')
            cv2.imwrite(local_file, img)
            img_paths.append(local_file)
            counter+=1


        except ModuleNotFoundError as ex:
            print('Your system is missing some dependencies please install then with pip install:')
            fastdup_capture_exception("visualize_top_components", ex)

        except Exception as ex:
            print('Failed on component', i, ex)
            fastdup_capture_exception("visualize_top_components", ex)

    print(f'Finished OK. Components are stored as image files {save_path}/components_[index].jpg')
    return top_components.head(num_components), img_paths


def read_clusters_from_file(work_dir, get_label_func, kwargs):
    nrows = None
    if 'nrows' in kwargs:
        nrows = kwargs['nrows']

    if isinstance(work_dir, str):
        if os.path.isdir(work_dir):
            work_dir = os.path.join(work_dir, FILENAME_KMEANS_ASSIGNMENTS)
        assert os.path.exists(work_dir), f'Failed to find work_dir {work_dir}'

        df = pd.read_csv(work_dir)
    elif isinstance(work_dir, pd.DataFrame):
        assert "filename" in df.columns, "Failed to find filename in dataframe columns"
        assert "cluster" in df.columns
        assert "distance" in df.columns
        df = work_dir
        assert len(df), f"Failed to read dataframe from {work_dir} or empty dataframe"

    if nrows is not None:
        df = df.head(nrows)

    if get_label_func is not None:
        df = find_label(get_label_func, df, 'filename', 'label')

    return df


def read_components_from_file(work_dir, get_label_func, kwargs):
    if isinstance(work_dir, str):
        assert os.path.exists(work_dir), 'Working directory work_dir does not exist'
        assert os.path.exists(os.path.join(work_dir, 'connected_components.csv')), "Failed to find fastdup output file. Please run using fastdup.run(..) to generate this file."
        assert os.path.exists(os.path.join(work_dir, 'atrain_features.dat.csv')), "Failed to find fastdup output file. Please run using fastdup.run(..) to generate this file."
    elif isinstance(work_dir, pd.DataFrame):
        assert len(work_dir), "Empty dataframe"
        assert 'input_dir' in kwargs and os.path.exists(kwargs['input_dir']), "Failed to find fastdup work_dir"
    else:
        assert False, f"Unknown work_dir type {type(work_dir)}"

    nrows = None
    if len(kwargs) and 'nrows' in kwargs:
        nrows = kwargs['nrows']

    debug_cc = kwargs is not None and "debug_cc" in kwargs and kwargs["debug_cc"]

    # read fastdup connected components, for each image id we get component id
    if isinstance(work_dir, str):
        components = pd.read_csv(os.path.join(work_dir, FILENAME_CONNECTED_COMPONENTS), nrows=nrows)
        if 'min_distance' not in components.columns:
            if kwargs is not None and "squeeze_cc" in kwargs:
                components["min_distance"] = 0
    elif isinstance(work_dir, pd.DataFrame):
        components = work_dir
        if nrows is not None:
            components = components.head(nrows)
    else:
        assert False, "Unknown work_dir type"

    if debug_cc:
        print(components.head())


    local_dir = work_dir if isinstance(work_dir, str) else kwargs['input_dir']
    local_filenames = os.path.join(local_dir, 'atrain_' + FILENAME_IMAGE_LIST)
    assert os.path.exists(local_filenames), "Error: Failed to find fastdup input file " + local_filenames + ". Please run using fastdup.run(..) to generate this file."
    filenames = pd.read_csv(local_filenames, nrows=nrows)
    if (len(components) != len(filenames)):
        if kwargs is not None and "squeeze_cc" in kwargs:
            pass
        else:
            assert(False), f"Error: number of rows in components file {work_dir}/connected_components.csv and number of rows in image file {work_dir}/atrain_features.dat.csv are not equal. This may occur if multiple runs where done on the same working folder overriding those files. Please rerun on a clen folder."

    assert 'filename' in filenames.columns, f"Error: Failed to find filename column in {work_dir}/atrain_features.dat.csv"

    if kwargs is not None and "squeeze_cc" in kwargs:
        components = components.merge(filenames, left_on="component_id", right_on="index", how="left")
    # now join the two tables to get both id and image name
    else:
        components['filename'] = filenames['filename']
    
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

def do_find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None, threshold=None, metric=None,
                           descending=True, min_items=None, max_items = None, keyword=None, return_stats=False, save_path=None,
                           comp_type="component", input_dir=None, kwargs=None):
    '''
    Function to find the largest components of duplicate images

    Args:
        work_dir (str): working directory where fastdup.run was run, alternatively a pd.DataFrame with the output of connected_components.csv

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): 'visual' or 'label'

        slice (str): optional label names or list of label names to slice the dataframe

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
    elif comp_type == "cluster":
        components = read_clusters_from_file(work_dir, get_label_func, kwargs)
        comp_col = "cluster"
        distance_col = "distance"
    else:
        assert False, f"Wrong component type {comp_type}"

    assert components is not None and len(components), f"Failed to read components file {work_dir} or empty dataframe read"

    if metric is not None:
        cols_to_use = ['index', metric]
        if metric != 'size':
            cols_to_use = ['index', 'width', 'height']
        local_dir = work_dir if isinstance(work_dir, str) else input_dir
        assert os.path.exists(os.path.join(local_dir, 'atrain_stats.csv')), f"Error: Failed to find stats file {os.path.join(local_dir)}/atrain_stats.csv. Please run using fastdup.run(..) to generate this file."
        stats = pd.read_csv(os.path.join(local_dir, 'atrain_stats.csv'), usecols=cols_to_use)

        if metric == 'size':
            stats['size'] = stats.apply(lambda x: x['width']*x['height'], axis=1)
        if len(stats) != len(components) and 'squeeze_cc' in kwargs:
            components = components.merge(stats, left_on='__id', right_on='index', how='left')
        else:
            components[metric] = stats[metric]


    # find the components that have the largest number of images included


    if (get_label_func is not None):
        assert 'label' in components.columns, "Failed to find label column in components dataframe"
        components = slice_df(components, slice, kwargs)
        if 'path' in group_by:
            components['path'] = components['filename'].apply(lambda x: os.path.dirname(x))

        if group_by == 'visual':
            top_labels = components.groupby(comp_col)['label'].apply(list)
            top_files = components.groupby(comp_col)['filename'].apply(list)
            dict_cols = {'files':top_files, 'label':top_labels}

            if kwargs and 'is_video' in kwargs:
                top_dirs = components.groupby(comp_col)['dirname'].apply(list)
                dict_cols['video'] = top_dirs

            #if threshold is not None or metric is not None
            if distance_col in components.columns:
                distance = components.groupby(comp_col)[distance_col].apply(np.min)
                dict_cols['distance'] = distance
            if metric is not None:
                top_metric = components.groupby(comp_col)[metric].apply(np.mean)
                dict_cols[metric] = top_metric

            comps = pd.DataFrame(dict_cols).reset_index()

        elif group_by == 'label':
            is_list = isinstance(components['label'].values[0], list)
            if is_list:
                 components = components.explode(column='label', ignore_index=True).reset_index()

            top_files = components.groupby('label')['filename'].apply(list)
            top_components = components.groupby('label')[comp_col].apply(list)
            dict_cols = {'files':top_files, comp_col:top_components}

            if kwargs and 'is_video' in kwargs:
                top_dirs = components.groupby('label')['dirname'].apply(list)
                dict_cols['video'] = top_dirs

            #if threshold is not None or metric is not None:
            if distance_col in components.columns:
                distance = components.groupby('label')[distance_col].apply(np.min)
                dict_cols['distance'] = distance
            if metric is not None:
                top_metric = components.groupby('label')[metric].apply(np.mean)
                dict_cols[metric] = top_metric
            comps = pd.DataFrame(dict_cols).reset_index()
        else:
            assert(False), "group_by should be visual or label, got " + group_by

    else:

        top_components = components.groupby(comp_col)['filename'].apply(list)
        if 'debug_cc' in kwargs:
            print(top_components.head())

        dict_cols = {'files':top_components}
        #if threshold is not None or metric is not None:

        if distance_col in components.columns:
            distance = components.groupby(comp_col)[distance_col].apply(np.min)
            dict_cols['distance'] = distance
        if metric is not None:
            top_metric = components.groupby(comp_col)[metric].apply(np.mean)
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

    stat_info = None
    if return_stats:
        stat_info = get_stats_df(comps, comps, 'len', save_path, max_width=None, input_dir=input_dir, kwargs=kwargs)

        # in case labels are list of lists, namely list of attributes per image, flatten the list
    if 'label' in comps.columns:
        try:
            print(comps['label'].values[0][0])
            if isinstance(comps['label'].values[0][0], list):
                comps['label'] = comps['label'].apply(lambda x: [item for sublist in x for item in sublist])
        except Exception as ex:
            print('Failed to flatten labels', ex)
            fastdup_capture_exception("find_top_components", e)
            pass


    if metric is None:
        comps = comps.sort_values('len', ascending=not descending)
    else:
        comps = comps.sort_values(metric, ascending=not descending)

    if threshold is not None:
        comps = comps[comps['distance'] > threshold]

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
        local_dir = work_dir if isinstance(work_dir, str) else input_dir
        if comp_type == "component":
            comps.to_pickle(f'{local_dir}/{FILENAME_TOP_COMPONENTS}')
        else:
            comps.to_pickle(f'{local_dir}/{FILENAME_TOP_CLUSTERS}')

    if not return_stats:
        return comps
    else:
        return comps, stat_info


def do_create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 group_by='visual', slice=None, max_width=None, max_items=None, min_items=None,
                                 get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_info_func=None,
                                 threshold=None ,metric=None, descending=True, keyword=None, comp_type="component", input_dir=None,
                                 kwargs=None):
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

        get_bounding_box_func (callable): optional function to get bounding box of an image and add them to the report

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

    
    if num_images > 1000 and not lazy_load:
        print("Warning: When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    assert num_images >= 1, "Please select one or more images"
    assert group_by == 'label' or group_by == 'visual', "Allowed values for group_by=[visual|label], got " + group_by
    if group_by == 'label':
        assert get_label_func is not None, "missing get_label_func, when grouping by labels need to set get_label_func"
    assert comp_type in ['component','cluster']
    num_items_title = 'num_images' if 'is_video' not in kwargs else 'num_videos'

    start = time.time()

    kwargs['lazy_load'] = lazy_load
    ret = visualize_top_components(work_dir, save_path, num_images,
                                                get_label_func, group_by, slice,
                                                get_bounding_box_func, max_width, threshold, metric,
                                                descending, max_items, min_items, keyword,
                                                return_stats=False, comp_type=comp_type, input_dir=input_dir, kwargs=kwargs)
    if ret is None:
        return None
    subdf, img_paths = ret
    if subdf is None or len(img_paths) == 0:
        return None

    assert len(subdf) == len(img_paths), "Number of components and number of images do not match"

    import fastdup.html_writer
    save_artifacts= 'save_artifacts' in kwargs and kwargs['save_artifacts']

    comp_col = "component_id" if comp_type == "component" else "cluster"

    cols_dict = {comp_col:subdf[comp_col].values,
                 num_items_title:subdf['len'].apply(lambda x: "{:,}".format(x)).values}
    if 'distance' in subdf.columns:
        cols_dict['distance'] = subdf['distance'].values
    if 'label' in subdf.columns:
        cols_dict['label'] = subdf['label'].values
    elif 'is_video' in kwargs:
        cols_dict['num_images'] = subdf['orig_len'].apply(lambda x: "{:,}".format(x)).values
        subdf['label'] = subdf['video']

    if metric in subdf.columns:
        cols_dict[metric] = subdf[metric].apply(lambda x: round(x,2)).values

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
            info_list.append(info_df.to_html(escape=True, header=False).replace('\n',''))
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
            info_list.append(info_df.to_html(escape=True, header=False).replace('\n',''))
        info_df.to_csv(f'{save_path}/component_{counter}_df.csv')
        counter += 1

    ret = pd.DataFrame({'info': info_list})

    if 'label' in subdf.columns:
        if group_by == 'visual':
            labels_table = []
            counter = 0
            for i,row in subdf.iterrows():
                labels = list(row['label'])
                if save_artifacts:
                    pd.DataFrame({'label':labels}).to_csv(os.path.join(save_path, f"component_{counter}_labels.csv"))
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
                    counts_df.to_csv(f'{save_path}/counts_{counter}.csv')
                counts_df = counts_df.head(lencount).to_html(escape=False).replace('\n','')
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
                if save_artifacts:
                    counts_df.to_csv(f'{save_path}/counts_{counter}.csv')
                counts_df = counts_df.head(lencount).to_html(escape=False).replace('\n','')
                comp_table.append(counts_df)
                counter+=1
            ret.insert(0, 'components', comp_table)

    if not lazy_load:
        ret.insert(0, 'image', [imageformatter(x, max_width) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        ret.insert(0, 'image', img_paths2)

    out_file = os.path.join(save_path, 'components.html')
    columns = ['info','image']
    if 'label' in subdf.columns:
        if group_by == 'visual':
            columns.append('label')
        elif group_by == 'label':
            columns.append('components')

    if comp_type == "component":
        if 'is_video' in kwargs:
            title = 'Fastdup tool - Video Components Report'
        else:
            title = 'Fastdup Tool - Components Report'
    else:
        title = "Fastdup Tool - KMeans Cluster Report"

    subtitle = None
    if slice is not None:
        subtitle = "slice: " + str(slice)
    if metric is not None:
        subtitle = "Sorted by " + metric + " descending" if descending else "Sorted by " + metric + " ascending"

    fastdup.html_writer.write_to_html_file(ret[columns], title, out_file, "", subtitle)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    if comp_type == "component":
        print("Stored components visual view in ", os.path.join(out_file))
    else:
        print("Stored KMeans clusters visual view in ", os.path.join(out_file))

    clean_images(lazy_load or save_artifacts or (threshold is not None), img_paths, "create_components_gallery")
    print('Execution time in seconds', round(time.time() - start, 1))
    return 0

def get_stats_df(df, subdf, metric, save_path, max_width=None, input_dir=None, kwargs=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        fastdup_capture_exception("get_stats_df", e, True)
        print("Warning: failed to import matplotlib, aggregate stats plot is not generated. Please pip install matplotlib if you like to view aggregate stats plots.")
        return ""

    stats_info = df[metric].describe().to_frame()

    plt.rcParams['axes.xmargin'] = 0
    minx = df[metric].min()
    maxx = df[metric].max()
    minx2 = subdf[metric].min()
    maxx2 = subdf[metric].max()
    line = None
    if minx2 > minx:
        line = minx2
    elif maxx2 < maxx:
        line = maxx2

    xlabel = None
    if metric in ['mean','max','stdv','min']:
        xlabel = metric +  ' (Color 0-255, larger is brighter)'
    elif metric == 'blur':
        xlabel = 'Blur (lower is blurry, higher is sharper)'
    elif metric == 'unique':
        xlabel = 'Number of unique colors 0-255, higher is better'
    elif metric == 'size':
        xlabel = 'Size (number of pixels - width x height)'

    ax = df[metric].plot.hist(bins=100, alpha=1, title=metric, fontsize=15, xlim=(minx,maxx), xlabel=xlabel)
    if line is not None:
        plt.axvline(line, color='r', linestyle='dashed', linewidth=2)
    fig = ax.get_figure()

    local_fig = f"{save_path}/stats.jpg"
    fig.savefig(local_fig ,dpi=100)
    try:
        img = fastdup_imread(local_fig, input_dir, kwargs)
    except Exception as ex:
        fastdup_capture_exception("get_stats_df", ex)
        return ""

    ret = pd.DataFrame({'stats':[stats_info.to_html(escape=False,index=True).replace('\n','')], 'image':[imageformatter(img, max_width)]})
    return ret.to_html(escape=False,index=False).replace('\n','')

def do_create_stats_gallery(stats_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            metric='blur', slice=None, max_width=None, descending=False, get_bounding_box_func=None,
                            get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None,
                            min_items=2, max_items=None, **kwargs):
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

        get_bounding_box_func (callable): Optional parameter to allow adding bounding box plot to the displayed image.
         This is a function the user implements that gets the full file path and returns a bounding box or an empty list if not available.

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the image file name. This is a function the user implements that gets the full file path and returns a new file name.

        get_extra_col_func (callable): Optional parameter to allow adding extra column to the report.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'
       
        min_items (int): Optional parameter to select components with min_items or more

        max_items (int): Optional parameter to limit the number of items displayed

     
     '''


    img_paths = []
    work_dir = None
    if isinstance(stats_file, pd.DataFrame):
        df = stats_file
    else:
        assert os.path.exists(stats_file), "Error: failed to find fastdup input file " + stats_file + " . Please run using fastdup.run(..) to generate this file."
        work_dir = os.path.dirname(os.path.abspath(stats_file))
        df = pd.read_csv(stats_file)
    assert len(df), "Failed to read stats file " + stats_file

    df = find_label(get_label_func, df, 'filename', 'label')

    if slice is not None:
        if isinstance(slice, str):
            if len(df) < 1000000:
                assert slice in df['label'].unique(), f"Failed to find {slice} in the list of available labels, can not visualize this label class. Example labels {df['label'].unique()[:10]}"
            df = df[df['label'] == slice]
        elif isinstance(slice, list):
            df = df[df['label'].isin(slice)]
        else:
            assert False, f"slice must be a string or list got {type(slice)}"

    if metric is not None and metric == 'size':
        df['size'] = df['width'] * df['height']

    assert metric in df.columns, "Failed to find metric " + metric + " in " + str(df.columns)

    if metric in ['unique', 'width', 'height', 'size']:
        df = df[df[metric] > DEFUALT_METRIC_ZERO]
    elif metric in ['blur', 'mean', 'min', 'max', 'stdv']:
        df = df[df[metric] != DEFAULT_METRIC_MINUS_ONE]

    subdf = df.sort_values(metric, ascending=not descending).head(num_images)
    stat_info = get_stats_df(df, subdf, metric, save_path, max_width)
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        try:
            filename = lookup_filename(row['filename'], work_dir)
            img = cv2.imread(filename)
            img = plot_bounding_box(img, get_bounding_box_func, filename)
            img = my_resize(img, max_width)

            imgpath = calc_image_path(lazy_load, save_path, filename)
            cv2.imwrite(imgpath, img)
            assert os.path.exists(imgpath), "Failed to save img to " + imgpath

        except Exception as ex:
            traceback.print_exc()
            print("Failed to generate viz for images", filename, ex)
            imgpath = None
        img_paths.append(imgpath)

    import fastdup.html_writer
    if not lazy_load:
        subdf.insert(0, 'Image', [imageformatter(x, max_width) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        subdf.insert(0, 'Image', img_paths2)

    cols = [metric,'Image','filename']

    if callable(get_extra_col_func):
        subdf['extra'] = subdf['filename'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')

    if callable(get_reformat_filename_func):
        subdf['filename'] = subdf['filename'].apply(lambda x: get_reformat_filename_func(x))

    out_file = os.path.join(save_path, metric + '.html')
    title = 'Fastdup Tool - ' + metric + ' Image Report'
    if slice is not None:
        title += ", " + str(slice)

    if metric == 'size':
        cols.append('width')
        cols.append('height')

    if 'label' in df.columns:
        cols.append('label')

    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file, stat_info)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored " + metric + " statistics view in ", os.path.join(out_file))
    clean_images(lazy_load, img_paths, "create_stats_gallery")

def do_create_similarity_gallery(similarity_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 slice=None, max_width=None, descending=False, get_bounding_box_func =None,
                                 get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None, min_items=2, 
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

        get_bounding_box_func (callable): Optional parameter to allow adding bounding box to the image. This is a function the user implements that gets the full file path and returns a bounding box or an empty list if not available.

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the filename before displaying it in the report. This is a function the user implements that gets the full file path and returns a string with the reformatted filename.

        get_extra_col_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        min_items (int): Minimal number of items in the similarity group (optional).

        max_items (int): Maximal number of items in the similarity group (optional).


    Returns:
        ret (pd.DataFrame): Dataframe with the image statistics
    '''


    from fastdup import generate_sprite_image
    img_paths = []
    img_paths2 = []
    info0 = []
    info = []
    label_score = []
    lengths = []

    work_dir = None
    if isinstance(similarity_file, pd.DataFrame):
        df = similarity_file
    elif isinstance(similarity_file, str):
        work_dir = os.path.dirname(os.path.abspath(similarity_file))
        assert os.path.exists(similarity_file), "Failed to find similarity file " + similarity_file + " . Please run using fastdup.run(..) to generate this file."
        df = pd.read_csv(similarity_file)
    else:
        assert False, f"Wrong type {type(work_dir)}"
    
    assert len(df), "Failed to read similarity file " + similarity_file

    if get_label_func is not None:
        df = find_label(get_label_func, df, 'from', 'label')
        df = find_label(get_label_func, df, 'to', 'label2')

        if slice != 'label_score':
            df = slice_df(df, slice)
            if df is None:
                return 1

    df = df.sort_values(['from','distance'], ascending= not descending)
    if 'label' in df.columns:
        top_labels = df.groupby('from')['label2'].apply(list)

    tos = df.groupby('from')['to'].apply(list)
    distances = df.groupby('from')['distance'].apply(list)

    if 'label' in df.columns:
        subdf = pd.DataFrame({'to':tos, 'label':top_labels,'distance':distances}).reset_index()
    else:
        subdf = pd.DataFrame({'to':tos, 'distance':distances}).reset_index()

    info_df = None

    if slice is None or slice != 'label_score':
        df2 = subdf.copy()
        subdf = subdf.head(num_images)
        stat_info = None
    else:
        for i, row in tqdm(subdf.iterrows(), total=len(subdf)):
            filename = lookup_filename(row['from'], work_dir)
            label = None
            if isinstance(get_label_func, dict):
                label = get_label_func.get(filename, MISSING_LABEL)
            elif callable(get_label_func):
                label = get_label_func(filename)
            else:
                assert False, "not implemented yet"
            similar = [x==label for x in list(row['label'])]
            similar = 100.0*sum(similar)/(1.0*len(row['label']))
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
        stat_info = get_stats_df(df2, subdf, metric='score', save_path=save_path, max_width=max_width, kwargs=kwargs)

    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        try:
            label = None
            filename = lookup_filename(row['from'], work_dir)
            if callable(get_label_func):
                label = get_label_func(filename)
            elif isinstance(get_label_func, dict):
                label = get_label_func.get(filename, MISSING_LABEL)
            elif isinstance(get_label_func, str):
                assert False, "Not implemented yet"

            if callable(get_reformat_filename_func):
                new_filename = get_reformat_filename_func(filename)
            else:
                new_filename = filename

            if label is not None:
                info0_df = pd.DataFrame({'label':[label],'from':[new_filename]}).T
            else:
                info0_df = pd.DataFrame({'from':[new_filename]}).T

            info0.append(info0_df.to_html(header=False,escape=False).replace('\n',''))


            img = fastdup_imread(filename, input_dir=input_dir, kwargs=kwargs)
            img = plot_bounding_box(img, get_bounding_box_func, filename)
            img = my_resize(img, max_width)

            imgpath = calc_image_path(lazy_load, save_path, filename)
            cv2.imwrite(imgpath, img)
            assert os.path.exists(imgpath), "Failed to save img to " + imgpath

            MAX_IMAGES = 10
            imgs = row['to'][:MAX_IMAGES]
            distances = row['distance'][:MAX_IMAGES]
            imgpath2 = f"{save_path}/to_image_{i}.jpg"
            info_df = pd.DataFrame({'distance':distances, 'to':[lookup_filename(im, work_dir) for im in imgs]})


            if callable(get_reformat_filename_func):
                info_df['to'] = info_df['to'].apply(lambda x: get_reformat_filename_func(x))

            if 'label2' in df.columns:
                info_df['label'] = row['label'][:MAX_IMAGES]
            info_df = info_df.sort_values('distance',ascending=False)
            info.append(info_df.to_html(escape=False).replace('\n',''))

            h = max_width if max_width is not None else 0
            w = h
            generate_sprite_image(imgs, min(len(imgs), MAX_IMAGES), save_path, get_label_func, h, w, imgpath2, min(len(imgs),MAX_IMAGES), max_width=max_width)
            assert os.path.exists(imgpath2)

        except Exception as ex:
            fastdup_capture_exception("create_similarity_gallery", ex)
            print("Failed to generate viz for images", filename, ex)
            imgpath = None
            imgpath2 = None

        img_paths.append(imgpath)
        img_paths2.append(imgpath2)

    import fastdup.html_writer
    if not lazy_load:
        subdf.insert(0, 'Image', [imageformatter(x, max_width) for x in img_paths])
        subdf.insert(0, 'Similar', [imageformatter(x, None) for x in img_paths2])
    else:
        img_paths3 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        img_paths4 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths2]
        subdf.insert(0, 'Image', img_paths3)
        subdf.insert(0, 'Similar', img_paths4)

    subdf['info_to'] = info
    subdf['info_from'] = info0

    out_file = os.path.join(save_path, 'topk_similarity.html')
    title = 'Fastdup Tool - Similarity Image Report'
    if slice is not None:
        title += ", " + str(slice)

    cols = ['info_from','info_to', 'Image','Similar']
    if slice is not None and slice == 'label_score':
        cols = ['score'] + cols
    if callable(get_extra_col_func):
        subdf['extra'] = subdf['from'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')


    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file, stat_info)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored similar images view in ", os.path.join(out_file))
    clean_images(lazy_load, img_paths, "create_similarity_gallery")

    return df2


def do_create_aspect_ratio_gallery(stats_file, save_path, get_label_func=None, max_width=None, num_images=0, slice=None,
                                   get_reformat_filename_func=None, input_dir=None, **kwargs):
    '''
    Create an html gallery of images with aspect ratio
     stats_file:
     save_path:
     get_label_func:
     max_width:
     num_images:
     slice:
     get_reformat_filename_func:
    Returns:
    '''

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        fastdup_capture_exception("create_aspect_ratio_gallery", e)
        print("Error: failed to import matplotlib pip package, can not generate aspect ratio plot without it.")
        return None

    from .html_writer import write_to_html_file
    from .image import imageformatter

    work_dir = None
    if isinstance(stats_file, pd.DataFrame):
        df = stats_file
    else:
        work_dir = os.path.dirname(os.path.abspath(stats_file))
        df = pd.read_csv(stats_file)
    assert len(df), "Zero rows found in " + stats_file

    if num_images is not None and num_images>0:
        df = df.head(num_images)

    if get_label_func is not None:
        df = find_label(get_label_func, df, 'filename', 'label')
        if slice is not None:
            df = df[df['label'] == slice]

    assert len(df), "Empty stats file " + stats_file
    df = df[df['width'] > DEFUALT_METRIC_ZERO]
    df = df[df['height'] > DEFUALT_METRIC_ZERO]

    shape = df[['width','height']].to_numpy()

    max_width_ = np.max(shape[:,0])
    max_height_ = np.max(shape[:,1])
    ret = shape[:,0]/shape[:,1]
    max_dim = max(max_height_, max_width_)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].scatter(shape[:,0], shape[:, 1])
    axs[0].plot(range(0, max_dim), range(0, max_dim), 'k')
    axs[0].set_ylabel('Width', fontsize=13)
    axs[0].set_xlabel('Height', fontsize=13)
    axs[0].grid()
    axs[0].set_title('Scatter of images shapes', fontsize=18)
    axs[0].set_xlim([0, max_width_])
    axs[0].set_ylim([0, max_height_])

    axs[1].hist(shape[:, 0]/shape[:, 1], bins=100)
    axs[1].grid()
    axs[1].set_xlabel('Aspect Ratio', fontsize=13)
    axs[1].set_ylabel('Frequency', fontsize=13)
    axs[1].set_title('Histogram of aspect ration for images', fontsize=18)
    axs[1].set_xlim([0, 2])

    local_fig = f"{save_path}/aspect_ratio.jpg"
    fig.savefig(local_fig ,dpi=100)
    img = cv2.imread(local_fig)

    max_width_img = df[df['width'] == max_width_]['filename'].values[0]
    max_width_img = lookup_filename(max_width_img, work_dir)
    max_height_img = df[df['height'] == max_height_]['filename'].values[0]
    max_height_img = lookup_filename(max_height_img, work_dir)

    try:
        img_max_width = fastdup_imread(max_width_img, input_dir, kwargs)
        img_max_height = fastdup_imread(max_height_img, input_dir, kwargs)
        if max_width is not None:
            img_max_width = my_resize(img_max_width, max_width)
            img_max_height = my_resize(img_max_height, max_width)
    except Exception as ex:
        print("Failed to read images ", max_width_img, max_height_img)
        fastdup_capture_exception("aspect ratio", ex)
        img_max_width = None
        img_max_height = None

    if get_reformat_filename_func is not None:
        max_width_img = get_reformat_filename_func(max_width_img)
        max_height_img = get_reformat_filename_func(max_height_img)

    aspect_ratio_info = pd.DataFrame({'Number of images':[len(df)],
                                      'Avg width':[np.mean(shape[0, :])],
                                      'Avg height':[np.mean(shape[1, :])],
                                      'Max width': [max_width_],
                                      'Max height': [max_height_],
                                      'Plot':[imageformatter(img, None)],
                                      'Max width Image<br>' + max_width_img+ f'<br>width: {max_width_}':[imageformatter(img_max_width, max_width)],
                                      'Max height Image<br>' + max_height_img + f'<br>height: {max_height_}':[imageformatter(img_max_height, max_width)]
                                      }).T

    ret = pd.DataFrame({'stats':[aspect_ratio_info.to_html(escape=False,index=False).replace('\n','')]})

    title = 'Fastdup tool - Aspect ratio report'
    out_file = os.path.join(save_path, 'aspect_ratio.html')
    print(f'Saved aspect ratio report to {out_file}')
    return write_to_html_file(ret, title, out_file, None)
