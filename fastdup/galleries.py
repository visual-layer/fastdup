
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import pandas as pd
import cv2
import numpy as np
import traceback
from fastdup.image import plot_bounding_box, my_resize, get_type, imageformatter, create_triplet_img, fastdup_imread
from fastdup.definitions import *
try:
    from tqdm import tqdm
except:
    tqdm = (lambda x: x)


def get_label(filename, get_label_func):
    ret = filename
    try:
        ret += "<br>" + "Label: " + get_label_func(filename)
    except Exception as ex:
        ret += "<br>Failed to get label for " + filename + " with error " + ex
    return ret



def slice_df(df, slice):
    if slice is not None:
        if isinstance(slice, str):
            # cover the case labels are string or lists of strings
            labels = df['label'].values
            is_list = isinstance(labels[0], list)
            if is_list:
                labels = [item for sublist in labels for item in sublist]
            unique, counts = np.unique(np.array(labels), return_counts=True)
            if slice not in unique:
                print(f"Failed to find {slice} in the list of available labels, can not visualize this label class")
                print("Example labels", df['label'].values[:10])
                return None
            if not is_list:
                df = df[df['label'] == slice]
            else:
                df = df[df['label'].apply(lambda x: slice in x)]
        elif isinstance(slice, list):
            if isinstance(df['label'].values[0], list):
                df = df[df['label'].apply(lambda x: len(set(x)&set(slice)) > 0)]
            else:
                df = df[df['label'].isin(slice)]
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
                                 get_extra_col_func=None, input_dir=None, work_dir=None):
    '''

    Function to create and display a gallery of images computed by the similarity metrics

    Parameters:
        similarity_file (str): csv file with the computed similarities by the fastdup tool, alternatively it can be a pandas dataframe with the computed similarities.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. An example list is [[100,100,200,200]] which contains a single bounding box.

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the gallery.

        input_dir (str): Optional parameter to allow reading images from a different path, or from webdataset tar files which are found on a different path

    Returns:
        ret (int): 0 if success, 1 if failed

    '''


    img_paths = []
    work_dir = None
    if isinstance(similarity_file, pd.DataFrame):
        df = similarity_file
    else:
        df = pd.read_csv(similarity_file)
        work_dir = os.path.dirname(similarity_file)
    assert len(df), "Failed to read similarity file"

    if slice is not None and callable(get_label_func):
        df['label'] = df['from'].apply(lambda x: get_label_func(x))
        if isinstance(slice, str):
            if slice == "diff":
                df['label2'] = df['to'].apply(lambda x: get_label_func(x))
                df = df[df['label'] != df['label2']]
            elif slice == "same":
                df['label2'] = df['to'].apply(lambda x: get_label_func(x))
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

    subdf = df.head(num_images) if descending else df.tail(num_images)
    subdf = subdf.reset_index()
    indexes = []
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        impath1, impath2, dist, ptype = extract_filenames(row, work_dir)
        if impath1 + '_' + impath2 in sets:
            continue
        try:
            img, imgpath = create_triplet_img(impath1, impath2, ptype, dist, save_path, get_bounding_box_func, input_dir)
            sets[impath1 +'_' + impath2] = True
            sets[impath2 +'_' + impath1] = True
            indexes.append(i)

        except Exception as ex:
            traceback.print_exc()
            print("Failed to generate viz for images", impath1, impath2, ex)
            imgpath = None
        img_paths.append(imgpath)

    subdf = subdf.iloc[indexes]
    import fastdup.html_writer

    if not lazy_load:
        subdf.insert(0, 'Image', [imageformatter(x, max_width) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.basename(x) + "\" loading=\"lazy\">" for x in img_paths]
        subdf.insert(0, 'Image', img_paths2)
    out_file = os.path.join(save_path, 'similarity.html')
    if get_label_func is not None and callable(get_label_func):
        subdf.insert(2, 'From', subdf['from'].apply(lambda x: get_label(x, get_label_func)))
        subdf.insert(3, 'To', subdf['to'].apply(lambda x: get_label(x, get_label_func)))
    else:
        subdf = subdf.rename(columns={'from':'From', 'to':'To'}, inplace=False)
    subdf = subdf.rename(columns={'distance':'Distance'}, inplace=False)
    fields = ['Image', 'Distance', 'From', 'To']

    if callable(get_extra_col_func):
        subdf['extra'] = subdf['From'].apply(lambda x: get_extra_col_func(x))
        subdf['extra2'] = subdf['To'].apply(lambda x: get_extra_col_func(x))
        fields.append('extra')
        fields.append('extra2')

    if get_reformat_filename_func is not None and callable(get_reformat_filename_func):
        subdf['From'] = subdf['From'].apply(lambda x: get_reformat_filename_func(x))
        subdf['To'] = subdf['To'].apply(lambda x: get_reformat_filename_func(x))

    title = 'Fastdup Tool - Similarity Report'
    if slice is not None:
        if slice == "diff":
            title += ", of different classes"
        else:
            title += ", for label " + str(slice)
    fastdup.html_writer.write_to_html_file(subdf[fields], title, out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file
    print("Stored similarity visual view in ", out_file)

    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as e:
                print("Warning, failed to remove image file ", i, " with error ", e)

    return 0


def do_create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None, max_width=None, get_bounding_box_func=None, get_reformat_filename_func=None,
                               get_extra_col_func=None, input_dir= None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        outliers_file (str): csv file with the computed outliers by the fastdup tool. Altenriously, this can be a pandas dataframe with the computed outliers.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

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
    if isinstance(outliers_file, pd.DataFrame):
        df = outliers_file
    else:
        df = pd.read_csv(outliers_file)
        work_dir = os.path.dirname(outliers_file)
    assert len(df), "Failed to read outliers file " + outliers_file

    if (how == 'all'):
        dups_file = os.path.join(os.path.dirname(outliers_file), FILENAME_SIMILARITY)
        if not os.path.exists(dups_file):
            print('Failed to find input file ', dups_file, ' which is needed for computing how=all similarities')

        dups = pd.read_csv(dups_file)
        assert len(dups), "Error: Failed to locate similarity file file " + dups_file
        dups = dups[dups['distance'] > dups['distance'].mean()]
        assert len(dups), "Did not find any images with similarity more than the mean {dups['distance'].mean()}"

        joined = df.merge(dups, on='from', how='left')
        joined = joined[pd.isnull(joined['distance_y'])]

        if (len(joined) == 0):
            print('Failed to find outlier images that are not included in the duplicates similarity files, run with how="one".')
            return 1

        subdf = joined.rename(columns={"distance_x": "distance", "to_x": "to"}).sort_values('distance', ascending=True)
    else:
        subdf = df.sort_values(by='distance', ascending=True)

    if callable(get_label_func):
        subdf['label'] = subdf['from'].apply(lambda x: get_label_func(x))
        subdf = slice_df(subdf, slice)
        if subdf is None:
            return 1

    subdf = subdf.drop_duplicates(subset='from').sort_values(by='distance', ascending=True).head(num_images)
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        impath1, impath2, dist, ptype = extract_filenames(row, work_dir)
        try:
            img = fastdup_imread(impath1, input_dir=input_dir)

            img = plot_bounding_box(img, get_bounding_box_func, impath1)
            img = my_resize(img, max_width=max_width)

            #consider saving second image as well!
            #make sure image file is unique, so add also folder name into the imagefile
            imgpath = os.path.join(save_path, impath1.replace('/',''))
            p, ext = os.path.splitext(imgpath)
            if ext is not None and ext != '' and ext.lower() not in ['png','tiff','tif','jpeg','jpg','gif']:
                imgpath += ".jpg"

            cv2.imwrite(imgpath, img)
            assert os.path.exists(imgpath), "Failed to save img to " + imgpath

        except Exception as ex:
            traceback.print_exc()
            print("Failed to generate viz for images", impath1, impath2, ex)
            imgpath = None
        img_paths.append(imgpath)

    import fastdup.html_writer
    if not lazy_load:
        subdf.insert(0, 'Image', [imageformatter(x, max_width) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        subdf.insert(0, 'Image', img_paths2)

    if get_label_func is not None and callable(get_label_func):
        subdf.insert(2, 'Path', subdf['from'].apply(lambda x: get_label(x, get_label_func)))
        subdf = subdf.rename(columns={'distance':'Distance'}, inplace=False)
    else:
        subdf = subdf.rename(columns={'from':'Path', 'distance':'Distance'}, inplace=False)

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

    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored outliers visual view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)


    return 0

def visualize_top_components(work_dir, save_path, num_components, get_label_func=None, group_by='visual', slice=None,
                             get_bounding_box_func=None, max_width=None, threshold=None, metric=None, descending=True,
                             max_items = None, min_items=None, keyword=None, return_stats=True, comp_type="component", input_dir=None):
    '''
    Visualize the top connected components

    Args:
        work_dir (str): directory with the output of fastdup run

        save_path (str): directory to save the output to

        num_components (int): number of top components to plot

        get_label_func (callable): option function to get label for each image given image filename

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
        print('Your system is missing some depdencies, please pip install matplotlib matplotlib-inline torchvision')
        print(ex)
        return None, None

    assert num_components > 0, "Number of components should be larger than zero"

    MAX_IMAGES_IN_GRID = 49

    ret = do_find_top_components(work_dir=work_dir, get_label_func=get_label_func, group_by=group_by,
                                            slice=slice, threshold=threshold, metric=metric, descending=descending,
                                            max_items=max_items,  min_items=min_items, keyword=keyword, save_path=save_path, return_stats=return_stats,
                                            comp_type=comp_type)
    if not return_stats:
        top_components = ret
    else:
        top_components, stats_html = ret
    top_components = top_components.head(num_components)

    if (top_components is None or len(top_components) == 0):
        print('Failed to find top components, try to reduce grouping threshold by running with turi_param="cchreshold=0.8" where 0.8 is an exmple value.')
        return None, None

    comp_col = "component_id" if comp_type == "component" else "cluster"

    # iterate over the top components
    index = 0
    img_paths = []
    for i,row in tqdm(top_components.iterrows(), total = len(top_components)):
        try:
            # find the component id
            component_id = row[comp_col]
            # find all the image filenames linked to this id
            files = row['files'][:MAX_IMAGES_IN_GRID]
            if (len(files) == 0):
                break

            tmp_images = []
            w,h = [], []
            for f in files:
                try:
                    img = fastdup_imread(f, input_dir)
                    img = plot_bounding_box(img, get_bounding_box_func, f)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tmp_images.append(img)
                    w.append(img.shape[1])
                    h.append(img.shape[0])
                except Exception as ex:
                    print("Warning: Failed to load image ", f, "skipping image due to error", ex)

            if len(tmp_images) == 0:
                print("Failed to read all images")
                return None, None

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

            local_file = os.path.join(save_path, f'component_{i}.jpg')
            cv2.imwrite(local_file, img)
            img_paths.append(local_file)
            index+=1


        except ModuleNotFoundError as ex:
            print('Your system is missing some dependencies please install then with pip install: ', ex)
            traceback.print_exc()
            return None, None

        except Exception as ex:
            print('Failed on component', i, ex)
            traceback.print_exc()
            return None, None

    print(f'Finished OK. Components are stored as image files {save_path}/components_[index].jpg')
    return top_components.head(num_components), img_paths, stats_html


def read_clusters_from_file(work_dir):
    if isinstance(work_dir, str):
        if os.path.isdir(work_dir):
            work_dir = os.path.join(work_dir, FILENAME_KMEANS_ASSIGNMENTS)
        if not os.path.exists(work_dir):
            print('Failed to find work_dir {work_dir')
            return None

        df = pd.read_csv(work_dir)
    elif isinstance(work_dir, pd.DataFrame):
        assert "filename" in df.columns, "Failed to find filename in dataframe columns"
        assert "cluster" in df.columns
        assert "distance" in df.columns
        df = work_dir
        assert len(df), f"Failed to read dataframe from {work_dir} or empty dataframe"

    return df


def read_components_from_file(work_dir):
    assert os.path.exists(work_dir), 'Working directory work_dir does not exist'
    assert os.path.exists(os.path.join(work_dir, 'connected_components.csv')), "Failed to find fastdup output file"
    assert os.path.exists(os.path.join(work_dir, 'atrain_features.dat.csv')), "Failed to find fastdup output file"

    # read fastdup connected components, for each image id we get component id
    components = pd.read_csv(os.path.join(work_dir, FILENAME_CONNECTED_COMPONENTS))

    filenames = pd.read_csv(os.path.join(work_dir, 'atrain_features.dat.csv'))
    if (len(components) != len(filenames)):
        print(f"Error: number of rows in components file {work_dir}/connected_components.csv and number of rows in image file {work_dir}/atrain_features.dat.csv are not equal")
        print("This may occur if multiple runs where done on the same working folder overriding those files. Please rerun on a clen folder")
        return None
    # now join the two tables to get both id and image name
    components['filename'] = filenames['filename']

    return components


def do_find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None, threshold=None, metric=None,
                           descending=True, min_items=None, max_items = None, keyword=None, return_stats=False, save_path=None,
                           comp_type="component", input_dir=None):
    '''
    Function to find the largest components of duplicate images

    Args:
        work_dir (str): working directory where fastdup.run was run.

        get_label_func (callable): optional function to get label for each image

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

    if comp_type == "component":
        components = read_components_from_file(work_dir)
        comp_col = "component_id"
        distance_col = "min_distance"
    elif comp_type == "cluster":
        components = read_clusters_from_file(work_dir)
        comp_col = "cluster"
        distance_col = "distance"
    else:
        assert False, f"Wrong component type {comp_type}"

    if components is None or len(components) == 0:
        print(f"Failed to read components file {work_dir} or empty dataframe read")
        return None

    if metric is not None:
        stats = pd.read_csv(os.path.join(work_dir, 'atrain_stats.csv'))
        assert len(stats) == len(components), "Number of rows in stats file and number of rows in components file are not equal"
        if metric == 'size':
            stats['size'] = stats.apply(lambda x: x['width']*x['height'], axis=1)
        components[metric] = stats[metric]

    # find the components that have the largest number of images included
    if callable(get_label_func):
        components['label'] = components['filename'].apply(get_label_func)
        components = slice_df(components, slice)

        if 'path' in group_by:
            components['path'] = components['filename'].apply(lambda x: os.path.dirname(x))


        if group_by == 'visual':
            top_labels = components.groupby(comp_col)['label'].apply(list)
            top_files = components.groupby(comp_col)['filename'].apply(list)
            dict_cols = {'files':top_files, 'label':top_labels}
            #if threshold is not None or metric is not None:
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
            #if threshold is not None or metric is not None:
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
        dict_cols = {'files':top_components}
        #if threshold is not None or metric is not None:
        distance = components.groupby(comp_col)[distance_col].apply(np.min)
        dict_cols['distance'] = distance
        if metric is not None:
            top_metric = components.groupby(comp_col)[metric].apply(np.mean)
            dict_cols[metric] = top_metric
        comps = pd.DataFrame(dict_cols).reset_index()

    if len(comps) == 0:
        print("No components found")
        return None

    comps['len'] = comps['files'].apply(lambda x: len(x))
    stat_info = None
    if return_stats:
        stat_info = get_stats_df(comps, comps, 'len', save_path, max_width=None, input_dir=input_dir)

        # in case labels are list of lists, namely list of attributes per image, flatten the list
    if 'label' in comps.columns:
        try:
            print(comps['label'].values[0][0])
            if isinstance(comps['label'].values[0][0], list):
                comps['label'] = comps['label'].apply(lambda x: [item for sublist in x for item in sublist])
        except Exception as ex:
            print('Failed to flatten labels', ex)
            pass


    if metric is None:
        comps = comps.sort_values('len', ascending=not descending)
    else:
        comps = comps.sort_values(metric, ascending=not descending)

    if threshold is not None:
        comps = comps[comps['distance'] > threshold]

    if keyword is not None:
        assert callable(get_label_func), "keyword can only be used with a callable get_label_func"
        assert group_by == 'visual', "keyword can only be used with group_by=visual"
        comps = comps[comps['label'].apply(lambda x: sum([1 if keyword in y else 0 for y in x]) > 0)]
        if len(comps) == 0:
            print("Failed to find any components with label keyword " + keyword)
            return None

    if min_items is not None:
        assert min_items > 1, "min_items should be a positive integer larger than 1"
        comps = comps[comps['len'] >= min_items]
        if len(comps) == 0:
            print(f"Failed to find any components with {min_items} or more items, try lowering the min_items threshold")
            return None

    if max_items is not None:
        assert max_items > 1, "min_items should be a positive integer larger than 1"
        comps = comps[comps['len'] <= max_items]
        if len(comps) == 0:
            print(f"Failed to find any components with {max_items} or less items, try lowering the max_items threshold")
            return None
        else:
            comps = comps[comps['len'] > 1] # remove any singleton components

    if threshold is not None or metric is not None or keyword is not None:
        if comp_type == "component":
            comps.to_pickle(f'{work_dir}/{FILENAME_TOP_COMPONENTS}')
        else:
            comps.to_pickle(f'{work_dir}/{FILENAME_TOP_CLUSTERS}')

    if not return_stats:
        return comps
    else:
        return comps, stat_info


def do_create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 group_by='visual', slice=None, max_width=None, max_items=None, min_items=None,
                                 get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_info_func=None,
                                 threshold=None ,metric=None, descending=True, keyword=None, comp_type="component", input_dir=None):
    '''

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the image label. Image label can be a string or a list of strings.

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

     '''

    if num_images > 1000 and not lazy_load:
        print("When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")


    assert num_images >= 1, "Please select one or more images"
    assert group_by == 'label' or group_by == 'visual', "Allowed values for group_by=[visual|label], got " + group_by
    if group_by == 'label':
        assert callable(get_label_func), "missing get_label_func, when grouping by labels need to set get_label_func"
    assert comp_type in ['component','cluster']


    subdf, img_paths, stats_html = visualize_top_components(work_dir, save_path, num_images,
                                                get_label_func, group_by, slice,
                                                get_bounding_box_func, max_width, threshold, metric,
                                                descending, max_items, min_items, keyword,
                                                return_stats=True, comp_type=comp_type, input_dir=input_dir)
    if subdf is None or len(img_paths) == 0:
        return None

    assert len(subdf) == len(img_paths), "Number of components and number of images do not match"

    import fastdup.html_writer

    comp_col = "component_id" if comp_type == "component" else "cluster"

    cols_dict = {comp_col:subdf[comp_col].values,
                 'num_images':subdf['len'].apply(lambda x: "{:,}".format(x)).values}
    if 'distance' in subdf.columns:
        cols_dict['distance'] = subdf['distance'].values
    if 'label' in subdf.columns:
    	cols_dict['label'] = subdf['label'].values
    if metric in subdf.columns:
        cols_dict[metric] = subdf[metric].apply(lambda x: round(x,2)).values

    ret2 = pd.DataFrame(cols_dict)
 
    info_list = []
    for i,row in ret2.iterrows():
        if group_by == 'visual':
            comp = row[comp_col]
            num = row['num_images']
            dict_rows = {'component':[comp], 'num_images':[num]}

            dist = row['distance']
            dict_rows['mean_distance'] = [np.mean(dist)]
            if metric is not None:
                dict_rows[metric] = [row[metric]]

            info_df = pd.DataFrame(dict_rows).T
            info_list.append(info_df.to_html(escape=True, header=False).replace('\n',''))
        elif group_by == 'label':
            label = row['label']
            num = row['num_images']
            dict_rows = {'label':[label], 'num_images':[num]}


            dist = row['distance']
            dict_rows['mean_distance'] = [np.mean(dist)]
            if metric is not None:
                dict_rows[metric] = [row[metric]]

            info_df = pd.DataFrame(dict_rows).T
            info_list.append(info_df.to_html(escape=True, header=False).replace('\n',''))
    ret = pd.DataFrame({'info': info_list})

    if 'label' in subdf.columns:
        if group_by == 'visual':
            labels_table = []
            for i,row in subdf.iterrows():
                unique, counts = np.unique(np.array(row['label']), return_counts=True)
                lencount = len(counts)
                if max_items is not None and max_items < lencount:
                    lencount = max_items;
                counts_df = pd.DataFrame({"counts":counts}, index=unique).sort_values('counts', ascending=False).head(lencount).to_html(escape=False).replace('\n','')
                labels_table.append(counts_df)
            ret.insert(0, 'label', labels_table)
        else:
            comp_table = []
            for i,row in subdf.iterrows():
                unique, counts = np.unique(np.array(row[comp_col]), return_counts=True)
                lencount = len(counts)
                if max_items is not None and max_items < lencount:
                    lencount = max_items;
                counts_df = pd.DataFrame({"counts":counts}, index=unique).sort_values('counts', ascending=False).head(lencount).to_html(escape=False).replace('\n','')
                comp_table.append(counts_df)
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
        title = 'Fastdup Tool - Components Report'
    else:
        title = "Fastdup Tool - KMeans Cluster Report"

    subtitle = None
    if slice is not None:
        subtitle = "slice: " + str(slice)
    if metric is not None:
        subtitle = "Sorted by " + metric + " descending" if descending else "Sorted by " + metric + " ascending"

    fastdup.html_writer.write_to_html_file(ret[columns], title, out_file, stats_html, subtitle)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    if comp_type == "component":
        print("Stored components visual view in ", os.path.join(out_file))
    else:
        print("Stored KMeans clusters visual view in ", os.path.join(out_file))

    if not lazy_load and threshold is None:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as e:
                print("Warning, failed to remove image file ", i, " with error ", e)

    return 0

def get_stats_df(df, subdf, metric, save_path, max_width=None, input_dir=None):
    stats_info = df[metric].describe().to_frame()
    import matplotlib.pyplot as plt
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
    img = fastdup_imread(local_fig, input_dir)

    ret = pd.DataFrame({'stats':[stats_info.to_html(escape=False,index=True).replace('\n','')], 'image':[imageformatter(img, max_width)]})
    return ret.to_html(escape=False,index=False).replace('\n','')

def do_create_stats_gallery(stats_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            metric='blur', slice=None, max_width=None, descending=False, get_bounding_box_func=None,
                            get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics.
    Note that fastdup generates a histogram of all the encountred valued for the specific metric. The red dashed line on this plot resulting in the number of images displayed in the report.
    For example, assume you have unique image values between 30-250, and the report displays 100 images with values betwewen 30-50. We plot a red line on the value 50.

    Parameters:
        stats_file (str): csv file with the computed image statistics by the fastdup tool. alternatively, a pandas dataframe can be passed in directly with the stats computed by fastdup.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the image label. Image label can be a string or a list of strings.

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
     '''


    img_paths = []
    work_dir = None
    if isinstance(stats_file, pd.DataFrame):
        df = stats_file
    else:
        work_dir = os.path.dirname(os.path.abspath(stats_file))
        df = pd.read_csv(stats_file)
    assert len(df), "Failed to read stats file " + stats_file

    if callable(get_label_func):
        df['label'] = df['filename'].apply(lambda x: get_label_func(x))

    if slice is not None:
        if isinstance(slice, str):
            if slice not in df['label'].unique():
                print(f"Failed to find {slice} in the list of available labels, can not visualize this label class")
                print("Example labels", df['label'].unique()[:10])
                return 1
            df = df[df['label'] == slice]
        elif isinstance(slice, list):
            df = df[df['label'].isin(slice)]
        else:
            assert False, "slice must be a string or list"

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

            #consider saving second image as well!
            #make sure image file is unique, so add also folder name into the imagefile
            imgpath = os.path.join(save_path, filename.replace('/',''))
            p, ext = os.path.splitext(imgpath)
            if ext is not None and ext != '' and ext.lower() not in ['png','tiff','tif','jpeg','jpg','gif']:
                imgpath += ".jpg"

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

    if callable(get_label_func):
        cols.append('label')

    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file, stat_info)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored " + metric + " statistics view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)

def do_create_similarity_gallery(similarity_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 slice=None, max_width=None, descending=False, get_bounding_box_func =None,
                                 get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        similarity_file (str): csv file with the computed image statistics by the fastdup tool, alternatively a pandas dataframe can be passed in directly.

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the image label. Image label can be a string or a list of strings.

        metric (str): Optional metric selection. One of blur, size, mean, min, max, width, height, unique.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels. A special value is 'label_score' which is used for comparing both images and labels of the nearest neighbors.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow adding bounding box to the image. This is a function the user implements that gets the full file path and returns a bounding box or an empty list if not available.

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the filename before displaying it in the report. This is a function the user implements that gets the full file path and returns a string with the reformatted filename.

        get_extra_col_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'


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
    else:
        work_dir = os.path.dirname(os.path.abspath(similarity_file))
        df = pd.read_csv(similarity_file)
    assert len(df), "Failed to read stats file " + similarity_file

    if callable(get_label_func):
        df['label'] = df['from'].apply(lambda x: get_label_func(x))
        df['label2'] = df['to'].apply(lambda x: get_label_func(x))
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
            label = get_label_func(filename)
            similar = [x==label for x in list(row['label'])]
            similar = 100.0*sum(similar)/(1.0*len(row['label']))
            lengths.append(len(row['label']))
            label_score.append(similar)
        subdf['score'] = label_score
        subdf['length'] = lengths

        subdf = subdf[subdf['length'] > 1]
        subdf = subdf.sort_values(['score','length'], ascending=not descending)
        df2 = subdf.copy()
        subdf = subdf.head(num_images)
        stat_info = get_stats_df(df2, subdf, metric='score', save_path=save_path, max_width=max_width)

    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        try:
            filename = lookup_filename(row['from'], work_dir)
            if callable(get_label_func):
                label = get_label_func(filename)
            if callable(get_reformat_filename_func):
                new_filename = get_reformat_filename_func(filename)
            else:
                new_filename = filename

            if callable(get_label_func):
                info0_df = pd.DataFrame({'label':[label],'from':[new_filename]}).T
            else:
                info0_df = pd.DataFrame({'from':[new_filename]}).T

            info0.append(info0_df.to_html(header=False,escape=False).replace('\n',''))


            img = fastdup_imread(filename, input_dir=input_dir)
            img = plot_bounding_box(img, get_bounding_box_func, filename)
            img = my_resize(img, max_width)

            imgpath = os.path.join(save_path, filename.replace('/',''))
            p, ext = os.path.splitext(imgpath)
            if ext is not None and ext != '' and ext.lower() not in ['png','tiff','tif','jpeg','jpg','gif']:
                imgpath += ".jpg"

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
            traceback.print_exc()
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
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)

    return df2


def do_create_aspect_ratio_gallery(stats_file, save_path, get_label_func=None, max_width=None, num_images=0, slice=None,
                                   get_reformat_filename_func=None, input_dir=None):
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

    import matplotlib.pyplot as plt
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

    if callable(get_label_func):
        df['label'] = df['filename'].apply(lambda x: get_label_func(x))
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
        img_max_width = fastdup_imread(max_width_img, input_dir)
        img_max_height = fastdup_imread(max_height_img, input_dir)
        if max_width is not None:
            img_max_width = my_resize(img_max_width, max_width)
            img_max_height = my_resize(img_max_height, max_width)
    except:
        print("Failed to read images ", max_width_img, max_height_img)
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
