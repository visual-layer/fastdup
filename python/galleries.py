
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import pandas as pd
import cv2
from fastdup import image
import numpy as np
import traceback
import fastdup
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


def extract_filenames(row):
    impath1 = row['from']
    impath2 = row['to']
    dist = row['distance']
    os.path.exists(impath1), "Failed to find image file " + impath1
    os.path.exists(impath2), "Failed to find image file " + impath2

    type1 = image.get_type(impath1)
    type2 = image.get_type(impath2)
    ptype = '{0}_{1}'.format(type1, type2)
    return impath1, impath2, dist, ptype

def do_create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True,
                              lazy_load=False, get_label_func=None, slice=None):
    '''

    Function to create and display a gallery of images computed by the similarity metrics

    Parameters:
        similarity_file (str): csv file with the computed similarities by the fastdup tool
        save_path (str): output folder location for the visuals
        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.
        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.
    '''
    assert os.path.exists(similarity_file), "Failed to find similarity file " + similarity_file
    if os.path.isdir(similarity_file):
        similarity_file = os.path.join(similarity_file, 'similarity.csv')

    assert num_images >= 1, "Please select one or more images"
    if num_images > 1000 and not lazy_load:
        print("When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        assert os.path.exists(save_path), "Failed to find save_path " + save_path

    if (get_label_func is not None):
        assert callable(get_label_func), "get_label_func has to be a collable function, given the filename returns the label of the file"

    img_paths = []
    df = pd.read_csv(similarity_file)
    assert len(df), "Failed to read similarity file"

    if slice is not None and callable(get_label_func):
        df['label'] = df['from'].apply(lambda x: get_label_func(x))
        if slice not in df['label'].unique():
            print(f"Slice label {slice} not found in the similarity file")
            return None
        df = df[df['label'] == slice]
        assert len(df), "Failed to find slice " + slice

    sets = {}

    subdf = df.head(num_images) if descending else df.tail(num_images)
    subdf = subdf.reset_index()
    indexes = []
    for i, row in tqdm(subdf.iterrows(), total=num_images):
        impath1, impath2, dist, ptype = extract_filenames(row)
        if impath1 + '_' + impath2 in sets:
            continue
        try:
            img, imgpath = image.create_triplet_img(impath1, impath2, ptype, dist, save_path)
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
        subdf.insert(0, 'Image', [image.imageformatter(x) for x in img_paths])
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

    fastdup.html_writer.write_to_html_file(subdf[fields], 'Fastdup tool - similarity report', out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file
    print("Stored similarity visual view in ", out_file)

    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as e:
                print("Warning, failed to remove image file ", i, " with error ", e)


def do_create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        outliers_file (str): csv file with the computed outliers by the fastdup tool
        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        how (str): Optional outlier selection method. one = take the image that is far away from any one image (but may have other images close to it).
                                                      all = take the image that is far away from all other images. Default is one.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label.
     '''
    assert os.path.exists(outliers_file), "Failed to find outliers file " + outliers_file
    if os.path.isdir(outliers_file):
        outliers_file = os.path.join(outliers_file, 'outliers.csv')

    if num_images > 1000 and not lazy_load:
        print("When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        assert os.path.exists(save_path), "Failed to find save_path " + save_path

    assert num_images >= 1, "Please select one or more images"
    assert how == 'one' or how == 'all', "Wrong argument to how=[one|all]"


    from fastdup import image
    img_paths = []
    df = pd.read_csv(outliers_file)
    assert len(df), "Failed to read outliers file " + outliers_file

    if (how == 'all'):
        dups_file = os.path.join(os.path.dirname(outliers_file), 'similarity.csv')
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

    if slice is not None:
        subdf['label'] = subdf['from'].apply(lambda x: get_label_func(x))
        if slice not in subdf['label'].unique():
            print(f"Failed to find {slice} in the list of available labels, can not visualize this label class")
            return 1
        subdf = subdf[subdf['label'] == slice]

    subdf = subdf.drop_duplicates(subset='from').sort_values(by='distance', ascending=True).head(num_images)
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        impath1, impath2, dist, ptype = extract_filenames(row)
        try:
            img = cv2.imread(impath1)
            h, w, c = img.shape
            w1 = 320
            if h > 320 or w > 320:
                img = cv2.resize(img, (w1, w1))
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
        subdf.insert(0, 'Image', [image.imageformatter(x) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        subdf.insert(0, 'Image', img_paths2)

    if get_label_func is not None and callable(get_label_func):
        subdf.insert(2, 'Path', subdf['from'].apply(lambda x: get_label(x, get_label_func)))
        subdf = subdf.rename(columns={'distance':'Distance'}, inplace=False)
    else:
        subdf = subdf.rename(columns={'from':'Path', 'distance':'Distance'}, inplace=False)

    out_file = os.path.join(save_path, 'outliers.html')
    fastdup.html_writer.write_to_html_file(subdf[['Image','Distance','Path']], 'Fastdup Tool - Outliers Report', out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored outliers visual view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)



def visualize_top_components(work_dir:str, save_path:str, num_components:int, get_label_func=None, group_by='visual', slice=None):
    '''
    Visualize the top connected components
    :param work_dir: output of fastdup run
    :param num_components: number of top components to plot
    :param get_label_func: function to get label for each image
    :param group_by: 'visual' or 'label'
    :return:
    pd.DataFreme with the top components
    img_list of the top components images
    '''

    try:
        from PIL import Image as PILImage
        import traceback
    except:
        print('Your system is missing some depdencies, please pip install matplotlib matplotlib-inline torchvision')
        return None, None

    assert os.path.exists(work_dir), "Failed to find work_dir " + work_dir
    assert num_components > 0, "Number of components should be larger than zero"

    MAX_IMAGES_IN_GRID = 48

    top_components = find_top_components(work_dir=work_dir, get_label_func=get_label_func, group_by=group_by, slice=slice).head(num_components)
    if (top_components is None or len(top_components) == 0):
        return None, None

    # iterate over the top components
    index = 0
    img_paths = []
    for i,row in tqdm(top_components.iterrows(), total = len(top_components)):
        try:
            # find the component id
            component_id = row['component_id']
            # find all the image filenames linked to this id
            files = row['files'][:MAX_IMAGES_IN_GRID]
            if (len(files) == 0):
                break

            tmp_images = []
            w,h = [], []
            for f in files:
                try:
                    img = PILImage.open(f).convert('RGB')
                    tmp_images.append(img)
                    w.append(img.size[0])
                    h.append(img.size[1])
                except Exception as ex:
                    print("Warning: Failed to load image ", f, "skipping image due to error", ex)

            if len(tmp_images) == 0:
                print("Failed to read all images")
                return None, None

            avg_h = int(np.mean(h))
            avg_w = int(np.mean(w))
            images = []
            for f in tmp_images:
                f = f.resize((avg_w,avg_h))
                images.append(f)

            img, labels = fastdup.generate_sprite_image(images,  len(images), '', None, h=avg_h, w=avg_w)
            local_file = os.path.join(save_path, 'component_' + str(component_id) + '.jpg')
            img.save(local_file)
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

    print(f'Finished OK. Components are stored as image files {save_path}/componentsXX.jpg')
    return top_components.head(num_components), img_paths

def find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None):
    '''
    Function to find the largest components of duplicate images

    Parameters
        work_dir (str) working directory where fastdup.run was run.

    Returns
        pd.DataFrame of top components. The column component_id includes the component name.
        The column files includes a list of all image files in this component.


    '''

    assert os.path.exists(work_dir), 'Working directory work_dir does not exist'
    assert os.path.exists(os.path.join(work_dir, 'connected_components.csv')), "Failed to find fastdup output file"
    assert os.path.exists(os.path.join(work_dir, 'atrain_features.dat.csv')), "Failed to find fastdup output file"


    # read fastdup connected components, for each image id we get component id
    components = pd.read_csv(os.path.join(work_dir, 'connected_components.csv'))

    filenames = pd.read_csv(os.path.join(work_dir, 'atrain_features.dat.csv'))
    if (len(components) != len(filenames)):
        print(f"Error: number of rows in components file {work_dir}/connected_components.csv and number of rows in image file {work_dir}/atrain_features.dat.csv are not equal")
        print("This may occur if multiple runs where done on the same working folder overriding those files. Please rerun on a clen folder")
        return None
    # now join the two tables to get both id and image name
    components['filename'] = filenames['filename']


    # find the components that have the largest number of images included
    if callable(get_label_func):
        components['labels'] = components['filename'].apply(get_label_func)
        if slice is not None:
            if slice not in components['labels'].unique():
                print(f"Slice {slice} is not a valid label in the components file ")
                return None
            components = components[components['labels'] == slice]

        if 'path' in group_by:
            components['path'] = components['filename'].apply(lambda x: os.path.dirname(x))

        if group_by == 'visual':
            top_labels = components.groupby('component_id')['labels'].apply(list)
            top_files = components.groupby('component_id')['filename'].apply(list)
            comps = pd.DataFrame({'files':top_files, 'labels':top_labels}).reset_index()
        elif group_by == 'label':
            top_files = components.groupby('labels')['filename'].apply(list)
            top_components = components.groupby('labels')['component_id'].apply(list)
            comps = pd.DataFrame({'files':top_files, 'component_id':top_components}).reset_index()
        else:
            assert(False), "group_by should be visual or label, got " + group_by

    else:
        top_components = components.groupby('component_id')['filename'].apply(list)
        comps = pd.DataFrame({'files':top_components}).reset_index()

    comps['len'] = comps['files'].apply(lambda x: len(x))
    comps = comps.sort_values('len', ascending=False)
    return comps

def do_create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None, group_by='visual', slice=None):
    '''

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the label for the html report

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

        slice(str) : optional label to draw only a subset of the components conforming to this label.

     '''
    assert os.path.exists(work_dir), "Failed to find work_dir " + work_dir
    if num_images > 1000 and not lazy_load:
        print("When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        if not os.path.exists(save_path):
            print(f"Failed to generate save_path directory {save_path}")
            return None

    assert num_images >= 1, "Please select one or more images"
    assert group_by == 'label' or group_by == 'visual', "Allowed values for group_by=[visual|label], got " + group_by
    if group_by == 'label':
        assert callable(get_label_func), "missing get_label_func, when grouping by labels need to set get_label_func"


    subdf, img_paths = visualize_top_components(work_dir, save_path, num_images, get_label_func, group_by, slice)
    if subdf is None or len(img_paths) == 0:
        return None

    assert len(subdf) == len(img_paths), "Number of components and number of images do not match"

    import fastdup.html_writer
    ret = pd.DataFrame({'component_id':subdf['component_id'].values, 'num_images':subdf['len'].values})
    if 'labels' in subdf.columns:
        ret.insert(0, 'labels', subdf['labels'].values)

    if not lazy_load:
        from fastdup import image
        ret.insert(0, 'image', [image.imageformatter(x) for x in img_paths])
    else:
        img_paths2 = ["<img src=\"" + os.path.join(save_path, os.path.basename(x)) + "\" loading=\"lazy\">" for x in img_paths]
        ret.insert(0, 'image', img_paths2)

    out_file = os.path.join(save_path, 'components.html')
    columns = ['component_id','num_images','image']
    if 'labels' in subdf.columns:
        columns.append('labels')

    fastdup.html_writer.write_to_html_file(ret[columns], 'Fastdup Tool - Components Report', out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored components visual view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as e:
                print("Warning, failed to remove image file ", i, " with error ", e)
