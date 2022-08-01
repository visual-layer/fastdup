
# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import os
import pandas as pd
import cv2
import numpy as np
import traceback
from fastdup.image import plot_bounding_box, my_resize, get_type, imageformatter, create_triplet_img

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
            if slice not in df['label'].unique():
                print(f"Failed to find {slice} in the list of available labels, can not visualize this label class")
                print("Example labels", df['label'].unique()[:10])
                return 1
            df = df[df['label'] == slice]
        elif isinstance(slice, list):
            df = df[df['label'].isin(slice)]
        else:
            assert False, "slice must be a string or a list of strings"

    return df


def extract_filenames(row):
    impath1 = row['from']
    impath2 = row['to']
    dist = row['distance']
    os.path.exists(impath1), "Failed to find image file " + impath1
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
                                 get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images computed by the similarity metrics

    Parameters:
        similarity_file (str): csv file with the computed similarities by the fastdup tool
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

    '''


    img_paths = []
    df = pd.read_csv(similarity_file)
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
    for i, row in tqdm(subdf.iterrows(), total=num_images):
        impath1, impath2, dist, ptype = extract_filenames(row)
        if impath1 + '_' + impath2 in sets:
            continue
        try:
            img, imgpath = create_triplet_img(impath1, impath2, ptype, dist, save_path, get_bounding_box_func)
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


def do_create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None, max_width=None, get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None):
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

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. An example list is [[100,100,200,200]] which contains a single bounding box.

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the gallery.
     '''



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

    if callable(get_label_func):
        subdf['label'] = subdf['from'].apply(lambda x: get_label(x, get_label_func))
        subdf = slice_df(subdf, slice)

    subdf = subdf.drop_duplicates(subset='from').sort_values(by='distance', ascending=True).head(num_images)
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        impath1, impath2, dist, ptype = extract_filenames(row)
        try:
            img = cv2.imread(impath1)

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



def visualize_top_components(work_dir:str, save_path:str, num_components:int, get_label_func=None, group_by='visual', slice=None,
                             get_bounding_box_func=None, max_width=None):
    '''
    Visualize the top connected components

    Args:
        work_dir (str): directory with the output of fastdup run
        num_components (int): number of top components to plot
        get_label_func (callable): option function to get label for each image given image filename
        group_by (str): 'visual' or 'label'
        slice (str): slice the datafrmae based on the label or a list of labels

    Returns:
        ret (pd.DataFrame): with the top components
        img_list (list): of the top components images
    '''

    try:
        from .tensorboard_projector import generate_sprite_image
        import traceback
    except Exception as ex:
        print('Your system is missing some depdencies, please pip install matplotlib matplotlib-inline torchvision')
        print(ex)
        return None, None

    assert os.path.exists(work_dir), "Failed to find work_dir " + work_dir
    assert num_components > 0, "Number of components should be larger than zero"

    MAX_IMAGES_IN_GRID = 48

    top_components = do_find_top_components(work_dir=work_dir, get_label_func=get_label_func, group_by=group_by, slice=slice).head(num_components)
    if (top_components is None or len(top_components) == 0):
        print('Failed to find top components, try to reduce grouping threshold by running with turi_param="cchreshold=0.8" where 0.8 is an exmple value.')
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
                    img = cv2.imread(f)
                    img = plot_bounding_box(img, get_bounding_box_func, f)
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

            local_file = os.path.join(save_path, 'component_' + str(i) + '.jpg')
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

    print(f'Finished OK. Components are stored as image files {save_path}/componentsXX.jpg')
    return top_components.head(num_components), img_paths

def do_find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None):
    '''
    Function to find the largest components of duplicate images

    Args:
        work_dir (str): working directory where fastdup.run was run.

        get_label_func (callable): optional function to get label for each image

        group_by (str): 'visual' or 'label'

        slice (str): optional label names or list of label names to slice the dataframe

    	Returns:
        ret (pd.DataFrame): of top components. The column component_id includes the component name.
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
            if isinstance(slice, str):
                if slice not in components['labels'].unique():
                    print(f"Slice {slice} is not a valid label in the components file ")
                    return None
                components = components[components['labels'] == slice]
            elif isinstance(slice, list):
                components = components[components['labels'].isin(slice)]
            else:
                assert(False), "slice should be a string or list of strings"

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
    comps = comps[comps['len'] > 1] # remove any singleton components
    return comps



def do_create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 group_by='visual', slice=None, max_width=None, max_items=None,
                                 get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_info_func=None):
    '''

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the label for the html report

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

        slice(str): optional label to draw only a subset of the components conforming to this label. Or a list of labels.

        max_width (int): optional parameter to control resulting html width. Default is None

        max_items (int): optional parameter to control th number of items displayed in statistics: top max_items labels (for group_by='visual')
            or top max_items components (for group_by='label'). Default is None namely show all items.

        get_bounding_box_func (callable): optional function to get bounding box of an image and add them to the report

        get_reformat_filename_func (callable): optional function to reformat the filename to be displayed in the report

        get_extra_col_func (callable): optional function to get extra column to be displayed in the report



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


    subdf, img_paths = visualize_top_components(work_dir, save_path, num_images, get_label_func, group_by, slice, get_bounding_box_func, max_width)
    if subdf is None or len(img_paths) == 0:
        return None

    assert len(subdf) == len(img_paths), "Number of components and number of images do not match"

    import fastdup.html_writer
    if 'labels' in subdf.columns:
    	ret2 = pd.DataFrame({'component_id':subdf['component_id'].values,
        	                 'num_images':subdf['len'].apply(lambda x: "{:,}".format(x)).values,
                	         'labels': subdf['labels']})
    else:
    	ret2 = pd.DataFrame({'component_id':subdf['component_id'].values,
        	                 'num_images':subdf['len'].apply(lambda x: "{:,}".format(x)).values})
 
    info_list = []
    for i,row in ret2.iterrows():
        if group_by == 'visual':
            comp = row['component_id']
            num = row['num_images']
            info_df = pd.DataFrame({'component':[comp], 'num_images':[num]}).T
            info_list.append(info_df.to_html(escape=True, header=False).replace('\n',''))
        elif group_by == 'label':
            label = row['labels']
            num = row['num_images']
            info_df = pd.DataFrame({'label':[label], 'num_images':[num]}).T
            info_list.append(info_df.to_html(escape=True, header=False).replace('\n',''))
    ret = pd.DataFrame({'info': info_list})

    if 'labels' in subdf.columns:
        if group_by == 'visual':
            labels_table = []
            for i,row in subdf.iterrows():
                unique, counts = np.unique(np.array(row['labels']), return_counts=True)
                lencount = len(counts)
                if max_items is not None and max_items < lencount:
                    lencount = max_items;
                counts_df = pd.DataFrame({"counts":counts}, index=unique).sort_values('counts', ascending=False).head(lencount).to_html(escape=False).replace('\n','')
                labels_table.append(counts_df)
            ret.insert(0, 'labels', labels_table)
        else:
            comp_table = []
            for i,row in subdf.iterrows():
                unique, counts = np.unique(np.array(row['component_id']), return_counts=True)
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
    if 'labels' in subdf.columns:
        if group_by == 'visual':
            columns.append('labels')
        elif group_by == 'label':
            columns.append('components')



    title = 'Fastdup Tool - Components Report'
    if slice is not None:
        title += ", " + str(slice)
    fastdup.html_writer.write_to_html_file(ret[columns], 'Fastdup Tool - Components Report', out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored components visual view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as e:
                print("Warning, failed to remove image file ", i, " with error ", e)





def do_create_stats_gallery(stats_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                               metric='blur', slice=None, max_width=None, descending=False, get_bounding_box_func=None,
                            get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        stats_file (str): csv file with the computed image statistics by the fastdup tool

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        metric (str): Optional metric selection. One of blur, size, mean_value

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow adding bounding box plot to the displayed image.
         This is a function the user implements that gets the full file path and returns a bounding box or an empty list if not available.

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the image file name. This is a function the user implements that gets the full file path and returns a new file name.

        get_extra_col_func (callable): Optional parameter to allow adding extra column to the report.
     '''


    img_paths = []
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
    subdf = df.sort_values(metric, ascending=not descending).head(num_images)
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        try:
            filename = row['filename']
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

    if metric is not None and metric == 'size':
        cols.append('width')
        cols.append('height')



    fastdup.html_writer.write_to_html_file(subdf[[metric,'Image','filename']], title, out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored" + metric + " statistics view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)

def do_create_similarity_gallery(similarity_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 slice=None, max_width=None, descending=False, get_bounding_box_func =None,
                                 get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        stats_file (str): csv file with the computed image statistics by the fastdup tool

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        metric (str): Optional metric selection. One of blur, size, mean_value

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow adding bounding box to the image. This is a function the user implements that gets the full file path and returns a bounding box or an empty list if not available.

        get_reformat_filename_func (callable): Optional parameter to allow reformatting the filename before displaying it in the report. This is a function the user implements that gets the full file path and returns a string with the reformatted filename.

        get_extra_col_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

     '''


    from fastdup import generate_sprite_image
    img_paths = []
    img_paths2 = []
    info0 = []
    info = []

    df = pd.read_csv(similarity_file)
    assert len(df), "Failed to read stats file " + stats_file

    if callable(get_label_func):
        df['label'] = df['from'].apply(lambda x: get_label_func(x))
        df['label2'] = df['to'].apply(lambda x: get_label_func(x))
        df = slice_df(df, slice)

    df = df.sort_values(['from','distance'], ascending=not descending)
    if 'label' in df.columns:
        top_labels = df.groupby('from')['label2'].apply(list)

    tos = df.groupby('from')['to'].apply(list)
    distances = df.groupby('from')['distance'].apply(list)
    
    if 'label' in df.columns:
        subdf = pd.DataFrame({'to':tos, 'label':top_labels,'distance':distances}).reset_index()
    else:
        subdf = pd.DataFrame({'to':tos, 'distance':distances}).reset_index()

    subdf = subdf.head(num_images)    
    for i, row in tqdm(subdf.iterrows(), total=min(num_images, len(subdf))):
        try:
            filename = row['from']

            if callable(get_reformat_filename_func):
                new_filename = get_reformat_filename_func(filename)
            else:
                new_filename = filename

            if 'label' in row:
                info0_df = pd.DataFrame({'label':[get_label_func(filename)],'from':[new_filename]}).T
            else:
                info0_df = pd.DataFrame({'from':[new_filename]}).T

            info0.append(info0_df.to_html(header=False,escape=False).replace('\n',''))


            img = cv2.imread(filename)
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
            info_df = pd.DataFrame({'distance':distances, 'to':imgs})


            if callable(get_reformat_filename_func):
                info_df['to'] = info_df['to'].apply(lambda x: get_reformat_filename_func(x))

            if 'label2' in df.columns:
                info_df['label'] = row['label'][:MAX_IMAGES]
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
    if callable(get_extra_col_func):
        subdf['extra'] = subdf['from'].apply(lambda x: get_extra_col_func(x))
        cols.append('extra')


    fastdup.html_writer.write_to_html_file(subdf[cols], title, out_file)
    assert os.path.exists(out_file), "Failed to generate out file " + out_file

    print("Stored similar images view in ", os.path.join(out_file))
    if not lazy_load:
        for i in img_paths:
            try:
                os.unlink(i)
            except Exception as ex:
                print("Failed to delete image file ", i, ex)

