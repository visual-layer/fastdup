#!/bin/python3

# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

import sys
import os
from ctypes import *
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import traceback
import platform
from .capture_io import *
from .galleries import *
import contextlib
from fastdup import coco

try:
	from tqdm import tqdm
except:
	tqdm = (lambda x: x)


__version__ = "0.24"
CONTACT_EMAIL="info@databasevisual.com"
if platform.system() == "Darwin":
	SO_SUFFIX=".dylib"
else:
	SO_SUFFIX=".so"

LOCAL_DIR=os.path.dirname(os.path.abspath(__file__))
so_file = os.path.join(LOCAL_DIR, 'libfastdup_shared' + SO_SUFFIX)
if not os.path.exists(so_file):
    print("Failed to find shared object", so_file);
    print("Current init file is on", __file__);
    sys.exit(1)
dll = CDLL(so_file)


model_path_full=os.path.join(LOCAL_DIR, 'UndisclosedFastdupModel.onnx')
if not os.path.exists(model_path_full):
    print("Failed to find onnx file", model_path_full);
    print("Current init file is on", __file__);
    sys.exit(1)


"""
This is a wrapper function to call the fastdup c++ code
"""

def run(input_dir='',             
        work_dir='.', 
        test_dir='',
        compute='cpu',    
        verbose=False,     
        num_threads=-1,     
        num_images=0,        
        turi_param='nnmodel=0',  
        distance='cosine',     #distance metric for the nearest neighbor model. Default "cosine".
        threshold=0.9,         #threshold for finding simiar images. Default is 0.85 (values 0->1)
        lower_threshold=0.05,   #lower threshold for finding simiar images. Default is 0.05 (values 0->1)
        model_path=model_path_full,
        #ONNX model path
        license='CC-BY-NC-ND-4.0',            #license string
        version=False,          #show version and exit      
        nearest_neighbors_k=2, 
        d=576,
        run_mode=0,
        nn_provider='nnf',
        min_offset=0,
        max_offset=0, 
        nnf_mode="HNSW32",
        nnf_param="",
        bounding_box="",
        batch_size = 1):

    '''
    Run fastdup tool for finding duplicate, near duplicates, outliers and clusters of related images in a corpus of images.
    The only mandatory argument is image_dir.

    Parameters:
        input_dir (str): Location of the images/videos to analyze.
        - A folder
        - A remote folder (s3 or minio starting with s3:// or minio://)
        - A file containing absolute filenames each on its own row.
        - A python list with absolute filenames
        - yolov5 yaml input file containing train and test folders (single folder supported for now)
        We support jpg, jpeg, tiff, tif, giff, png, mp4, avi. In addition we support tar, tar.gz, tgz and zip files containing images.
        If you have other image extensions that are readable by opencv imread() you can give them in a file and then we do not check for the
        known extnsions.
        Note: it is not possible to mix compressed (videos or tars/zips) and regular images. Use the flag turi_param='tar_only=1' if you want to ignore images and run from compressed files.

        work_dir (str): Optional path for storing intermediate files and results. Default is "."

        test_dir (str): Optional path for test data. When given similarity of train and test images is compared (vs. train/train or test/test which are not performed).

        compute (str): Compute type [cpu|gpu] default is cpu.

        verbose (boolean): Verbosity. Default is False.

        num_threads (int): Number of threads. Default is -1 to be auto configured by the number of cores.

        num_images (unsigned long long): Number of images to run on. Default is 0 which means run on all the images in the image_dir folder.

        turi_param (str): Optional turi parameters seperated by command. 
            nnmodel=xx, Nearest Neighbor model for clustering the features together. Supported options are 0 = brute_force (exact), 1 = ball_tree and 2 = lsh (both approximate).
            ccthreshold=xx, Threshold for running conected components to find clusters of similar images. Allowed values 0->1.
            store_int=0|1 store the similarity as string filenames or string index of the file id (to save space)
            run_cc=0|1 run connected components on the resulting similarity graph. Default is 1.
            run_pagerank=0|1 run pagerank on the resulting similarity graph. Default is 1.
            delete_tar=0|1 when working with tar files obtained from cloud storage delete the tar after download
            delete_img=0|1 when working with images obtained from cloud storage delete the image after download
            tar_only=0|1 run only on tar files and ignore images in folders. Default is 0.
	    Example run: turi_param='nnmodel=0,ccthreshold=0.99'

        distance (str): Distance metric for the Nearest Neighbors algorithm. Default is cosine. Other distances are euclidean, squared_euclidean, manhattan.

        threshold (float): Similarity measure in the range 0->1, where 1 is totally identical, 0.98 and above is almost identical, and 0.85 and above is very similar. Default is 0.85 which means that only image pairs with similarity larger than 0.85 are stored.

        lower_threshold (float): Similarity measure to outline images that are far away (outliers) vs. the total distribution. Default value is 0.05 (means 5% out of the total similarities computed).

        model_path(str): Optional location of ONNX model file, should not be used.

        version(bool): Print out the version number. This function takes no argument.

        nearest_neighbors_k (int): For each image, how many similar images to look for. Default is 2.

        d (int): Length of the feature vector. Default is 576. Change this parameter only when providing precomputed features instead of images.

        run_mode (int): run_mode=0 (the default) does the feature extraction and NN embedding to compute all pairs similarities. It uses the input_dir command line argument for finding the directory to run on (or a list of files to run on). The features are extracted and saved into the working_dir path (the default features out file nme is features.dat in the same folder for storing the numpy features and features.dat.csv for storing the image file names corresponding to the numpy features). For larger dataset it may be wise to split the run into two, to make sure intermediate results are stored in case you encounter an error.
        run_mode=1 computes the extracted features and stores them, does not compute the NN embedding. For large datasets, it is possible to run on a few computing nodes, to extract the features, in parallel. Use the min_offset and max_offset flags to allocate a subset of the images for each computing node. Offsets start from 0 to n-1 where n is the number of images in the input_dir folder.
        run_mode=2 reads a stored feature file and computes the NN embedding to provide similarities. The input_dir param is ignored, and the work_dir is used to point to the numpy feature file. (Give a full path and filename).
        run_mode=3 Reads the NN model stored by faiss.index from the work_dir and computes all pairs similarity on all inages give by the test_dir parameter. input_dir should point to the location of the train data. This mode is used for scoring similarities on a new test dataset given a precomputed simiarity index on a train dataset.
        run_mode=4 reads the NN model stored by `nnf.index` from the `work_dir` and computes all pairs similarity on pre extracted feature vectors computer by run_mode=1.

        nn_provider (string): Provider of the nearest neighbor algorithm, allowed values are turi|nnf.

        min_offset (unsigned long long): Optional min offset to start iterating on the full file list. Default is -1.

        max_offset (unsigned long long): Optional max offset to start iterating on the full file list. Default is -1.

        nnf_mode (str): When nn_provider='nnf' selects the nnf model mode. Default option is HNSW32.

        nnf_param (str): When nn_provider='nnf' selects assigns optional parameters. 

        bounding_box (str): Optional bouding box to crop images, given as bounding_box='rows=xx,cols=xx,height=xx,width=xx'.

        batch_size (int): Optional batch size when computing inferenc, default = 1. Allowed values < 200.

    Returns:
        Status code 0 = success, 1 = error.

    '''

    print("FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.");

    if (version):
        print("This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives 4.0 "
                "International license. Please reach out to %s for licensing options.", CONTACT_EMAIL);
        return 0

    if isinstance(input_dir, list):
        files = pd.DataFrame({'filenames':input_dir})
        files.to_csv('files.txt')
        input_dir = 'files.txt'


    elif (input_dir.strip() == '' and run_mode != 2):
        print("Found an empty input directory, please point to the directory where you are images are found");
        return 1

    elif not os.path.exists(input_dir):
        print("Failed to find input dir ", input_dir, " please check your input");
        return 1

    # support for YOLO dataset format
    if (input_dir.endswith('.yaml')):
        import yaml
        with open(input_dir, "r") as stream:
            try:
                print('Detected yolo config file')
                config = yaml.safe_load(stream)
                if 'path' not in config or 'train' not in config:
                    print('Failed to find path or train in yolo config file')
                    return 1
                if isinstance(config['train'], list):
                    print('Train location folder list is not supported, please create a single train folder')
                # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
                #path: ../datasets/coco128  # dataset root dir
                #train: images/train2017  # train images (relative to 'path') 128 images
                #val: images/train2017  # val images (relative to 'path') 128 images
                #test:  # test images (optional)
                input_dir = os.path.join(config['path'], config['train'])
                if 'test' in config and config['test'].strip() != '':
                    test_dir = os.path.join(config['path'], config['test'])

            except Exception as exc:
                print('Error when loading yolo .yaml config', exc)
                return 1


    if batch_size < 1 or batch_size > 200:
        print("Allowed values for batch size 1->200.")
        return 1

    if (run_mode == 3 and test_dir == ''):
        print('For run_mode=3 test_dir parameter needs to point to the location of the test batch of images compred to the train images')
        return 1

    #Calling the C++ side
    dll.do_main.restype = c_int
    dll.do_main.argtypes = [c_char_p, 
            c_char_p,
            c_char_p,
            c_char_p,
            c_bool,
            c_int,
            c_ulonglong, 
            c_char_p,
            c_char_p,
            c_float,
            c_float,
            c_char_p,
            c_char_p,
            c_bool,
            c_int,
            c_int,
            c_int,
            c_char_p,
            c_ulonglong,
            c_ulonglong,
            c_char_p,
            c_char_p, 
            c_char_p,
            c_int]

    cm = contextlib.nullcontext()

    if 'JPY_PARENT_PID' in os.environ:
        print("On Jupyter notebook running on large datasets, there may be delay getting the console output. We recommend running using python shell.")
        from IPython.utils.capture import capture_output
        cm = capture_output()

    with cm as c:
        ret = dll.do_main(bytes(input_dir, 'utf-8'),
        bytes(work_dir, 'utf-8'),
        bytes(test_dir.strip(), 'utf-8'),
        bytes(compute, 'utf-8'),
        verbose,
        num_threads,
        num_images,
        bytes(turi_param, 'utf-8'),
        bytes(distance, 'utf-8'),
        threshold,
        lower_threshold,
        bytes(license, 'utf-8'),
        bytes(model_path, 'utf-8'),
        version,
        nearest_neighbors_k,
        d,
        run_mode,
        bytes(nn_provider, 'utf-8'),
        min_offset,
        max_offset,
        bytes(nnf_mode, 'utf-8'),
        bytes(nnf_param, 'utf-8'),
        bytes(bounding_box, 'utf-8'),
        batch_size)

        if hasattr(c, 'stdout'):
            print(c.stdout)
            print(c.stderr)

        return ret

    return 1


def load_binary_feature(filename, d=576):
    '''
    Python function for loading the stored binary features written by fastdup and their matching filenames and analyzing them in Python.

    Parmaeters:
        filename(str):The binary feature file location
        d(int): Feature vector length

    Returns:
        A list of with all image file names of length X.
        An np matrixy of shape rows x d cols (default d is 576). Each row conform to feature vector os a single image.

    Example:
        import fastdup
        file_list, mat_features = fastdup.load_binary('features.dat')

    '''
	
    if not os.path.exists(filename) or not os.path.exists(filename + '.csv'):
        print("Error: failed to find the binary feature file:", filename, ' and the filenames csv file:', filename + '.csv')
        return None
    assert(d > 0), "Feature vector length d has to be larger than zero"

    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype='<f')

    df = pd.read_csv(filename + '.csv')['filename'].values
    num_images = len(df);
    print('Read a total of ', num_images, 'images')

    data = np.reshape(data, (num_images, d))
    assert data.shape[1] == d
    return list(df), data



def save_binary_feature(save_path, filenames, np_array):
    '''
    Function for saving data to be used by fastdup. Given a list of images and their matching feature vectors in a numpy array,
    function saves data in a format readable by fastdup. This saves the image extraction step, to be used with run_mode=1 namely perform
    nearest neighbor model on the feature vectors.

    Parmaeters:
        save_path(str): Working folder to save the files to
        filenames(list): A list of file location of the images (absolute paths) of length n images
        np_array(np.array): Numpy array of size n x d. Each row is a feature vector of one file.

    Returns:
        0 in case of sucess, otherwise 1

    '''

    assert isinstance(save_path, str)  and save_path.strip() != "", "Save path should be a non empty string"
    assert isinstance(filenames, list), "filenames should be a list of image files"
    assert len(filenames), "filenames should be a non empty list"
    assert isinstance(filenames[0], str), 'filenames should contain strings with the image absolute paths'
    assert isinstance(np_array, np.ndarray),  "np_array should be a numpy array"
    assert np_array.dtype == 'float32', "np_array dtype must be float32. You can generate the array using the" \
                              "command: features = np.zeros((rows, cols), dtype='float32')"
    assert np_array.shape[0] == len(filenames), "np_array should contain rows matching to the filenames list"

    
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        df = pd.DataFrame({'filename': filenames})
        local_filename = os.path.join(save_path, 'atrain_features.dat')
        df.to_csv(local_filename + '.csv', index=False)
        bytes_array = np_array.tobytes()
        with open(local_filename, 'wb') as f:
            f.write(bytes_array)
        assert os.path.exists(local_filename), "Failed to save file " + local_filename

    except Exception as ex:
        print("Failed to save to " + save_path + " Exception: " + ex)
        return 1

    return 0

def create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True,
                              lazy_load=False, get_label_func=None):
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
    do_create_duplicates_gallery(similarity_file, save_path, num_images, descending, lazy_load, get_label_func)


def create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one'):
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
     '''

    do_create_outliers_gallery(outliers_file, save_path, num_images, lazy_load, get_label_func, how)



def create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None, group_by='visual'):
    '''

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the label for the html report

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

     '''
    do_create_components_gallery(work_dir, save_path, num_images, lazy_load, get_label_func, group_by)

def delete_components(top_components: pd.DataFrame, to_delete: list,  how: int = 'one', dry_run: bool = True):
    '''
      function to automate deletion of duplicate images using the connected components analysis.

      Parameters
          top_components (pd.DataFrame): largest components as found by the function find_top_components().
          to_delete (list): a list of integer component ids to delete
          how (int): either 'all' (deletes all the component) or 'one' (leaves one image and delete the rest of the duplicates)
          dry_run: if True does not delete but print the rm commands used, otherwise deletes


      Return
          None

    '''

    assert isinstance(top_components, pd.DataFrame), "top_components should be a pandas dataframe"
    assert len(top_components), "top_components should not be enpty"
    assert isinstance(to_delete, list), "to_delete should be a list of integer component ids"
    assert len(to_delete), "to_delete should not be empty"
    assert isinstance(to_delete[0], int) or isinstance(to_delete[0], np.int64), "to_delete should be a list of integer component ids"
    assert how == 'one' or how == 'all', "how should be one of 'one'|'all'"
    assert isinstance(dry_run, bool)

    for comp in tqdm(to_delete):
        subdf = top_components[top_components['component_id'] == comp]
        if (len(subdf) == 0):
            print("Warning: failed to find image files for component id", comp)
            continue

        files = subdf['files'].values[0]
        if (len(files) == 1):
            print('Warning: component id ', comp, ' has no related images, please check..')
            continue

        if (how == 'one'):
            files = files[1:]

        for f in files:
            if (dry_run):
                print(f'rm -f {f}')
            else:
                try:
                    os.unlink(f)
                except Exception as ex:
                    print('Failed to remove file', f, ' with exception', ex)

def export_to_tensorboard_projector(work_dir:str, log_dir:str, sample_size:int = 900,
                                    sample_method:str='random', with_images=True, get_label_func=None):
    '''
    Export feature vector embeddings to be visualized using tensorboard projector app.
    :param work_dir: work_dir where fastdup results are stored
    :param log_dir: output dir where tensorboard will read from
    :param sample_size: how many images to view
    :param sample_method: how to sample, currently 'random' is supported
    :param with_images:bool add images to the visualization (default True)
    :param get_label_func (callable): Optional parameter to allow adding class label name to the image. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.
    :return:
    '''


    from .tensorboard_projector import export_to_tensorboard_projector_inner
    assert os.path.exists(work_dir), 'Failed to find work_dir' + work_dir
    assert os.path.exists(os.path.join(work_dir, 'atrain_features.dat')), f'Faild to find fastdup output {work_dir}atrain_features.dat'
    assert sample_size <= 5000, f'Tensorboard projector is limited by 5000 images'

    imglist, features = load_binary_feature(os.path.join(work_dir, 'atrain_features.dat'))
    export_to_tensorboard_projector_inner(imglist, features, log_dir, sample_size, sample_method, with_images, get_label_func)




def read_coco_labels(path):
    assert(os.path.exists(path)), "Failed to find path " + path
    return coco.read_coco_labels(path)



def generate_sprite_image(img_list, sample_size, log_dir, get_label_func=None, h=0, w=0):
    '''
    Generate a sprite image of images for tensorboard projector    
    Parameters
    img_list (list): list of image filenames (full path)
    sample_size (int):  how many images in to plot
    log_dir (str): directory to save the sprite image
    get_label_func (callable): optional function given a full path filename outputs its label	
    h (int): optional requested hight of each subimage
    w (int): optional requested width of each subimage    

    return
    path to sprite image
    list of labels

    '''

    assert len(img_list), "Image list is empty"
    assert sample_size > 0
    from .tensorboard_projector import generate_sprite_image as tgenerate_sprite_image
    return tgenerate_sprite_image(img_list, sample_size, log_dir, get_label_func, h=h, w=w)
