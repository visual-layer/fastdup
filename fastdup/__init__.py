#!/bin/python3.8
#FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
#This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
#4.0 International license. Please reach out to info@databasevisual.com for licensing options.

#__init__.py file wraps the main calls to our c++ engine.



import sys
import os
from ctypes import *
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import platform
from fastdup.galleries import do_create_similarity_gallery, do_create_outliers_gallery, do_create_stats_gallery, do_create_components_gallery, do_create_duplicates_gallery
import contextlib
from fastdup import coco
from fastdup.version_check import check_for_update

try:
	from tqdm import tqdm
except:
	tqdm = (lambda x: x)


__version__="0.112"
CONTACT_EMAIL="info@databasevisual.com"
if platform.system() == "Darwin":
	SO_SUFFIX=".dylib"
else:
	SO_SUFFIX=".so"

# check if more recent fastdup version is available
check_for_update(__version__)

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
        batch_size = 1,
        resume = 0):

    '''
    Run fastdup tool for finding duplicate, near duplicates, outliers and clusters of related images in a corpus of images.
    The only mandatory argument is image_dir.

    Args:
        input_dir (str):
            Location of the images/videos to analyze.
                * A folder
                * A remote folder (s3 or minio starting with s3:// or minio://)
                * A file containing absolute filenames each on its own row.
                * A python list with absolute filenames
                * yolov5 yaml input file containing train and test folders (single folder supported for now)
                * We support jpg, jpeg, tiff, tif, giff, png, mp4, avi. In addition we support tar, tar.gz, tgz and zip files containing images.
            If you have other image extensions that are readable by opencv imread() you can give them in a file and then we do not check for the
            known extensions.
            Note: it is not possible to mix compressed (videos or tars/zips) and regular images. Use the flag turi_param='tar_only=1' if you want to ignore images and run from compressed files.

        work_dir (str): Optional path for storing intermediate files and results. Default is "."

        test_dir (str): Optional path for test data. When given similarity of train and test images is compared (vs. train/train or test/test which are not performed).

        compute (str): Compute type [cpu|gpu] default is cpu. Note: gpu is supported only in the enterprise version.

        verbose (boolean): Verbosity. Default is False.

        num_threads (int): Number of threads. Default is -1 to be auto configured by the number of cores.

        num_images (unsigned long long): Number of images to run on. Default is 0 which means run on all the images in the image_dir folder.

        turi_param (str): Optional turi parameters seperated by command. \
            ==nnmodel=xx==, Nearest Neighbor model for clustering the features together. Supported options are 0 = brute_force (exact),
            1 = ball_tree and 2 = lsh (both approximate).\
            ==ccthreshold=xx==, Threshold for running conected components to find clusters of similar images. Allowed values 0->1.\
            ==run_cc=0|1== run connected components on the resulting similarity graph. Default is 1.\
            ==run_pagerank=0|1== run pagerank on the resulting similarity graph. Default is 1.\
            ==delete_tar=0|1== when working with tar files obtained from cloud storage delete the tar after download\
            ==delete_img=0|1== when working with images obtained from cloud storage delete the image after download\
            ==tar_only=0|1== run only on tar files and ignore images in folders. Default is 0.\
	        Example run: turi_param='nnmodel=0,ccthreshold=0.99'

        distance (str): Distance metric for the Nearest Neighbors algorithm. Default is cosine. Other distances are euclidean, squared_euclidean, manhattan.

        threshold (float): Similarity measure in the range 0->1, where 1 is totally identical, 0.98 and above is almost identical, and 0.85 and above is very similar. Default is 0.85 which means that only image pairs with similarity larger than 0.85 are stored.

        lower_threshold (float): Similarity measure to outline images that are far away (outliers) vs. the total distribution. Default value is 0.05 (means 5% out of the total similarities computed).

        model_path (str): Optional location of ONNX model file, should not be used.

        version (bool): Print out the version number. This function takes no argument.

        nearest_neighbors_k (int): For each image, how many similar images to look for. Default is 2.

        d (int): Length of the feature vector. Default is 576. Change this parameter only when providing precomputed features instead of images.

        run_mode (int):
            ==run_mode=0== (the default) does the feature extraction and NN embedding to compute all pairs similarities.
            It uses the input_dir command line argument for finding the directory to run on (or a list of files to run on).
            The features are extracted and saved into the working_dir path (the default features out file nme is
            `features.dat` in the same folder for storing the numpy features and features.dat.csv for storing the
            image file names corresponding to the numpy features).
            For larger dataset it may be wise to split the run into two, to make sure intermediate results are stored in case you encounter an error.\
            \
            ==run_mode=1== computes the extracted features and stores them, does not compute the NN embedding.
            For large datasets, it is possible to run on a few computing nodes, to extract the features, in parallel.
            Use the min_offset and max_offset flags to allocate a subset of the images for each computing node.
            Offsets start from 0 to n-1 where n is the number of images in the input_dir folder.\
            \
            ==run_mode=2== reads a stored feature file and computes the NN embedding to provide similarities.
            The input_dir param is ignored, and the work_dir is used to point to the numpy feature file. (Give a full path and filename).\
            \
            ==run_mode=3== Reads the NN model stored by nnf.index from the work_dir and computes all pairs similarity on all images
            given by the test_dir parameter. input_dir should point to the location of the train data.
            This mode is used for scoring similarities on a new test dataset given a precomputed simiarity index on a train dataset.\
            \
            ==run_mode=4== reads the NN model stored by `nnf.index` from the `work_dir` and computes all pairs similarity on pre extracted feature vectors computer by run_mode=1.\

        nn_provider (string): Provider of the nearest neighbor algorithm, allowed values are turi|nnf.

        min_offset (unsigned long long): Optional min offset to start iterating on the full file list.

        max_offset (unsigned long long): Optional max offset to start iterating on the full file list.

        nnf_mode (str): When nn_provider='nnf' selects the nnf model mode.

        nnf_param (str): When nn_provider='nnf' selects assigns optional parameters. 

        bounding_box (str): Optional bouding box to crop images, given as bounding_box='rows=xx,cols=xx,height=xx,width=xx'.

        batch_size (int): Optional batch size when computing inference, default = 1. Allowed values < 200. Note: batch_size > 1 is enabled in the enterprise version.

        resume (int): Optional flag to resume from a previous run.

    Returns:
        ret (int): Status code 0 = success, 1 = error.

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

    if os.path.abspath(input_dir) == os.path.abspath(work_dir.strip()):
        print("Input and work_dir output directories are the same, please point to different directories")
        return 1

    if (resume == 0 and (os.path.exists(os.path.join(work_dir, 'atrain_features.dat')) or \
        os.path.exists(os.path.join(work_dir, 'features.dat')))  and (run_mode == 0 or run_mode == 1)):
        print("Found existing atrain_features.dat file in the working directory, please remove it before running the program or run in a fresh directory.")
        print("If you like to resume a prevuously stopped run, please run with resume=1.")
        return 1

    if (work_dir.startswith('./')):
        work_dir = work_dir[2:]
    if (input_dir.startswith('./')):
        input_dir = input_dir[2:]

    cwd = os.getcwd()
    if (work_dir.startswith(cwd + '/')):
        work_dir = work_dir.replace(cwd + '/', '')

    if (run_mode == 3 and not os.path.exists(os.path.join(work_dir, 'index.nnf'))):
        print("An index.nnf file is required for run_mode=3, please run with run_mode=0 to generate this file")
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
            c_int,
            c_int]

    cm = contextlib.nullcontext()

    if 'JPY_PARENT_PID' in os.environ:
        print("On Jupyter notebook running on large datasets, there may be delay getting the console output. We recommend running using python shell.")
        from IPython.utils.capture import capture_output
        cm = capture_output(stdout=True, stderr=True,display=True)

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
        batch_size,
        resume)

        if hasattr(c, 'stdout'):
            print(c.stdout)
            print(c.stderr)

        return ret

    return 1



def run_on_webdataset(input_dir='',
                      work_dir='.',
                      test_dir='',
                      compute='cpu',
                      verbose=False,
                      num_threads=-1,
                      num_images=0,
                      turi_param='nnmodel=0',
                      distance='cosine',
                      threshold=0.9,
                      lower_threshold=0.05,
                      model_path=model_path_full,
                      license='CC-BY-NC-ND-4.0',
                      version=False,
                      nearest_neighbors_k=2,
                      d=576,
                      nn_provider='nnf',
                      min_offset=0,
                      max_offset=0,
                      nnf_mode="HNSW32",
                      nnf_param="",
                      bounding_box="",
                      batch_size = 1):
    '''
    Run the FastDup software on a web dataset.
    This run is composed of two stages. First extract all feature vectors using run_mode=1, then run the nearest neighbor model using run_mode=2.
    Make sure that work_dir has enough free space for extracting tar files. Tar files are extracted temporarily into work_dir/tmp folder.
    You can control the free space using the flags turi_param='delete_tar=1|0' and delete_img='1|0'.  When delete_tar=1 the tars are processed one by one and deleted after processing.
    When delete_img=1 the images are processed one by one and deleted after processing.
    '''

    ret = run(input_dir=input_dir, work_dir=work_dir, test_dir=test_dir, compute=compute, verbose=verbose, num_threads=num_threads, num_images=num_images,
              turi_param=turi_param, distance=distance, threshold=threshold,
              lower_threshold=lower_threshold, license=license, model_path=model_path, version=version, nearest_neighbors_k=nearest_neighbors_k, d=d,
              nn_provider=nn_provider, min_offset=min_offset, max_offset=max_offset, nnf_mode=nnf_mode, nnf_param=nnf_param, bounding_box=bounding_box,
                batch_size=batch_size, run_mode=1)
    if ret != 0:
        return ret

    return run(input_dir=input_dir, work_dir=work_dir, test_dir=test_dir, compute=compute, verbose=verbose, num_threads=num_threads, num_images=num_images,
              turi_param=turi_param, distance=distance, threshold=threshold,
              lower_threshold=lower_threshold, license=license, model_path=model_path, version=version, nearest_neighbors_k=nearest_neighbors_k, d=d,
              nn_provider=nn_provider, min_offset=min_offset, max_offset=max_offset, nnf_mode=nnf_mode, nnf_param=nnf_param, bounding_box=bounding_box,
              batch_size=batch_size, run_mode=2)


def load_binary_feature(filename, d=576):
    '''
    Python function for loading the stored binary features written by fastdup and their matching filenames and analyzing them in Python.

    Args:
        filename (str): The binary feature file location
        d (int): Feature vector length

    Returns:
        filenames (list): A list of with all image file names of length X.
        np_array (np.array): An np matrix of shape rows x d cols (default d is 576). Each row conform to feature vector os a single image.

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

    Args:
        save_path (str): Working folder to save the files to
        filenames (list): A list of file location of the images (absolute paths) of length n images
        np_array (np.array): Numpy array of size n x d. Each row is a feature vector of one file.

    Returns:
        ret (int): 0 in case of success, otherwise 1

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
                              lazy_load=False, get_label_func=None, slice=None, max_width=None,
                              get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images computed by the similarity metrics

        Example:
	 	import fastdup
		fastdup.run('input_folder', 'output_folder')
		fastdup.create_duplicates_gallery('output_folder', save_path='.', get_label_func = lambda x: x.split('/')[1], slice='hamburger')

    Regarding get_label_func, this example assumes that the second folder name is the class name for example my_data/hamburger/image001.jpg. You can change it to match your own labeling convention.


    Args:
        similarity_file (str): csv file with the computed similarities by the fastdup tool

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label.
            This is a function the user implements that gets the full image file path and returns html string with the label or any other metadata desired.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.
            slice could be a specific label i.e. slice='haumburger' and in that case only similarities between hamburger and other classes are presented.
            Two reserved arguments for slice are "diff" and "same". When using "diff" the report only shows similarities between classes. When using "same" the report will show only similarities inside same class.
            Note that when using slice, the function get_label_function should be implmeneted.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding additional column to the report
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

    if slice is not None and get_label_func is None:
        print("When slicing on specific labels need to provide a function to get the label (using the parameter get_label_func)")
        return 1

    return do_create_duplicates_gallery(similarity_file, save_path, num_images, descending, lazy_load, get_label_func, slice, max_width, get_bounding_box_func,
                                        get_reformat_filename_func, get_extra_col_func)


def create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None, max_width=None, get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        outliers_file (str): csv file with the computed outliers by the fastdup tool

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        how (str): Optional outlier selection method. one = take the image that is far away from any one image (but may have other images close to it).
                                                      all = take the image that is far away from all other images. Default is one.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding additional column to the report

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

    if slice is not None and get_label_func is None:
        print("When slicing on specific labels need to provide a function to get the label (using the parameter get_label_func)")
        return 1

    if (get_label_func is not None):
        assert callable(get_label_func), "get_label_func has to be a collable function, given the filename returns the label of the file"

    return do_create_outliers_gallery(outliers_file, save_path, num_images, lazy_load, get_label_func, how, slice,
                                      max_width, get_bounding_box_func, get_reformat_filename_func, get_extra_col_func)



def create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                              group_by='visual', slice=None, max_width=None, max_items=None, get_bounding_box_func=None,
                              get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional label string, given a absolute path to an image return the label for the html report

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional parameter to set the max html width of images in the gallery. Default is None.

        max_items (int): Optional parameter to limit the number of items displayed (labels for group_by='visual' or components for group_by='label'). Default is None.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.  The input is an absolute path to the image and the output is a list of bounding boxes.  Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.  The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding more information to the report.


    Returns:
        ret (int): 0 in case of success, otherwise 1
    '''

    if slice is not None and get_label_func is None:
        print("When slicing on specific labels need to provide a function to get the label (using the parameter get_label_func)")
        return 1

    if (get_label_func is not None):
        assert callable(get_label_func), "get_label_func has to be a collable function, given the filename returns the label of the file"

    if max_width is not None:
        assert isinstance(max_width, int), "html image width should be an integer"
        assert max_width > 0, "html image width should be > 0"

    if max_items is not None:
        assert isinstance(max_items, int), "max items should be an integer"
        assert max_items > 0, "html image width should be > 0"

    return do_create_components_gallery(work_dir, save_path, num_images, lazy_load, get_label_func, group_by, slice,
                                        max_width, max_items, get_bounding_box_func,
                                        get_reformat_filename_func, get_extra_col_func)


def delete_components(top_components, to_delete,  how = 'one', dry_run = True):
    '''
    function to automate deletion of duplicate images using the connected components analysis.

    Args:
        top_components (pd.DataFrame): largest components as found by the function find_top_components().
        to_delete (list): a list of integer component ids to delete
        how (int): either 'all' (deletes all the component) or 'one' (leaves one image and delete the rest of the duplicates)
        dry_run (bool): if True does not delete but print the rm commands used, otherwise deletes



    '''

    assert isinstance(top_components, pd.DataFrame), "top_components should be a pandas dataframe"
    assert len(top_components), "top_components should not be enpty"
    assert isinstance(to_delete, list), "to_delete should be a list of integer component ids"
    assert len(to_delete), "to_delete should not be empty"
    assert isinstance(to_delete[0], int) or isinstance(to_delete[0], np.int64), "to_delete should be a list of integer component ids"
    assert how == 'one' or how == 'all', "how should be one of 'one'|'all'"
    assert isinstance(dry_run, bool)

    for comp in (to_delete):
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

def export_to_tensorboard_projector(work_dir, log_dir, sample_size = 900,
                                    sample_method='random', with_images=True, get_label_func=None, d=576, file_list=None):
    '''
    Export feature vector embeddings to be visualized using tensorboard projector app.

    Args:
        work_dir (str): work_dir where fastdup results are stored

        log_dir (str): output dir where tensorboard will read from

        sample_size (int): how many images to view. Default is 900.

        sample_method (str): how to sample, currently 'random' is supported.

        with_images (bool): add images to the visualization (default True)

        get_label_func (callable): Optional parameter to allow adding class label name to the image. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        d (int): dimension of the embedding vector. Default is 576.

        file_list (list): Optional parameter to specify a list of files to be used for the visualization. If not specified, filenames are taken from the work_dir/atrain_features.dat.csv file
                      Note: be careful here as the order of the file_list matters, need to keep the exact same order as the atrain_features.dat.csv file!
    Returns:
        ret (int): 0 in case of success, 1 in case of failure
    '''


    from .tensorboard_projector import export_to_tensorboard_projector_inner
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
        assert os.path.exists(work_dir), 'Failed to create work_dir ' + work_dir
    assert os.path.exists(os.path.join(work_dir, 'atrain_features.dat')), f'Faild to find fastdup output {work_dir}atrain_features.dat'
    assert sample_size <= 5000, f'Tensorboard projector is limited by 5000 images'

    imglist, features = load_binary_feature(os.path.join(work_dir, 'atrain_features.dat'), d=d)
    if file_list is not None:
        assert isinstance(file_list, list), 'file_list should be a list of absolute file names given in the same order'
        assert len(file_list) == len(imglist), "file_list should be the same length as imglist got " + str(len(file_list)) + " and " + str(len(imglist))
    export_to_tensorboard_projector_inner(imglist, features, log_dir, sample_size, sample_method, with_images, get_label_func, d=d)




def read_coco_labels(path):
    assert(os.path.exists(path)), "Failed to find path " + path
    return coco.read_coco_labels(path)



def generate_sprite_image(img_list, sample_size, log_dir, get_label_func=None, h=0, w=0, alternative_filename=None, alternative_width = None, max_width=None):
    '''
    Generate a sprite image of images for tensorboard projector. A sprite image is a large image composed of grid of smaller images.

    Parameters:
        img_list (list): list of image filenames (full path)

        sample_size (int):  how many images in to plot

        log_dir (str): directory to save the sprite image

        get_label_func (callable): optional function given a full path filename outputs its label

        h (int): optional requested hight of each subimage

        w (int): optional requested width of each subimage

        alternative_filename (str): optional parameter to save the resulting image to a different name

        alternative_width (int): optional parameter to control the number of images per row

        max_width (int): optional parameter to control the rsulting width of the image

    Returns:
        path (str): path to sprite image
        labels (list): list of labels

    '''

    assert len(img_list), "Image list is empty"
    assert sample_size > 0
    from .tensorboard_projector import generate_sprite_image as tgenerate_sprite_image
    return tgenerate_sprite_image(img_list, sample_size, log_dir, get_label_func, h=h, w=w,
                                  alternative_filename=alternative_filename, alternative_width=alternative_width, max_width=max_width)



def find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None):
    '''
    Function to find the largest components of duplicate images

    Args:
        work_dir (str): working directory where fastdup.run was run.

        get_label_func (callable): optional function given a full path filename outputs its label

        group_by (str): optional parameter to group by 'visual' or 'label'. When grouping by visual fastdup aggregates visually similar images together.
            When grouping by 'label' fastdup aggregates images with the same label together.

        slice (str): optional parameter to slice the results by a specific label. For example, if you want to slice by 'car' then pass 'car' as the slice parameter.

    Returns:
        df (pd.DataFrame): of top components. The column component_id includes the component name.
            The column files includes a list of all image files in this component.


    '''
    from .galleries import do_find_top_components
    return do_find_top_components(work_dir, get_label_func, group_by, slice)



def init_search(k, work_dir, d = 576, model_path = model_path_full):
    '''
    Initialize real time search and precomputed nnf data.
    This function should be called only once before running searches. The search function is search().

    Args:
        k (int): number of nearest neighbors to search for
        work_dir (str): working directory where fastdup.run was run.
        d (int): dimension of the feature vector. Defualt is 576.
        model_path (str): path to the onnx model file. Optional.

    Returns:
        ret (int): 0 in case of success, otherwise 1.

    '''

    assert os.path.exists(model_path), "Failed to find model_path " + model_path
    assert d > 0, "d must be greater than 0"

    fun = dll.init_search
    fun.restype = c_int
    fun.argtypes = [c_int,
		            c_char_p,
                    c_int,
                    c_char_p]

    ret = fun(k, bytes(work_dir, 'utf-8'), d, bytes(model_path, 'utf-8'))
    if ret != 0:
        print("Failed to initialize search")
        return ret

    return 0




def search(img, size, verbose=0):
    '''
    Search for similar images in the image database.

    Args:
        img (str): the image to search for
        size (int): image size width x height
        verbose (int): run in verbose mode
    Returns:
        ret (int): 0 = in case of success, 1 = in case of failure
            The output file is created on work_dir/similrity.csv as initialized by init_search
    '''

    from numpy.ctypeslib import ndpointer
    fun = dll.search
    fun.restype = c_int
    fun.argtypes = [ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
                    c_int, c_int]

    #img_arr = np.ascontiguousarray(img, dtype=np.uint8)
    img_arr = np.ascontiguousarray(np.array(img),dtype=np.uint8)
    ret = fun(img_arr, size, verbose)
    if ret != 0:
        print("Failed to search for image")
        return ret

    return 0

def create_stats_gallery(stats_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            metric='blur', slice=None, max_width=None, descending= False, get_bounding_box_func=None,
                         get_reformat_filename_func=None, get_extra_col_func=None):
    '''
    Function to create and display a gallery of images computed by the outliers metrics.
    For example, most blurry images, most dark images etc.

    Args:
        stats_file (str): csv file with the computed image statistics by the fastdup tool

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        metric (str): Optional metric selection. One of 'blur','size','mean','min','max','unique','stdv'.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Option parameter to select the maximal image width in the report

        descending (bool): Optional parameter to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image. The input is an absolute path to the image and the output is a list of bounding boxes. Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the gallery.

    Returns:
        ret (int): 0 in case of success, otherwise 1.
    '''
    if slice is not None and get_label_func is None:
        print("When slicing on specific labels need to provide a function to get the label (using the parameter get_label_func)")
        return 1

    if (get_label_func is not None):
        assert callable(get_label_func), "get_label_func has to be a collable function, given the filename returns the label of the file"

    if max_width is not None:
        assert isinstance(max_width, int), "html image width should be an integer"
        assert max_width > 0, "html image width should be > 0"

    assert metric in ['blur','size','mean','min','max','unique','stdv'], "Unknown metric value"

    assert os.path.exists(stats_file), "Failed to find outliers file " + stats_file
    if os.path.isdir(stats_file):
        stats_file = os.path.join(stats_file, 'similarity.csv')

    if num_images > 1000 and not lazy_load:
        print("When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        assert os.path.exists(save_path), "Failed to find save_path " + save_path

    assert num_images >= 1, "Please select one or more images"

    return do_create_stats_gallery(stats_file, save_path, num_images, lazy_load, get_label_func, metric, slice, max_width,
                                   descending, get_bounding_box_func, get_reformat_filename_func, get_extra_col_func)


def create_similarity_gallery(similarity_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 slice=None, max_width=None, descending=False, get_bounding_box_func=None,
                                 get_reformat_filename_func=None, get_extra_col_func=None):
    '''

    Function to create and display a gallery of images computed by the outliers metrics

    Args:
        similarity_file (str): csv file with the computed image statistics by the fastdup tool

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the report

    Returns:
        ret (int): 0 in case of success, otherwise 1.

     '''

    assert os.path.exists(similarity_file), "Failed to find outliers file " + similarity_file
    if os.path.isdir(similarity_file):
        similarity_file = os.path.join(similarity_file, 'similarity.csv')

    if num_images > 1000 and not lazy_load:
        print("When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        assert os.path.exists(save_path), "Failed to find save_path " + save_path

    assert num_images >= 1, "Please select one or more images"

    return do_create_similarity_gallery(similarity_file, save_path, num_images, lazy_load, get_label_func, 
        slice, max_width, descending, get_bounding_box_func, get_reformat_filename_func, get_extra_col_func)
