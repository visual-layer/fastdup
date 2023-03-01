#!/bin/python3.8
#FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
#This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
#4.0 International license. Please reach out to info@databasevisual.com for licensing options.

#__init__.py file wraps the main calls to our c++ engine.



import sys
import os
import json
import shutil
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] ="offscreen"
from ctypes import *
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import platform
from fastdup.galleries import do_create_similarity_gallery, do_create_outliers_gallery, do_create_stats_gallery, \
    do_create_components_gallery, do_create_duplicates_gallery, do_create_aspect_ratio_gallery
import contextlib
import time
from fastdup import coco
from fastdup.sentry import init_sentry, fastdup_capture_exception, fastdup_performance_capture, fastdup_capture_log_debug_state
from fastdup.definitions import *
from fastdup.utils import *
from datetime import datetime
from fastdup.sentry import v1_sentry_handler

try:
    from tqdm import tqdm
except:
    tqdm = (lambda x: x)


__version__="0.211"
CONTACT_EMAIL="info@databasevisual.com"

init_sentry()
#record_time()

ret = check_latest_version(__version__)
if ret:
    raise RuntimeError(f"fastdup detected your are running an old version {__version__} (10 versions or more vs. the latest) please upgrade fastdup")



LOCAL_DIR=os.path.dirname(os.path.abspath(__file__))
os.environ['FASTDUP_LOCAL_DIR'] = LOCAL_DIR
is_windows = False

if platform.system() == "Windows":
    is_windows = True
    import struct
    assert struct.calcsize("P") * 8 == 64, "Detected 32 bit windows, not supported, please run with 64 bits windows"
    SO_SUFFIX=".dll"
    so_file = os.path.join(LOCAL_DIR, 'fastdup_shared' + SO_SUFFIX)
    # https://docs.sentry.io/platforms/native/configuration/backends/crashpad/
    if os.path.exists(os.path.join(LOCAL_DIR, 'lib\\crashpad_handler.exe')):
        os.environ['SENTRY_CRASHPAD'] = os.path.join(LOCAL_DIR, 'lib\\crashpad_handler.exe')
elif platform.system() == "Darwin":
    SO_SUFFIX=".dylib"
    # https://docs.sentry.io/platforms/native/configuration/backends/crashpad/
    if os.path.exists(os.path.join(LOCAL_DIR, 'lib/crashpad_handler')):
        os.environ['SENTRY_CRASHPAD'] = os.path.join(LOCAL_DIR, 'lib/crashpad_handler')
    else:
        print('Failed to find crashpad handler on ', os.path.join(LOCAL_DIR, 'lib/crashpad_handler'))
    so_file = os.path.join(LOCAL_DIR, 'libfastdup_shared' + SO_SUFFIX)
else:
    SO_SUFFIX=".so"
    so_file = os.path.join(LOCAL_DIR, 'libfastdup_shared' + SO_SUFFIX)


if not os.path.exists(so_file):
    print("Failed to find shared object", so_file);
    print("Current init file is on", __file__);
    sys.exit(1)

try:
    # this should be supported only from python3.8 and up
    if platform.system() == "Windows":
        os.add_dll_directory(LOCAL_DIR)
        os.add_dll_directory(LOCAL_DIR + "\\lib")
        os.add_dll_directory(os.path.join(os.environ['SystemRoot'], 'System32'))
        #os.add_dll_directory("C:\\Program Files\\PowerShell\\7")
        dll = WinDLL(so_file)
    else:
        dll = CDLL(so_file)
except Exception as ex:
    fastdup_capture_exception("__init__", ex)
    print("Please reach out to fastdup support, it seems installation is missing critical files to start fastdup.")
    print("We would love to understand what has gone wrong.")
    print("You can open an issue here: " +  GITHUB_URL + " or email us at " + CONTACT_EMAIL)
    find_command = "\"find " + LOCAL_DIR + " \""
    if platform.system() == "Windows":
        find_command = "\"tree " + LOCAL_DIR + " \""
    print("Share out output of the command " + find_command)
    sys.exit(1)

model_path_full=os.path.join(LOCAL_DIR, 'UndisclosedFastdupModel.ort')
if not os.path.exists(model_path_full):
    fastdup_capture_exception("Bad Install",  RuntimeError("Failed to find ort model on init " + __file__))
    print("Failed to find ort model on init " + __file__)
    sys.exit(1)


#if 'conda' in sys.version.lower() and 'clang' in sys.version.lower():
#    print("Warning: detected conda environment on Mac, this may lead to unstable behavior. It is recommended to switch to python. You can install python using 'brew install python@3.8'")


def do_run(input_dir='',
           work_dir='.',
           test_dir='',
           compute='cpu',
           verbose=False,
           num_threads=-1,
           num_images=0,
           turi_param='nnmodel=0',
           distance='cosine',     #distance metric for the nearest neighbor model.
           threshold=0.9,         #threshold for finding simiar images. (allowed values 0->1)
           lower_threshold=0.05,   #lower percentile threshold for finding simiar images (values 0->1)
           model_path=model_path_full,
           license='',            #license string
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
           resume = 0,
           high_accuracy=False):

    fastdup_capture_log_debug_state(locals())
    start_time = time.time()

    print("FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.")
    if (version):
        print("This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives 4.0 "
              "International license. Please reach out to %s for licensing options.", CONTACT_EMAIL);
        return 0

    if 'sync_s3_to_local=1' in turi_param:
        assert input_dir.startswith('s3://') or input_dir.startswith('minio://'), 'sync_s3_to_local=1 can only be used with s3:// or minio:// input_dir'
        if 'delete_img=1' in turi_param:
            print('Warning: delete_img=1 is not supported with sync_s3_to_local=1, please delete images manually from the work_dir after their download')
            turi_param = turi_param.replace('delete_img=1', '')
        if 'delete_img=0' not in turi_param:
            turi_param += ',delete_img=0'

    _model_path = model_path
    if (bounding_box == "face"):
        _model_path=os.path.join(LOCAL_DIR, 'UndisclosedFDModel.onnx')
        assert os.path.exists(_model_path), "Failed to find FD model"
        turi_param += ",save_crops=1"
        run_mode = 1

    input_dir2 = input_dir
    if isinstance(input_dir, list):
        input_dir2 = None
    config = {"input_dir": input_dir2, "work_dir": work_dir, "test_dir": test_dir, "compute": compute, "verbose": verbose, "num_threads": num_threads,
              "num_images": num_images, "turi_param": turi_param, "distance": distance, "threshold": threshold, "lower_threshold": lower_threshold,
              "model_path": _model_path, "version": version, "nearest_neighbors_k": nearest_neighbors_k, "d": d, "run_mode": run_mode,
              "nn_provider": nn_provider, "min_offset": min_offset, "max_offset": max_offset, "nnf_mode": nnf_mode, "nnf_param": nnf_param,
              "bounding_box": bounding_box, "batch_size": batch_size, "resume": resume, "high_accuracy": high_accuracy}

    # in case of failure crash report store current config
    fastdup_capture_log_debug_state(config)
    work_dir = shorten_path(work_dir)
    try:
        if not work_dir.startswith('smb://'):
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)
            with open(f"{work_dir}/config.json", "w") as f:
                json.dump(config, f, indent=4)
    except Exception as ex:
        print(f"Warning: error writing config file: {ex} to file {work_dir}/config.json")

    if isinstance(input_dir, list):
        os.makedirs(work_dir, exist_ok=True)
        # saves the path to the list of files
        files = expand_list_to_files(input_dir)
        input_dir = save_as_csv_file_list(files, os.path.join(work_dir, 'files.txt'))

    elif (input_dir.strip() == '' and run_mode != RUN_NN):
        print("Found an empty input directory, please point to the directory where you are images are found")
        return 1

    input_dir = shorten_path(input_dir)
    if work_dir.startswith('smb://'):
        assert input_dir != work_dir, "Input and work dir should point to different folders"
    else:
        assert os.path.abspath(input_dir) != os.path.abspath(work_dir.strip()), "Input and work_dir output directories are the same, " \
                "please point to different directories"

    if isinstance(test_dir, list):
        files = expand_list_to_files(test_dir)
        test_dir = save_as_csv_file_list(files, os.path.join(work_dir, INPUT_TEST_FILE_LOCATION))

    if not os.path.exists(input_dir):
        if input_dir.startswith('s3://') or input_dir.startswith('minio://') or input_dir.startswith('smb://'):
            pass
        else:
            assert False, f"Failed to find input dir {input_dir} please check your input."
    else:
        if os.path.isfile(input_dir):
            if check_if_folder_list(input_dir):
                files = list_subfolders_from_file(input_dir)
                input_dir = save_as_csv_file_list(files, os.path.join(work_dir, "dirfiles.txt"))

    if not work_dir.startswith('smb://'):
        if resume == 0 and (os.path.exists(os.path.join(work_dir, 'atrain_features.dat')) or \
                             os.path.exists(os.path.join(work_dir, FILENAME_FEATURES)))  and \
                             (run_mode == RUN_ALL or run_mode == RUN_EXTRACT):
            assert False, "Found existing atrain_features.dat file in the working directory, please remove it before running the program or run in a fresh directory."

    assert nn_provider in ['nnf'], "Nearest neighbor implementation should be nnf."
    if nn_provider == 'nnf':
        if nnf_mode == "Flat":
            assert distance in ['cosine', 'euclidean', 'l1','linf','canberra','braycurtis','jensenshannon'], f"Distance metric {distance} not supported for nnf provider nnf when nnf_mode=Flat"
        else:
            assert distance in ['euclidean', 'cosine'], "Distance should be either euclidean or cosine when nn_provider='nnf'"

    if ((run_mode == RUN_NNF_SEARCH_IMAGE_DIR or run_mode == RUN_NNF_SEARCH_STORED_FEATURES or run_mode == RUN_KMEANS_STORED_FEATURES) \
            and not os.path.exists(os.path.join(work_dir, FILENAME_NNF_INDEX))):
        print(f"An {FILENAME_NNF_INDEX} file is required for run_mode=3, please run with run_mode=0 to generate this file")
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
                if 'test' in config and config['test'] is not None and config['test'].strip() != '':
                    test_dir = os.path.join(config['path'], config['test'])

            except Exception as exc:
                import traceback
                traceback.print_exc();
                print('Error when loading yolo .yaml config', input_dir, exc)
                return 1


    if batch_size < 1 or batch_size > 200:
        print("Allowed values for batch size 1->200.")
        return 1


    if run_mode == RUN_NNF_SEARCH_IMAGE_DIR:
        assert test_dir != '', 'For run_mode=3 test_dir parameter needs to point to the location of the test batch of images ' \
                               'compared to the train images'

    if high_accuracy:
        if d != DEFAULT_MODEL_FEATURE_WIDTH:
            if d != HIGH_ACCURACY_MODEL_FEATURE_WIDTH:
                assert False, "Can not run high accuracy model when using user provided d, please run with high_accuracy=False, or undefine d"

        if _model_path != model_path_full:
            if _model_path.endswith('UndisclosedFastdupModel2.ort'):
                d = HIGH_ACCURACY_MODEL_FEATURE_WIDTH
            else:
                assert False, "Error: Can not run high accuracy model when using user provided model_path, please run with high_accuracy=False"
        else:
            _model_path = model_path_full.replace('l.ort', 'l2.ort')
            d = HIGH_ACCURACY_MODEL_FEATURE_WIDTH

    if not os.path.exists(_model_path):
        try:
            err_msg = f"Model path folder exists? {os.path.isdir(os.path.dirname(_model_path))} \n"
            f"List of files in model path folder {os.listdir(os.path.dirname(_model_path))}\n"
            f"Please file a github issue: https://github.com/visual-layer/fastdup/issues to report this issue."
        except:
            err_msg = f"Failed to list model_path folder {_model_path}, folder does not exist"

        if _model_path != model_path_full:
            assert False, f"Failed to find user provided ORT model path {_model_path}.\n{err_msg}\n"
        else:
            assert False, f"Failed to find fastdup ORT model path {_model_path}.\n{err_msg}\n" \
                          f"It looks like a corrupted fastdup install. "

    # When working with s3 remote folder allow loading it first to disk to improve performance
    if 'sync_s3_to_local=1' in turi_param:
        assert input_dir.startswith('s3://') or input_dir.startswith('minio://'), 'sync_s3_to_local=1 can only be used with s3:// or minio:// input_dir'
        if 'delete_img=1' in turi_param:
            print('Warning: delete_img=1 is not supported with sync_s3_to_local=1, please delete images manually from the work_dir after their download')
            turi_param = turi_param.replace('delete_img=1', '')
        if 'delete_img=0' not in turi_param:
            turi_param += ',delete_img=0'
        input_dir = download_from_s3(input_dir, work_dir, verbose, False)

        if test_dir.startswith('s3://') or test_dir.startswith('minio://'):
            test_dir = download_from_s3(test_dir, work_dir, verbose, True)

    local_error_file = os.path.join(work_dir, FILENAME_ERROR_MSG)
    if os.path.exists(local_error_file):
        os.unlink(local_error_file)


    char_type = c_char_p if not is_windows else c_wchar_p
    encoding = lambda x: bytes(x, 'utf-8') if not is_windows else x
   
    #Calling the C++ side
    dll.do_main.restype = c_int
    dll.do_main.argtypes = [char_type,
                            char_type,
                            char_type,
                            c_char_p,
                            c_bool,
                            c_int,
                            c_ulonglong,
                            c_char_p,
                            c_char_p,
                            c_float,
                            c_float,
                            c_char_p,
                            char_type,
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

    with cm as c:
        ret = dll.do_main(encoding(input_dir),
                          encoding(work_dir),
                          encoding(test_dir.strip()),
                          bytes(compute, 'utf-8'),
                          verbose,
                          num_threads,
                          num_images,
                          bytes(turi_param, 'utf-8'),
                          bytes(distance, 'utf-8'),
                          threshold,
                          lower_threshold,
                          bytes(license, 'utf-8'),
                          encoding(_model_path),
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

        if ret != 0 and 'JPY_PARENT_PID' in os.environ:
            if os.path.exists(local_error_file):
                with open(local_error_file, "r") as f:
                    error = f.read()
                    print("fastdup C++ error received: ", error, "\n")
                    fastdup_capture_exception("C++ error", RuntimeError(error))

        fastdup_performance_capture("do_run", start_time)
        return ret

    return 1



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
        distance='cosine',     #distance metric for the nearest neighbor model.
        threshold=0.9,         #threshold for finding simiar images. (allowed values 0->1)
        lower_threshold=0.05,   #lower percentile threshold for finding simiar images (values 0->1)
        model_path=model_path_full,
        license='',            #license string
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
        resume = 0,
        high_accuracy=False):

    '''
    Run fastdup tool for finding duplicate, near duplicates, outliers and clusters of related images in a corpus of images.
    The only mandatory argument is image_dir.

    Args:
        input_dir (str):
            Location of the images/videos to analyze.
                * A folder
                * A remote folder (s3 or minio starting with s3:// or minio://). When using minio append the minio server name for example minio://google/visual_db/sku110k.
                * A file containing absolute filenames each on its own row.
                * A file containing s3 full paths or minio paths each on its own row.
                * A python list with absolute filenames.
                * A python list with absolute folders, all images and videos on those folders are added recusively
                * For run_mode=2, a folder containing fastdup binary features or a file containing list of atrain_feature.dat.csv files in multiple folders
                * yolov5 yaml input file containing train and test folders (single folder supported for now)
                * We support jpg, jpeg, tiff, tif, giff, heif, heic, bmp, png, mp4, avi. In addition we support tar, tar.gz, tgz and zip files containing images.
            If you have other image extensions that are readable by opencv imread() you can give them in a file (each image on its own row) and then we do not check for the
            known extensions and use opencv to read those formats.
            Note: It is not possible to mix compressed (videos or tars/zips) and regular images. Use the flag turi_param='tar_only=1' if you want to ignore images and run from compressed files.
            Note: We assume image sizes should be larger or equal to 10x10 pixels. Smaller images (either on width or on height) will be ignored with a warning shown.
            Note: It is possible to skip small images also by defining minimum allowed file size using turi_param='min_file_size=1000' (in bytes).
            Note: For performance reasons it is always preferred to copy s3 images from s3 to local disk and then run fastdup on local disk. Since copying images from s3 in a loop is very slow.
            Alternatively you can use the flag turi_param='sync_s3_to_local=1' to copy ahead all images on the remote s3 bucket to disk.
            Note: fastdup plus beta version now supports bounding boxes on the c++ side. To use it prepare an input file with the following csv header: filename,col_x,row_y,width,height where each row as an image file
            and bounding box information in the above format. Fastdup will run on the bounding box level and the reports will be generated on the bounding box level. For using bounding boxes please sign up
            for our free beta program at https://visual-layer.com or send an email to info@databasevisual.com.

        work_dir (str): Path for storing intermediate files and results.

        test_dir (str): Optional path for test data. When given similarity of train and test images is compared (vs. train/train or test/test which are not performed).
            The following options are supported.
                * test_dir can be a local folder path
                * An s3:// or minio:// path.
                * A python list with absolute filenames
                * A file containing absolute filenames each on its own row.

        compute (str): Compute type [cpu|gpu] Note: gpu is supported only in the enterprise version.

        verbose (boolean): Verbosity.

        num_threads (int): Number of threads. If no value is specified num threads is auto configured by the number of cores.

        num_images (unsigned long long): Number of images to run on. On default, run on all the images in the image_dir folder.

        turi_param (str): Optional turi parameters seperated by command. Example run: turi_param='nnmodel=0,ccthreshold=0.99'
            The following parameters are supported.
                * nnmodel=xx, Nearest Neighbor model for clustering the features together. Supported options are 0 = brute_force (exact), 1 = ball_tree and 2 = lsh (both approximate).
                * ccthreshold=xx, Threshold for running connected components to find clusters of similar images. Allowed values 0->1. The default ccthreshold is 0.96. This groups very similar images together, for example identical images or images that went
                    simple transformations like scaling, flip, zoom in. As higher the score the more similar images are grouped by and you will get \
                    smaller clusters. Score 0.9 is pretty broad and will clsuter images together even if they fine details are not similar. \
                    It is recommended to experiment with this parameter based on your dataset and then visualize the results using `fastdup.create_components_gallery()`.
                * run_cc=0|1 run connected components on the resulting similarity graph. Default is 1.
                * run_pagerank=0|1 run pagerank on the resulting similarity graph. Default is 1.
                * delete_tar=0|1 when working with tar files obtained from cloud storage delete the tar after download
                * delete_img=0|1 when working with images obtained from cloud storage delete the image after download
                * tar_only=0|1 run only on tar files and ignore images in folders. Default is 0.
                * run_stats=0|1 compute image statistics. Default is 1.
                * sync_s3_to_local=0|1 In case of using s3 bucket sync s3 to local folder to improve performance. Assumes there is enough local disk space to contain the dataDefault is 0.\


        distance (str): Distance metric for the Nearest Neighbors algorithm. The default is 'cosine' which works well in most cases.
            For nn_provider='nnf' the following distance metrics are supported.
            When using nnf_mode='Flat': 'cosine', 'euclidean', 'l1','linf','canberra','braycurtis','jensenshannon' are supported.
            Otherwise 'cosine' and 'euclidean' are supported.

        threshold (float): Similarity measure in the range 0->1, where 1 is totally identical, 0.98 and above is almost identical.

        lower_threshold (float): Similarity percentile measure to outline images that are far away (outliers) vs. the total distribution. (means 5% out of the total similarities computed).

        model_path (str): Optional location of ONNX model file, should not be used.

        version (bool): Print out the version number. This function takes no argument.

        nearest_neighbors_k (int): For each image, how many similar images to look for.

        d (int): Length of the feature vector. On default it is 576. When you use your own onnx model, change this parameter to the output model feature vector length.

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

        nn_provider (string): Provider of the nearest neighbor algorithm, allowed values are nnf.

        min_offset (unsigned long long): Optional min offset to start iterating on the full file list.

        max_offset (unsigned long long): Optional max offset to start iterating on the full file list.

        nnf_mode (str): When nn_provider='nnf' selects the nnf model mode.
            default is HSNW32. More accurate is Flat.

        nnf_param (str): When nn_provider='nnf' selects assigns optional parameters.
            ==num_em_iter=XX==, number of KMeans EM iterations to run. Default is 20.\
            ==num_clusters=XX==, number of KMeans clusters to use. Default is 100.\

        bounding_box (str): Optional bounding box to crop images, given as bounding_box='row_y=xx,col_x=xx,height=xx,width=xx'. This defines a global bounding box to be used for all images.
            Beta release features (need to sign up at https://visual-layer.com): Tt is possible to set bounding_box='face' to crop the face from the image (in case a face is present).
            In addition, you can set bounding_box='yolov5s' and we will run yolov5s to create and crop bounding boxes on your data. (We do not host this model, it is downloaded from the relevant github proejct).
            For the face/yolov5 crop the margin around the face is defined by turi_param='augmentation_horiz=0.2,augmentation_vert=0.2' where 0.2 mean 20% additional margin around the face relative to the width and height respectively.
            It is possible to change the margin, the lowest value is 0 (no margin) and upper allowed value is 1. Default is 0.2.

        batch_size (int): Optional batch size when computing inference. Allowed values < 200. Note: batch_size > 1 is enabled in the enterprise version.

        resume (int): Optional flag to resume from a previous run.

        high_accuracy (bool): Compute a more accurate model. Runtime is increased about 15% and feature vector storage size/ memory is increased about 60%. The upside is the model can distinguish better of minute details in images with many objects.

    Returns:
        ret (int): Status code 0 = success, 1 = error.

    '''
    fastdup_capture_log_debug_state(locals())

    _input_dir = input_dir
    fd_model = False
    if bounding_box == 'face' or bounding_box == 'yolov5s' or bounding_box == "rotated":
        local_model = os.path.join(LOCAL_DIR, 'UndisclosedFDModel.onnx')
        if bounding_box == 'yolov5s':
            local_model = find_model(YOLOV5S_MODEL)
        elif bounding_box == 'rotated':
            local_model = model_path_full

        bounding_box = ''
        turi_param += ",save_crops=1"
        if 'augmentation_horiz' not in turi_param and 'augmentation_vert' not in turi_param:
            turi_param += ",augmentation_horiz=0.2,augmentation_vert=0.2"
        ret = do_run(input_dir=input_dir,
                     work_dir=work_dir,
                     test_dir=test_dir,
                     compute=compute,
                     verbose=verbose,
                     num_threads=num_threads,
                     num_images=num_images,
                     turi_param=turi_param.replace(',shorten_filenames=1','').replace('shorten_filenames=1',''),
                     distance=distance,
                     threshold=threshold,
                     lower_threshold=lower_threshold,
                     model_path=local_model,
                     license=license,
                     version=version,
                     nearest_neighbors_k=nearest_neighbors_k,
                     d=d,
                     run_mode=1,
                     nn_provider=nn_provider,
                     min_offset=min_offset,
                     max_offset=max_offset,
                     nnf_mode=nnf_mode,
                     nnf_param=nnf_param,
                     bounding_box=bounding_box,
                     batch_size = batch_size,
                     resume = resume,
                     high_accuracy=high_accuracy)
        if (ret != 0):
            print("Failed to run fastdup")
            return ret
        try:
            os.unlink(os.path.join(work_dir, 'atrain_features.dat'))
        except:
            pass
        input_dir = os.path.join(tempfile.gettempdir(), 'yolo_crops_input.csv')
        pd.read_csv(os.path.join(work_dir, 'atrain_crops.csv'))[[
            'index', 'filename', 'col_x', 'row_y', 'width', 'height']].to_csv(input_dir, index=False)

        turi_param = turi_param.replace(',save_crops=1', '')

    ret = do_run(input_dir=input_dir,
             work_dir=work_dir,
             test_dir=test_dir,
             compute=compute,
             verbose=verbose,
             num_threads=num_threads,
             num_images=num_images,
             turi_param=turi_param if not fd_model else turi_param.replace(',save_crops=1','').replace('save_crops=1',''),
             distance=distance,
             threshold=threshold,
             lower_threshold=lower_threshold,
             model_path=model_path,
             license=license,
             version=version,
             nearest_neighbors_k=nearest_neighbors_k,
             d=d,
             run_mode=run_mode,
             nn_provider=nn_provider,
             min_offset=min_offset,
             max_offset=max_offset,
             nnf_mode=nnf_mode,
             nnf_param=nnf_param,
             bounding_box=bounding_box,
             batch_size = batch_size,
             resume = resume,
             high_accuracy=high_accuracy)
    return ret

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
                      license='',
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
    fastdup_capture_log_debug_state(locals())

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
        >>> import fastdup
        >>> file_list, mat_features = fastdup.load_binary(FILENAME_FEATURES)

    '''

    if not os.path.exists(filename) or not os.path.exists(filename + '.csv'):
        print("Error: failed to find the binary feature file:", filename, ' and the filenames csv file:', filename + '.csv')
        return None
    assert(d > 0), "Feature vector length d has to be larger than zero"

    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype='<f')

    df = pd.read_csv(filename + '.csv')['filename'].values
    assert df is not None, "Failed to read input file " + filename
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
    fastdup_capture_log_debug_state(locals())

    assert isinstance(save_path, str)  and save_path.strip() != "", "Save path should be a non empty string"
    assert isinstance(filenames, list), "filenames should be a list of image files"
    assert filenames is not None and len(filenames), "filenames should be a non empty list"
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


def load_config(work_dir):
    try:
        if work_dir == '':
            work_dir = '.'
        if os.path.exists(f'{work_dir}/config.json'):
            with open(f'{work_dir}/config.json') as f:
                return json.load(f)
    except:
        print(f"Failed to read config file {work_dir}/config.json")
        return None

def check_params(work_dir, num_images, lazy_load, get_label_func, slice, save_path, max_width):
    assert num_images >= 1, "Please select one or more images using num_images=xx flag"
    if num_images > 1000 and not lazy_load:
        print("Warning: When plotting more than 1000 images, please run with lazy_load=True. Chrome and Safari support lazy loading of web images, otherwise the webpage gets too big")

    if (get_label_func is not None):
        assert callable(get_label_func) or isinstance(get_label_func, dict) or (isinstance(get_label_func, str) and \
        os.path.exists(get_label_func)), "get_label_func has to be a callable function or a dictionary, given the filename returns the "
        "label of the file. Alternatively get_label_func can be a file with header of index,label and a single line of labels per"
        "image. The label file has to have the `atrain_features.dat.csv`"

    if slice is not None and get_label_func is None:
        assert False, "When slicing on specific labels need to provide a function to get the label (using the parameter get_label_func"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        assert os.path.exists(save_path), f"Failed to generate save_path directory {save_path}"

    if isinstance(work_dir, str):
        assert os.path.exists(work_dir), "Failed to find file or work_dir (__init__) " + work_dir

        subfolder = os.path.join(save_path, "images")
        if lazy_load and not os.path.exists(subfolder):
            os.mkdir(subfolder)
            assert os.path.exists(subfolder), f"Failed to generate sub folder images in {subfolder} when lazy_load=True"

    if max_width is not None:
        assert isinstance(max_width, int), "html image width should be an integer"
        assert max_width > 0, "html image width should be > 0"


    return 0



def load_dataframe(file_type, type, input_dir, work_dir, kwargs, cols):
    fastdup_capture_log_debug_state(locals())

    assert type in ["similarity","outliers"]
    nrows = None
    if 'nrows' in kwargs:
        nrows = kwargs['nrows']

    if isinstance(file_type, pd.DataFrame):
        if nrows is not None and file_type is not None and len(file_type) > nrows:
            file_type = file_type.head(nrows)

    elif isinstance(file_type, str):
        assert os.path.exists(file_type), "Failed to find similarity file " + file_type
        if os.path.isdir(file_type):
            file_type = os.path.join(file_type, FILENAME_SIMILARITY if type == "similarity" else FILENAME_OUTLIERS)

        if file_type.endswith('.csv'):
            hierarchical_run = 'hierarchical' in file_type
            kwargs['hierarchical_run'] = hierarchical_run

            if hierarchical_run:
                assert work_dir is not None and os.path.isdir(work_dir), "When running hierarchical clustering, need to provide the work_dir"
            kwargs['hierarchical_threshold'] = os.path.basename(file_type).split('_')[-1].replace('.csv', '')
            if 'debug_hierarchical' in kwargs:
                print('Found debug hierarchical', kwargs['hierarchical_threshold'], kwargs['hierarchical_run'])

        config = load_config(os.path.dirname(file_type))
        file_type = pd.read_csv(file_type, nrows=nrows)
        if input_dir is None and config is not None  and 'input_dir' in config:
            input_dir = config['input_dir']

    else:
        print('wrong type of similarity file', type(file_type))
        return None, None

    assert isinstance(file_type, pd.DataFrame)
    assert len(file_type), "Found empty dataframe"
    for col in cols:
        assert col in file_type.columns, f"Failed to find column {col} in dataframe"
    return file_type, input_dir



def remove_duplicate_video_distances(df, kwargs):
    fastdup_capture_log_debug_state(locals())

    #remove duplicate indications into the same video
    df['subfolder1'] = df['from'].apply(lambda x: os.path.dirname(x))
    df['subfolder2'] = df['to'].apply(lambda x: os.path.dirname(x))
    df = df[df['subfolder1'] != df['subfolder2']]
    if len(df) == 0:
        print("Error: failed to find links pointing between videos")
        return None

    if 'debug_video_size' in kwargs:
        print(df.head())

    #sort the dataframe by similarity
    df = df.sort_values(by=['distance'], ascending=False)
    assert len(df), "Empty dataframe"
    if 'debug_video_size' in kwargs:
        print(df.head())


    sizes = df.groupby(['subfolder1', 'subfolder2']).size().reset_index(name='counts')

    #keep only one example image across videos
    df = df.drop_duplicates(subset=['subfolder1','subfolder2'], keep='first')
    df = df.merge(sizes, how='left', left_on=['subfolder1', 'subfolder2'], right_on=['subfolder1', 'subfolder2'])
    if 'debug_video_size' in kwargs:
        print('Video size', df.head())

    if len(df) == 0:
        print("Error, failed to find any duplicate videos")
        return None

    return df

def create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True,
                              lazy_load=False, get_label_func=None, slice=None, max_width=None,
                              get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None,
                              input_dir=None, work_dir=None, threshold=None, **kwargs):
    '''

    Function to create and display a gallery of duplicate/near duplicate images as computed by the similarity metric.

    In addition, it is possible to compute hierarchical gallery of duplicate/near duplicate clusters. For doing so need to
        (A) Run fastdup to compute similarity on work_dir
        (B) Run connected components on the work_dir saving the component results to save_path (need to run with lazy_load=True)
        (C) Run create_duplicates_gallery() on the components to find pairs of similar components. Point the similarity_file to similarity_hierarchical_XX.csv file where XX is the
        connected components threshold (ccthreshold=XX).

    Example:
        >>> import fastdup
        >>> fastdup.run('input_folder', 'output_folder')
        >>> fastdup.create_duplicates_gallery('output_folder', save_path='.', get_label_func = lambda x: x.split('/')[1], slice='hamburger')

    Regarding get_label_func, this example assumes that the second folder name is the class name for example my_data/hamburger/image001.jpg. You can change it to match your own labeling convention.


    Args:
        similarity_file (str): csv file with the computed similarities by the fastdup tool, or a work_dir path, or a pandas dataframe containing the similarities.

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.
            slice could be a specific label i.e. slice='haumburger' and in that case only similarities between hamburger and other classes are presented.
            Two reserved arguments for slice are "diff" and "same". When using "diff" the report only shows similarities between classes. When using "same" the report will show only similarities inside same class.
            Note that when using slice, the function get_label_function should be implmeneted.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding additional column to the report

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        work_dir (str): Optional parameter to specify fastdup work_dir, when using a pd.DataFrame instead of a duplicate file path

        threshold (float): Optional parameter to specify the threshold for similarity score to be considered as duplicate. Values above the threshold will be considered as duplicate.
            Allowed values are between 0 and 1.

        save_artifacts (boolean): Optional parameter to allow saving the intermediate artifacts (raw images, csv with results) to the output folder

   '''
    fastdup_capture_log_debug_state(locals())

    try:
        start_time = time.time()
        ret = check_params(similarity_file, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret;

        similarity_file, input_dir = load_dataframe(similarity_file, "similarity", input_dir, work_dir, kwargs, ["from", "to", "distance"])


        ret = do_create_duplicates_gallery(similarity_file, save_path, num_images, descending, lazy_load, get_label_func, slice, max_width, get_bounding_box_func,
                                            get_reformat_filename_func, get_extra_col_func, input_dir, work_dir, threshold, kwargs)
        fastdup_performance_capture("create_duplicates_gallery", start_time)
        return ret

    except Exception as ex:
        fastdup_capture_exception("create_duplicates_gallery", ex)


def create_duplicate_videos_gallery(similarity_file, save_path, num_images=20, descending=True,
                              lazy_load=False, get_label_func=None, slice=None, max_width=None,
                              get_bounding_box_func=None, get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None, work_dir=None, threshold=None, **kwargs):
    '''

    Function to create and display a gallery of duplicaate videos computed by the similarity metrics

    Example:
        >>> import fastdup
        >>> fastdup.run('input_folder', 'output_folder', run_mode=1)  # extract frames from videos
        >>> fastdup.run('input_folder', 'output_folder', run_mode=2)  # run fastdup
        >>> fastdup.create_duplicates_videos_gallery('output_folder', save_path='.')


    Args:
        similarity_file (str): csv file with the computed similarities by the fastdup tool, or a work_dir path, or a pandas dataframe containing the similarities.

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.
            slice could be a specific label i.e. slice='haumburger' and in that case only similarities between hamburger and other classes are presented.
            Two reserved arguments for slice are "diff" and "same". When using "diff" the report only shows similarities between classes. When using "same" the report will show only similarities inside same class.
            Note that when using slice, the function get_label_function should be implmeneted.

        max_width (int): Optional parameter to set the max width of the gallery.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists


        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding additional column to the report

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        work_dir (str): Optional parameter to specify fastdup work_dir, when using a pd.DataFrame instead of a duplicate file path

        threshold (float): Optional parameter to specify the threshold for similarity score to be considered as duplicate. Values above the threshold will be considered as duplicate.
            Allowed values are between 0 and 1.

        save_artifacts (boolean): Optional parameter to allow saving the intermediate artifacts (raw images, csv with results) to the output folder

   '''

    try:
        fastdup_capture_log_debug_state(locals())
        start_time = time.time()
        ret = check_params(similarity_file, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret

        if threshold:
            assert threshold >= 0 and threshold <= 1, "threshold should be between 0 and 1"

        if work_dir is None and isinstance(similarity_file, str):
            if  os.path.isdir(similarity_file):
                work_dir = similarity_file
            else:
                work_dir = os.path.dirname(os.path.abspath(similarity_file))

        df, input_dir = load_dataframe(similarity_file, "similarity", input_dir, work_dir, kwargs, ["from", "to", "distance"])
        if threshold is not None:
            df = df[df['distance'] >= threshold]

        df = remove_duplicate_video_distances(df, kwargs)
        kwargs['is_video'] = True

        ret = create_duplicates_gallery(df, save_path, num_images, descending, lazy_load, get_label_func, slice, max_width, get_bounding_box_func,
                                                get_reformat_filename_func, get_extra_col_func, input_dir, work_dir, threshold, **kwargs)
        fastdup_performance_capture("create_duplicates_gallery", start_time)
        return ret
    except Exception as ex:
        fastdup_capture_exception( "create_duplicates_gallery", ex)

def create_outliers_gallery(outliers_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            how='one', slice=None, max_width=None, get_bounding_box_func=None,
                            get_reformat_filename_func=None, get_extra_col_func=None, input_dir =None, work_dir=None, **kwargs):
    '''

    Function to create and display a gallery of images computed by the outliers metrics.
    Outliers are computed using the fastdup tool, by embedding each image to a short feature vector, finding top k similar neighbors
    and finding images that are further away from all other images, i.e. outliers.
    On default fastdup saves the outliers into a file called `outliers.csv` inside the `work_dir` folder.
    It is possible to load this file using pandas to get the list of outlir images.
    Note that the number of images included in the outliers file depends on the `lower_threshold` parameter in the fastdup run. This command line argument is a percentile
    i.e. 0.05 means top 5% of the images that are further away from the rest of the images are considered outliers.

    Parameters:
        outliers_file (str): csv file with the computed outliers by the fastdup tool, or a work_dir path, or a pandas dataframe contraining the outliers

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        how (str): Optional outlier selection method. one = take the image that is far away from any one image (but may have other images close to it).
                                                      all = take the image that is far away from all other images. Default is one.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional parameter to set the max width of the gallery.

         get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.
            The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding additional column to the report

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        work_dir (str): Optional parameter to specify fastdup work_dir, when using a pd.DataFrame instead of a outliers file path

     '''

    try:
        fastdup_capture_log_debug_state(locals())

        start_time = time.time()
        ret = check_params(outliers_file, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret;

        if work_dir is None and isinstance(outliers_file, str):
            if os.path.isdir(outliers_file):
                work_dir = outliers_file
            else:
                work_dir = os.path.dirname(outliers_file)
        outliers_file, input_dir = load_dataframe(outliers_file, "outliers", input_dir, work_dir, kwargs, ['from', "to", "distance"])
        assert how == 'one' or how == 'all', "Wrong argument to how=[one|all]"

        ret = do_create_outliers_gallery(outliers_file, save_path, num_images, lazy_load, get_label_func, how, slice,
                                          max_width, get_bounding_box_func, get_reformat_filename_func, get_extra_col_func, input_dir, work_dir,
                                        **kwargs)
        fastdup_performance_capture("create_outliers_gallery", start_time)
        return ret


    except Exception as ex:
            fastdup_capture_exception("create_outliers_gallery", ex)

def create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                              group_by='visual', slice=None, max_width=None, max_items=None, get_bounding_box_func=None,
                              get_reformat_filename_func=None, get_extra_col_func=None, threshold=None, metric=None,
                              descending=True, min_items=None, keyword=None, input_dir=None, **kwargs):
    '''

    Function to create and display a gallery of images for the largest graph components

    Args:
        work_dir (str): path to fastdup work_dir, or a path to connected component csv file. Altenatively dataframe with connected_compoennts.csv content from previous fastdup run.

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional parameter to set the max html width of images in the gallery. Default is None.

        max_items (int): Optional parameter to limit the number of items displayed (labels for group_by='visual' or components for group_by='label'). Default is None.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.  The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding more information to the report.

        threshold (float): Optional parameter to set the treshold for chosing components. Default is None.

        metric (str): Optional parameter to set the metric to use (like blur) for chose components. Default is None.

        descending (boolean): Optional parameter to set the order of the components. Default is True namely list components from largest to smallest.

        min_items (int): Optional parameter to select components with min_items or more items. Default is None.

        keyword (str): Optional parameter to select components with keyword asa subset of the label. Default is None.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        kwargs (dict): Optional parameter to pass additional parameters to the function.
        
        split_sentence_to_label_list (boolean): Optional parameter to split the label into a list of labels. Default is False.
           
        limit_labels_printed (int): Optional parameter to limit the number of labels printed in the html report. Default is max_items.
        nrows (int): limit the number of read rows for debugging purposes of the report
        save_artifacts (bool): Optional param to save intermediate artifacts like image paths used for generating the component

    Returns:
        ret (int): 0 in case of success, otherwise 1
    '''

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())
        ret = check_params(work_dir, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret

        if max_items is not None:
            assert isinstance(max_items, int), "max items should be an integer"
            assert max_items > 0, "html image width should be > 0"

        if isinstance(work_dir, str):
            config = load_config(os.path.dirname(work_dir))
            if input_dir is None and config is not None and 'input_dir' in config:
                input_dir = config['input_dir']
        elif isinstance(work_dir, pd.DataFrame):
            assert input_dir is not None, "When passing dataframe need to point input_dir to the previous work_dir"
            assert len(work_dir), "Empty dataframe encountered"
            assert 'component_id' in work_dir.columns, "Connected components dataframe should contain 'component_id' column"
            assert '__id' in work_dir.columns or 'len' in work_dir.columns, "Connected components dataframe should contain '__id' column"
        else:
            assert False, f"Wrong work_dir type {type(work_dir)}"

        ret = do_create_components_gallery(work_dir, save_path, num_images, lazy_load, get_label_func, group_by, slice,
                                            max_width, max_items, min_items, get_bounding_box_func,
                                            get_reformat_filename_func, get_extra_col_func, threshold, metric=metric,
                                            descending=descending, keyword=keyword, comp_type="component", input_dir=input_dir, kwargs=kwargs)
        fastdup_performance_capture("create_components_gallery", start_time)
        return ret

    except Exception as ex:
        fastdup_capture_exception("create_components_gallery", ex)


def create_component_videos_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                              group_by='visual', slice=None, max_width=None, max_items=None, get_bounding_box_func=None,
                              get_reformat_filename_func=None, get_extra_col_func=None, threshold=None, metric=None,
                              descending=True, min_items=None, keyword=None, input_dir=None, **kwargs):
    '''

    Function to create and display a gallery of similar videos based on the graph components

    Args:
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional parameter to set the max html width of images in the gallery. Default is None.

        max_items (int): Optional parameter to limit the number of items displayed (labels for group_by='visual' or components for group_by='label'). Default is None.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.  The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding more information to the report.

        threshold (float): Optional parameter to set the treshold for chosing components. Default is None.

        metric (str): Optional parameter to set the metric to use (like blur) for chose components. Default is None.

        descending (boolean): Optional parameter to set the order of the components. Default is True namely list components from largest to smallest.

        min_items (int): Optional parameter to select components with min_items or more items. Default is None.

        keyword (str): Optional parameter to select components with keyword asa subset of the label. Default is None.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

    Returns:
        ret (int): 0 in case of success, otherwise 1
    '''

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        kwargs['is_video'] = True
        df, input_dir = load_dataframe(work_dir, "similarity", input_dir, work_dir, kwargs, ["from", "to", "distance"])
        df = remove_duplicate_video_distances(df, kwargs)
        if df is None:
            return 1

        ret = create_components_gallery(work_dir, save_path=save_path, num_images=num_images, lazy_load=lazy_load,
                                         get_label_func=get_label_func, group_by=group_by, slice=slice,
                                         max_width=max_width, max_items=max_items, get_bounding_box_func=get_bounding_box_func,
                                         get_reformat_filename_func=get_reformat_filename_func, get_extra_col_func=get_extra_col_func, threshold=threshold, metric=metric,
                                         descending=descending, min_items=min_items, keyword=keyword, comp_type="component",
                                         input_dir=input_dir, **kwargs)
        fastdup_performance_capture("create_component_video_gallery", start_time)
        return ret

    except Exception as ex:
            fastdup_capture_exception("create_component_videos_gallery", ex)

def create_kmeans_clusters_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            slice=None, max_width=None, max_items=None, get_bounding_box_func=None,
                              get_reformat_filename_func=None, get_extra_col_func=None, threshold=None, metric=None,
                              descending=True, min_items=None, keyword=None, input_dir=None, **kwargs):
    '''
    Function to visualize the kmeans clusters.

    Args:
    
        work_dir (str): path to fastdup work_dir

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        slice (str or list): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Optional parameter to set the max html width of images in the gallery. Default is None.

        max_items (int): Optional parameter to limit the number of items displayed (labels for group_by='visual' or components for group_by='label'). Default is None.

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.  The input is an absolute path to the image and the output is the string to display instead of the filename.

        get_extra_col_func (callable): Optional parameter to allow adding more information to the report.

        threshold (float): Optional parameter to set the treshold for chosing components. Default is None.

        metric (str): Optional parameter to set the metric to use (like blur) for chose components. Default is None.

        descending (boolean): Optional parameter to set the order of the components. Default is True namely list components from largest to smallest.

        min_items (int): Optional parameter to select components with min_items or more items. Default is None.

        keyword (str): Optional parameter to select components with keyword asa subset of the label. Default is None.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

    Returns:
         ret (int): 0 in case of success, otherwise 1
    '''

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        if isinstance(work_dir, str):
            config = load_config(os.path.dirname(work_dir))
            if input_dir is None and config is not None and 'input_dir' in config:
                input_dir = config['input_dir']

        ret = check_params(work_dir, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret

        ret = do_create_components_gallery(work_dir, save_path, num_images, lazy_load, get_label_func,
                                            'visual', slice, max_width, max_items, min_items, get_bounding_box_func,
                                            get_reformat_filename_func, get_extra_col_func, threshold, metric=metric,
                                            descending=descending, keyword=keyword, comp_type="cluster",
                                            input_dir=input_dir, kwargs=kwargs)
        fastdup_performance_capture("create_components_gallery", start_time)
        return ret

    except Exception as ex:
        fastdup_capture_exception("create_kmeans_clusters_gallery", ex)

def inner_delete(files, dry_run, how, save_path=None):
    fastdup_capture_log_debug_state(locals())

    if how == 'move':
        assert save_path is not None and os.path.exists(save_path)

    count = 0
    for f in files:
        if (dry_run):
            if how == 'delete':
                print(f'rm -f {f}')
            elif how == 'move':
                print(f'mv {f} {save_path}')
        else:
            try:
                if how == 'delete':
                    os.unlink(f)
                elif how == 'move':
                    os.rename(f, os.path.join(save_path, os.path.basename(f)))
                else:
                    assert False, "Wrong argument to how=[delete|move]"
                count+=1

            except Exception as ex:
                print(f'Failed to {how} file', f, ' with exception', ex)
    if not dry_run:
        print(f'total {how}d', count, 'files')
    return 0



def inner_retag(files, labels=None, how='retag=labelImg', save_path=None):
    fastdup_capture_log_debug_state(locals())

    assert files is not None and len(files)
    assert how == 'retag=labelImg' or how == 'retag=cvat', "Currently only retag=labelImg is supported"
    if save_path:
        assert os.path.exists(save_path)

    from fastdup.label_img import do_export_to_labelimg
    from fastdup.cvat import do_export_to_cvat


    if how == 'retag=labelImg':
        return do_export_to_labelimg(files, labels, save_path)
    elif how == 'retag=cvat':
        return do_export_to_cvat(files, labels, save_path)
    else:
        assert False, "not supported"


def delete_components(top_components, to_delete = None,  how = 'one', dry_run = True):
    '''
    function to automate deletion of duplicate images using the connected components analysis.

        Example:
        >>> import fastdup
        >>> fastdup.run('/path/to/data', '/path/to/output')
        >>> top_components = fastdup.find_top_components('/path/to/output')
        >>> delete_components(top_components, None, how = 'one', dry_run = False)

    Args:
        top_components (pd.DataFrame): largest components as found by the function find_top_components().
        to_delete (list): a list of integer component ids to delete. On default None which means delete duplicates from all components.
        how (int): either 'all' (deletes all the component) or 'one' (leaves one image and delete the rest of the duplicates)
        dry_run (bool): if True does not delete but print the rm commands used, otherwise deletes

    Returns:
        ret (list): list of deleted files

    '''

    try:
        start_time = time.time()
        assert isinstance(top_components, pd.DataFrame), "top_components should be a pandas dataframe"
        assert len(top_components), "top_components should not be enpty"
        assert to_delete is None or isinstance(to_delete, list), "to_delete should be a list of integer component ids"
        if isinstance(to_delete, list):
            assert len(to_delete), "to_delete should not be empty"
            assert isinstance(to_delete[0], int) or isinstance(to_delete[0], np.int64), "to_delete should be a list of integer component ids"
        assert how == 'one' or how == 'all', "how should be one of 'one'|'all'"
        assert isinstance(dry_run, bool)

        if to_delete is None:
            to_delete = top_components['component_id'].tolist()

        total_deleted = []

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

            inner_delete(files, how='delete', dry_run=dry_run)
            total_deleted += files

        fastdup_performance_capture("delete_components", start_time)
        return total_deleted
    except Exception as ex:
        fastdup_capture_exception("delete_components", ex)


def delete_components_by_label(top_components_file,  min_items=10, min_distance=0.96,  how = 'majority', dry_run = True):
    '''
    function to automate deletion of duplicate images using the connected components analysis.

    Args:
        top_components (pd.DataFrame): largest components as found by the function find_top_components().
        to_delete (list): a list of integer component ids to delete
        how (int): either 'all' (deletes all the component) or 'majority' (leaves one image with the dominant label count and delete the rest)
        dry_run (bool): if True does not delete but print the rm commands used, otherwise deletes

    Returns:
        ret (list): list of deleted files

    '''
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        assert os.path.exists(top_components_file), "top_components_file should be a path to a file"

        # label is ,component_id,files,labels,to,distance,blur,len
        df = pd.read_pickle(top_components_file)
        df = df[df['distance'] >= min_distance]
        df = df[df['len'] >= min_items]

        total = []

        for i, comp in (df.iterrows()):
            files = comp['files']
            labels = comp['labels']
            assert len(files) == len(labels)
            assert len(files)>= min_items
            subdf = pd.DataFrame({'files':files, 'labels':labels})
            unique, counts = np.unique(np.array(labels), return_counts=True)
            counts_df = pd.DataFrame({"counts":counts}, index=unique).sort_values('counts', ascending=False)
            if (how == 'majority'):
                if (np.max(counts) >= len(subdf)/ 2):
                    sample = counts_df.index.values[0]
                    files = subdf[subdf['labels'] == sample]['files'].values
                    assert len(files)
                    subfile = files[1:]
                    inner_delete(subfile, how='delete', dry_run=dry_run)
                    total.extend(subfile)
                else:
                    inner_delete(files, how='delete', dry_run=dry_run)
                    total.extend(files)
            elif (how == 'all'):
                inner_delete(files, how='delete', dry_run=dry_run)
                total.extend(files)
            else:
                assert False, "how should be one of 'majority'|'all'"

        fastdup_performance_capture("delete_components_by_label", start_time)
        return total
    except Exception as e:
        fastdup_capture_exception("delete_components_by_label", e)

def delete_or_retag_stats_outliers(stats_file, metric, filename_col = 'filename', label_col=None, lower_percentile=None, upper_percentile=None,
                          lower_threshold=None, upper_threshold=None, get_reformat_filename_func=None, dry_run=True,
                          how='delete', save_path=None, work_dir=None):
    '''
    function to automate deletion of outlier files based on computed statistics.

    Example:
        >>> import fastdup
        >>> fastdup.run('/my/data/", work_dir="out")
        delete 5% of the brightest images and delete 2% of the darkest images
        >>> fastdup.delete_or_retag_stats_outliers("out", metric="mean", lower_percentile=0.05, dry_run=False)

        It is recommended to run with dry_run=True first, to see the list of files deleted before actually deleting.

    Example:
        This example first find wrong labels using similarity gallery and then deletes anything with score < 51.
        Score is in range 0-100 where 100 means this image is similar only to images from the same class label.
        Score 0 means this image is only similar to images from other class labels.
        >>> import fastdup
        >>> df2 = create_similarity_gallery(..., get_label_func=...)
        >>>fastdup.delete_or_retag_stats_outliers(df2, metric='score', filename_col = 'from', lower_threshold=51, dry_run=True)

    Note: it is possible to run with both `lower_percentile` and `upper_percentile` at once. It is not possible to run with `lower_percentile` and `lower_threshold` at once since they may be conflicting.

    Args:
        stats_file (str):
          * folder pointing to fastdup workdir or
          * file pointing to work_dir/atrain_stats.csv file or
          * pandas DataFrame containing list of files giveb in the filename_col column and a metric column.

        metric (str): statistic metric, should be one of "blur", "mean", "min", "max", "stdv", "unique", "width", "height", "size"

        filename_col (str): column name in the stats_file to use as the filename

        lower_percentile (float): lower percentile to use for the threshold. Values are 0->1, where 0.05 means remove 5% of the lowest values.

        upper_percentile (float): upper percentile to use for the threshold. Values are 0->1, where 0.95 means remove 5% of the upper values.

        lower_threshold (float): lower threshold to use for the threshold. Only used if lower_percentile is None.

        upper_threshold (float): upper threshold to use for the threshold. Only used if upper_percentile is None.

        get_reformat_filename_func (callable): Optional parameter to allow changing the  filename into another string. Useful in the case fastdup was run on a different folder or machine and you would like to delete files in another folder.

        dry_run (bool): if True does not delete but print the rm commands used, otherwise deletes

        how (str): either 'delete' or 'move' or 'retag'. In case of retag allowed value is retag=labelImg or retag=cvat

        save_path (str): optional. In case of a folder and how == 'retag' the label files will be moved to this folder.

        work_dir (str): optional. In case of stats dataframe, point to fastdup work_dir.


      Returns:
          ret (list): list of deleted files (or moved or retagged files)

    '''
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        assert isinstance(dry_run, bool)
        assert how == 'delete' or how == 'move' or how == 'retag', "how should be one of 'delete'|'move'|'retag'"
        if how == 'move':
            assert save_path is not None, "When how='move' need to provide save_path to move the files to"

        if lower_threshold is not None and lower_percentile is not None:
            assert False, 'You should only specify one of lower_threshold or lower_percentile'

        if upper_threshold is not None and upper_percentile is not None:
            assert False,  'You should only specify one of upper_threshold or upper_percentile'

        if isinstance(stats_file, pd.DataFrame):
            assert isinstance(work_dir, str) and os.path.exists(work_dir), "When providing pandas dataframe need to set work_dir to point to fastdup work_dir"
            df = stats_file
        else:
            df = load_stats(stats_file, work_dir, {})
        if metric == "score" and metric not in df.columns:
            assert False, "For removing wrong labels created by the create_similarity_gallery() need to run stats_file=df where df is the output of create_similarity_gallery()"


        assert metric in df.columns or metric=='size', f"Unknown metric {metric} options are {df.columns}"
        assert filename_col in df.columns
        if label_col:
            assert label_col in df.columns, f"{label_col} column should be in the stats_file"

        if metric == 'size':
            df['size'] = df.apply(lambda x: x['width'] * x['height'], axis=1)

        if lower_percentile is not None:
            assert lower_percentile >= 0 and lower_percentile <= 1, "lower_percentile should be between 0 and 1"
            lower_threshold = df[metric].quantile(lower_percentile)
        if upper_percentile is not None:
            assert upper_percentile >= 0 and upper_percentile <= 1, "upper_percentile should be between 0 and 1"
            upper_threshold = df[metric].quantile(upper_percentile)

        orig_df = df.copy()
        orig_len = len(df)

        if (lower_threshold is not None):
            print(f"Going to delete any images with {metric} < {lower_threshold}")
            df = orig_df[orig_df[metric] < lower_threshold]
            if (upper_threshold is not None):
                print(f"Going to delete any images with {metric} > {upper_threshold}")
                df = pd.concat([df, orig_df[orig_df[metric] > upper_threshold]], axis=0)
        elif (upper_threshold is not None):
                print(f"Going to delete any images with {metric} > {upper_threshold}")
                df = orig_df[orig_df[metric] > upper_threshold]
        else:
            assert(False), "You should specify either lower_threshold or upper_threshold or lower_percetiel or upper_percentile"


        if orig_len == len(df):
            print('Warning: current request to delete all files, please select a subset of files to delete.', orig_len, len(df))
            print(df[metric].describe(), lower_threshold, upper_threshold)
            return 0
        elif len(df) == 0:
            print('Did not find any items to delete, please check your selection')
            return 0


        if get_reformat_filename_func is None:
            files = df[filename_col].values
        else:
            files = df[filename_col].apply(get_reformat_filename_func).values

        if how == 'delete' or how == 'move':
            return inner_delete(files, how=how, dry_run=dry_run, save_path=save_path)
        elif how.startswith('retag'):
            if label_col is not None:
                label = df[label_col].values
            else:
                label = None
            return inner_retag(files, label, how, save_path)
        else:
            assert(False), "How should be one of 'delete'|'move'|'retag'"

        fastdup_performance_capture("delete_or_retag_stats_outliers", start_time)
        return files
    except Exception as e:
        fastdup_capture_exception("delete_or_retag_stats_outliers", e)

def export_to_tensorboard_projector(work_dir, log_dir, sample_size = 900,
                                    sample_method='random', with_images=True, get_label_func=None, d=576, file_list=None):
    '''
    Export feature vector embeddings to be visualized using tensorboard projector app.

    Example:
        >>> import fastdup
        >>> fastdup.run('/my/data/', work_dir='out')
        >>> fastdup.export_to_tensorboard_projector(work_dir='out', log_dir='logs')

        After data is exporeted run tensorboard projector
        >>> %load_ext tensorboard
        >>> %tensorboard --logdir=logs

    Args:
        work_dir (str): work_dir where fastdup results are stored

        log_dir (str): output dir where tensorboard will read from

        sample_size (int): how many images to view. Default is 900.

        sample_method (str): how to sample, currently 'random' is supported.

        with_images (bool): add images to the visualization (default True)

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        d (int): dimension of the embedding vector. Default is 576.

        file_list (list): Optional parameter to specify a list of files to be used for the visualization. If not specified, filenames are taken from the work_dir/atrain_features.dat.csv file
                      Note: be careful here as the order of the file_list matters, need to keep the exact same order as the atrain_features.dat.csv file!
    Returns:
        ret (int): 0 in case of success, 1 in case of failure
    '''

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        try:
            import tensorflow
            from tensorboard.plugins import projector
        except Exception as ex:
            print('For saving information for tensorboard project you need to install tensorflow. Please pip install tensorflow and tensorbaord and try again')
            fastdup_capture_exception("tensorflow import", ex)
            return 1


        from fastdup.tensorboard_projector import export_to_tensorboard_projector_inner
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
        fastdup_performance_capture("export_to_tensorboard_projector", start_time)

    except Exception as ex:
        fastdup_capture_exception("export_to_tensorboard_projector", ex)


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

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        h (int): optional requested hight of each subimage

        w (int): optional requested width of each subimage

        alternative_filename (str): optional parameter to save the resulting image to a different name

        alternative_width (int): optional parameter to control the number of images per row

        max_width (int): optional parameter to control the rsulting width of the image

    Returns:
        path (str): path to sprite image
        labels (list): list of labels

    '''
    try:
        assert len(img_list), "Image list is empty"
        assert sample_size > 0
        from fastdup.tensorboard_projector import generate_sprite_image as tgenerate_sprite_image
        ret = tgenerate_sprite_image(img_list, sample_size, log_dir, get_label_func, h=h, w=w,
                                      alternative_filename=alternative_filename, alternative_width=alternative_width, max_width=max_width)
        return ret
    except Exception as ex:
        fastdup_capture_exception("generate_sprite_image", ex)


def find_top_components(work_dir, get_label_func=None, group_by='visual', slice=None, threshold=None, metric=None,
                        descending=True, min_items=None, max_items = None, keyword=None,  save_path=None,
                        comp_type="component", **kwargs):
    '''
    Function to find the largest components of duplicate images

    Args:
        work_dir (str): working directory where fastdup.run was run.

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        group_by (str): optional parameter to group by 'visual' or 'label'. When grouping by visual fastdup aggregates visually similar images together.
            When grouping by 'label' fastdup aggregates images with the same label together.

        slice (str): optional parameter to slice the results by a specific label. For example, if you want to slice by 'car' then pass 'car' as the slice parameter.

        threshold (float): optional threshold to select only distances larger than the treshold

        metric (str): optional metric to sort by. Valid values are mean,min,max,unique,blur,size

        descending (bool): optional value to sort the components, default is True

        min_items (int): optional value, select only components with at least min_items

        max_items (int): optional value, select only components with at most max_items

        keyword (str): optional, select labels with keyword  value inside

        save_path (str): optional, save path

        comp_type (str): optional, either component or cluster

    Returns:
        df (pd.DataFrame): of top components. The column component_id includes the component name.
            The column files includes a list of all image files in this component.


    '''
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        from .galleries import do_find_top_components
        ret = do_find_top_components(work_dir, get_label_func, group_by, slice, threshold=threshold,
                                      metric=metric, descending=descending, min_items=min_items, max_items = max_items,
                                      keyword=keyword, save_path=save_path, comp_type=comp_type, kwargs=kwargs)
        fastdup_performance_capture("find_top_components", start_time)
        return ret
    except Exception as ex:
        fastdup_capture_exception("find_top_components", ex)

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

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

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

        fastdup_performance_capture("init_search", start_time)
    except Exception as e:
        fastdup_capture_exception("init_search", e)
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

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

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

        fastdup_performance_capture("search", start_time)
        return 0
    except Exception as e:
        fastdup_capture_exception("search", e)


def create_stats_gallery(stats_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                            metric='blur', slice=None, max_width=None, descending= False, get_bounding_box_func=None,
                         get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None, work_dir=None, **kwargs):
    '''
    Function to create and display a gallery of images computed by the statistics metrics.
    Supported metrics are: mean (color), max (color), min (color), stdv (color), unique (number of unique colors), bluriness (computed by the variance of the laplpacian method
    see https://theailearner.com/2021/10/30/blur-detection-using-the-variance-of-the-laplacian-method/.
    The metrics are created by fastdup.run() and stored into the `work_dir` into a file named `atrain_stats.csv`. Note that the metrics are computed
    on the fly fastdup loads and resizes every image only once.

    Args:
        stats_file (str): csv file with the computed image statistics by the fastdup tool, alternatively a pandas dataframe. Default stats file is saved by fastdup.run() into the folder `work_dir` as `atrain_stats.csv`.

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        metric (str): Optional metric selection. Supported metrics are:
            * width - of original image before resize
            * height - of original image before resize
            * size - area
            * file_size - file size in bytes
            * blur - variance of the laplacian
            * unique - number of unique colors, 0..255
            * mean - mean color 0..255
            * max - max color 0..255
            * min - min color 0..255
            Advanced metris include (for running advanced metrics, run with turi_param='run_advanced_stats=1')
            * contrast
            * rms_contrast - square root of mean sum of stdv/mean per channel
            * mean_rel_intensity_r
            * mean_rel_intensity_b
            * mean_rel_intensity_g
            * mean_hue - transform to HSV and compute mean H
            * mean_saturation - transform to HSV and compute mean S
            * mean_val - transform to HSV and compute mean V
            * edge_density - using canny filter
            * mean_r - mean of R channel
            * mean_g - mean of G channel
            * mean_b - mean of B channel


        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.

        max_width (int): Option parameter to select the maximal image width in the report

        descending (bool): Optional parameter to control the order of the metric

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

        work_dir (str): Optional parameter to fastdup work_dir. Needed when stats file is a pd.DataFrame.


    Returns:
        ret (int): 0 in case of success, otherwise 1.
    '''

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        ret = check_params(stats_file, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret

        assert metric in ['blur','size','mean','min','max','unique','stdv', 'file_size','rms_contrast','mean_rel_intensity_r',
                          'mean_rel_intensity_b','mean_rel_intensity_g','contrast','mean_saturation','mean_hue', 'mean_val', 'edge_density','mean_r', 'mean_g','mean_b'], "Unknown metric value: " + metric

        stats_file = load_stats(stats_file, work_dir, kwargs)
        try:
            import matplotlib
        except Exception as ex:
            print("Failed to import matplotlib. Please install matplotlib using 'python3.8 -m pip install matplotlib'")
            fastdup_capture_exception("create_stats_gallery", ex)
            return 1


        ret = do_create_stats_gallery(stats_file, save_path, num_images, lazy_load, get_label_func, metric, slice, max_width,
                                       descending, get_bounding_box_func, get_reformat_filename_func, get_extra_col_func, input_dir, work_dir, kwargs=kwargs)
        fastdup_performance_capture("create_stats_gallery", start_time)
        return ret

    except Exception as e:
        fastdup_capture_exception("create_stats_gallery", e)

def create_similarity_gallery(similarity_file, save_path, num_images=20, lazy_load=False, get_label_func=None,
                                 slice=None, max_width=None, descending=False, get_bounding_box_func=None,
                                 get_reformat_filename_func=None, get_extra_col_func=None, input_dir=None, work_dir=None,
                                 min_items=2, max_items=None, **kwargs):
    '''

    Function to create and display a gallery of images computed by the similarity metric. In each table row one query image is
    displayed and `num_images` most similar images are displayed next to it on the right.

    In case the dataset is labeled, the user can specify the label using the function `get_label_func`. In this case a `score` metric is computed to reflect how similar the query image to the most similar images in terms of class label.
    Score 100 means that out of the top k num_images similar images, all similar images are from the same class. Score 0 means that the image is similar only to images which are from different class.
    Score 50 means that the query image is similar to the same number of images from the same class and from other classes. The report is sorted by the score metric.
    For high quality labeled dataset we expect the score to be high, low score may indicate class label issues.

    Args:
        similarity_file (str): csv file with the computed image statistics by the fastdup tool, or a path to the work_dir,
            alternatively a pandas dataframe. In case of a pandas dataframe need to set work_dir to point to fastdup work_dir.

        save_path (str): output folder location for the visuals

        num_images (int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        slice (str): Optional parameter to select a slice of the outliers file based on a specific label or a list of labels.
            A special value is 'label_score' which is used for comparing both images and labels of the nearest neighbors. The score values are 0->100 where 0 means the query image is only similar to images outside its class, 100 means the query image is only similar to images from the same class.

        max_width (int): Optional param to limit the image width

        descending (bool): Optional param to control the order of the metric

        get_bounding_box_func (callable): Optional parameter to allow plotting bounding boxes on top of the image.
            The input is an absolute path to the image and the output is a list of bounding boxes.
            Each bounding box should be 4 integers: x1, y1, x2, y2. Example of valid bounding box list: [[0, 0, 100, 100]]
            Alternatively, get_bounding_box_func could be a dictionary returning the bounding box list for each filename.
            Alternatively, get_bounding_box_func could be a csv containing index,filename,col_x,row_y,width,height or a work_dir where the file atrain_crops.csv exists

        get_reformat_filename_func (callable): Optional parameter to allow changing the presented filename into another string.

        get_extra_col_func (callable): Optional parameter to allow adding extra columns to the report

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

        work_dir (str): Optional parameter to fastdup work_dir. Needed when similarity_file is a pd.DataFrame.

        min_items (int): Optional parameter to select components with min_items or more

        max_items (int): Optional parameter to limit the number of items displayed

    Returns:
        ret (pd.DataFrame): similarity dataframe, for each image filename returns a list of top K similar images.
            each row has the columns 'from', 'to', 'label' (optional), 'distance'
     '''

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        ret = check_params(similarity_file, num_images, lazy_load, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret

        similarity_file, input_dir = load_dataframe(similarity_file, "similarity", input_dir, work_dir, kwargs, ["from", "to", "distance"])

        ret = do_create_similarity_gallery(similarity_file, save_path, num_images, lazy_load, get_label_func,
            slice, max_width, descending, get_bounding_box_func, get_reformat_filename_func, get_extra_col_func, 
            input_dir,  work_dir, min_items, max_items, kwargs=kwargs)
        fastdup_performance_capture("create_similarity_gallery", start_time)
        return ret

    except Exception as e:
        fastdup_capture_exception("create_similarity_gallery", e)
        return None


def create_aspect_ratio_gallery(stats_file, save_path, get_label_func=None, lazy_load=False, max_width=None, num_images=0, slice=None,
                                get_filename_reformat_func=None, input_dir=None, **kwargs):
    '''
    Function to create and display a gallery of aspect ratio distribution.

    Args:
         stats_file (str): csv file with the computed image statistics by the fastdup tool, or work_dir path or a pandas dataframe with the stats compouted by fastdup.

        save_path (str): output folder location for the visuals

        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).


         max_width (int): optional parameter to limit the plot image width



         num_images (int): optional number of images to compute the statistics on (default computes on all images)

         slice (str): optional parameter to slice the stats file based on a specific label or a list of labels.

         get_filename_reformat_func (callable): optional function to reformat the filename before displaying it.

        input_dir (str): Optional parameter to specify the input directory of webdataset tar files,
            in case when working with webdataset tar files where the image was deleted after run using turi_param='delete_img=1'

    Returns:
        ret (int): 0 in case of success, otherwise 1.
    '''
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        ret = check_params(stats_file, 1, False, get_label_func, slice, save_path, max_width)
        if ret != 0:
            return ret

        stats_file = load_stats(stats_file, kwargs)

        try:
            import matplotlib
        except Exception as e:
            fastdup_capture_exception("create_aspect_ratio_gallery", e)
            print("Failed to import matplotlib. Please install matplotlib using 'python3.8 -m pip install matplotlib'")
            return 1


        ret = do_create_aspect_ratio_gallery(stats_file, save_path, get_label_func, lazy_load, max_width, num_images, slice, input_dir, kwargs=kwargs)
        fastdup_performance_capture("create_aspect_ratio_gallery", start_time)
        return ret

    except Exception as e:
        fastdup_capture_exception("create_aspect_ratio_gallery", e)


def export_to_cvat(files, labels, save_path):
    """
    Function to export a collection of files that needs to be annotated again to cvat batch job format.
    This creates a file named fastdup_label.zip in the directory save_path.
    The files can be retagged in cvat using Tasks -> Add (plus button) -> Create from backup -> choose the location of the fastdup_label.zip file.

    Args:
        files (str):
        labels (str):
        save_path (str):

    Returns:
        ret (int): 0 in case of success, otherwise 1.
    """
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        assert len(files), "Please provide a list of files"
        assert labels is None or isinstance(labels, list), "Please provide a list of labels"

        from fastdup.cvat import do_export_to_cvat
        ret =  do_export_to_cvat(files, labels, save_path)
        fastdup_performance_capture("export_to_cvat", start_time)
        return ret
    except Exception as e:
        fastdup_capture_exception("export_to_cvat", e)

def export_to_labelImg(files, labels, save_path):
    """
    Function to export a collection of files that needs to be annotated again to cvat batch job format.
    This creates a file named fastdup_label.zip in the directory save_path.
    The files can be retagged in cvat using Tasks -> Add (plus button) -> Create from backup -> choose the location of the fastdup_label.zip file.

    Args:
        files (str):
        labels (str):
        save_path (str):

    Returns:
        ret (int): 0 in case of success, otherwise 1.
    """
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        assert len(files), "Please provide a list of files"
        assert labels is None or isinstance(labels, list), "Please provide a list of labels"

        from fastdup.label_img import do_export_to_labelimg
        ret =  do_export_to_labelimg(files, labels, save_path)
        fastdup_performance_capture("export_to_labelImg", start_time)
        return ret
    except Exception as e:
        fastdup_capture_exception("export_to_labelImg", e)
        return 1

def top_k_label(labels_col, distance_col, k=10, threshold = None,  min_count=None, unknown_class=None):
    '''
    Function to classify examples based on their label using the top k nearest neighbors.
    Decision is made by accounting for the majority of the neighbors.

    Args:
        labels_col (list): list of labels
        distance_col (list): list of distances
        k (int): optional parameter
        threshold (float): optional parameter to consder neighbors with simiarity larger than threshold
        min_count (int): optional parameter to consider only examples with at least min_count neighbors with the same label
        unknown_class: optional parameter to add decisions to unknown class in cases there is no majority
    Returns:
        computed label
    '''
    assert len(labels_col), "Empty dataframe recevieved"
    df = pd.DataFrame({'labels':labels_col, 'distance':distance_col})

    if threshold is not None:
        df = df[df['distance'] >= threshold]

    ret = df.groupby('labels').agg('count')[['distance']]
    ret = ret.rename({'distance':'count'}, axis=1)
    ret2 = df.groupby('labels')['distance'].apply(list).to_frame()
    ret2 = ret2.rename({'distance':'distance_list'},axis=1)

    #print(ret)
    #print(ret2)
    ret = ret.join(ret2)
    ret = ret.sort_values('count', ascending=False)
    #print(ret)

    label = ret.index.values[0]
    count = ret['count'].values[0]
    if len(ret) == 1:
        return label
    else:
        second_label = ret.index.values[1]
        second_count = ret['count'].values[1]
        if min_count is not None and count >= min_count:
            return label

        if count > second_count:
            return label

        if unknown_class is None:
            return label


        return unknown_class




def create_knn_classifier(work_dir, k, get_label_func, threshold=None):
    '''
    Function to create a knn classifier out of fastdup run. We assume there are existing labels to the datapoints.

    Args:
        work_dir (str): fastdup work_dir, or location of a similarity file, or a pandas DataFrame with the computed similarities
        k (int): (unused)
        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.
        threshold (float): optional threshold to consider neighbors with similarity larger than threshold
            prediction per image to one of the given classes.

    Returns:
        df (pd.DataFrame): List of predictions using knn method
    '''
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        from fastdup.confusion_matrix import classification_report

        assert os.path.exists(work_dir), "Failed to find work directory " + work_dir
        assert callable(get_label_func) or isinstance(get_label_func, dict) or (isinstance(get_label_func, str) and os.path.exists(get_label_func)), \
            "Please provide a valid callable function get_label_func, given a filename returns its string label or a list of labels, " \
            "or a dictionary where the key is the absolute file name and the value is the label or list of labels or a labels file with header index,label where" \
            "each row is a label corresponding to the image in the atrain_features_data.csv file"

        if threshold is not None:
            assert threshold >= 0 and threshold <= 1, "Please provide a valid threshold 0->1"

        if isinstance(work_dir, pd.DataFrame):
            df = work_dir
            assert len(df), "Empty dataframe received"
        else:
            if os.path.isdir(work_dir):
                similarity_file = os.path.join(work_dir, FILENAME_SIMILARITY)
            df = pd.read_csv(similarity_file)

        labels_dict = None
        if callable(get_label_func):
            df['to_label'] = df['to'].apply(get_label_func)
        elif isinstance(get_label_func, dict):
            df['to_label'] = df['to'].apply(lambda x: get_label_func.get(x, MISSING_LABEL))
        elif isinstance(get_label_func, str):
            labels_df = pd.read_csv(get_label_func)
            filenames_df = pd.read_csv(os.path.join(work_dir, FILENAME_IMAGE_LIST))
            if len(labels_df) != len(filenames_df):
                print('Error: labels file length does not match the number of images in the similarity file', get_label_func, len(labels_df), len(df))
                return None
            if 'label' not in labels_df.columns:
                print('Error: labels file does not contain a label column', get_label_func)
                return None
            filenames_df['label'] = labels_df['label']
            labels_dict = pd.Series(filenames_df.label.values,index=filenames_df.filename).to_dict()
            df['to_label'] = df['to'].apply(lambda x: labels_dict.get(x, MISSING_LABEL))

        from_list = df.groupby(by='from', axis=0)['to'].apply(list)
        distance_list = df.groupby(by='from', axis=0)['distance'].apply(list)
        to_label_list = df.groupby(by='from', axis=0)['to_label'].apply(list)

        df_from = from_list.to_frame()
        df_dist = distance_list.to_frame()
        df_label = to_label_list.to_frame()

        df_merge = df_from.merge(df_dist, on='from')
        df_merge = df_merge.merge(df_label, on='from')

        if callable(get_label_func):
            df_merge['from_label'] = df_merge.index.map(get_label_func)
        elif isinstance(get_label_func, dict):
            df_merge['from_label'] = df_merge.index.map(lambda x: get_label_func.get(x, MISSING_LABEL))
        elif isinstance(get_label_func, str):
            assert labels_dict is not None
            df_merge['from_label'] = df_merge.index.map(lambda x: labels_dict.get(x, MISSING_LABEL))


        df_merge['top_k'] = df_merge.apply(lambda x:
                                           top_k_label(x['to_label'], x['distance'], k, threshold), axis=1)

        y_values = df_merge['from_label'].tolist()
        p1_values = df_merge['top_k'].tolist()
        filenames = df_merge.index.tolist()
        print(classification_report(y_values, p1_values))
        fastdup_performance_capture("create_knn_classifier", start_time)
        return pd.DataFrame({'filename':filenames, 'prediction':p1_values, 'label':y_values})
    except Exception as ex:
        fastdup_capture_exception("create_knn_classifier", ex)
        return pd.DataFrame({'filename':[]})


def create_kmeans_classifier(work_dir, k, get_label_func, threshold=None):
    '''
    Function to create a knn classifier out of fastdup run. We assume there are existing labels to the datapoints.

    Args:
        work_dir (str): fastdup work_dir, or location of a similarity file, or a pandas DataFrame with the computed similarities
        k (int): (unused)
        get_label_func (callable): optional function given an absolute path to an image return the image label.
            Image label can be a string or a list of strings. Alternatively, get_label_func can be a dictionary where the key is the absolute file name and the value is the label or list of labels.
            Alternatively, get_label_func can be a filename containing string label for each file. First row should be index,label. Label file should be same length and same order of the atrain_features_data.csv image list file.
        threshold (float): (unused)

    Returns:
        df (pd.DataFrame): dataframe with filename, label and predicted label. Row per each image
    '''
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        from fastdup.confusion_matrix import classification_report

        assert callable(get_label_func) or isinstance(get_label_func, dict) or (isinstance(get_label_func, str) and os.path.exists(get_label_func)), \
            "Please provide a valid callable function get_label_func, given a filename returns its string label or a list of labels, " \
            "or a dictionary where the key is the absolute file name and the value is the label or list of labels or a labels file with header index,label where" \
            "each row is a label corresponding to the image in the atrain_features_data.csv file"

        comps = find_top_components(work_dir, get_label_func, 'visual', slice=None, comp_type='cluster')
        print(comps.columns)
        comps['top_k'] = comps.apply(lambda x:
                                           top_k_label(x['label'], x['distance'], k, threshold=threshold), axis=1)
        files = []
        y_values = []
        p1_values = []
        for i,row in comps.iterrows():
            cluster_label = row['top_k']
            for f,l in zip(row['files'], row['label']):
                files.append(f)
                y_values.append(l)
                p1_values.append(cluster_label)

        print(classification_report(y_values, p1_values))
        fastdup_performance_capture("create_kmeans_classifier", start_time)
        return pd.DataFrame({'prediction':p1_values, 'label':y_values, 'filename':files})

    except Exception as ex:
        fastdup_capture_exception("create_kmeans_classifier", ex)
        return pd.DataFrame({'filename':[]})

def run_kmeans(input_dir='',
               work_dir='.',
               verbose=False,
               num_clusters=100,
               num_em_iter=20,
               num_threads=-1,
               num_images=0,
               model_path=model_path_full,
               license='',            #license string
               nearest_neighbors_k=2,
               d=576,
               bounding_box="",
               high_accuracy=False):
    """
    Run KMeans algorithm on a folder of images given by `input_dir` and save the results to `work_dir`.
    Fastdup will extract feature vectors using the model specified by `model_path` and then run KMeans to cluster the vectors.
    The results will be saved to `work_dir` in the following format:
    - `kmeans_centroids.csv`: a csv file containing the centroids of the clusters.
    - `kmeans_assignments.csv`: assignment of each data point to the closet centroids (number of centroids given by `nearest_neighbors_k`).
    After running kmeans you can use `create_kmeans_clusters_gallery` to view the results.

    Args:
        input_dir (str): path to the folder containing the images to be clustered. See `fastdup.run` for more details.
        work_dir (str): path to the folder where the results will be saved.
        verbose (bool): verbosity level, default False
        num_clusters (int): Number of KMeans clusters to use
        num_em_iter (int): Number of em iterations
        num_threads (int): Number of threads for performing the feature vector extraction
        num_images (int): Limit the number of images
        model_path (str): Model path for the model to be used for feature vector extraction
        license (str): License string
        nearest_neighbors_k (int): When assigning an image into a cluster, how many clusters to assign to (starting from the closest)
        d (int): Dimension of the feature vector
        bounding_box (str): Optional bounding box see fastdup:::run for more details
        high_accuracy (bool): Use higher accuracy model for the feature extraction

    Returns:
        ret (int): 0 in case of success, 1 in case of error
    """
    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        assert num_clusters >= 2, "Number of clusters must be at least 2, got {}".format(num_clusters)
        assert num_em_iter >=1, "Number of EM iterations must be at least 1, got {}".format(num_em_iter)

        ret = run(input_dir=input_dir,
                work_dir=work_dir,
                verbose=verbose,
                num_threads=num_threads,
                num_images=num_images,
                model_path=model_path,
                license=license,            #license string
                nearest_neighbors_k=nearest_neighbors_k,
                d=d,
                run_mode=5,
                nnf_param=f"num_clusters={num_clusters},num_em_iter={num_em_iter}",
                bounding_box=bounding_box,
                high_accuracy=high_accuracy)
        fastdup_performance_capture("run_kmeans", start_time)
        return ret

    except Exception as ex:
        fastdup_capture_exception("run_kmeans", ex)


def run_kmeans_on_extracted(input_dir='',
               work_dir='.',
               verbose=False,
               num_clusters=100,
               num_em_iter=20,
               num_threads=-1,
               num_images=0,
               model_path=model_path_full,
               license='',            #license string
               nearest_neighbors_k=2,
               d=576):
    """
    Run KMeans algorithm on a folder of extracted feature vectors (created on default when running fastdup:::run).
    The results will be saved to `work_dir` in the following format:
    - `kmeans_centroids.csv`: a csv file containing the centroids of the clusters. In each row one centroid. In total `num_clusters` rows.
    - `kmeans_assignments.csv`: assignment of each data point to the closet centroids (number of centroids given by `nearest_neighbors_k`). In each row the image filename is listed, centoid id (starting from zero) and the L2 distance to the centroid.
    After running kmeans you can use `fastdup:::create_kmeans_clusters_gallery` to view the results.

    Args:
        input_dir (str): path to the folder containing the images to be clustered. See fastup:::run for more details.
        work_dir (str): path to the folder where the results will be saved.
        verbose (bool): verbosity level, default False
        num_clusters (int): Number of KMeans clusters to use
        num_em_iter (int): Number of em iterations
        num_threads (int): Number of threads for performing the feature vector extraction
        num_images (int): Limit the number of images
        model_path (str): Model path for the model to be used for feature vector extraction
        license (str): License string
        nearest_neighbors_k (int): When assigning an image into a cluster, how many clusters to assign to (starting from the closest)
        d (int): Dimension of the feature vector

    Returns:
        ret (int): 0 in case of success, 1 in case of error
    """

    try:
        start_time = time.time()
        fastdup_capture_log_debug_state(locals())

        assert num_clusters >= 2, "Number of clusters must be at least 2, got {}".format(num_clusters)
        assert num_em_iter >=1, "Number of EM iterations must be at least 1, got {}".format(num_em_iter)

        ret = run(input_dir=input_dir,
                   work_dir=work_dir,
                   verbose=verbose,
                   num_threads=num_threads,
                   num_images=num_images,
                   model_path=model_path,
                   license=license,            #license string
                   nearest_neighbors_k=nearest_neighbors_k,
                   d=d,
                   run_mode=6,
                   nnf_param=f"num_clusters={num_clusters},num_em_iter={num_em_iter}")
        fastdup_performance_capture("run_kmeans_on_extracted", start_time)
        return ret
    except Exception as ex:
        fastdup_capture_exception("run_kmeans_on_extracted", ex)



def extract_video_frames(input_dir, work_dir, verbose=False, num_threads=-1, num_images=0, min_offset=0, max_offset=0, turi_param="",
                         model_path = model_path_full, d=576, resize_video=0, keyframes_only=1, license=""):
    """
    A function to go over a collection of videos and etract them into frames. The output is saved to the work_dir/tmp
    subfolder.

    Args:
        input_dir (str):
        Location of the videos to extract.
            * A folder
            * A remote folder (s3 or minio starting with s3:// or minio://). When using minio append the minio server name for example minio://google/visual_db/sku110k.
            * A file containing absolute filenames each on its own row.
            * A file containing s3 full paths or minio paths each on its own row.
            * A python list with absolute filenames
            * We support api/mp4 video formats.

        work_dir (str): Optional path for storing intermediate files and results.

        verbose (boolean): Verbosity.

        num_threads (int): Number of threads. If no value is specified num threads is auto configured by the number of cores.

        num_images (unsigned long long): Number of images to run on. On default, run on all the images in the image_dir folder.

        turi_param (str): Optional turi parameters seperated by command. Example run: turi_param='nnmodel=0,ccthreshold=0.99'
        The following parameters are supported.
            * nnmodel=xx, Nearest Neighbor model for clustering the features together. Supported options are 0 = brute_force (exact), 1 = ball_tree and 2 = lsh (both approximate).
            * ccthreshold=xx, Threshold for running connected components to find clusters of similar images. Allowed values 0->1. The default ccthreshold is 0.96. This groups very similar images together, for example identical images or images that went
            simple transformations like scaling, flip, zoom in. As higher the score the more similar images are grouped by and you will get \
                smaller clusters. Score 0.9 is pretty broad and will clsuter images together even if they fine details are not similar. \
                                                                                                                                   It is recommended to experiment with this parameter based on your dataset and then visualize the results using `fastdup.create_components_gallery()`.
            * run_cc=0|1 run connected components on the resulting similarity graph. Default is 1.
            * run_pagerank=0|1 run pagerank on the resulting similarity graph. Default is 1.
            * delete_tar=0|1 when working with tar files obtained from cloud storage delete the tar after download
            * delete_img=0|1 when working with images obtained from cloud storage delete the image after download
            * tar_only=0|1 run only on tar files and ignore images in folders. Default is 0.
            * run_stats=0|1 compute image statistics. Default is 1.
            * sync_s3_to_local=0|1 In case of using s3 bucket sync s3 to local folder to improve performance. Assumes there is enough local disk space to contain the dataDefault is 0. \


        min_offset (unsigned long long): Optional min offset to start iterating on the full file list.

        max_offset (unsigned long long): Optional max offset to start iterating on the full file list.

        resize_video (int): 0 = do not resize video, 1 = resize video based on the model_path dimensions

        keyframes_only (int): 0 = extract all frames, 1 = extract only keyframes

        model_path (str): optional string to point to alternatiuve onnx or ort model

        d (int): output feature vector for model


    Returns:
        ret (int): Status code 0 = success, 1 = error.
    """
    fastdup_capture_log_debug_state(locals())

    t_param = f"video_keyframe_only={keyframes_only},video_no_resize={int(resize_video == 0)},run_video_extraction_only=1"
    if (turi_param != ""):
        t_param += "," + turi_param


    return run(input_dir=input_dir, work_dir=work_dir, verbose=verbose, run_mode=1,
               turi_param=t_param, num_images=num_images, num_threads=num_threads,
               min_offset=min_offset, max_offset=max_offset, model_path=model_path, d=d, license=license)

# give access to the main class
# at the end of the file to solve circular dependencies
from fastdup.engine import Fastdup
from typing import Union
import fastdup.fastdup_controller as FD

@v1_sentry_handler
def create(work_dir: Union[str, Path], input_dir: Union[str, Path] = None) -> Fastdup:
    fd = Fastdup(work_dir=work_dir, input_dir=input_dir)
    return fd
