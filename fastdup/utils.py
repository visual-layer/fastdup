import os
import glob
import random
import platform
from fastdup.definitions import *
from datetime import datetime
from fastdup.sentry import fastdup_capture_exception
import warnings
import itertools
import pathlib
import subprocess
import time
import os

def read_local_error_file(ret, local_error_file):
    if (ret != 0 and 'JPY_PARENT_PID' in os.environ) or 'COLAB_JUPYTER_IP' in os.environ:
        if os.path.exists(local_error_file):
            # windows can generate non ascii printouts
            with open(local_error_file, "r", encoding="utf-8") as f:
                error = f.read()
                data_type = "error" if ret != 0 else "info"
                print(f"fastdup C++ {data_type} received: ", error[:5000], "\n")
                if ret != 0:
                    fastdup_capture_exception("C++ error", RuntimeError(error[:5000]))

def download_from_s3(input_dir, work_dir, verbose, is_test=False):
    """
    Download files from S3 to local disk (called only in case of turi_param='sync_s3_to_local=1')
    Note: we assume there is enough local disk space otherwise the download may fail
     input_dir: input directory on s3 or minio
     work_dir: local working directory
     verbose: if verbose show progress
     is_test: If this is a test folder save it on S3_TEST_TEMP_FOLDER otherwise on S3_TEMP_FOLDER
    Returns: The local download directory
    """
    print(f'Going to download s3 files from {input_dir} to local {work_dir}')

    local_folder = S3_TEST_TEMP_FOLDER if is_test else S3_TEMP_FOLDER
    if platform.system() == "Windows":
        local_folder = 'testtemp' if is_test else 'temp'
    if input_dir.startswith('s3://'):
        endpoint = "" if "FASTDUP_S3_ENDPOINT_URL" not in os.environ else f"--endpoint-url={os.environ['FASTDUP_S3_ENDPOINT_URL']}"
        command = f'aws s3 {endpoint} sync ' + input_dir + ' ' + f'{work_dir}/{local_folder}'
        if not verbose:
            command += ' --no-progress'
        ret = os.system(command)
        if ret != 0:
            print('Failed to sync s3 to local. Command was ' + command)
            return ret
    elif input_dir.startswith('minio://'):
        if platform.system() == "Windows":
            assert "FASTDUP_MC_PATH" in os.environ, "Have to define FASTUP_MC_PATH environment variable to point to minio client full_path. For example C:\\Users\\danny_bickson\\mc.exe"
            mc_path = os.environ["FASTDUP_MC_PATH"]
            assert os.path.exists(mc_path), "Failed to find minio client on " + mc_path
            command = f'{mc_path} cp --recursive ' + input_dir.replace('minio://', '') + ' ' + f'{work_dir}\\{local_folder}'

        else:
            command = 'mc cp --recursive ' + input_dir.replace('minio://', '') + ' ' + f'{work_dir}/{local_folder} '
        if not verbose:
            command += ' --quiet'
        ret = os.system(command)
        if ret != 0:
            print('Failed to sync s3 to local. Command was: ' + command)
            return ret

    input_dir = f'{work_dir}/{local_folder}'
    return input_dir


def download_from_web(url, local_model=None):
    import urllib.request
    local_file = os.path.expanduser((os.environ["USERPROFILE"] + get_sep() if platform.system() == "Windows" else "~/") + os.path.basename(url if local_model is None else local_model))
    urllib.request.urlretrieve(url, local_file)
    assert os.path.isfile(local_file), f"Failed to download url {url} to local file {local_file}, please try to download manually"
    return local_file
    #url = "https://github.com/itsnine/yolov5-onnxruntime/raw/master/models/yolov5s.onnx"

def find_model(model_name, url):
    local_model = os.path.expanduser((os.environ["USERPROFILE"] + get_sep() if platform.system() == "Windows" else "~/") + os.path.basename(url))
    #handle clip model where all models are called visual.onnx and thus may override which other
    if 'clip336' in model_name:
        local_model = local_model.replace('visual.onnx', 'visual336.onnx')
    elif 'clip14' in model_name:
        local_model = local_model.replace('visual.onnx', 'visual14.onnx')

    if not os.path.isfile(os.path.expanduser(local_model)):
        print(f"Trying to download {model_name} model from {url} to {local_model}")
        local_model = download_from_web(url, local_model)
    return local_model

def check_latest_version(curversion):
    try:
        if 'FASTDUP_PRODUCTION' in os.environ:
            return False

        import requests
        try:
            from packaging.version import parse
        except ModuleNotFoundError as ex:
            print("Failed to find packaging module, please install via `pip install setuptools`")
            fastdup_capture_exception("check_latest_version", ex, True)
            return False

        # Search for the package on PyPI using the PyPI API
        response = requests.get('https://pypi.org/pypi/fastdup/json', timeout=2)

        # Get the latest version number from the API response
        latest_version = parse(response.json()['info']['version'])
        latest_version_num = int(str(latest_version).split(".")[0])
        latest_version_frac = int(str(latest_version).split(".")[1])

        latest_version = latest_version_num*1000+latest_version_frac
        cur_version_num = int(curversion.split(".")[0])
        cur_version_frac = int(curversion.split(".")[1])
        if latest_version > cur_version_num*1000+cur_version_frac+25:
            return True

    except Exception as e:
        fastdup_capture_exception("check_latest_version", e, True)


    return False



def convert_v1_to_v02(df):
    if 'filename_from' in df.columns and 'filename_to' in df.columns:
        del df['from']
        del df['to']
        df = df.rename(columns={'filename_from':'from', 'filename_to':'to'})
    if 'filename_outlier' in df.columns and 'filename_nearest' in df.columns:
        df = df.rename(columns={'filename_outlier': 'from', 'filename_nearest': 'to'})
    if 'label_from' in df.columns and 'label_to' in df.columns:
        df = df.rename(columns={'label_from':'label', 'label_to':'label2'})
    if 'label_outlier' in df.columns :
        df = df.rename(columns={'label_outlier': 'label'})
    return df


def record_time():
    try:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d")
        with open("/tmp/.timeinfo", "w") as f:
            if date_time.endswith('%'):
                date_time = date_time[:len(date_time)-1]
            f.write(date_time)
    except Exception as ex:
        fastdup_capture_exception("Timestamp", ex)

def calc_save_dir(save_path):
    save_dir = save_path
    if save_dir.endswith(".html"):
        save_dir = os.path.dirname(save_dir)
        if save_dir == "":
            save_dir = "."
    return save_dir

def get_images_from_path(path):
    "List a subfoler recursively and get all image files supported by fastdup"
    # create list to store results

    assert os.path.isdir(path), "Failed to find directory " + path
    filenames = []
    ret = []
    # get all image files
    image_extensions = SUPPORTED_IMG_FORMATS
    image_extensions.extend(SUPPORTED_VID_FORMATS)
    filenames += glob.glob(f'{path}/**/*', recursive=True)

    for r in filenames:
        ext = os.path.splitext(r)
        if len(ext) < 2:
            continue
        ext = ext[1]
        if ext in image_extensions:
            ret.append(r)

    if len(ret) == 0:
        print("Warning: failed to find any image/video files in folder " + path)
    return ret


def list_subfolders_from_file(file_path):
    assert os.path.isfile(file_path)
    ret = []

    with open(file_path, "r") as f:
        for line in f:
            if os.path.isdir(line.strip()):
               ret += get_images_from_path(line.strip())

    assert len(ret), "Failed to find any folder listing from file " + file_path
    return ret


def shorten_path(path):
    if isinstance(path, pathlib.Path):
        path = str(path)
    if path.startswith('~'):
        path = os.path.expanduser(path)
    elif path.startswith('./'):
        path = path[2:]

    if path.endswith('/'):
        path = path[:-1]

    cwd = os.getcwd()
    if (path.startswith(cwd + '/')):
        path = path.replace(cwd + '/', '')

    return path

def check_if_folder_list(file_path):
    assert os.path.isfile(file_path), "Failed to find file " + file_path
    if file_path.endswith('yaml'):
        return False
    with open(file_path, "r") as f:
        for line in f:
            return os.path.isdir(line.strip())
    return False


def save_as_csv_file_list(filenames, files_path):
     import pandas as pd
     files = pd.DataFrame({'filename':filenames})
     files.to_csv(files_path)
     return files_path


def expand_list_to_files(the_list):
    assert len(the_list), "Got an empty list for input"
    files = []
    for f in the_list:
        if isinstance(f, str) or isinstance(f, pathlib.PosixPath):
            f = str(f)
            if f.startswith("s3://") or f.startswith("minio://"):
                if os.path.splitext(f.lower()) in SUPPORTED_IMG_FORMATS or os.path.splitext(f.lower()) in SUPPORTED_VID_FORMATS:
                    files.append(f)
                    break

                assert False, f"Unsupported mode: can not run on lists of s3 folders, please list all image or video files " \
                              f"in s3 (using `aws s3 ls <bucket name>` into a text file, and run fastdup pointing to this text file. " \
                              f"File was {f}, supported image and video formats are {SUPPORTED_IMG_FORMATS}, {SUPPORTED_VID_FORMATS}"
            elif os.path.isfile(f):
                files.append(f)
            elif os.path.isdir(f):
                files.extend(get_images_from_path(f))
            else:
                assert False, f"Unknown file type encountered in list: {f}"

    assert len(files), "Failed to extract any files from list"
    return files

def ls_crop_folder(path):
    assert os.path.isdir(path), "Failed to find directlry " + path
    files = os.listdir(path)
    import pandas as pd
    df = pd.DataFrame({'filename':files})
    assert len(df), "Failed to find any crops in folder " + path

def find_nrows(kwargs):
    nrows = None
    if kwargs is not None and isinstance(kwargs, dict) and 'nrows' in kwargs:
        nrows = kwargs['nrows']
        assert isinstance(nrows, int) and nrows > 0, "Wrong 'nrows' parameter, should be integer > 0"
    return nrows

def load_filenames(work_dir, kwargs):
    assert work_dir is not None and isinstance(work_dir, str) and os.path.exists(work_dir), \
        f"Need to specify work_dir to point to the location of fastdup work_dir, got {work_dir}"
    load_crops = 'load_crops' in kwargs and kwargs['load_crops']
    draw_bbox = 'draw_bbox' in kwargs and kwargs['draw_bbox']

    if work_dir.endswith('.csv'):
        local_filenames = work_dir
    elif load_crops or draw_bbox:
        local_filenames = os.path.join(work_dir, "atrain_" + FILENAME_CROP_LIST)
    else:
        local_filenames = os.path.join(work_dir, "atrain_" + FILENAME_IMAGE_LIST)
    assert os.path.isfile(local_filenames), "Failed to find fastdup input file " + local_filenames
    nrows = find_nrows(kwargs)
    import pandas as pd
    filenames = pd.read_csv(local_filenames, nrows=nrows)
    assert len(filenames), "Empty dataframe found " + local_filenames
    assert 'filename' in filenames.columns, f"Error: Failed to find filename column in {work_dir}/atrain_{FILENAME_IMAGE_LIST}"
    if load_crops and not draw_bbox:
        assert 'crop_filename' in filenames.columns, f"Failed to load crop filename {local_filenames}"
        filenames["filename"] = filenames["crop_filename"]
    return filenames

def merge_stats_with_filenames(df, filenames):
    df = df.merge(filenames, left_on='index', right_on='index')
    assert len(df), "Failed to merge stats input with filenames"
    return df

def load_stats(stats_file, work_dir, kwargs={}, usecols=None):
    assert stats_file is not None, "None stat file"
    nrows = find_nrows(kwargs)
    stats = stats_file
    import pandas as pd

    if isinstance(stats_file, pd.DataFrame):
        if nrows is not None:
            stats = stats_file.head(nrows)
        assert work_dir is not None, "When calling with stats_file which is a pd.DataFrame need to point work_dir to the fastdup work_dir folder"
        kwargs["external_df"] = True

    elif isinstance(stats_file, str):
        assert stats_file is not None and isinstance(stats_file, str) and os.path.exists(stats_file), \
            "Need to specify work_dir to point to the location of fastdup atrain_stats.csv stats file"
        if stats_file.endswith(".csv") and os.path.isfile(stats_file):
            local_filenames = stats_file
            if work_dir is None:
                work_dir = os.path.dirname(local_filenames)
        elif os.path.isdir(stats_file):
            local_filenames = os.path.join(stats_file, "atrain_" + FILENAME_IMAGE_STATS)
            if work_dir is None:
                work_dir = stats_file
        else:
            assert False, "Failed to find stats file " + stats_file

        assert os.path.exists(local_filenames), f"Failed to read stats file {local_filenames} please make sure fastdup was run and this file was created."
        stats = pd.read_csv(local_filenames, nrows=nrows, usecols=usecols)
        assert len(stats), "Empty dataframe found " + local_filenames
    else:
        assert False, "wrong type " + stats_file

    assert stats is not None, "Failed to find stats file " + str(stats_file) + " " + str(work_dir)
    if 'filename' not in stats.columns and 'from' not in stats.columns and 'to' not in stats.columns:
        assert 'index' in stats.columns, "Failed to find index columns" + str(stats.columns)
        filenames = load_filenames(work_dir, kwargs)
        if len(filenames) == len(stats):
            assert 'filename' in filenames.columns, "Failed to find filename column in atrain_features.dat.csv file"
            stats['filename'] = filenames['filename']
        else:
            stats = merge_stats_with_filenames(stats, filenames)

    assert stats is not None, "Failed to read stats"
    assert 'filename' in stats.columns, f"Error: Failed to find filename column"

    return stats

def load_labels(get_label_func, kwargs={}):
    assert isinstance(get_label_func, str)
    assert os.path.isfile(get_label_func), "Failed to find file " + get_label_func
    nrows = find_nrows(kwargs)

    # use quoting=csv.QUOTE_NONE for LAOIN
    quoting = 0
    if len(kwargs) and 'quoting' in kwargs:
        quoting = kwargs['quoting']
    import pandas as pd
    df_labels = pd.read_csv(get_label_func, nrows=nrows, quoting=quoting)
    assert len(df_labels), "Found empty file "+ get_label_func
    assert 'label' in df_labels.columns, f"Error: wrong columns in labels file {get_label_func} expected 'label' column"
    return df_labels

def merge_with_filenames(df, filenames):
    df2 = df.merge(filenames, left_on='from', right_on='index').merge(filenames, left_on='to', right_on='index')
    assert len(df2), f"Failed to merge similarity/outliers with atrain_features.dat.csv file, \n{df.head()}, \n{filenames.head()}"
    df = df2
    del df['from']
    del df['to']
    del df['index_x']
    del df['index_y']
    df = df.rename(columns={'filename_x': 'from', 'filename_y': 'to'})
    return df

def merge_with_filenames_one_sided(df, filenames):
    df2 = df.merge(filenames, left_on='from', right_on='index')
    assert len(df2), f"Failed to merge similarity/outliers with atrain_features.dat.csv file \n{df.head()}, \n{filenames.head()}"
    df = df2
    return df

def merge_with_filenames_search(df, filenames):
    df2 = df.merge(filenames, left_on='to', right_on='index')
    assert len(df2), f"Failed to merge similarity/outliers with atrain_features.dat.csv file \n{df.head()}, \n{filenames.head()}"
    df = df2
    return df
def get_bounding_box_func_helper(get_bounding_box_func):
    if get_bounding_box_func is None:
        return None
    import pandas as pd
    if callable(get_bounding_box_func) or isinstance(get_bounding_box_func, dict):
        return get_bounding_box_func
    elif isinstance(get_bounding_box_func, str):
        if os.path.isfile(get_bounding_box_func):
            df = pd.read_csv(get_bounding_box_func)
        elif os.path.isdir(get_bounding_box_func):
            local_file = os.path.join(get_bounding_box_func, "atrain_crops.csv")
            assert os.path.exists(local_file), "Failed to find bounding box file in " + local_file
            df = pd.read_csv(os.path.join(get_bounding_box_func, "atrain_crops.csv"))
        else:
            assert False, "Failed to find input file/folder " + get_bounding_box_func
    elif isinstance(get_bounding_box_func, pd.DataFrame):
        df = get_bounding_box_func
    else:
        assert False, "get_bounding_box_func should be a callable function, a dictionary, a file with bounding box info or a dataframe"

    assert len(df), "Empty dataframe with bounding box information"
    assert "filename" in df.columns
    assert "row_y" in df.columns
    assert "col_x" in df.columns
    assert "width" in df.columns
    assert "height" in df.columns
    df["bbox"] = df.apply(lambda x: [x["col_x"], x["row_y"], x["col_x"] + x["width"], x["row_y"] + x["height"]], axis=1)
    df = df.groupby('filename')['bbox'].apply(list).reset_index()
    my_dict = df.set_index('filename')['bbox'].to_dict()
    return my_dict

def sample_from_components(row, metric, kwargs, howmany):
    selection_strategy = kwargs['selection_strategy']
    if selection_strategy == SELECTION_STRATEGY_FIRST:
        return list(itertools.islice(zip(row['files'],row['files_ids']), howmany))
    elif selection_strategy == SELECTION_STRATEGY_RANDOM:
        return list(itertools.islice(random.sample(zip(row['files'], row['files_ids'])), howmany))
    elif selection_strategy == SELECTION_STRATEGY_UNIFORM_METRIC:
        assert metric in row, "When using selection_strategy=2 (SELECTION_STRATEGY_UNIFORM_METRIC) need to call with metric=metric."
        assert len(row[metric]) == len(row['files'])
        #Combine the lists into a list of tuples
        combined = zip(zip(row['files'],row['files_ids']), row[metric])

        # Sort the list of tuples by the float value
        sorted_combined = sorted(combined, key=lambda x: x[1])
        if len (sorted_combined) < howmany:
            sindices = range(0, len(sorted_combined))
        else:
            sindices = range(0, max(1, int(len(sorted_combined)/howmany)), len(sorted_combined))

        # Extract the filenames from the selected subset
        filenames = [sorted_combined[t][0] for t in sindices]
        return filenames





def s3_partial_sync(uri: str, work_dir: str, num_images: int, verbose:bool, check_interval: int, *args) -> None:
    from tqdm import tqdm
    assert os.path.exists(work_dir)

    local_dir = os.path.join(work_dir, "tmp")
    if os.path.exists(local_dir):
        assert False, f"Error: found folder {local_dir}, please remove it and try again"

    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
        assert os.path.exists(local_dir), "Failed to find work dir"

    if not verbose:
        arglist = ['aws', 's3', 'sync', uri, local_dir, '--quiet', *args]
    else:
        arglist = ['aws', 's3', 'sync', uri, local_dir, *args]

    if verbose:
      print('Going to run', arglist)
    process = subprocess.Popen(arglist)
    pbar = tqdm(desc='files', total=num_images)

    while process.poll() is None:
        time.sleep(check_interval)
        files = os.listdir(local_dir)
        #files = [f for f in files if (os.path.splitext(f.lower()) in SUPPORTED_IMG_FORMATS) or (os.path.splitext(f.lower()) in SUPPORTED_VID_FORMATS)]
        pbar.update(len(files) - pbar.n)

        if len(files) >= num_images:
            process.terminate()
            return_code = process.wait(5)

            if return_code != 0:
                process.kill()

            break
    return local_dir




def convert_coco_dict_to_df(coco_dict: dict, input_dir: str):
    """
    Convert dictionary in COCO format object annotations to a Fastdup DF
    :param coco_dict:
    :return: a Dataframe in the expected format for fastdup bboxes.
    """

    # merge between bounding box annotations and their image ids
    assert "images" in coco_dict, f"Invalid coco format, expected 'images' field inside the dictionary, {str(coco_dict)[:250]}"
    assert "annotations" in coco_dict, f"Invalid coco format, expected 'annotations' field inside the dictionary {str(coco_dict)[:250]}"
    assert "categories" in coco_dict, f"Failed to find categories in dict {str(coco_dict)[:250]}"
    assert isinstance(input_dir, str) or isinstance(input_dir, pathlib.Path), f"input_dir should be a str pointing to the absolute path of image location, got {input_dir}"
    import pandas as pd
    df = pd.merge(pd.DataFrame(coco_dict['images']).rename(columns={'width':'img_w', 'height':'img_h'}),
                  pd.DataFrame(coco_dict['annotations']),
                  left_on='id', right_on='image_id')
    assert len(df), f"Failed to merge coco dict {str(coco_dict)[:250]}"
    if 'rot_bb_view' in df.columns:
        rotated_bb = list(df['rot_bb_view'].apply(lambda x: {'x1':x[0][0], 'y1':x[1][0],
                                                           'x2':x[0][1], 'y2':x[1][1],
                                                           'x3':x[0][2], 'y3':x[1][2],
                                                           'x4':x[0][3], 'y4':x[1][3]}).values)
        assert len(rotated_bb) == len(df), f"Failed to find any bounding boxes {str(coco_dict)[:250]}"
        df = pd.concat([df, pd.DataFrame(rotated_bb)], axis=1)
        assert len(df), f"Failed to add rotated cols {str(coco_dict)[:250]}"
    else:
        bbox_df = list(df['bbox'].apply(lambda x: {'col_x': x[0], 'row_y': x[1], 'width': x[2], 'height': x[3]}).values)
        assert len(bbox_df), f"Failed to find any bounding boxes {str(coco_dict)[:250]}"
        df = pd.concat([df, pd.DataFrame(bbox_df)], axis=1)
        assert len(df), f"Failed to add bbox cols {str(coco_dict)[:250]}"

    #merge category id to extrac the category name
    df = df.merge(pd.DataFrame(coco_dict['categories']), left_on='category_id', right_on='id')
    assert len(df), f"Failed to merge coco dict with labels {str(coco_dict)[:250]}"
    df = df.rename(columns={'file_name':'filename','name':'label'})



    #df['filename'] = df['filename'].apply(lambda x: os.path.join(input_dir, x))
    #those are the required fields needed by fastdup
    assert 'filename' in df.columns, f"Failed to find columns in coco label dataframe {str(coco_dict)[:250]}"
    if 'col_x' in df.columns:
        df = df[['filename','col_x','row_y','width','height','label']]
    else:
        assert 'label' in df.columns, "When working with rotated bounding boxes, fastdup requires label column : <name>"
        df = df[['filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'label']]

    return df

def find_model_path(model_path, d):
    if model_path.lower().startswith('dinov2') or model_path.lower() in ['efficientnet', 'resnet50', 'clip', 'clip336', 'clip14']:
        # use DINOv2s/DINOv2b to run with DINOv2 models,
        # case insensitive naming, e.g., dinov2s, DINOv2s, ...
        if model_path.lower() == 'dinov2s':
            model_path = find_model(model_path.lower(), DINOV2S_MODEL)
            d = DINOV2S_MODEL_DIM
        elif model_path.lower() == 'dinov2b':
            model_path = find_model(model_path.lower(), DINOV2B_MODEL)
            d = DINOV2B_MODEL_DIM
        elif model_path.lower() == 'clip':
            model_path = find_model(model_path.lower(), CLIP_MODEL)
            d = CLIP_MODEL_DIM
        elif model_path.lower() == 'clip336':
            model_path = find_model(model_path.lower(), CLIP_MODEL2)
            d = CLIP_MODEL2_DIM
        elif model_path.lower() == 'clip14':
            model_path = find_model(model_path.lower(), CLIP_MODEL14)
            d = CLIP_MODEL14_DIM
        elif model_path.lower() == "resnet50":
            model_path = find_model(model_path.lower(), RESNET50_MODEL)
            d = RESNET50_MODEL_DIM
        elif model_path.lower() == "efficientnet":
            model_path = find_model(model_path.lower(), EFFICIENTNET_MODEL)
            d = EFFICIENTNET_MODEL_DIM
        else:
            assert False, f"Supporting dinov2 models are dinov2s and dinov2b, got {model_path}"

    return model_path, d


if __name__ == "__main__":
    import fastdup
    fd = fastdup.create(input_dir='/mnt/data/sku110k', work_dir='abcd')
    fd.run(num_images=10, overwrite=True)
