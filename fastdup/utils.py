import os
import glob
import random
import platform
from fastdup.definitions import *
from datetime import datetime
from fastdup.sentry import fastdup_capture_exception
import warnings


IMAGE_SUFFIXES = ['jpg', 'jpeg','png','gif','tif', 'tiff', 'bmp', 'heif', 'heic']

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

        command = 'aws s3 sync ' + input_dir + ' ' + f'{work_dir}/{local_folder}'
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


def download_from_web(url):
    import urllib.request
    local_file = os.path.expanduser((os.environ["USERPROFILE"] + get_sep() if platform.system() == "Windows" else "~/") + "yolov5s.onnx")
    urllib.request.urlretrieve(url, local_file)
    return local_file
    #url = "https://github.com/itsnine/yolov5-onnxruntime/raw/master/models/yolov5s.onnx"

def find_model(model_name):
    local_model = os.path.expanduser((os.environ["USERPROFILE"] + get_sep() if platform.system() == "Windows" else "~/") + os.path.basename(model_name))
    
    if not os.path.isfile(local_model):
        print("Trying to download yolov5s model from ", YOLOV5S_MODEL)
        local_model = download_from_web(YOLOV5S_MODEL)
    return local_model

def check_latest_version(curversion):
    try:
        import requests
        from packaging.version import parse

        # Search for the package on PyPI using the PyPI API
        response = requests.get('https://pypi.org/pypi/fastdup/json')

        # Get the latest version number from the API response
        latest_version = parse(response.json()['info']['version'])

        latest_version = (int)(float(str(latest_version))*1000)
        if latest_version > (int)(float(curversion)*1000)+10:
            return True

    except Exception as e:
        fastdup_capture_exception("check_latest_version", e, True)


    return False



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
        if f.startswith("s3://") or f.startswith("minio://"):
            if f.lower().endswith(IMAGE_SUFFIXES):
                files.extend(f)
            else:
                assert False, "Unsupported mode: can not run on lists of s3 folders, please list all files in s3 and give a list of all files each one in a new row"
        if os.path.isfile(f):
            files.append(f)
        elif os.path.isdir(f):
            files.extend(get_images_from_path)
        else:
            warnings.warn(f"Failed to find file {f}")

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
        "Need to specify work_dir to point to the location of fastdup work_dir"

    if work_dir.endswith('.csv'):
        local_filenames = work_dir
    else:
        local_filenames = os.path.join(work_dir, "atrain_" + FILENAME_IMAGE_LIST)
    assert os.path.isfile(local_filenames), "Failed to find fastdup input file " + local_filenames
    nrows = find_nrows(kwargs)
    import pandas as pd
    filenames = pd.read_csv(local_filenames, nrows=nrows)
    assert len(filenames), "Empty dataframe found " + local_filenames
    assert 'filename' in filenames.columns, f"Error: Failed to find filename column in {work_dir}/atrain_{FILENAME_IMAGE_LIST}"

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
    df = df.merge(filenames, left_on='from', right_on='index').merge(filenames, left_on='to', right_on='index')
    assert len(df), "Failed to merge similarity/outliers with atrain_features.dat.csv file"
    del df['from']
    del df['to']
    del df['index_x']
    del df['index_y']
    df = df.rename(columns={'filename_x': 'from', 'filename_y': 'to'})
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
        return row['files'][:howmany]
    elif selection_strategy == SELECTION_STRATEGY_RANDOM:
        return random.sample(row['files'], howmany)
    elif selection_strategy == SELECTION_STRATEGY_UNIFORM_METRIC:
        assert metric in row, "When using selection_strategy=2 (SELECTION_STRATEGY_UNIFORM_METRIC) need to call with metric=metric."
        assert len(row[metric]) == len(row['files'])
        #Combine the lists into a list of tuples
        combined = zip(row['files'], row[metric])

        # Sort the list of tuples by the float value
        sorted_combined = sorted(combined, key=lambda x: x[1])
        if len (sorted_combined) < howmany:
            sindices = range(0, len(sorted_combined))
        else:
            sindices = range(0, max(1, int(len(sorted_combined)/howmany)), len(sorted_combined))

        # Extract the filenames from the selected subset
        filenames = [sorted_combined[t][0] for t in sindices]
        return filenames

if __name__ == "__main__":
    #print(get_bounding_box_func_helper("../t1/atrain_crops.csv"))
    file = ["a","b","c","d", "e","f","g","h"]
    floats = [1,2,3,4,1,2,3,1]
    row = {}
    row['blur'] = floats
    row['files'] = file
    sample_from_components(row, 'blur', {}, 2)
