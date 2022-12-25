import os
from fastdup.definitions import S3_TEMP_FOLDER, S3_TEST_TEMP_FOLDER
from datetime import datetime
from fastdup.sentry import fastdup_capture_exception

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
    if input_dir.startswith('s3://'):
        command = 'aws s3 sync ' + input_dir + ' ' + f'{work_dir}/{local_folder}'
        if not verbose:
            command += ' --no-progress'
        ret = os.system(command)
        if ret != 0:
            print('Failed to sync s3 to local. Command was aws s3 sync ' + input_dir + ' ' + f'{work_dir}/{local_folder}')
            return ret
    elif input_dir.startswith('minio://'):
        command = 'mc cp --recursive ' + input_dir.replace('minio://', '') + ' ' + f'{work_dir}/{local_folder} '
        if not verbose:
            command += ' --quiet'
        ret = os.system(command)
        if ret != 0:
            print('Failed to sync s3 to local. Command was: mc cp --recursive ' + input_dir.replace('minio://', '') + ' ' + f'{work_dir}/{local_folder}')
            return ret
    input_dir = f'{work_dir}/{local_folder}'
    return input_dir

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