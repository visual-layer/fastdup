from fastdup import definitions
import os
from tqdm import tqdm




def merge_webdataset_low_memory(work_dir, test_dir='', num_images=None, num_images_test=None, merge_labels=False, merge_stats=False ):
    """
    Function to merge multiple image lists obtained from webdataset format by running fastdup with run_mode=1
    into a single list.
    The following files will be created under work_dir:
    * atrain_features.dat.csv - list of all filenames
    * atrain_stats.csv - list of all image statistics (optional)
    * atrain_labels.csv - list of all labels (optional)

    Arguments:
        work_dir: fastdup working dir
        test_dir (str):  test dir (optional)
        num_images (int): optional number of images in the work_dir to verify the number of images
        num_images_test (int): optional the number oif images in the test_dir to verify the number of images
        merge_labels (bool): if true, merges the label files
        merge_stats (bool) if ture, merges the image statistics files
    Returns:
        None
    """

    print('Going to merge filenames')
    fa =open('atrain_' + FILENAME_IMAGE_LIST, 'w')
    fa.write(f"{IMAGELIST_HEADER}\n")
    counter = 0
    files = 0
    for i in tqdm(sorted(os.listdir(work_dir))):
        if i.endswith(FILENAME_IMAGE_LIST) and i != 'atrain_' + FILENAME_IMAGE_LIST:
            files+=1
            with open(i) as f:
                line='aa'
                while(line is not None and line != '') :
                    line = f.readline().strip()
                    if line == IMAGELIST_HEADER:
                        continue
                    ret = line.split(',')
                    if len(ret) == 2:
                        fa.write(f'{counter},{ret[1]}\n')
                        counter+=1
    print('Total files', files, 'total lines', counter)
    fa.close()
    if num_images:
        assert counter == num_images

    if test_dir != '':
        fa =open('atrain_' + IMAGELIST_HEADER, 'a')
        files = 0
        for i in tqdm(sorted(os.listdir(test_dir))):
            if i.endswith('features.dat.csv') and i != 'atrain_features.dat.csv':
                files+=1
                with open(os.path.join(test_dir , i)) as f:
                    line='aa'
                    while(line is not None and line != '' ) :
                        line = f.readline().strip()
                        if line == IMAGELIST_HEADER:
                            continue
                        ret = line.split(',')
                        if len(ret) == 2:
                            fa.write(f'{counter},{ret[1]}\n')
                            counter+=1
        print('Total files', files, 'total lines', counter)
        fa.close()

    if num_images and num_images_test:
        assert counter == num_images+num_images_test, "Wrong number of images"


    if merge_labels:
        print("Going to merge labels")
        fa =open('atrain_' + FILENAME_LABELS, 'w')
        fa.write(f"{LABEL_HEADER}\n")
        counter = 0
        files = 0
        for i in tqdm(sorted(os.listdir(work_dir))):
            if i.endswith(FILENAME_LABELS) and i != 'atrain_' + FILENAME_LABELS:
                files+=1
                with open(os.path.join(work_dir, i), 'r', encoding='latin') as f:
                    with open(i.replace('labels', 'features.dat')) as f1:
                        line = f.readline()
                        line0 = f1.readline()
                        while(line0 is not None and line0 != '' ) :
                            try:
                                line2 = line.strip()
                                if line2 == LABEL_HEADER:
                                    line = f.readline()
                                    line0 = f1.readline()
                                ret = line2.find(',')
                                if ret >= 1:
                                    line2 = line2[ret+1:].replace(',','')
                                    fa.write(f'{counter},{line2}\n')
                                    counter+=1
                            except:
                                fa.write(f'{counter},N/A\n')
                                counter+=1
                            line = f.readline()
                            line0 = f1.readline()

        print('Total files', files, 'total lines', counter)
        fa.close()
        if num_images:
            assert counter == num_images

        if test_dir != '':
            fa =open('atrain_' + LABEL_HEADER, 'a')
            files = 0
            for i in tqdm(sorted(os.listdir(work_dir))):
                if i.endswith(FILENAME_LABELS) and i != 'atrain_' + FILENAME_LABELS:
                    files+=1
                    with open(os.path.join(test_dir,  i), 'r', encoding='latin') as f:
                        with open(os.path.join(test_dir, i.replace('labels', 'features.dat'))) as f1:
                            line = f.readline()
                            line0 = f1.readline()
                            while(line0 is not None and line0 != '' ) :
                                try:
                                    line2 = line.strip()
                                    if line2 == STATS_HEADER:
                                        line = f.readline()
                                        line0 = f1.readline()
                                    ret = line2.find(',')
                                    if ret >= 1:
                                        line2 = line2[ret+1:].replace(',','')
                                        fa.write(f'{counter},{line2}\n')
                                        counter+=1
                                except:
                                    fa.write(f'{counter},N/A\n')
                                    counter+=1
                                line = f.readline()
                                line0 = f1.readline()

        print('Total files', files, 'total lines', counter)
        fa.close()
        if num_images and num_images_test:
            assert counter == num_images+num_images_test


    if merge_stats:
        print("Going to merge stats")
        fa =open('atrain_stats.csv', 'w')
        fa.write(f"{STATS_HEADER}\n")
        counter = 0
        files = 0
        for i in tqdm(sorted(os.listdir('.'))):
            if i.endswith('stats.csv') and i != 'atrain_stats.csv':
                files+=1
                with open(os.path.join(work_dir, i)) as f:
                    line='aa'
                    while(line is not None and line != '' ) :
                        line = f.readline().strip()
                        if line == STATS_HEADER:
                            continue
                        ret = line.find(',')
                        if ret >= 1:
                            fa.write(f'{counter},{line[ret+1:]}\n')
                            counter+=1
        print('Total files', files, 'total lines', counter)
        fa.close()

        if num_images:
            assert counter == num_images


        fa =open('atrain_' + FILENAME_STATS, 'a')
        files = 0
        for i in tqdm(sorted(os.listdir(test_dir))):
            if i.endswith(FILENAME_STATS) and i != 'atrain_' + FILENAME_STATS:
                files+=1
                with open(os.psth.join(test_dir,  i)) as f:
                    line='aa'
                    while(line is not None and line != '') :
                        line = f.readline().strip()
                        if line == HEADER_STATS:
                            continue
                        ret = line.find(',')
                        if ret >= 1:
                            fa.write(f'{counter},{line[ret+1:]}\n')
                            counter+=1
        print('Total files', files, 'total lines', counter)
        fa.close()
        if num_images and num_images_test:
            assert counter == num_images + num_images_test


def filter_similarity_low_memory(work_dir, out_file, threshold):
    """
    One fastdup runs and creates similarity.csv file, select a subset of similaritis > threshold and put them in the out_file
    Arguments:
        work_dir (str): fastdup working dir where the file similarity/csv is found, or a full path pointing to similarity.csv file
        out_file (str): name of output similarity file
        threshold: 0->1 take images >= threshold
    Returns
        None
    """
    assert isinstance(threshold, float)
    assert threshold < 1 and threshold > 0
    sim_file = os.path.join(work_dir, FILENAME_SIMILARITY) if os.path.isdir(work_dir) else work_dir
    assert os.path.exists(sim_file)
    assert out_file != sim_file
    fa = open(out_file, 'w')
    fa.write(f"{SIMILARITY_HEADER, EADER}\n")
    f =open(sim_file, 'r')
    line = 'aa'
    counter = 0
    while(line is not None and line != ''):
        line = f.readline().strip()
        if line == SIMILARITY_HEADER:
            continue
        ret = line.split(',')
        if len(ret) == 3:
            distance = float(ret[2])
            if distance > threshold:
                fa.write(line + "\n")
            counter +=1
        if counter % 1000000 == 0:
            print(counter)

    fa.close()
