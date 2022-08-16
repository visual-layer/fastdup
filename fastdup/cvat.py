"""
Cvat

git clone https://github.com/opencv/cvat
cd cvat

docker-compose up -d


docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'
"""

import os
import random
import json
import glob
import cv2
import shutil

MANIFEST_FILE = 'data/manifest.jsonl'
INDEX_FILE = 'data/index.json'
ANNOTATIONS_FILE = 'annotations.json'
TASKS_FILE = 'task.json'
ZIP_FILE = 'fastdup_label.zip'

""" List of needed files 

annotations.json
data/index.json
data/manifest.jsonl
data/ < images>
tasks.json
"""

"""
example annotations.json:


[
    {
        "version":0,
        "tags":[],
        "shapes":[
            {"type":"rectangle",
             "occluded":false,
             "z_order":0,
             "rotation":0.0,
             "points":[511.811764705888,456.6588235294166,1234.6352941176538,1171.952941176476],
             "frame":0,
             "group":0,
             "source":"manual",
             "attributes":[],
             "label":"an"
            },
            {"type":"rectangle",
            "occluded":false,
            "z_order":0,
            "rotation":0.0,
            "points":[438.4230492196875,261.7085234093647,1039.1961584633846,875.1442977190891],
            "frame":1,
            "group":0,
            "source":
            "manual",
            "attributes":[],
            "label":"an"
            }
            ],
        "tracks":[]
    }
]
"""


def create_annotations_file(files, labels, save_path):
    assert len(files) == len(labels)
    assert len(files) > 0



    shape = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        h, w, c = img.shape
        shape.append(
            {
                "type":"rectangle",
                "occluded":False,
                "z_order":0,
                "rotation":0.0,
                "points":[0,0,w,h],
                "frame":i,
                "group":0,
                "source":"fastdup",
                "attributes":[],
                "label":labels[i]
            }
        )

    assert len(shape) == len(files)
    annotations = [
        {
            "version":0,
            "tags":[],
            "shapes":shape,
            "tracks":[]
        }
    ]

    local_file = os.path.join(save_path, ANNOTATIONS_FILE)
    with open(local_file, "w") as mf:
        mf.write(json.dumps(annotations))
    assert os.path.exists(local_file)
    return local_file



"""
Example labels
 [
            {"name":"an","color":"#b4a3d8","attributes":[]},
            {"name":"ct","color":"#f337a4","attributes":[]},
            {"name":"cp","color":"#b4a24a","attributes":[]}
        ],
"""
def format_labels(labels):
    assert len(labels)
    ret = []
    for i,l in enumerate(labels):
        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        assert len(color) == 7
        ret.extend([{"name": l, "color":color , "attributes": []}])
    assert len(ret) == len(labels)
    return ret


"""
Example tasks.json:

tasks.json
{
    "name":"config",
    "bug_tracker":"",
    "status":"annotation",
    "labels":
        [
            {"name":"an","color":"#b4a3d8","attributes":[]},
            {"name":"ct","color":"#f337a4","attributes":[]},
            {"name":"cp","color":"#b4a24a","attributes":[]}
        ],
    "subset":"Train",
    "version":"1.0",
    "data":
    {
        "chunk_size":36,
        "image_quality":70,
        "start_frame":0,
        "stop_frame":7,
        "storage_method":"cache",
        "storage":"local",
        "sorting_method":"lexicographical",
        "chunk_type":"imageset"
    },
    "jobs":
        [{"start_frame":0,"stop_frame":7,"status":"annotation"}]}
"""
def create_tasks_file(files, labels, save_path):
    tasks = {
        "name": "fastdup_task",
        "bug_tracker": "",
        "status": "annotation",
        "labels": format_labels(labels),
        "subset": "Train",
        "version": "1.0",
        "data": {
            "chunk_size": 36,
            "image_quality": 70,
            "start_frame": 0,
            "stop_frame": len(files),
            "storage_method": "cache",
            "storage": "local",
            "sorting_method": "lexicographical",
            "chunk_type": "imageset"
        },
        "jobs": [{"start_frame": 0, "stop_frame": len(files), "status": "annotation"}]
    }
    with open(os.path.join(save_path, TASKS_FILE), 'w') as f:
        json.dump(tasks, f, indent=4)
    assert os.path.exists(os.path.join(save_path, TASKS_FILE))
    return tasks

""" Example index.json
{"0":36,
"1":140,
"2":244,
"3":347,
"4":450,
"5":552,
"6":656,
"7":759}
"""



def create_cvat_index(index, save_path):
    """
    Create an index.json file for cvat
    """
    with open(os.path.join(save_path, INDEX_FILE), "w") as mf:
        mf.write('{')
        for i, f in enumerate(index):
            mf.write('"{}":{}'.format(i, f))
            if i < len(index)-1:
                mf.write(',')
            else:
                mf.write('}')
    return INDEX_FILE

""" Example manifest.save_path
manifest.jsonl
{"version":"1.1"}
{"type":"images"}
{"name":"145-8_5_big_494_0","extension":".jpg","width":1308,"height":1568,"meta":{"related_images":[]}}
{"name":"145-8_5_big_494_1","extension":".jpg","width":1175,"height":1172,"meta":{"related_images":[]}}
{"name":"145-8_5_big_494_2","extension":".jpg","width":1240,"height":826,"meta":{"related_images":[]}}
{"name":"145-8_5_big_494_3","extension":".jpg","width":900,"height":1285,"meta":{"related_images":[]}}
{"name":"145-8_5_big_497_0","extension":".jpg","width":971,"height":654,"meta":{"related_images":[]}}
{"name":"145-8_5_big_497_1","extension":".jpg","width":1206,"height":1000,"meta":{"related_images":[]}}
{"name":"145-8_5_big_497_2","extension":".jpg","width":1120,"height":679,"meta":{"related_images":[]}}
{"name":"145-8_5_big_497_3","extension":".jpg","width":1057,"height":1327,"meta":{"related_images":[]}}
"""

def create_cvat_manifest(files, save_path):
    """
    Create a manifest.jsonl file for cvat
    """
    index = []
    local_file = os.path.join(save_path, MANIFEST_FILE)
    with open(local_file, "w") as mf:
        mf.write('{"version":"1.1"}\n')
        mf.write('{"type":"images"}\n')
        cnt = 36
        for f in files:
            filename, ext = os.path.splitext(os.path.basename(f))
            img = cv2.imread(f)
            h, w, c = img.shape
            cstr = "{"
            cstr += "\"name\":\"{}\",\"extension\":\".{}\",\"width\":{},\"height\":{}".format(filename, ext[1:], w, h)
            cstr += ",\"meta\":{\"related_images\":[]}}\n"
            mf.write(cstr)
            index.append(cnt)
            cnt += len(cstr)

    assert os.path.exists(local_file)
    assert len(index) == len(files)
    return index



def init_cvat_dir(save_path):
    if not os.path.exists(save_path):
        os.system(f'mkdir -p {save_path}/data')
    assert os.path.exists(save_path), "Failed to create save_path " + save_path


def copy_images_and_zip(files, save_path):
    for f in files:
        shutil.copy(f, os.path.join(save_path, 'data'))
    local_file = os.path.join(save_path, ZIP_FILE)
    if os.path.exists(local_file):
        os.remove(local_file)
    os.system(f'cd {save_path} && zip -r {ZIP_FILE} .')
    assert os.path.exists(local_file)
    print('Zipped file:', local_file, ' for cvat')
    
    
def do_export_to_cvat(files, labels, save_path):
    init_cvat_dir(save_path)
    create_annotations_file(files, labels, save_path)
    create_tasks_file(files, labels, save_path)
    index = create_cvat_manifest(files, save_path)
    create_cvat_index(index, save_path)
    copy_images_and_zip(files, save_path)





if __name__ == '__main__':
    files = sorted(glob.glob('/mnt/data/sku110k/*')[:10])
    label_func = lambda x: x.split('/')[-1].split('.')[0].split('_')[-1]
    labels = [label_func(f) for f in files]
    do_export_to_cvat(files, labels,'/users/mikasnoopy/Downloads/cvat/')
        
