from enum import Enum
import os
import sys
import tempfile

FILENAME_SIMILARITY = "similarity.csv"
FILENAME_SEARCH = "search.csv"
FILENAME_OUTLIERS = "outliers.csv"
FILENAME_NNF_INDEX = "nnf.index"
FILENAME_FEATURES = "features.dat"
FILENAME_IMAGE_LIST = "features.dat.csv"
FILENAME_IMAGE_STATS = "stats.csv"
FILENANE_BAD_IMAGE_LIST = "features.bad.csv"
FILENAME_COMPONENT_INFO = "component_info.csv"
FILENAME_CONNECTED_COMPONENTS = "connected_components.csv"
FILENAME_LABELS = "labels.csv"
FILENAME_KMEANS_CENTROIDS = "kmeans_centroids.csv"
FILENAME_CROP_LIST = "crops.csv"
FILENAME_KMEANS_ASSIGNMENTS = "kmeans_assignments.csv"
FILENAME_ERROR_MSG = "error.msg"
FILENAME_DUPLICATES_HTML = "duplicates.html"
FILENAME_OUTLIERS_HTML = "outliers.html"
FILENAME_COMPONENTS_HTML = "components.html"
FOLDER_FULL_IMAGE_RUN = "full_image_run"

IMAGELIST_HEADER="index,filename"
LABEL_HEADER="index.label"
ALLOWED_STATS = ['blur','size','mean','min','max','unique','stdv', 'file_size','rms_contrast','mean_rel_intensity_r',
        'mean_rel_intensity_b','mean_rel_intensity_g','contrast,mean_saturation','edge_density','dominant_r','dominant_g','dominant_b']
STATS_HEADER="index,filename,width,height,unique,blur,mean,min,max,stdv,file_size,rms_contrast,mean_rel_intensity_r,mean_rel_intensity_b,mean_rel_intensity_g,contrast,mean_saturation,edge_density,dominant_r,dominant_g,dominant_b"
SIMILARITY_HEADER="from,to,distance"

FILENAME_TOP_COMPONENTS = "top_components.pkl"
FILENAME_TOP_CLUSTERS = "top_clusters.pkl"
MISSING_LABEL = "N/A"

if sys.platform == "win32":
    S3_TEMP_FOLDER = tempfile.gettempdir()
    S3_TEST_TEMP_FOLDER = tempfile.gettempdir()
else:
    S3_TEMP_FOLDER = "tmp"
    S3_TEST_TEMP_FOLDER = "testtmp"
INPUT_FILE_LOCATION = "files.txt"
INPUT_TEST_FILE_LOCATION = "testfiles.txt"

DEFAULT_MODEL_FEATURE_WIDTH = 576
HIGH_ACCURACY_MODEL_FEATURE_WIDTH = 960

PRINTOUT_BAR_WIDTH = 88

DEFUALT_METRIC_ZERO = 0
DEFAULT_METRIC_MINUS_ONE = -1
# Version is dynamically inserted during build process from FASTDUP_VERSION file (line below will be replaced)
VERSION__ = "2.30"
"like to view aggregate stats plots. Matplotlib is deliberately not included as a requirement since it has multiple backends "
"and special care needs to select the right backend for your OS/Hardware combination. You can install matplot lib using "
"python3.8 -m pip install matplotlib matplotlib-inline. (change the python3.8 to your python version). "

SUPPORTED_IMG_FORMATS = [".png", ".jpg", ".jpeg", ".giff", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".bmp", ".webp", ".jp2", ".jfif", ".pdf", ".dcm", ".dicom", ".qaf"]
SUPPORTED_VID_FORMATS = [".mp4", ".avi", ".dav", ".m4v", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", ".3gp"]

RUN_ALL = 0
RUN_EXTRACT = 1
RUN_NN = 2
RUN_NNF_SEARCH_IMAGE_DIR = 3
RUN_NNF_SEARCH_STORED_FEATURES = 4
RUN_KMEANS = 5
RUN_KMEANS_STORED_FEATURES = 6

SELECTION_STRATEGY_FIRST = 0
SELECTION_STRATEGY_RANDOM = 1
SELECTION_STRATEGY_UNIFORM_METRIC = 2

YOLOV5S_MODEL = "https://github.com/itsnine/yolov5-onnxruntime/raw/master/models/yolov5s.onnx"
DINOV2S_MODEL = "https://vl-company-website.s3.us-east-2.amazonaws.com/model_artifacts/dinov2/dinov2_vits14.onnx"
DINOV2S_MODEL_DIM = 384
DINOV2B_MODEL = "https://vl-company-website.s3.us-east-2.amazonaws.com/model_artifacts/dinov2/dinov2_vitb14.onnx"
DINOV2B_MODEL_DIM = 768
CLIP_MODEL = "https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/ViT-B-32/visual.onnx"
CLIP_MODEL_DIM = 512
CLIP_MODEL2 = "https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/ViT-L-14@336px/visual.onnx"
CLIP_MODEL2_DIM = 768
CLIP_MODEL14 = "https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/ViT-L-14/visual.onnx"
CLIP_MODEL14_DIM = 768

EFFICIENTNET_MODEL = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"
EFFICIENTNET_MODEL_DIM = 1000
RESNET50_MODEL = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-12.onnx"
RESNET50_MODEL_DIM = 1000

CAPTION_MODEL_NAMES = ['automatic', 'vitgpt2', 'blip', 'blip2']
VQA_MODEL1_NAME = "vqa"
AGE_LABEL1_NAME = 'age'

# dtypes
IMG = 'image'
BBOX = 'bbox'

# run modes
MODE_DEFAULT = 'default'
MODE_EMBEDDING = 'embedding'
MODE_CROP = 'crop'
MODE_ROTATED_BBOX = 'rotated'

# fastdup files
SPLITS_CSV = 'splits_found.json'
MAPPING_CSV = 'atrain_features.dat.csv'
BAD_CSV = 'atrain_features.bad.csv'
FEATURES_DATA = 'atrain_features.dat'
STATS_CSV = 'atrain_stats.csv'
CONFIG_JSON = 'config.json'
CROPS_CSV = 'atrain_crops.csv'

# extra files
BBOX_INPUT_CSV = 'objects_annot_fastdup_input.csv'
ANNOT_PKL = 'full_annot.pkl.gz'
IMG_GRP_ANNOT_PKL = 'img_grouped_annot.pkl.gz'

# annotation expected columns
ANNOT_FILENAME = 'filename'
ANNOT_CROP_FILENAME = 'crop_filename'
ANNOT_IMG_ID = 'img_id'
ANNOT_IMG_H = 'img_h'
ANNOT_IMG_W = 'img_w'
ANNOT_BBOX_X = 'col_x'
ANNOT_BBOX_Y = 'row_y'
ANNOT_BBOX_W = 'width'
ANNOT_BBOX_H = 'height'
ANNOT_ROT_BBOX_X1 = 'x1'
ANNOT_ROT_BBOX_Y1 = 'y1'
ANNOT_ROT_BBOX_X2 = 'x2'
ANNOT_ROT_BBOX_Y2 = 'y2'
ANNOT_ROT_BBOX_X3 = 'x3'
ANNOT_ROT_BBOX_Y3 = 'y3'
ANNOT_ROT_BBOX_X4 = 'x4'
ANNOT_ROT_BBOX_Y4 = 'y4'
ANNOT_SPLIT = 'split'
ANNOT_ERROR = 'error_code'
ANNOT_LABEL = 'label'

# extended annotation columns
ANNOT_VALID = 'is_valid'
ANNOT_FD_ID = 'index'

# Connected components columns
CC_INST_ID = '__id'
CC_UNI_SPLIT = 'uni_split'
CC_BI_SPLIT = 'bi_split'

# bad files columns
BAD_FILENAME = 'filename'
BAD_ERROR = 'error_code'
BAD_FD_ID = 'index'

# similarity columns
SIM_SRC_IMG = 'from'
SIM_DST_IMG = 'to'
SIM_SCORE = 'distance'

# outliers columns
OUT_ID = 'outlier'
OUT_NEAREST_NEIGHBOR = 'nearest'
OUT_SCORE = 'distance'

# stats columns
STATS_INST_ID = 'index'

# map file columns
MAP_INST_ID = 'index'
MAP_FILENAME = 'filename'

ERROR_MISSING_IMAGE = 'ERROR_MISSING_FILE'
ERROR_BAD_BOUNDING_BOX = 'ERROR_BAD_BOUNDING_BOX'

def get_sep():
    return os.sep
