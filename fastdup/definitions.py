from enum import Enum

FILENAME_SIMILARITY = "similarity.csv"
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
FILENAME_KMEANS_ASSIGNMENTS = "kmeans_assignments.csv"
FILENAME_ERROR_MSG = "error.msg"

IMAGELIST_HEADER="index,filename"
LABEL_HEADER="index.label"
ALLOWED_STATS = ['blur','size','mean','min','max','unique','stdv', 'file_size','rms_contrast','mean_rel_intensity_r',
        'mean_rel_intensity_b','mean_rel_intensity_g','contrast,mean_saturation','edge_density','dominant_r','dominant_g','dominant_b']
STATS_HEADER="index,filename,width,height,unique,blur,mean,min,max,stdv,file_size,rms_contrast,mean_rel_intensity_r,mean_rel_intensity_b,mean_rel_intensity_g,contrast,mean_saturation,edge_density,dominant_r,dominant_g,dominant_b"
SIMILARITY_HEADER="from,to,distance"

FILENAME_TOP_COMPONENTS = "top_components.pkl"
FILENAME_TOP_CLUSTERS = "top_clusters.pkl"
MISSING_LABEL = "N/A"

S3_TEMP_FOLDER = "tmp"
S3_TEST_TEMP_FOLDER = "testtmp"
INPUT_FILE_LOCATION = "files.txt"
INPUT_TEST_FILE_LOCATION = "testfiles.txt"

DEFAULT_MODEL_FEATURE_WIDTH = 576
HIGH_ACCURACY_MODEL_FEATURE_WIDTH = 960

DEFUALT_METRIC_ZERO = 0
DEFAULT_METRIC_MINUS_ONE = -1
VERSION__ = "0.204"

GITHUB_URL = "https://github.com/visual-layer/fastdup/issues"

MATPLOTLIB_ERROR_MSG = "Warning: failed to import matplotlib, plot is not generated. Please pip install matplotlib if you "
"like to view aggregate stats plots. Matplotlib is deliberately not included as a requirement since it has multiple backends "
"and special care needs to select the right backend for your OS/Hardware combination. You can install matplot lib using "
"python3.8 -m pip install matplotlib matplotlib-inline. (change the python3.8 to your python version). "

SUPPORTED_IMG_FORMATS = [".png", ".jpg", ".jpeg", ".giff", ".jpeg", ".tif", ".heic", ".heif"]
SUPPORTED_VID_FORMATS = ["mp4", ".avi"]

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