
# dtypes
IMG = 'images'
BBOX = 'bbox'

# run modes
MODE_DEFAULT = 'default'
MODE_EMBEDDING = 'embedding'
MODE_CROP = 'crop'

# fastdup files
SPLITS_CSV = 'splits_found.json'
SIMILARITY_CSV = 'similarity.csv'
OUTLIERS_CSV = 'outliers.csv'
MAPPING_CSV = 'atrain_features.dat.csv'
BAD_CSV = 'atrain_features.bad.csv'
FEATURES_DATA = 'atrain_features.dat'
STATS_CSV = 'atrain_stats.csv'
CONFIG_JSON = 'config.json'
CC_CSV = 'connected_components.csv'
CROPS_CSV = 'atrain_crops.csv'
CC_INFO_CSV = 'component_info.csv'
NNF_INDEX = 'nnf.index'

# extra files
BBOX_INPUT_CSV = 'objects_annot_fastdup_input.csv'
ANNOT_PKL = 'full_annot.pkl.gz'
IMG_GRP_ANNOT_PKL = 'img_grouped_annot.pkl.gz'

# annotation expected columns
ANNOT_FILENAME = 'img_filename'
ANNOT_CROP_FILENAME = 'crop_filename'
ANNOT_IMG_ID = 'img_id'
ANNOT_IMG_H = 'img_h'
ANNOT_IMG_W = 'img_w'
ANNOT_BBOX_X = 'bbox_x'
ANNOT_BBOX_Y = 'bbox_y'
ANNOT_BBOX_W = 'bbox_w'
ANNOT_BBOX_H = 'bbox_h'
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
ANNOT_FD_ID = 'fastdup_id'

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
