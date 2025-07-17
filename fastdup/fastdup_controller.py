import json
import os
import tempfile
from typing import List, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import fastdup
from pandas.errors import EmptyDataError
import shutil
import fastdup.definitions as FD
#import boto3
from fastdup import _LOGGER, _create_fastdup_logger
from fastdup.sentry import v1_sentry_handler, fastdup_capture_exception, fastdup_capture_log_debug_state, fastdup_metrics_increment
from fastdup.definitions import FOLDER_FULL_IMAGE_RUN
from fastdup.utilities import convert_coco_dict_to_df, shorten_path, _create_template_msg, _POST_RUN_MSG
import pathlib
import re
import logging

class FastdupController:
    def __init__(self, work_dir: Union[str, Path], input_dir: Union[str, Path, list] = None, log_level: Union[str, int] = None):
        """
        This class serves as a proxy for fastdup basic usage,
        the class wraps fastdup-run call provides quick access to
        fastdup files such as: similarity,  csv outlier csv, etc...

        Moreover, the class provides several extra features:
            - Ability to run connected component analysis on splits without calling fastdup run again
            - Ability to add annotation file and quickly merge it to any of fastdup inputs
        Currently the class support running fastdup on images and object
        :param work_dir: target output dir or existing output dir
        :param input_dir: (Optional) path to data dir, or a list of files
        """

        # allow for users to run without specifying work_dir, in this case a default work dir named work_dir is created
        if log_level is not None:
            self._logger = _create_fastdup_logger(f'fastdup.{__name__}.{id(self)}', log_level)
        else:
            self._logger = _LOGGER

        if work_dir is None:
            self._logger.warning('Warning: fastdup create() without work_dir argument, output is stored in a folder named work_dir in your current working path.')
            work_dir = "work_dir"

        # check if fastdup was already applied
        self._fastdup_applied = is_fastdup_dir(work_dir)
        if isinstance(work_dir, str):
            work_dir = shorten_path(work_dir)
        self._work_dir = Path(work_dir)
        self._input_dir = input_dir if input_dir is None else get_input_dir(input_dir)

        # set default arguments
        self._df_annot = None
        self._run_mode = FD.MODE_DEFAULT
        self._embeddings_dim = 576
        self._max_fd_id = 0
        self._config = None
        self._dtype = None
        self._bbox = None
        self._valid_bbox_annot = False
        self._valid_rotated_annot = False
        self._run_stats = True

        if self._fastdup_applied:
            assert not isinstance(input_dir, list), \
                'input_dir must be a single path when fastdup was already applied'
            # load fastdup state
            self._get_fastdup_state()

        self._has_split = self._df_annot is not None and FD.ANNOT_SPLIT in self._df_annot
        self._has_label = self._df_annot is not None and FD.ANNOT_LABEL in self._df_annot
        fastdup_capture_log_debug_state({"fastdup_applied": self._fastdup_applied, "run_mode":self._run_mode,
                                         "bbox": self._bbox, "has_split": self._has_split, "has_label": self._has_label})

        if self._logger.getEffectiveLevel() <= logging.INFO:
            print(_create_template_msg(self._work_dir, self._input_dir))

    def get_status_string(self):
        return f"self._run_mode={self._run_mode} self._dtype={self._dtype} self._bbox={self._bbox} self._has_label={self._has_label}"

    def get_annotations_dataframe(self):
        return self._df_annot

    def _get_fastdup_state(self):
        if os.path.isfile(self._work_dir / FD.CONFIG_JSON):
            self._config = json.load(open(self._work_dir / FD.CONFIG_JSON))
        else:
            assert False, f"Failed to read config file from {self._work_dir / FD.CONFIG_JSON}"
        self._input_dir = Path(self._config.get('input_dir', '.')) if self._input_dir is None else self._input_dir
        self._df_annot = pd.read_pickle(self._work_dir / FD.ANNOT_PKL, compression='gzip')
        assert self._df_annot is not None and not self._df_annot.empty, f"Failed to read pickle {str(self._work_dir / FD.ANNOT_PKL)}"
        self._dtype = self._infer_dtype(requested_dtype='image')
        self._filename_prefix = self._config.get('filename_prefix', self._get_filename_prefix(self._input_dir))
        self._embeddings_dim = self._config.get('embeddings_dim', 576)
        self._max_fd_id = self._config.get('max_fd_id', 0)
        self._run_mode = self._config.get('run_mode', FD.MODE_DEFAULT)

    def _init_run(self, input_dir: Union[str, Path] = None, df_annot: pd.DataFrame = None,
                  subset: list = None, embeddings=None, data_type: str = 'image', overwrite: bool = False,
                  fastdup_args: dict = None):
        """
        Initialize fastdup run arguments, unlike the constructor that tries to load an existing state from work_dir,
        this method allows to change the input_dir, df_annot, subset, embeddings, data_type arguments, checks they are
        valid, before executing fastdup run
        :param input_dir: folder containing the data
        :param df_annot: df with annotations
        :param subset: subset of files to run fastdup on
        :param embeddings: pre-calculated embeddings for images/bounding boxes, np.ndarray of type float32
        :param data_type: image or bbox
        :param overwrite: overwrite existing fastdup state (delete work_dir)
        """
        fastdup_capture_log_debug_state(locals())
        if overwrite:
            clean_work_dir(self._work_dir)
            #shutil.rmtree(self._work_dir, ignore_errors=True)
            self._fastdup_applied = False

        # set run mode & data type
        if df_annot is not None:
            assert isinstance(df_annot, pd.DataFrame) and not df_annot.empty, f"wrong df_annot type {df_annot}"
        self._dtype = self._infer_dtype(requested_dtype=data_type, df_annot=df_annot)
        self._run_mode = FD.MODE_DEFAULT

        bbox_mode = fastdup_args.get('bounding_box', 'none')
        if bbox_mode in ['yolov5s', 'face', 'ocr']:
            if bbox_mode == "ocr":
                try:
                    import paddle
                    import paddleocr
                except Exception as ex:
                    fastdup_capture_exception("Failed import paddle/paddleoecr", ex)
                    assert False, "For running with bounding_box='ocr' need to install paddlepaddle and paddleocr, using 'pip install -U paddlepaddle \"paddleocr>=2.0.6\" \"numpy==1.23\" \"PyMuPDF==1.21.1\""
            self._run_mode = FD.MODE_CROP
            self._dtype = FD.BBOX
            self._bbox = bbox_mode
        elif bbox_mode == 'rotated':
            self._run_mode = FD.MODE_ROTATED_BBOX
            self._dtype = FD.BBOX
            self._bbox = bbox_mode
        elif self._dtype == FD.BBOX:
            self._bbox = 'xywh_bbox'
        else:
            self._bbox = None

        if self._dtype == FD.BBOX:
            fastdup_args['bounding_box'] = self._bbox

        if embeddings is not None:
            assert isinstance(embeddings, np.ndarray), "embedding parameter must be np.ndarray"
            assert embeddings.dtype == 'float32', "embeddings dtype must be float32. You can generate the array using the" \
                                                "command: features = np.zeros((rows, cols), dtype='float32')"
            self._run_mode = FD.MODE_EMBEDDING
            self._embeddings_dim = embeddings.shape[1]
            assert df_annot is not None, "When running with embeddings, must provide annotations list with image names per each embedding row"

        self._fastdup_applied = is_fastdup_dir(self._work_dir)

        # set basic arguments and verify they are valid
        self._verify_fastdup_run_args(input_dir, self._work_dir, df_annot, subset, self._dtype, embeddings)
        self._pre_calc_features = embeddings
        self._subset = subset

        # check if annot has split and label and set filename prefix
        self._input_dir = get_input_dir(input_dir)
        self._df_annot = df_annot
        self._max_fd_id = 0
        self._filename_prefix = self._get_filename_prefix(self._input_dir)
        self._has_split = self._df_annot is not None and FD.ANNOT_SPLIT in self._df_annot
        self._has_label = self._df_annot is not None and FD.ANNOT_LABEL in self._df_annot

    def __getitem__(self, instance_id):
        # anything above max_fd_id is an artificial id - not in fastdup-mapping - invalid instance
        if instance_id not in self._df_annot.index:
            raise IndexError(f'instance_id: {instance_id} is invalid')
        return dict(self._df_annot.loc[instance_id])

    @property
    def run_mode(self):
        return self._run_mode

    @property
    def work_dir(self):
        return str(self._work_dir)

    @property
    def input_dir(self):
        return str(self._input_dir)

    def num_instances(self, valid_only=True):
        """
        Get number of instances in the dataset
        :param valid_only: if True, return only valid annotations
        """
        if self._df_annot is None:
            return 0
        df_annot = self._df_annot.query(f'{FD.ANNOT_VALID}') if valid_only and self._df_annot is not None \
            else self._df_annot
        return len(df_annot[FD.ANNOT_FILENAME].unique()) if self._df_annot is not None else 0


    def annotations(self, valid_only=True):
        """
        Get annotation as data frame
        :param valid_only: if True, return only valid annotations
        :return: pandas dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        df_annot = self._df_annot.query(f'{FD.ANNOT_VALID}') if valid_only and self._df_annot is not None \
            else self._df_annot
        return df_annot


    def init_search(self, k, verbose=False, license=""):
        """
        Initialize search
        Args:
            k (int): number of returned images
            verbose (bool): verbose level
            license (str): optional license string

        Returns:

        """
        return fastdup.init_search(k=k,
                                   work_dir=self.config['work_dir'],
                                   d=self.config['d'],
                                   model_path=self.config['model_path'],
                                   verbose=verbose,
                                   license=license
                                   )

    def search(self, filename, img=None, verbose=False):
        '''
           Search for similar images in the image database.

           Args:
               filename (str): full path pointing to an image.
               img (PIL.Image): (Optional) loaded and resized PIL.Image, in case given it is not red from filename
               verbose (bool): (Optiona) run in verbose mode, default is False
           Returns:
               ret (pd.DataFrame): None in case of error, otherwise a pd.DataFrame with from,to,distance columns
           '''
        return fastdup.search(filename=filename, img=img, verbose=verbose)

    def vector_search(self, filename="query_vector", vec=None, verbose=False):
        '''
        Search for similar embeddings to a given vector

        Args:
            filename: vector name (used for debugging)
            vec (numpy): Mandatory numpy matrix of size 1xd or a vector of size d
            verbose (bool): (Optiona) run in verbose mode, default is False
        Returns:
            ret (pd.DataFrame): None in case of error, otherwise a pd.DataFrame with from,to,distance columns
        '''
        return fastdup.vector_search(filename, vec, verbose=verbose)

    def similarity(self, data: bool = True, split: Union[str, List[str]] = None,
                   include_unannotated: bool = False, load_crops: bool = False) -> pd.DataFrame:
        """
        Get fastdup similarity file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :param load_crops: return similarity of crops (if load_crops=True) otherwise return similarity of full images
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        df = self._fetch_df(csv_name=FD.FILENAME_SIMILARITY, load_crops=load_crops, force_error=False)

        if df is None or df.empty:
            self._logger.debug(f'No similarities found, try using a lower threshold')
            return df
        merged = self._add_annot_and_split(df, data, merge_on=[FD.SIM_SRC_IMG, FD.SIM_DST_IMG],
                                         split=split, unannotated=include_unannotated, load_crops=load_crops)
        assert len(merged), f"Failed to merge {df.head()} {data.head()}"
        return merged

    def outliers(self, data: bool = True, split: Union[str, List[str]] = None,
                 include_unannotated: bool = False, load_crops: bool = False, fast_mode: bool = False) -> pd.DataFrame:
        """
        Get fastdup outlier file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :param load_crops: return outliers of crops (if load_crops=True) otherwise return similarity of full images

        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        # get df and rename columns (from, to, score) -> (outlier, nearest_neighbor, score)
        df = self._fetch_df(csv_name=FD.FILENAME_OUTLIERS, load_crops=load_crops)
        if df is None or df.empty:
            self._logger.debug(f'No outliers found')
            return None
        df = df.rename({
            FD.SIM_SRC_IMG: FD.OUT_ID, FD.SIM_DST_IMG: FD.OUT_NEAREST_NEIGHBOR
        }, axis=1)
        if not fast_mode:
            df = self._add_annot_and_split(df, data, merge_on=[FD.OUT_ID, FD.OUT_NEAREST_NEIGHBOR], split=split,
                                           unannotated=include_unannotated, suffix=True, load_crops=load_crops)
        assert len(df), "df has no rows"
        if FD.OUT_SCORE not in df.columns: #no outliers
            return df
        df = df.sort_values(FD.OUT_SCORE).groupby(FD.OUT_ID).head(1).reset_index(drop=True)
        return df


    def embeddings(self, d = 576):
        """
        Get fastdup embeddings
        :param d: feature vector width, on default 576

        :return: tuple: (list, np.array) list of filenamed with their features, np matrix of size len(list) with one feature per row
        """
        from fastdup import load_binary_feature
        return load_binary_feature(self._work_dir, d)

    def feature_vector(self, img_path, model_path=None, d=576):
        """
        Compute feature vector for a single image

        Args:
            img_path (str): a path pointing to an image, could be local or s3 or minio path, see run() documentation
            model_path: optional path pointing to onnx/ort model, see run() documentation
            d:o ptional feature vector width of the onnx/ort model, see run() documentation

        Returns:
            embeddings: 1 x d numpy matrix contains feature embeddings (row vector)
            files: the image filename used to genrate the embedding (should be equal to img_path)


        """
        assert isinstance(img_path, str) or isinstance(img_path, pathlib.Path)
        ret = self.run(input_dir=[img_path], model_path=model_path, d=d, overwrite=True, print_summary=False, run_mode=1, run_explore=False)
        if ret != 0:
            return ret
        files, embs = self.embeddings()
        assert len(files) != 0, "Failed to extract embeddings"

        return embs, files

    def feature_vectors(self, img_paths, model_path=None, d=576):
        """
        Compute feature vectors to a list of images

        Args:
            img_path (list): a list of local images, or a folder name
            model_path: optional path pointing to onnx/ort model, see run() documentation
            d:o ptional feature vector width of the onnx/ort model, see run() documentation

        Returns:
            embeddings: num_images x d numpy matrix contains feature embeddings, one per each row
            files: list of filenames used to create the embedding. Note that when some images are corrupted they will not appear in the valid filenames returned.
                   Use fd.annotations(valid_only=False) to get the bad image files report

        """
        if isinstance(img_paths, pd.DataFrame):
            assert "filename" in img_paths.columns, "Need to provide a dataframe with 'filename' column pointing to the full path of the images"
        elif isinstance(img_paths, list):
            assert len(img_paths), "Found ana empty list on img_paths2 variable"
            img_paths = pd.DataFrame({'filename': img_paths})
        ret = self.run(input_dir='.', model_path=model_path, d=d, overwrite=True, annotations=img_paths, print_summary=False, run_mode=1)
        if ret != 0:
            return ret
        files, embs = self.embeddings()
        if len(files) != len(img_paths):
            self._logger.debug("failed to generate feature vectors for some images. Run ")
        return embs, files

    def invalid_instances(self):
        """
        Get fastdup invalid file
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        return self._df_annot.query(f'not {FD.ANNOT_VALID}')

    def img_stats(self, data: bool = True, split: bool = None,
                  include_unannotated: bool = False, load_crops: bool = None) -> pd.DataFrame:
        """
        Get fastdup stats file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :param load_crops: Load crop data in case present (default is None)
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')

        if load_crops is None:
            load_crops = self._dtype == FD.BBOX
        df = self._fetch_df(csv_name=FD.STATS_CSV, load_crops=load_crops)
        if df is None:
            self._logger.info(f'Warning: No stats file found in {self._work_dir}')
            return pd.DataFrame()
        if df.empty:
            return df
        assert 'width' in df.columns and 'height' in df.columns, "df is missing needed columns: width and height"
        df = df.rename({'width': FD.ANNOT_IMG_W, 'height': FD.ANNOT_IMG_H}, axis=1)
        return self._add_annot_and_split(df, data, merge_on=[FD.ANNOT_FD_ID], split=split, suffix=False,
                                         unannotated=include_unannotated)

    @property
    def config(self) -> dict:
        """
        Get fastdup config file
        :return: config dict
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        return self._config

    def connected_components(self, data: bool = True, split: Union[str, List[str]] = None,
                             include_unannotated: bool = False, load_crops: bool = False, fast_mode:bool = False) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get fastdup connected components file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :param load_crops: return components of crops (if load_crops=True) otherwise return similarity of full images

        :return: requested dataframe, Each raw contains a single image, and colum contains component_id for that image.
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        # get connected components and add annotation
        df_cc = self._fetch_df(csv_name=FD.FILENAME_CONNECTED_COMPONENTS, load_crops=load_crops)
        if df_cc is None:
            self._logger.debug(f'No connected components found, try using a lower threshold')
            return pd.DataFrame(), None
        assert FD.CC_INST_ID in df_cc.columns, f'Missing column {FD.CC_INST_ID} in connected component file'
        df_cc = df_cc[df_cc['count'] != 0] # remove singletons

        if df_cc.empty:
            self._logger.debug(f'No connected components found, try using a lower threshold')
            return df_cc, None
        assert FD.CC_INST_ID in df_cc.columns, f'Missing column {FD.CC_INST_ID} in connected component file'
        df_cc = df_cc.rename({FD.CC_INST_ID: FD.ANNOT_FD_ID}, axis=1)
        if not fast_mode:
            df_cc = self._add_annot_and_split(df_cc, data, merge_on=[FD.ANNOT_FD_ID], split=split, suffix=False,
                                              unannotated=include_unannotated, load_crops=load_crops)
        df_info = self._fetch_df(csv_name=FD.FILENAME_COMPONENT_INFO, load_crops=load_crops)
        if df_info is None or df_info.empty:
            self._logger.debug(f"Warning: Missing file {self.work_dir}/{FD.FILENAME_COMPONENT_INFO}")
        return df_cc, df_info


    def connected_components_grouped(self, sort_by: str = 'comp_size', ascending: bool = True, metric = None, load_crops: bool = None, group_by: str = "visual", kwargs = {}) -> pd.DataFrame:
        """
        Get a dataframe where each row is a component (=cluster) and in the column list of images belonging to that cluster
        :param sort_by: parameter to sort by, on default by comp_size
        :param asecnding: sorting order
        :param metric: include image statistic metric like blur, mean, min, max, etc
        :param load_crops: load crops

        :return: requested dataframe, Each raw contains a component with all images belonging to that cocmponent
        """
        assert group_by in ["visual", "label"], "group_by can be either 'visual' (grouping by similar images) or 'label' (groupinbg by labels) "

        if 'draw_bbox' in kwargs and kwargs['draw_bbox']:
            load_crops = False
        external_df = self.connected_components(load_crops=load_crops)

        assert isinstance(external_df, tuple), f"Wrong return values from connected components {external_df}"
        external_df = external_df[0]
        if group_by == "label":
            assert "label" in external_df.columns, "Failed to find label columns so can not group by labels."

        external_df = external_df.dropna(subset='filename')
        from fastdup.galleries import  load_and_merge_stats
        external_df = load_and_merge_stats(external_df, metric, self.work_dir, kwargs)
        assert 'filename' in external_df.columns, f"Failed to find filename column in {external_df.head()}"
        if metric is not None:
            assert metric in external_df.columns, f"Failed to find metric {metric} in columns {external_df.head()}"

        group_by_col = "component_id" if group_by == "visual" else "label"
        files = external_df.groupby(group_by_col)['filename'].apply(list)
        files_ids = external_df.groupby(group_by_col)['index'].apply(list)
        distance = external_df.groupby(group_by_col)['min_distance'].apply(np.min)
        crop_files = None
        if 'crop_filename' in external_df.columns:
            crop_files = external_df.groupby(group_by_col)['crop_filename'].apply(list)
        labels = None
        if group_by == 'visual' and 'label' in external_df.columns:
            labels = external_df.groupby(group_by_col)['label'].apply(list)
        elif group_by == "label":
            components = external_df.groupby(group_by_col)['component_id'].apply(list)

        if metric is not None:
            metrics = external_df.groupby(group_by_col)[metric].apply(np.mean)

        external_df = pd.DataFrame({'files': files.values, 'component_id': files.index.values if group_by == "visual" else components,
                                    'files_ids': files_ids.values if crop_files is None else crop_files.values,
                                    'distance': distance.values})

        if crop_files is not None:
            external_df['crop_filename'] = crop_files.values
        if labels is not None and group_by == "visual":
            external_df['label'] = labels.values
        elif group_by == "label":
            external_df = external_df.reset_index()
        external_df['len'] = external_df['files'].apply(lambda x: len(x))
        if metric is not None:
            external_df[metric] = metrics.values
        external_df = external_df[external_df['len'] >= 2]
        if sort_by == 'comp_size':
            external_df = external_df.sort_values('len', ascending=ascending)
        else:
            assert sort_by in external_df.columns, f"Wrong sort_by column, missing column {sort_by} available columns: {external_df.columns}"
            external_df = external_df.sort_values(sort_by, ascending=ascending)

        return external_df

    @v1_sentry_handler
    def run(self, input_dir: Union[str, Path] = None, annotations: Union[pd.DataFrame,list] = None, subset: list = None,
            embeddings=None, data_type: str = FD.IMG, overwrite: bool = False,
            print_summary: bool = False, print_vl_datasets_ref: bool = False, run_explore: bool = True, dataset_name: str = None, verbose: bool=False,
            run_fast: bool=False,
            **fastdup_kwargs):
        """
        This function
            1. calculate subset of images to analyze
            2. run fastdup
            3. map images/bboxes to fastdup index (on bbox this is done in self._set_fastdup_input)
            4. expand annotation csv to include files that are not in annotation but is in subset
            5. create a version of annotation that is grouped by image
        :param input_dir: input directory containing images
        :param annotations: (Optional) annotations dataframe, the expected column convention is:
             - filename: input_dir-relative filenames
             - img_h, img_w (Optional): image height and width
             - bbox_x, bbox_y, bbox_h, bbox_w (Optional): bounding box arguments. Alternatively x1,y2,x2,y2,x3,y3,x4,x4 for rotated bounding box.
             - split (Optional): data split, e.g. train, test, etc ...
             Alternatively, a list of filenames
             Alternatively, a filename of json coco format contains bounding box annotations
             Alternatively, a dictionary containing coco format annotations
        :param subset: (Optional) subset of images to analyze
        :param embeddings: (Optional) pre-calculated feature embeddings. Data type of np.ndarray of size n x d, n is the number of data points, d is the feature vector length.
            data type must be 'float32'. When embedding is given, must send annotations which are list of filenames matching the same length of the embeddings.
        :param data_type: (Optional) data type, one of 'image', 'bbox'
        :param overwrite: (Optional) overwrite existing files
        :param print_summary: Print summary report of fastdup run results
        :param run_fast: Skip backward compatible code to run faster
        :param fastdup_kwargs: (Optional) fastdup run arguments, see fastdup.run() documentation

        """
        fastdup_capture_log_debug_state(locals())
        fastdup_metrics_increment('run')

        if self._fastdup_applied and not overwrite:
            self._logger.info('Fastdup was already applied, use overwrite=True to re-run')
            return
        if annotations is not None:
            if isinstance(annotations, list):
                annotations = pd.DataFrame({'filename':annotations})
            elif isinstance(annotations, dict):
                assert isinstance(self.input_dir, str), f"When working with COCO annotations need to provide fastdup.create(input_dur=...) with input_dir which is a single assolute path pointing to root folder with all images, got {self._input_dir}"
                annotations = convert_coco_dict_to_df(annotations, self._input_dir)
            elif isinstance(annotations, str) or isinstance(annotations, pathlib.Path):
                if isinstance(annotations, str):
                    annotations = shorten_path(annotations)
                assert os.path.isfile(annotations), f"Failed to find annotations file {annotations}"
                if annotations.endswith('.csv'):
                    annotations = pd.read_csv(annotations)
                elif annotations.endswith('.json'):
                    import json
                    label = json.loads(open(annotations, 'r').read())
                    annotations = convert_coco_dict_to_df(label, self._input_dir)
                else:
                    assert False, "Unknown annotation file format, should end with .csv or .json"
            elif isinstance(annotations, pd.DataFrame):
                annotations = annotations.copy() # fastdup changes the input

            assert isinstance(annotations, pd.DataFrame) and not annotations.empty and "filename" in annotations.columns, f"Got wrong annotation parameter, should be pd.DataFrame with the mandatory columns: filename {annotations}"
            first_filename = annotations['filename'].values[0]
            user_columns = list(annotations.columns)
        else:
            user_columns = None
        self._init_run(input_dir, annotations, subset, embeddings, data_type, overwrite, fastdup_kwargs)

        # get user's fastdup kwargs or use default
        fastdup_kwargs = {} if fastdup_kwargs is None else fastdup_kwargs
        if embeddings is not None:
            if 'run_stats'  in fastdup_kwargs:
                assert not fastdup_kwargs['run_stats'], "When computing a model on embeddings stats are not computed. If you like to compute stats run with run_stats_only=True without embeddings on a clean work_dir."
            if 'run_advanced_stats' in fastdup_kwargs:
                assert not fastdup_kwargs['run_advanced_stats'], "When computing a model on embeddings advanced_stats are not computed. If you like to compute stats run with run_stats_only=True without embeddings on a clean work_dir."
            self._run_stats = False
            run_explore = False

        if 'bounding_box' in fastdup_kwargs:
            run_explore = False

        if self._pre_calc_features is not None:
            fastdup_kwargs['run_mode'] = 2
            fastdup_kwargs['d'] = self._embeddings_dim
        fastdup_kwargs = set_fastdup_kwargs(fastdup_kwargs)
        if 'run_stats' in fastdup_kwargs and not fastdup_kwargs['run_stats']:
            self._run_stats = False

        assert not os.path.isfile(self._work_dir), f"Work dir {self._work_dir} is pointing to a file"
        os.makedirs(self._work_dir, exist_ok=True)
        assert os.path.exists(self._work_dir), "Failed to create folder " + str(self._work_dir)

        if overwrite and os.path.isfile(os.path.join(self._work_dir, 'atrain_features.dat.csv')):
            os.unlink(os.path.join(self._work_dir, 'atrain_features.dat.csv'))
        # run fastdup - create embeddings
        fastdup_input = self._set_fastdup_input()
        if not run_fast:
            if fastdup.run(fastdup_input, work_dir=str(self._work_dir), logger=self._logger, **fastdup_kwargs) != 0:
                raise RuntimeError('Fastdup execution failed')
        
            # post process - map fastdup-id to image (for bbox this is done in self._set_fastdup_input)
            if self._dtype == FD.IMG or self._run_mode == FD.MODE_CROP:
                self._create_img_mapping()

            # expand annotation csv to include files that are not in annotation but is in subset
            self._expand_annot_df()
            if self._dtype != FD.BBOX:
                self._index_annot_df()

            self._save_artifacts(fastdup_kwargs)
            self._fastdup_applied = True

        if run_explore:
            fastdup_metrics_increment('run_with_explore')
            from fastdup.fastdup_runner.run import do_visual_layer
            vl_input = self._input_dir if user_columns is None else self.annotations()[user_columns]
            num_images = fastdup_kwargs.get('num_images', None)
            if num_images is not None:
                assert isinstance(num_images, int), "num_images argument should be int"
                os.environ["FASTDUP_NUM_IMAGES"] = str(num_images)
            do_visual_layer(work_dir=self._work_dir, input_dir=vl_input,
                            dataset_name=dataset_name, overwrite=overwrite, run_server=False, verbose=verbose)
        else:
            fastdup_metrics_increment('run_without_explore')

        if print_summary:
            self.summary(show_comp = fastdup_kwargs.get('run_stats_only', 0 ) == 0)
        if print_vl_datasets_ref:
            self.vl_datasets_ref_printout()

        if self._logger.getEffectiveLevel() <= logging.INFO:
            print(_POST_RUN_MSG)

        return 0
    
    def explore(self, verbose=False) -> None:
        from fastdup.fastdup_runner.run import do_visual_layer
        fastdup_metrics_increment('explore')
        do_visual_layer(work_dir=self._work_dir, overwrite=False, run_server=True, verbose=verbose)

    def summary(self, verbose=True, blur_threshold: float = 150.0, brightness_threshold: float = 253.0,
                darkness_threshold: float = 4.0, show_comp: bool = True) -> List[str]:
        """
        This function provides a summary of the dataset statistics and issues uncovered
        using fastdup. This includes total number of images, invalid images, duplicates, outliers,
        blurry and dark/bright images. Count and percent for each.
        :param verbose:
        :param blur_threshold:
        :param brightness_threshold:
        :param darkness_threshold:
        :return:
        """
        summary_stats = []
        # Image count and valid count, by reason
        total_image_count = len(self.annotations(valid_only=False))

        def pct(x):
            return 100 * x / total_image_count

        object = "images" if (self._bbox is None or (self._bbox not in ['xywh_bbox','rotated','yolov5s', 'face', 'ocr'])) else "objects"
        summary_stats.append(f"Dataset contains {total_image_count} {object}")

        invalid_data_df = self.invalid_instances()
        invalid_image_count = len(invalid_data_df)
        valid_image_count = total_image_count - invalid_image_count
        invalid_stats = f"Valid {object} are {pct(valid_image_count):.2f}% ({valid_image_count:,d}) of the data, "\
                        f"invalid are {pct(invalid_image_count):.2f}% ({invalid_image_count:,d}) of the data"
        summary_stats.append(invalid_stats)
        if invalid_image_count:
            summary_stats.append("For a detailed analysis, use `.invalid_instances()`.\n")
        # Images belonging to clusters
        # X percent of images belong to Y clusters, the largest of ZZ images.

        if show_comp:
            try:
                cc_df, cc_info_df = self.connected_components(fast_mode=True)
                number_of_ccs = len(cc_info_df)
                images_in_ccs = int(cc_df)
                images_not_in_ccs = total_image_count - images_in_ccs
                largest_cc_image_count = int(cc_df['count'].max())
                summary_stats.append(f"Similarity:  {pct(images_in_ccs):.2f}% ({images_in_ccs:,d}) belong to "\
                                     f"{number_of_ccs} similarity clusters (components).")
                summary_stats.append(f"{pct(images_not_in_ccs):.2f}% ({images_not_in_ccs:,d}) images do not "
                                     f"belong to any similarity cluster.")
                summary_stats.append(f"Largest cluster has {largest_cc_image_count:,d} "
                                     f"({pct(largest_cc_image_count):.2f}%) edges.")
                sim_thresh = self.config['fastdup_kwargs']['threshold']
                cc_thresh = self.config['fastdup_kwargs']['turi_param']['ccthreshold']
                summary_stats.append(f"For a detailed analysis, use `.connected_components()`\n"
                                     f"(similarity threshold used is {sim_thresh}, "
                                     f"connected component threshold used is {cc_thresh}).\n")
            except Exception as e:
                # summary_stats.append(f"Components:  failed to find images clustered into components, try to run with lower cc_threshold.")
                pass

        if show_comp:
            try:
                # Outlier counts
                outlier_df = self.outliers(fast_mode=True)
                if outlier_df is not None:
                    number_of_outliers = len(outlier_df)
                    outlier_threhold = 100 * self._config.get('lower_threshold', 0.05)
                    summary_stats.append(f"Outliers: {pct(number_of_outliers):.2f}% ({number_of_outliers:,d}) of images are "\
                                  f"possible outliers, and fall in the bottom {outlier_threhold:.2f}% of "\
                                  f"similarity values.")
                    summary_stats.append(f"For a detailed list of outliers, use `.outliers()`.\n")
            except Exception as e:
                fastdup_capture_exception("failed to calc outliers", e)
                summary_stats.append(f"Outliers: Unable to calculate outliers.")

        try:
            # Blurry, dark and bright images
            if self._run_stats:
                stats_df = self.img_stats()
                blurry_images = (stats_df.blur < blur_threshold).sum()
                bright_images = (stats_df['mean'] > brightness_threshold).sum()
                dark_images = (stats_df['mean'] < darkness_threshold).sum()
                stats_str = f"Blur: found {pct(blurry_images):.2f}% of images ({blurry_images:,d}) "\
                            f"that are blurry.\n "\
                            f"Brightness: found {pct(bright_images):.2f}% of images ({bright_images:,d}) " \
                            f"that are bright.\n "\
                            f"Darkness: found {pct(dark_images):.2f}% of images ({dark_images:,d}) " \
                            f"that are dark.\n "\
                            f"For more information, use `.img_stats()`.\n"
            else:
                stats_str = "Skipped running stats\n"
        except Exception as e:
            stats_str = f"Unable to calculate blur, brightness and darkness.\n"
            fastdup_capture_exception(stats_str, e)

        if verbose:
            print('\n', FD.PRINTOUT_BAR_WIDTH * '#')
            print(f"\nDataset Analysis Summary: \n")
            for line in summary_stats:
                print(f"    {line}")

        return summary_stats

    def vl_datasets_ref_printout(self, vl_datasets_link='https://app.visual-layer.com/vl-datasets?utm_source=fastdup'):
        """
        Next steps:
            1. Add UTM tracking
            2. Make this visible during the progress bar for long-running data ("Meanwhile, why won't you check out...")
            3. Make this visible after a run is completed
            4. Make this visible during gallery generation (Optional: add a deprecation warning for the HTML galleries)
        """
        print(FD.PRINTOUT_BAR_WIDTH * "#")
        print(f"Would you like to see awesome visualizations for some of the most popular academic datasets?")
        print(f"Click here to see and learn more: {vl_datasets_link}")
        print(FD.PRINTOUT_BAR_WIDTH * "#")

    def img_grouped_annot(self, image_level_columns=None) -> pd.DataFrame:
        """
        This function groups annotation according to fastdup-im-id and returns the grouped file.
        Because this process takes significant time the process is done once and the result is cached
        :return: grouped annotation dataframe
        """
        image_level_columns = [] if image_level_columns is None else image_level_columns
        # load in case file exist
        img_grp_annot_csv = self._work_dir / FD.IMG_GRP_ANNOT_PKL
        if os.path.exists(img_grp_annot_csv):
            return pd.read_pickle(img_grp_annot_csv, compression='gzip')

        # merge on img column
        merge_col = FD.ANNOT_FILENAME
        df_annot = self._df_annot.copy()

        # define image level columns - for which all values are the same for a given image
        # define other columns - such as bbox for which one image can have several values
        agg_cols = list(set(df_annot.columns).difference({merge_col}))
        first_val_cols = {FD.ANNOT_IMG_ID, FD.ANNOT_SPLIT, FD.ANNOT_SPLIT, FD.ANNOT_IMG_H, FD.ANNOT_IMG_W
                          }.union(set(image_level_columns))
        # is_valid could be image-level or bbox-level depending on the input data type
        if self._dtype == FD.IMG:
            first_val_cols = first_val_cols.union({FD.ANNOT_VALID, FD.ANNOT_FD_ID})
        first_val_cols = first_val_cols.intersection(set(df_annot.columns))
        to_list_cols = set(self._df_annot.columns).difference(first_val_cols).difference({merge_col})

        # define aggregation and group
        agg_func = dict(**{col: lambda x: x.iloc[0] for col in first_val_cols},
                        **{col: pd.Series.tolist for col in to_list_cols})
        df_annot = df_annot.groupby(merge_col)[agg_cols].agg(agg_func).reset_index()
        df_annot.to_pickle(img_grp_annot_csv, compression='gzip')
        return df_annot

    def _index_annot_df(self):
        """
        add fastdup-id to rows with NA-id and set index
        """
        self._max_fd_id = self._df_annot[FD.ANNOT_FD_ID].max()
        self._df_annot['fd_index'] = self._df_annot[FD.ANNOT_FD_ID].fillna(
            pd.Series(np.arange(len(self._df_annot))) + self._max_fd_id + 1
        )
        self._df_annot = self._df_annot.set_index('index', drop=False)
        self._df_annot.index.name = None

    def _save_artifacts(self, fastdup_kwargs):
        """
        This function saves artifacts that are created during fastdup run
        """
        # read config
        fastdup_kwargs['turi_param'] = dict([arg.split("=") for arg in fastdup_kwargs['turi_param'].split(',')])
        config_json = self._work_dir / FD.CONFIG_JSON
        assert os.path.exists(config_json), f"Failed to find config json file {str(self._work_dir / FD.CONFIG_JSON)}"
        with open(config_json, 'rt') as fp:
            config = json.load(fp)

            # update config
        config['max_fd_id'] = int(self._max_fd_id)
        config['data_type'] = self._dtype
        config['filename_prefix'] = str(self._filename_prefix)
        config['input_dir'] = str(self._input_dir)
        config['embeddings_dim'] = self._embeddings_dim
        config['run_mode'] = self._run_mode
        config['fastdup_kwargs'] = fastdup_kwargs
        self._config = config
        # save config
        with open(config_json, 'wt') as fp:
            json.dump(config, fp, indent=4)
        assert os.path.isfile(config_json), f"Failed to write to config.json file {str(config.json)}"

        # save image mapping
        self._df_annot.to_pickle(self._work_dir / FD.ANNOT_PKL)

    def _set_fastdup_input(self):
        """
        Set input for fastdup according to data type, this function takes into account that the inputs are validated.
        The following options are supported:
            1. image data:
                a. w/wo annot, no subset: return input dir (input dir can also be a list of dirs in that case)
                b. w/wo annot, subset: (edit annot - intersect over subset) and return subset as list
            2. bbox data:
                a. with annot, no subset: create input-csv from annot and return path to it
                b. with annot, subset: intersect annot over subset, create input-csv from annot and return path to it
        ***************************************************************************************************************
        * for bounding box data type this function creates the fastdup indices as well                                *
        * for image data type indexing should be done after fastdup run                                               *
        * this is because on image d-type the input can be a directory, and it's not scalable to index it on python.  *
        ***************************************************************************************************************
        :return: first input argument for fastdup
        """
        if self._pre_calc_features is not None:
            n_instances, embeddings_dim = self._pre_calc_features.shape
            self._embeddings_dim = embeddings_dim
            if self._df_annot is None:
                self._df_annot = pd.DataFrame([f'{i}' for i in np.arange(n_instances)], columns=[FD.ANNOT_FILENAME])
            if self._subset is not None:
                self._df_annot = self._df_annot.iloc[self._subset]
                self._pre_calc_features = self._pre_calc_features[self._subset]
            assert FD.ANNOT_FILENAME in self._df_annot.columns, f"Failed to find {FD.ANNOT_FILENAME} in annotations dataframe, this colum should contain the path (relative or absolute) to image or video filenamesfilename"
            fastdup.save_binary_feature(self.work_dir, self._df_annot[FD.ANNOT_FILENAME].to_list(), self._pre_calc_features)
            assert self.input_dir is not None, "Failed to find input dir"
            return self.input_dir
            # image data type - return input dir or subset (and edit annot by intersecting over subset)
        elif self._dtype == FD.IMG or self._run_mode == FD.MODE_CROP:
            # In case we have annotations, but no bounding box just class labels
            if self._df_annot is not None:
                assert FD.ANNOT_FILENAME in self._df_annot.columns, f"Failed to find {FD.ANNOT_FILENAME} in annotations dataframe, this colum should contain the path (relative or absolute) to image or video filenamesfilename"
                assert not self._df_annot.empty, "Found empty annotations df"
                if self._subset is not None:
                    df_annot = self._df_annot[self._df_annot[FD.ANNOT_FILENAME].isin(self._subset)]
                    assert not self._df_annot.empty, "Failed to find any annotations in the subset"
                else:
                    df_annot = self._df_annot

                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    temp_name = f'{temp.name}.csv'
                    df_annot[[FD.ANNOT_FILENAME]].to_csv(temp_name, index=False)
                    assert os.path.exists(temp_name), f"Failed to write annotations to file {temp_name}"
                    return temp_name

            else:
                if self._subset is None:
                    return str(self._input_dir) if not isinstance(self._input_dir, list) else self._input_dir
                else:
                    subset = [str(self._input_dir / s) for s in
                              self._subset] if self._subset is not None else self._subset
                    assert subset is not None and subset != [], "Failed to find input dir"
                    return subset

        elif self._dtype == FD.BBOX:
            if self._subset is not None:
                subset = []
                assert FD.ANNOT_SPLIT in self._df_annot.columns, f"Failed to find column {FD.ANNOT_SPLIT} in annotations columns"
                self._df_annot = self._df_annot[self._df_annot[FD.ANNOT_SPLIT].isin(subset)]

            # set fastdup instance id
            self._df_annot[FD.ANNOT_FD_ID] = np.arange(len(self._df_annot)).astype(int)

            # save bbox csv & add input-dir images filename
            df_annot = self._df_annot.copy()
            assert FD.ANNOT_FILENAME in self._df_annot.columns, f"Failed to find column {FD.ANNOT_FILENAME} in annotations columns"

            df_annot['filename'] = df_annot[FD.ANNOT_FILENAME].apply(lambda fname: Path(self._input_dir) / fname)
            df_annot = df_annot.astype({FD.ANNOT_FD_ID: int})

            # convert column names input expected: "filename,col_x,row_y,width,height"
            bbox_cols = [FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H]
            rotated_bbox_cols = [FD.ANNOT_ROT_BBOX_X1, FD.ANNOT_ROT_BBOX_Y1, FD.ANNOT_ROT_BBOX_X2, FD.ANNOT_ROT_BBOX_Y2,
                                 FD.ANNOT_ROT_BBOX_X3, FD.ANNOT_ROT_BBOX_Y3, FD.ANNOT_ROT_BBOX_X4, FD.ANNOT_ROT_BBOX_Y4, 
                                 FD.ANNOT_LABEL]
            if set(rotated_bbox_cols).issubset(df_annot.columns):
                main_cols = [FD.ANNOT_FD_ID, 'filename'] + rotated_bbox_cols
                df_annot = df_annot[main_cols]
            elif set(bbox_cols).issubset(df_annot.columns):
                main_cols = [FD.ANNOT_FD_ID, 'filename'] + bbox_cols
                df_annot = df_annot[main_cols]
            else:
                assert False, f"Found wrong columns in annotation files {df_annot.columns} should include bounding box in the format {bbox_cols} or {rotated_bbox_cols}"
            # run fastdup on a file with full path
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp_name = f'{temp.name}.csv'
                df_annot.to_csv(temp_name, index=False)
                assert os.path.exists(temp_name), f"Failed to write annotations to file {temp_name}"
                return temp_name


    def _add_annot_and_split(self, df: pd.DataFrame, data: bool, merge_on: List[str],
                             split: Union[str, List[str]] = None, suffix: bool = True,
                             unannotated: bool = False, load_crops: bool = False) -> pd.DataFrame:
        """
        Merge df according to context (IMG/OBJ) and filter according to split
        :param df: input df
        :param data: (Optional) if true - add annotation
        :param merge_on: merge on columns - merged with fastdup-id
        :param split: (Optional) filtered by split
        :param suffix: (Optional) add suffix of merged on column
        :return: merged and filtered df
        """


        # yolo and face detection do not get annotationm, instead they run on the data and produce output annotations
        # those annotations are read back
        if self._bbox in ["face", "yolov5s", "ocr"]:
            if self._bbox == "face":
                df_annot = self._fetch_df(FD.MAPPING_CSV, True, force_error=True)
            elif self._bbox == "yolov5s":
                df_annot = self._fetch_df(FD.MAPPING_CSV, False, force_error=True)
                df_crops = self._fetch_df(FD.CROPS_CSV, False, force_error=True)
                assert len(df_crops) == len(df_annot), f"Wrong length detected"
                if "filename" in df_annot:
                    del df_annot["filename"]
                df_annot2 = df_annot.merge(df_crops, on=FD.ANNOT_FD_ID)
                assert len(
                    df_annot2), f"Failed to merge {df_annot.head()} {df_crops.head()} " + self.get_status_string()
                df_annot = df_annot2
            elif self._bbox == "ocr":
                self._df_annot = self._fetch_df(FD.CROPS_CSV, True, force_error=True)
                self._df_annot[FD.ANNOT_VALID] = True
                df_annot = self._df_annot

            #self._df_annot = df_annot
            original_col_names = df_annot.columns
            for col_name in merge_on:
                # merge to left side with appropriate suffix (forced)
                if len(merge_on) > 1:
                    df_annot.columns = df_annot.columns.map(
                    lambda x: f'{str(x)}_{col_name}' if (x != FD.ANNOT_FD_ID) else str(x))
                if df.empty or df_annot.empty:
                    return df
                df2 = pd.merge(df, df_annot, left_on=col_name, right_on=FD.ANNOT_FD_ID, how='left')
                assert len(df2), f"Failed to merge {df.head()} {df_annot.head()}" + self.get_status_string()
                df = df2
                df_annot.columns = original_col_names
                if len(df) == 0:
                    assert len(
                        df), f"Failed to merge df with annotations {col_name} {df_annot.columns} {df.head()} {df_annot.head()}"
            return df

        else:
            # get split and annotations
            splits = split if isinstance(split, list) or split is None else [split]
            if df is None or (not data and split is None):
                return df

            df_annot = self._merge_df_with_annot(df, left_on=merge_on, suffix=suffix, unannotated=unannotated, load_crops=load_crops)
            assert len(df_annot), "Failed to merge"

            # filter by split
            if split is not None and self._has_split:
                df_annot = df_annot[np.logical_and.reduce([df_annot[f'{FD.ANNOT_SPLIT}_{col}' if suffix else FD.ANNOT_SPLIT]
                                                          .isin(splits) for col in merge_on])]
                assert len(df_annot), "Failed to find split"
            elif split is not None and not self._has_split:
                raise ValueError(f'No split column found in annotation file')

            # merge annotations
            if not data:
                df_annot = df_annot[df.columns]

        assert len(df_annot), "Failed to find annotations"
        return df_annot

    def _merge_df_with_annot(self, df: pd.DataFrame, left_on: List[str],
                            inplace: bool = False, suffix: bool = True, unannotated: bool = False, load_crops: bool = False) -> pd.DataFrame:
        """
        Merge any df with img-annotation-file (grouped) /  object-annotation-file. by default the merge is done
        according to fastdup-id, to override use right_on argument
        :param df: df to merge from left
        :param left_on: list of columns to merge on
        :param right_on: (Optional) column name to merge on in df-annotation (fastdup-id)
        :param inplace: (Optional) override df
        :param suffix: (Optional) if True left_on - column names are attached as suffix to merged annotation
        :return: merged dataframe
        """

        assert len(df), "Bug: got empty df"
        df2 = df if inplace else df.copy()
        df_annot = self._df_annot.copy()
        df_annot = df_annot[~df_annot[FD.ANNOT_FD_ID].isna()]
        if not unannotated:
            df_annot = df_annot[df_annot[FD.ANNOT_VALID]]
            assert len(df_annot), f"Failed to find valid annotations"
            #df2 = df2[df2[left_on].isin(df_annot.index).all(axis=1)]
            #assert len(df2), f"Failed to find index in col {left_on} {df_annot.head()} {df.head()}"
        #df_annot = df_annot.reset_index()
        original_col_names = df_annot.columns

        for col_name in left_on:
            # merge to left side with appropriate suffix (forced)
            df_annot.columns = df_annot.columns.map(lambda x: f'{str(x)}_{col_name}' if (suffix and x !=FD.ANNOT_FD_ID) else str(x))
            df3 = pd.merge(df2, df_annot, left_on=col_name, right_on=FD.ANNOT_FD_ID, how='left')
            df_annot.columns = original_col_names
            if len(df3) == 0:
                assert len(df3), f"Failed to merge df with annotations {col_name} {df_annot.columns} {df2.head()} {df_annot.head()}"
            df2 = df3
        return df2

    def _fetch_df(self, csv_name: str, load_crops: bool = False, force_error: bool = False) -> pd.DataFrame:
        """
        Retrieve fastdup file from relevant workdir
        :param csv_name: basename of required file
        :return: pd.DataFrame
        """

        filename = csv_name if csv_name.startswith(self.work_dir) else os.path.join(self._work_dir,csv_name)
        if not load_crops and os.path.exists(os.path.join(self._work_dir, FOLDER_FULL_IMAGE_RUN, csv_name)):
             filename = os.path.join(self._work_dir, FOLDER_FULL_IMAGE_RUN, csv_name)

        if not os.path.exists(filename):
            if force_error:
                assert os.path.exists(filename), f"Failed to read {filename}"
            else:
                return None
        ret =  pd.read_csv(filename)
        if ret.empty and force_error:
            assert not ret.empty, f"Found empty file {filename}"
        return ret

    def _create_img_mapping(self):
        """
        Map a df to fastdup-img-id according to fname column (inner merge)
        The function can also create generate missing and filtered csv in the work-dir
        """

        # get mapping df from fastdup
        df_mapping = self._fetch_df(FD.MAPPING_CSV)
        assert df_mapping is not None and not df_mapping.empty, f"Failed to find {FD.MAPPING_CSV} in work_dir"
        df_mapping = df_mapping.reset_index()
        if FD.MAP_INST_ID not in df_mapping.columns:
            df_mapping[FD.MAP_INST_ID] = df_mapping.index
        df_mapping = df_mapping[[FD.ANNOT_FILENAME, FD.ANNOT_FD_ID]]
        df_bad_files = self._fetch_df(FD.BAD_CSV)

        if self._df_annot is None:
            if df_bad_files is not None and not df_bad_files.empty:
                df_bad_files = df_bad_files[['index', 'filename']]
                self._df_annot = pd.concat([df_mapping, df_bad_files])
            else:
                self._df_annot = df_mapping

        else:
            assert "index" not in self._df_annot, f"Found index in self._df_annot {self._df_annot.head()}"
            if df_bad_files is not None and not df_bad_files.empty:
                total = pd.concat([df_mapping, df_bad_files])
            else:
                total = df_mapping
            assert len(total), "Failed to find annotations"
            df_annot = pd.merge(self._df_annot, total, on=FD.ANNOT_FILENAME, how='left')
            assert len(
                df_annot), f"Failed to merge to find indexes. First dataframe was \n{self._df_annot.head()}\nSecond dataframe:\n{total.head()}" + self.get_status_string()

            if len(df_annot) > 1 and df_annot['index'].isnull().sum() == len(df_annot):
                assert False, f"Failed to merge to find indexes. First dataframe was \n{self._df_annot.head()}\n Second dataframe:\n{total.head()}\n please check your input_dir" + self.get_status_string()

            self._df_annot = df_annot

    # self._df_annot[FD.ANNOT_FD_ID] = self._df_annot[FD.ANNOT_FD_ID].astype(pd.UInt32Dtype())


    def _expand_annot_df(self):
        """
        Expand annotation df to include all relevant information:
            1. collect subsets for error types analysis (MISSING_ANNOTATION, MISSING_IMAGE, FD-ERROR, VALID)
            2. set NaN fastdup-id for instances that are not in df_mapping
            3. add not-in-annot instances (but in fastdup mapping) to df_annot
            4. mark default error type as VALID
            5. mark error type of not-in-annot as  MISSING_ANNOTAION
            6. mark error type of not-in-mapping as MISSING_IMAGE
            7. mark error types from bad-files
            8. convert fastdup-id to UInt32Dtype and label to categorical (for memory efficiency)
            9. add is-valid column and set to True for VALID instances
        """
        assert self._df_annot is not None, "Fastdup got to a bad state, no image info found please file a github issue"

        # add crops to annotation df (available in case bounding box is not available=='face'/'yolov5s')
        if (self._run_mode == FD.MODE_CROP or self._run_mode == FD.MODE_ROTATED_BBOX or self._dtype == FD.BBOX) and self._bbox != "ocr":
            # complication: crops are not sorted according their input order due to multithreading in the c side, need to rearrange
            # according to the input order
            df_crops_annot = self._fetch_df(FD.CROPS_CSV, False)
            assert df_crops_annot is not None and not df_crops_annot.empty, f"Failed to find {self.work_dir}/{FD.CROPS_CSV} in work_dir"
            assert 'index' in df_crops_annot.columns, f"Failed to find 'index' column in {self.work_dir}/{FD.CROPS_CSV}"
            assert "crop_filename" in df_crops_annot.columns, "Failed to fidn crop_filename in columns"
            if "filename" in self._df_annot:
                del self._df_annot["filename"]
            df_annot = pd.merge(self._df_annot, df_crops_annot[["crop_filename", "filename", "index"]],
                                on=FD.ANNOT_FD_ID
                                , how='left')
            assert len(
                df_annot), f"Empty merge result when reading crop file from {df_crops_annot} first was {self._df_annot.head()} second was {df_crops_annot.head()}" + self.get_status_string()
            self._df_annot = df_annot
        elif self._bbox == "ocr":
            df_crops_annot = self._fetch_df(FD.CROPS_CSV, False)
            assert df_crops_annot is not None and not df_crops_annot.empty, f"Failed to find {self.work_dir}/{FD.CROPS_CSV} in work_dir"
            assert 'index' in df_crops_annot.columns, f"Failed to find 'index' column in {self.work_dir}/{FD.CROPS_CSV}"
            assert "crop_filename" in df_crops_annot.columns, "Failed to fidn crop_filename in columns"
            if "filename" in self._df_annot:
                del self._df_annot["filename"]
            df_annot = pd.merge(self._df_annot, df_crops_annot[["crop_filename", "filename", "index", "label"]],
                                on=FD.ANNOT_FD_ID, how='left')
            assert len(df_annot), f"Empty merge result when reading crop file from {df_crops_annot} first was {self._df_annot.head()} second was {df_crops_annot.head()}" + self.get_status_string()
            self._df_annot = df_annot

        if self._run_mode == FD.MODE_ROTATED_BBOX:
            self._df_annot = self._df_annot[list(set(self._df_annot.columns).difference({
                FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_H, FD.ANNOT_BBOX_W,
            }))]

        # 1. collect subsets for error types analysis (MISSING_ANNOTATION, MISSING_IMAGE, FD-ERROR, VALID)
        df_bad_files = self._fetch_df(FD.BAD_CSV, False)
        df_mapping = self._fetch_df(FD.MAPPING_CSV, False)
        assert df_mapping is not None and not df_mapping.empty,  f"Failed to find {FD.MAPPING_CSV} in work_dir"
        assert 'index' in df_mapping.columns and 'filename' in df_mapping.columns, f"Failed to find index and filename columns is df_mapping. available cols are: {df_mapping.head()}"

        if FD.MAP_INST_ID not in df_mapping.columns:
            df_mapping[FD.MAP_INST_ID] = df_mapping.index
        df_mapping = df_mapping.reset_index()[[FD.ANNOT_FILENAME, FD.ANNOT_FD_ID]]

        if self._dtype == FD.BBOX and self._bbox == "xywh_bbox":
            assert 'crop_filename' not in df_mapping.columns
            df_mapping = df_mapping.rename(columns={'index': 'new_index', 'filename': 'crop_filename'})
            # del self._df_annot['filename']
            annot = self._df_annot.merge(df_mapping, left_on='crop_filename', right_on='crop_filename', how='left')
            del annot['index']
            annot = annot.rename(columns={'new_index': 'index'})
            annot[FD.ANNOT_VALID] = annot['index'].apply(lambda x: False if pd.isnull(x) else True)
            self._df_annot = annot
        else:
            if df_bad_files is None:
                seen_by_fastdup = set(df_mapping[FD.ANNOT_FD_ID])
            else:
                seen_by_fastdup = set(df_bad_files[FD.BAD_FD_ID]).union(set(df_mapping[FD.ANNOT_FD_ID]))
            df_mapping_not_in_annot = df_mapping[~df_mapping[FD.ANNOT_FD_ID].isin(self._df_annot[FD.ANNOT_FD_ID])]
            df_annot_in_subset = self._df_annot[self._df_annot[FD.ANNOT_FD_ID].isin(seen_by_fastdup)]
            df_annot_not_in_subset = self._df_annot[~self._df_annot[FD.ANNOT_FD_ID].isin(seen_by_fastdup)]

            # 2. set NaN fastdup-id for instances that are not in df_mapping
            self._df_annot[FD.ANNOT_FD_ID] = self._df_annot[FD.ANNOT_FD_ID].mask(~self._df_annot[FD.ANNOT_FD_ID].
                                                                                 isin(df_mapping[FD.ANNOT_FD_ID]))

            # 3. add not-in-annot instances (but in fastdup mapping) to df_annot
            merged =  pd.merge(df_annot_in_subset, df_mapping,
                         on=FD.ANNOT_FD_ID, how='outer', suffixes=('', '_map'))
            assert len(merged), f"Failed to merge {df_annot_in_subset.head()} {df_mapping.head()}" + self.get_status_string()
            self._df_annot = pd.concat([merged,
                df_annot_not_in_subset])
            self._df_annot[FD.ANNOT_FILENAME] = self._df_annot[FD.ANNOT_FILENAME].fillna(self._df_annot[FD.ANNOT_FILENAME + '_map'])
            self._df_annot.drop(FD.ANNOT_FILENAME + '_map', axis=1, inplace=True)

            # 4. mark default error type as VALID
            self._df_annot[FD.ANNOT_ERROR] = 'VALID'

            # 5. mark error type of not-in-annot as  MISSING_ANNOTAION
            self._df_annot[FD.ANNOT_ERROR] = self._df_annot[FD.ANNOT_ERROR].mask(
                self._df_annot[FD.ANNOT_FD_ID].isin(df_mapping_not_in_annot[FD.ANNOT_FD_ID]), 'MISSING_ANNOTATION')

            # 6. mark error type of not-in-mapping as MISSING_IMAGE
            self._df_annot[FD.ANNOT_ERROR] = self._df_annot[FD.ANNOT_ERROR].mask(
                ~self._df_annot[FD.ANNOT_FD_ID].isin(seen_by_fastdup), FD.ERROR_MISSING_IMAGE)
            self._df_annot[FD.ANNOT_ERROR] = self._df_annot.apply(lambda x: FD.ERROR_MISSING_IMAGE if ('crop_filename' in x
                            and pd.isnull(x['crop_filename'])) else x[FD.ANNOT_ERROR], axis=1)

            # 7. mark error types from bad-files
            if df_bad_files is not None and not df_bad_files.empty:
                fd_errors = df_bad_files.set_index('index')[FD.ANNOT_ERROR]
                self._df_annot[FD.ANNOT_ERROR] = self._df_annot.apply(
                    lambda row: fd_errors.get(row[FD.ANNOT_FD_ID], row[FD.ANNOT_ERROR]), axis=1)

            # 8. convert fastdup-id to UInt32Dtype and label to categorical (for memory efficiency)
            #self._df_annot[FD.ANNOT_FD_ID] = self._df_annot[FD.ANNOT_FD_ID].astype(dtype=pd.UInt32Dtype())
            if self._has_label:
                self._df_annot[FD.ANNOT_LABEL] = self._df_annot[FD.ANNOT_LABEL].astype(dtype="category")

            # 9. add is-valid column and set to True for VALID instances
            self._df_annot[FD.ANNOT_VALID] = self._df_annot[FD.ANNOT_ERROR] == 'VALID'

    def _verify_bbox(self):
        invalid_valid_col = (
                (self._df_annot.bbox_x < 0) | (self._df_annot.bbox_y < 0) |
                (self._df_annot.bbox_x + self._df_annot.bbox_w > self._df_annot.img_w) |
                (self._df_annot.bbox_y + self._df_annot.bbox_h > self._df_annot.img_h) |
                (self._df_annot.bbox_w < 10) | (self._df_annot.bbox_h < 10)
        )
        if FD.ANNOT_VALID in self._df_annot.columns:
            self._df_annot[FD.ANNOT_VALID] = self._df_annot[FD.ANNOT_VALID] & ~invalid_valid_col
        else:
            self._df_annot[FD.ANNOT_VALID] = ~invalid_valid_col
        return self._df_annot

    def _get_filename_prefix(self, input_dir: Path):
        """
        get fastdup filename-prefix (relative path to input dir)
        :param input_dir: input dir
        :return: relative path to input dir
        """
        is_s3_input = False if input_dir is None else (str(input_dir).startswith('s3://') or str(input_dir).startswith('minio://'))
        if is_s3_input:
            return os.path.join(self._work_dir ,"tmp", str(input_dir).split("/", 3)[-1])
        else:
            return input_dir

    def _infer_dtype(self, requested_dtype: str, df_annot: pd.DataFrame = None) -> str:
        """
        Infer run-type - images/bboxes from.
        :param requested_dtype: users requested data type
        :param df_annot: data annotation
        :return: run-type
        """
        if df_annot is not None:
            assert isinstance(df_annot, pd.DataFrame) and len(df_annot), f"Invalid df_annot {df_annot}"
        if self._fastdup_applied:
            return json.load(open(self._work_dir / FD.CONFIG_JSON, 'rt')).get('data_type', FD.IMG)

        bbox_cols = {FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y,
                     FD.ANNOT_BBOX_H, FD.ANNOT_BBOX_W}
        rotated_bbox_cols = {FD.ANNOT_ROT_BBOX_X1, FD.ANNOT_ROT_BBOX_Y1,
                             FD.ANNOT_ROT_BBOX_X2, FD.ANNOT_ROT_BBOX_Y2,
                             FD.ANNOT_ROT_BBOX_X3, FD.ANNOT_ROT_BBOX_Y3,
                             FD.ANNOT_ROT_BBOX_X4, FD.ANNOT_ROT_BBOX_Y4}
                             
        bbox_cols_available = df_annot is not None and \
                              (bbox_cols.issubset(df_annot.columns) or rotated_bbox_cols.issubset(df_annot.columns))

        if requested_dtype == FD.IMG and not isinstance(self._input_dir, list):
            return FD.BBOX if bbox_cols_available else FD.IMG

        assert requested_dtype != FD.BBOX or bbox_cols_available, f"missing df-annotations or missing bounding box columns in annotation df: {bbox_cols} or {rotated_bbox_cols}"
        return requested_dtype

    def _verify_fastdup_run_args(self, input_dir, work_dir, df_annot, subset, data_type, embeddings):
        """
        Verify constructor arguments and raise exception if invalid.
        This method is called by the constructor, and checks that the requested procedure is supported.
        the following checks are performed:
            1. arguments have valid values
                a. input_dir is None and is a string/pathlib.Path or a list of strings/pathlib.Path
                b. work_dir is not None and is a string/pathlib.Path
                c. data_type is one of FD.IMG, FD.BBOX
                d. df_annot is None or a pandas.DataFrame
            2. check that if fastdup was already applied, df_annot is, subset and data_type are None
            3. verify input contains is supported - in case allowing more option in the future this is the place to add.
                a. image data:
                    i. input dir is list of strings/pathlib.Path, without df_annot, without subset
                    ii. input dir is string/pathlib.Path, w/wo df_annot, w/wo subset
                b. bbox data:
                    i. input dir is string/pathlib.Path, with df_annot, w/wo subset
            4. verify df_annot contains required columns
                a. image data: FD.ANNOT_FILENAME
                b. bbox data: FD.ANNOT_FILENAME, FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H
            5. verify input_dir/s are directories

        :param input_dir: input directory or list of input directories
        :param work_dir: target directory for saving results
        :param df_annot: annotation dataframe
        :param subset: subset of data to process
        :param data_type: data type to process (img, bbox)
        :param is_applied: check if fastdup was already applied
        :return:
        """

        # do not allow change annot/subset/data_type if fastdup was already applied
        assert not self._fastdup_applied, \
            'there is already an active fastup run on the working dir, change work_dir or run with overwrite=True'

        # verify arguments contains valid values
        assert isinstance(embeddings, np.ndarray) or embeddings is None, \
            'pre_calc_features must be a numpy array'
        assert embeddings is not None or (isinstance(input_dir, list) or isinstance(input_dir, str) or
                                                 isinstance(input_dir, Path)), \
            'input_dir must be provided and be a string/pathlib.Path or a list of strings/pathlib.Path'
        assert work_dir is not None and (isinstance(work_dir, str) or isinstance(work_dir, Path)), \
            'work_dir must be provided and be a string or pathlib.Path'
        assert data_type in [FD.IMG, FD.BBOX], \
            f'invalid data_type, found: {data_type} supported: img, bbox'
        assert df_annot is None or isinstance(df_annot, pd.DataFrame), 'df_annot must be a pandas DataFrame'

        if embeddings is not None and df_annot is not None:
            assert embeddings.shape[0] == df_annot.shape[0], \
                f'pre_calc_features and df_annot must have the same number of rows, got {embeddings.shape[0]} vs. {df_annot.shape[0]}, {df_annot.head()}'

        # verify arguments combinations
        assert any(
            # list of input dirs, without annotations, without subset - image data-type only
            [isinstance(input_dir, list) and df_annot is None and subset is None and data_type in [FD.IMG],
             # single input dir, without annotations, with/without subset - image data-type only
             (isinstance(input_dir, str) or isinstance(input_dir, Path)) and df_annot is None and (data_type in [FD.IMG] or self._run_mode == FD.MODE_CROP ),
             # single input dir, with annotations, with/without subset - image/bbox
             (isinstance(input_dir, str) or isinstance(input_dir, Path)) and df_annot is not None,
             # no input dir, with/without annotations, with/without subset - with-pre-calc-features
             input_dir is None and embeddings is not None]
        ),\
            'invalid input (input-dir, annotation, subset, data-type) ' \
            '\nsupported options: ' \
            '\n1. single input dir, with annotations, with/without subset' \
            '\n2. single input dir, without annotations, with/without subset - image data-type only' \
            '\n3. list of input dirs, without annotations, without subset - image data-type only' \
            '\n4. given embeddings, in this case input_dir should be done.'

        # verify df_annot columns
        if df_annot is not None:
            assert FD.ANNOT_FILENAME in df_annot, \
                'when running on images df_annot must contain a column named "filename" ' \
                'if you wish to run on bounding boxes, make sure to provide the relevant columns'
            if data_type == FD.IMG:
                assert df_annot[FD.ANNOT_FILENAME].nunique() == len(df_annot[FD.ANNOT_FILENAME]), \
                    'df_annot must contain unique filenames, found repeating filenames'
            elif data_type == FD.BBOX:
                bbox_cols = {FD.ANNOT_FILENAME, FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H}
                rotated_bbox_cols = {FD.ANNOT_FILENAME, FD.ANNOT_ROT_BBOX_X1, FD.ANNOT_ROT_BBOX_Y1, FD.ANNOT_ROT_BBOX_X2,
                                     FD.ANNOT_ROT_BBOX_Y2, FD.ANNOT_ROT_BBOX_X3, FD.ANNOT_ROT_BBOX_Y3,
                                     FD.ANNOT_ROT_BBOX_X4, FD.ANNOT_ROT_BBOX_Y4}
                assert bbox_cols.issubset(df_annot.columns) or rotated_bbox_cols.issubset(df_annot.columns), \
                    f'df_annot must contain columns named, {FD.ANNOT_FILENAME}, {FD.ANNOT_BBOX_X}, ' \
                    f'{FD.ANNOT_BBOX_Y}, {FD.ANNOT_BBOX_H}, {FD.ANNOT_BBOX_W} or {FD.ANNOT_ROT_BBOX_X1}, {FD.ANNOT_ROT_BBOX_Y1}, {FD.ANNOT_ROT_BBOX_X2}. ' \
                    f'{FD.ANNOT_ROT_BBOX_Y2}, {FD.ANNOT_ROT_BBOX_X3}, {FD.ANNOT_ROT_BBOX_Y3}, {FD.ANNOT_ROT_BBOX_X4}, {FD.ANNOT_ROT_BBOX_Y4}'

                len_df_annot = len(df_annot)

                if bbox_cols.issubset(df_annot.columns):
                    df_annot = df_annot.drop_duplicates(subset=bbox_cols)

                else:
                    df_annot = df_annot.drop_duplicates(subset=rotated_bbox_cols)

                if len(df_annot) < len_df_annot:
                    self._logger.warn(f"Warning: removed {len_df_annot - len(df_annot)} duplicate bounding boxes")
            else:
                assert False, f"Wrong data type {data_type}"


    def enrich(self, task, model, input_df, input_col, num_rows=None, device=None):

        self._fastdup_applied = True # Hack: Allow users to run enrichment without first running fastdup
        if num_rows:
            df = input_df.head(num_rows)
        else: df = input_df

        if task == "zero-shot-classification":
            if model == "recognize-anything-model":
                from fastdup.models_ram import RecognizeAnythingModel

                enrichment_model = RecognizeAnythingModel(device=device)
                df["ram_tags"] = df[input_col].apply(enrichment_model.run_inference)

            elif model == "tag2text":
                from fastdup.models_tag2text import Tag2TextModel

                enrichment_model = Tag2TextModel(device=device)
                df["tag2text_tags"] = df[input_col].apply(
                    lambda x: enrichment_model.run_inference(x)[0].replace(" | ", " . ")
                )
                df["tag2text_caption"] = df[input_col].apply(
                    lambda x: enrichment_model.run_inference(x)[2]
                )

        elif task == "zero-shot-detection":
            if model == "grounding-dino":
                from fastdup.models_grounding_dino import GroundingDINO

                enrichment_model = GroundingDINO(device=device)

                def compute_bbox(row):
                    results = enrichment_model.run_inference(
                        row["filename"], text_prompt=row[input_col]
                    )
                    return results["boxes"], results["scores"], results["labels"]

                df["grounding_dino_bboxes"], df["grounding_dino_scores"], df["grounding_dino_labels"] = zip(
                    *df.apply(compute_bbox, axis=1)
                )

        elif task == "zero-shot-segmentation":
            if model == "segment-anything":
                import torch
                from fastdup.models_sam import SegmentAnythingModel

                enrichment_model = SegmentAnythingModel(device=device)

                try:
                    tensor_list = [
                        torch.tensor(bbox, dtype=torch.float32)
                        for bbox in df[input_col]
                    ]
                except Exception as e:
                    raise KeyError(
                        f"Column `{input_col}` does not exist."
                    )

                def preprocess_and_run(filename, bbox):
                    try:
                        result = enrichment_model.run_inference(filename, bboxes=bbox)

                        if isinstance(result, torch.Tensor) and result.device.type == "cuda":
                            result = result.cpu()

                        return result
                    except Exception as e:
                        self._logger.error(f'{e}')
                        
                df["sam_masks"] = [
                    preprocess_and_run(filename, bbox)
                    for filename, bbox in zip(df["filename"], tensor_list)
                ]

        try:
            enrichment_model.unload_model()
        except Exception as e:
            raise ValueError("Please select a valid enrichment model")
        return df


    def caption(self, model_name='automatic', device = 'cpu', batch_size: int = 8, subset: list = None, vqa_prompt: str = None, kwargs=None) -> pd.DataFrame:
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')

        if subset:
            df = pd.DataFrame(subset , columns =['filename'])
        else:
            df = self.annotations(valid_only=True)
        assert len(df), "No images found."

        if model_name in FD.CAPTION_MODEL_NAMES:
            from fastdup.captions import generate_labels
            df['caption'] = generate_labels(df['filename'].tolist(), model_name, device, batch_size)
        elif model_name == FD.VQA_MODEL1_NAME:
            from fastdup.captions import generate_vqa_labels
            df['caption'] = generate_vqa_labels(df['filename'], vqa_prompt, kwargs)
        elif model_name == FD.AGE_LABEL1_NAME:
            from fastdup.captions import generate_age_labels
            df['caption'] = generate_age_labels(df['filename'], kwargs)
        else:
            assert False, f"Unknown model name provided {model_name}. Available models for caption generation are 'vitgpt2', 'blip2', and 'blip'.\n Available models for VQA are 'vqa' and 'age'."

        return df


def is_fastdup_dir(work_dir):
    return os.path.exists(Path(work_dir) / FD.MAPPING_CSV) and \
           os.path.exists(Path(work_dir) / FD.FILENAME_NNF_INDEX) and \
           os.path.exists(Path(work_dir) / FD.ANNOT_PKL)


def clean_work_dir(work_dir):
    if os.path.isfile(str(Path(work_dir) / FD.MAPPING_CSV)):
        os.remove(str(Path(work_dir) / FD.MAPPING_CSV))
    if os.path.isfile(str(Path(work_dir) / FD.FILENAME_NNF_INDEX)):
        os.remove(str(Path(work_dir) / FD.FILENAME_NNF_INDEX))
    if os.path.isfile(str(Path(work_dir) / FD.ANNOT_PKL)):
        os.remove(str(Path(work_dir) / FD.ANNOT_PKL))
    if os.path.isfile(str(Path(work_dir) / FD.FEATURES_DATA)):
        os.remove(str(Path(work_dir) / FD.FEATURES_DATA))
    if os.path.isfile(str(Path(work_dir) / FD.FILENAME_ERROR_MSG)):
        os.remove(str(Path(work_dir) / FD.FILENAME_ERROR_MSG))


def set_fastdup_kwargs(input_kwargs: dict) -> dict:
    """
    override default arguments in fastdup args with users-input
    :param input_kwargs:iunput kwargs to init function
    :return: updated dict
    """
    fastdup_params = {
        'input_dir', 'work_dir', 'test_dir', 'compute', 'verbose', 'num_threads', 'num_images', 'distance',
        'threshold', 'lower_threshold', 'model_path', 'license', 'version', 'nearest_neighbors_k', 'd', 'run_mode',
        'nn_provider', 'min_offset', 'max_offset', 'nnf_mode', 'nnf_param', 'bounding_box', 'batch_size', 'resume',
        'high_accuracy'
    }
    turi_params = {
        'nnmodel': {'map': {'brute_force': 0, 'ball_tree': 1, 'lsh': 2}, 'default': 'brute_force'},
        'ccthreshold': {'map': None, 'default': 0.96},
        'run_cc': {'map': {True: 1, False: 0}, 'default': True},
        'run_sentry': {'map': {True: 1, False: 0}, 'default': True},
        'delete_tar': {'map': {True: 1, False: 0}, 'default': False},
        'delete_img': {'map': {True: 1, False: 0}, 'default': False},
        'tar_only': {'map': {True: 1, False: 0}, 'default': False},
        'run_stats': {'map': {True: 1, False: 0}, 'default': True},
        'run_stats_only': {'map': {True: 1, False: 0}, 'default': False},
        'run_advanced_stats': {'map': {True: 1, False: 0}, 'default': False},
        'sync_s3_to_local': {'map': {True: 1, False: 0}, 'default': False},
        'store_int': {'map': {True: 1, False: 0}, 'default': True},
        'shorten_filenames': {'map': {True: 1, False: 0}, 'default': False},
        'save_crops': {'map': {True: 1, False: 0}, 'default': False},
        'augmentation_horiz': {'map': None, 'default': 0.2},
        'augmentation_vert': {'map': None, 'default': 0.2},
        'augmentation_additive_margin': {'map': None, 'default': 0},
        'num_onnx_inter_threads': {'map': None, 'default': 0},
        'num_onnx_intra_threads': {'map': None, 'default': 0},
        'is_clip14_model':  {'map': {True: 1, False: 0}, 'default': False},
        #'run_labels': {'map': {True: 1, False: 0}, 'default': True},
        #'run_read_filenames': {'map': {True: 1, False: 0}, 'default': True},
        #'min_file_size': {'map': None, 'default': 0},
        #'read_features_parallel': {'map': None, 'default': 0},
        #'test_correction_offset': {'map': None, 'default': 0},
        #'max_augmentations': {'map': None, 'default': 1},
        #'augmentation_type': {'map': None, 'default': 0},
        #'is_ultraface_model': {'map': {True: 1, False: 0}, 'default': False},
        #'is_yolo_model': {'map': {True: 1, False: 0}, 'default': False},
        'min_input_image_height': {'map': None, 'default': 10},
        'min_input_image_width': {'map': None, 'default': 10},
        'save_thumbnails': {'map': {True: 1, False: 0}, 'default': False},
        'find_regex': {'map': None, 'default': ""},
        'no_sort': {'map': {True: 1, False: 0}, 'default': False},
        'quiet': {'map': {True: 1, False: 0}, 'default': False},
        'fastdup_ocr_lang': {'map': None, 'default': "en"},
        'fastdup_ocr_no_crop': {'map': {True: 1, False: 0}, 'default': False},
        'global_log_error_level': {'map': None, 'default': 3}

    }

    for key, value in input_kwargs.items():
        if key not in fastdup_params and key not in turi_params:
            raise ValueError(f'invalid argument {key}, allowed fastdup params are {fastdup_params}, allowed turi_param values are {turi_params}')

    turi_kwargs = []
    for arg_name, param in turi_params.items():
        map_dict = param['map']
        map_func = lambda x: x if map_dict is None else map_dict[x]
        value = input_kwargs.get(arg_name, param['default'])
        turi_kwargs.append(f'{arg_name}={map_func(value)}')

    fastdup_kwargs = {key: value for key, value in input_kwargs.items() if key in fastdup_params}
    fastdup_kwargs['turi_param'] = ','.join(turi_kwargs)
    return fastdup_kwargs


def get_input_dir(_input_dir):
    def pathlib_or_s3_cast(c):
        c = str(c)
        return c if (c.startswith('s3://') or c.startswith("smb://") or c.startswith("minio://")) else Path(c)
    return [pathlib_or_s3_cast(f) for f in _input_dir] if isinstance(_input_dir, list)\
        else pathlib_or_s3_cast(_input_dir)
