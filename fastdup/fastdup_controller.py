import json
import os
import tempfile
import warnings
from typing import List, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import fastdup
from pandas.errors import EmptyDataError
import shutil
import fastdup.fastup_constants as FD
#import boto3
from fastdup.sentry import v1_sentry_handler
import re


class FastdupController:
    @v1_sentry_handler
    def __init__(self, work_dir: Union[str, Path], input_dir: Union[str, Path] = None):
        """
        This class serves as a proxy for fastdup basic usage,
        the class wraps fastdup-run call provides quick access to
        fastdup files such as: similarity,  csv outlier csv, etc...

        Moreover, the class provides several extra features:
            - Ability to run connected component analysis on splits without calling fastdup run again
            - Ability to add annotation file and quickly merge it to any of fastdup inputs
        Currently the class support running fastdup on images and object
        :param work_dir: target output dir or existing output dir
        :param input_dir: (Optional) path to data dir
        """
        # check if fastdup was already applied
        self._fastdup_applied = is_fastdup_dir(work_dir)
        self._work_dir = Path(work_dir)
        self._input_dir = input_dir if input_dir is None else get_input_dir(input_dir)

        # set default arguments
        self._df_annot = None
        self._run_mode = FD.MODE_DEFAULT
        self._embeddings_dim = 576
        self._max_fd_id = 0
        self._config = None
        self._dtype = None

        if self._fastdup_applied:
            assert not isinstance(input_dir, list), \
                'input_dir must be a single path when fastdup was already applied'
            # load fastdup state
            self._get_fastdup_state()

        self._has_split = self._df_annot is not None and FD.ANNOT_SPLIT in self._df_annot
        self._has_label = self._df_annot is not None and FD.ANNOT_LABEL in self._df_annot

    def _get_fastdup_state(self):
        self._config = json.load(open(self._work_dir / FD.CONFIG_JSON))
        self._input_dir = Path(self._config.get('input_dir', '.')) if self._input_dir is None else self._input_dir
        self._df_annot = pd.read_pickle(self._work_dir / FD.ANNOT_PKL, compression='gzip')
        self._dtype = self._infer_dtype(requested_dtype='infer')
        self._filename_prefix = self._config.get('filename_prefix', self._get_filename_prefix(self._input_dir))
        self._embeddings_dim = self._config.get('embeddings_dim', 576)
        self._max_fd_id = self._config.get('max_fd_id', 0)
        self._run_mode = self._config.get('run_mode', FD.MODE_DEFAULT)

    def _init_run(self, input_dir: Union[str, Path] = None, df_annot: pd.DataFrame = None,
                  subset: list = None, embeddings=None, data_type: str = 'infer', overwrite: bool = False,
                  fastdup_args: dict = None):
        """
        Initialize fastdup run arguments, unlike the constructor that tries to load an existing state from work_dir,
        this method allows to change the input_dir, df_annot, subset, embeddings, data_type arguments, checks they are
        valid, before executing fastdup run
        :param input_dir: folder containing the data
        :param df_annot: df with annotations
        :param subset: subset of files to run fastdup on
        :param embeddings: pre-calculated embeddings for images/bounding boxes
        :param data_type: image or bbox
        :param overwrite: overwrite existing fastdup state (delete work_dir)
        """
        if overwrite:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._fastdup_applied = False

        # set run mode & data type
        self._dtype = self._infer_dtype(requested_dtype=data_type, df_annot=df_annot)
        self._run_mode = FD.MODE_DEFAULT

        if 'bounding_box' in fastdup_args:
            self._run_mode = FD.MODE_CROP
            self._dtype = FD.BBOX

        if embeddings is not None:
            self._run_mode = FD.MODE_EMBEDDING

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
        if instance_id not in self._df_annot.index or instance_id > self._max_fd_id:
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

    @v1_sentry_handler
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

    @v1_sentry_handler
    def similarity(self, data: bool = True, split: str = None, include_unannotated=False) -> pd.DataFrame:
        """
        Get fastdup similarity file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        df = self._fetch_df(csv_name=FD.SIMILARITY_CSV)
        if df is None or df.empty:
            print('No similarity file found')
            return None
        return self._add_annot_and_split(df, data, merge_on=[FD.SIM_SRC_IMG, FD.SIM_DST_IMG],
                                         split=split, unannotated=include_unannotated)

    @v1_sentry_handler
    def outliers(self, data: bool = True, split: str = None, include_unannotated=False) -> pd.DataFrame:
        """
        Get fastdup outlier file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        # get df and rename columns (from, to, score) -> (outlier, nearest_neighbor, score)
        df = self._fetch_df(csv_name=FD.OUTLIERS_CSV)
        if df is None or df.empty:
            print('No outliers found')
            return None
        df = df.rename({
            FD.SIM_SRC_IMG: FD.OUT_ID, FD.SIM_DST_IMG: FD.OUT_NEAREST_NEIGHBOR
        }, axis=1)
        df = self._add_annot_and_split(df, data, merge_on=[FD.OUT_ID, FD.OUT_NEAREST_NEIGHBOR], split=split,
                                       unannotated=include_unannotated, suffix=True)
        df = df.sort_values(FD.OUT_SCORE).groupby(FD.OUT_ID).head(1).reset_index()
        return df

    @v1_sentry_handler
    def invalid_instances(self):
        """
        Get fastdup invalid file
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        return self._df_annot.query(f'not {FD.ANNOT_VALID}').reset_index(drop=True)

    @v1_sentry_handler
    def img_stats(self, data: bool = True, split: bool = None, include_unannotated=False) -> pd.DataFrame:
        """
        Get fastdup stats file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        df = self._fetch_df(csv_name=FD.STATS_CSV)
        if df is None or df.empty:
            print(f'No stats file found in {self._work_dir}')
            return None
        df = df.rename({'width': FD.ANNOT_IMG_W, 'height': FD.ANNOT_IMG_H, FD.STATS_INST_ID: FD.ANNOT_FD_ID}, axis=1)
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

    @v1_sentry_handler
    def connected_components(self, data: bool = True, split: str = None, include_unannotated=False) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get fastdup connected components file
        :param data: add annotation
        :param split: filter by split
        :param include_unannotated: include instances that are not represented in the annotations
        :return: requested dataframe
        """
        if not self._fastdup_applied:
            raise RuntimeError('Fastdup was not applied yet, call run() first')
        # get connected components and add annotation
        df_cc = self._fetch_df(csv_name=FD.CC_CSV)
        if df_cc is None or df_cc.empty:
            print('No connected components found')
            return None, None
        df_cc = df_cc.rename({FD.CC_INST_ID: FD.ANNOT_FD_ID}, axis=1)
        df_cc = self._add_annot_and_split(df_cc, data, merge_on=[FD.ANNOT_FD_ID], split=split, suffix=False,
                                          unannotated=include_unannotated)
        df_info = self._fetch_df(csv_name=FD.CC_INFO_CSV)
        return df_cc, df_info

    @v1_sentry_handler
    def run(self, input_dir: Union[str, Path] = None, annotations: pd.DataFrame = None, subset: list = None,
            embeddings=None, data_type: str = 'infer', overwrite: bool = False,
            print_summary: bool = True, **fastdup_kwargs):
        """
        This function
            1. calculate subset of images to analyze
            2. run fastdup
            3. map images/bboxes to fastdup index (on bbox this is done in self._set_fastdup_input)
            4. expand annotation csv to include files that are not in annotation but is in subset
            5. create a version of annotation that is grouped by image
        :param input_dir: input directory containing images
        :param annotations: (Optional) annotations file, the expected column convention is:
             - img_filename: input_dir-relative filenames
             - img_h, img_w (Optional): image height and width
             - bbox_x, bbox_y, bbox_h, bbox_w (Optional): bounding box arguments
             - split (Optional): data split, e.g. train, test, etc ...
        :param subset: (Optional) subset of images to analyze
        :param embeddings: (Optional) pre-calculated embeddings
        :param data_type: (Optional) data type, one of 'infer', 'image', 'bbox'
        :param overwrite: (Optional) overwrite existing files
        :param print_summary: Print summary report of fastdup run results
        :param fastdup_kwargs: (Optional) fastdup run arguments, see fastdup.run() documentation
        :return:
        """
        if self._fastdup_applied and not overwrite:
            warnings.warn('Fastdup was already applied, use overwrite=True to re-run')
            return
        self._init_run(input_dir, annotations, subset, embeddings, data_type, overwrite, fastdup_kwargs)

        # get user's fastdup kwargs or use default
        fastdup_kwargs = {} if fastdup_kwargs is None else fastdup_kwargs
        if self._pre_calc_features is not None:
            fastdup_kwargs['run_mode'] = 2
            fastdup_kwargs['d'] = self._embeddings_dim
        fastdup_kwargs = set_fastdup_kwargs(fastdup_kwargs)

        os.makedirs(self._work_dir, exist_ok=True)

        # run fastdup - create embeddings
        fastdup.run(self._set_fastdup_input(), work_dir=str(self._work_dir), **fastdup_kwargs)
        fastdup_convert_to_relpath(self._work_dir, self._filename_prefix)

        # post process - map fastdup-id to image (for bbox this is done in self._set_fastdup_input)
        if self._dtype == FD.IMG or self._run_mode == FD.MODE_CROP:
            self._create_img_mapping()

        # expand annotation csv to include files that are not in annotation but is in subset
        self._expand_annot_df()
        self._index_annot_df()

        self._save_artifacts(fastdup_kwargs)
        self._fastdup_applied = True
        if print_summary:
            self.summary()

    @v1_sentry_handler
    def summary(self, verbose=True, blur_threshold: float = 150.0, brightness_threshold: float = 253.0,
                darkness_threshold: float = 4.0) -> List[str]:
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

        summary_stats.append(f"Dataset contains {total_image_count} images")

        invalid_data_df = self.invalid_instances()
        invalid_image_count = len(invalid_data_df)
        valid_image_count = total_image_count - invalid_image_count
        invalid_stats = f"Valid images are {pct(valid_image_count):.2f}% ({valid_image_count:,d}) of the data, "\
                        f"invalid are {pct(invalid_image_count):.2f}% ({invalid_image_count:,d}) of the data"
        summary_stats.append(invalid_stats)
        if invalid_image_count:
            summary_stats.append("For a detailed analysis, use `.invalids()`.\n")
        # Images belonging to clusters
        # X percent of images belong to Y clusters, the largest of ZZ images.
        try:
            cc_df, _ = self.connected_components()
            number_of_ccs = cc_df[cc_df['count'] != 0]['count'].nunique()
            images_in_ccs = int((cc_df['count'] != 0).sum() / 2)
            images_not_in_ccs = total_image_count - images_in_ccs
            largest_cc_image_count = int(cc_df['count'].max())
            summary_stats.append(f"Similarity:  {pct(images_in_ccs):.2f}% ({images_in_ccs:,d}) belong to "\
                                 f"{number_of_ccs} similarity clusters (components).")
            summary_stats.append(f"{pct(images_not_in_ccs):.2f}% ({images_not_in_ccs:,d}) images do not "
                                 f"belong to any similarity cluster.")
            summary_stats.append(f"Largest cluster has {largest_cc_image_count:,d} "
                                 f"({pct(largest_cc_image_count):.2f}%) images.")
            sim_thresh = self.config['fastdup_kwargs']['threshold']
            cc_thresh = self.config['fastdup_kwargs']['turi_param']['ccthreshold']
            summary_stats.append(f"For a detailed analysis, use `.connected_components()`\n"
                                 f"(similarity threshold used is {sim_thresh}, "
                                 f"connected component threshold used is {cc_thresh}).\n")
        except Exception as e:
            summary_stats.append(f"Similarity:  Unable to calculate similarity clusters.")

    
        try:
            # Outlier counts
            outlier_df = self.outliers()
            number_of_outliers = len(outlier_df)
            outlier_threhold = 100 * self._config.get('lower_threshold', 0.05)
            summary_stats.append(f"Outliers: {pct(number_of_outliers):.2f}% ({number_of_outliers:,d}) of images are "\
                          f"possible outliers, and fall in the bottom {outlier_threhold:.2f}% of "\
                          f"similarity values.")
            summary_stats.append(f"For a detailed list of outliers, use `.outliers(data=True)`.")
        except Exception as e:
            summary_stats.append(f"Outliers: Unable to calculate outliers.")
        
        try:
            # Blurry, dark and bright images
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
        except Exception as e:
            stats_str = f"Unable to calculate blur, brightness and darkness.\n"

        if verbose:
            print('\n', 88 * '#')
            print(f"\nDataset Analysis Summary: \n")
            for line in summary_stats:
                print(f"    {line}")

        return summary_stats

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
        self._df_annot = self._df_annot.set_index('fd_index')

    def _save_artifacts(self, fastdup_kwargs):
        """
        This function saves artifacts that are created during fastdup run
        """
        # read config
        fastdup_kwargs['turi_param'] = dict([arg.split("=") for arg in fastdup_kwargs['turi_param'].split(',')])
        config_json = self._work_dir / FD.CONFIG_JSON
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
            fastdup.save_binary_feature(self.work_dir, self._df_annot[FD.ANNOT_FILENAME].to_list(), self._pre_calc_features)
            return self.input_dir
        # image data type - return input dir or subset (and edit annot by intersecting over subset)
        elif self._dtype == FD.IMG or self._run_mode == FD.MODE_CROP:
            if self._subset is None:
                return str(self._input_dir)
            else:
                if self._df_annot is not None:
                    self._df_annot = self._df_annot[self._df_annot[FD.ANNOT_FILENAME].isin(self._subset)]
                subset = [str(self._input_dir / s) for s in self._subset] if self._subset is not None else self._subset
                return subset

        elif self._dtype == FD.BBOX:
            if self._subset is not None:
                subset = []
                self._df_annot = self._df_annot[self._df_annot[FD.ANNOT_SPLIT].isin(subset)]

            # set fastdup instance id
            self._df_annot[FD.ANNOT_FD_ID] = np.arange(len(self._df_annot)).astype(int)

            # save bbox csv & add input-dir images filename
            df_annot = self._df_annot.copy()
            df_annot['full_path'] = df_annot[FD.ANNOT_FILENAME].apply(lambda fname: self._input_dir / fname)
            df_annot = df_annot.astype({FD.ANNOT_FD_ID: int})

            # convert column names input expected: "filename,col_x,row_y,width,height"
            bbox_cols = [FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H]
            rotated_bbox_cols = [FD.ANNOT_ROT_BBOX_X1, FD.ANNOT_ROT_BBOX_Y1, FD.ANNOT_ROT_BBOX_X2, FD.ANNOT_ROT_BBOX_Y2,
                                 FD.ANNOT_ROT_BBOX_X3, FD.ANNOT_ROT_BBOX_Y3, FD.ANNOT_ROT_BBOX_X4, FD.ANNOT_ROT_BBOX_Y4, 
                                 FD.ANNOT_LABEL]
            if set(rotated_bbox_cols).issubset(df_annot.columns):
                main_cols = [FD.ANNOT_FD_ID, 'full_path'] + rotated_bbox_cols
                df_annot = df_annot[main_cols].rename({FD.ANNOT_FD_ID: 'index', 'full_path': 'filename'}, axis=1)
            else:
                main_cols = [FD.ANNOT_FD_ID, 'full_path'] + bbox_cols
                fast_dup_cols = ['index', 'filename', 'col_x', 'row_y', 'width', 'height']
                df_annot = df_annot[main_cols].rename(dict(zip(main_cols, fast_dup_cols)), axis=1)
            # run fastdup on a file with full path
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp_name = f'{temp.name}.csv'
                df_annot.to_csv(temp_name, index=False)
                return temp_name

    def _add_annot_and_split(self, df: pd.DataFrame, data: bool, merge_on: List[str],
                             split: Union[str, List[str]] = None, suffix: bool = True,
                             unannotated: bool = False) -> pd.DataFrame:
        """
        Merge df according to context (IMG/OBJ) and filter according to split
        :param df: input df
        :param data: (Optional) if true - add annotation
        :param merge_on: merge on columns - merged with fastdup-id
        :param split: (Optional) filtered by split
        :param suffix: (Optional) add suffix of merged on column
        :return: merged and filtered df
        """

        # get split and annotations
        splits = split if isinstance(split, list) or split is None else [split]
        if df is None or (not data and split is None):
            return df
        df_annot = self._merge_df_with_annot(df, left_on=merge_on, suffix=suffix, unannotated=unannotated)

        # filter by split
        if split is not None and self._has_split:
            df_annot = df_annot[np.logical_and.reduce([df_annot[f'{FD.ANNOT_SPLIT}_{col}' if suffix else FD.ANNOT_SPLIT]
                                                      .isin(splits) for col in merge_on])]
        elif split is not None and not self._has_split:
            raise ValueError(f'No split column found in annotation file')

        # merge annotations
        if not data:
            df_annot = df_annot[df.columns]
        return df_annot

    def _merge_df_with_annot(self, df: pd.DataFrame, left_on: List[str],
                            inplace: bool = False, suffix: bool = True, unannotated: bool = False) -> pd.DataFrame:
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

        df = df if inplace else df.copy()
        df_annot = self._df_annot.copy()
        df_annot = df_annot[~df_annot[FD.ANNOT_FD_ID].isna()].set_index(FD.ANNOT_FD_ID)
        if not unannotated:
            df_annot = df_annot[df_annot[FD.ANNOT_VALID]]
            df = df[df[left_on].isin(df_annot.index).all(axis=1)]
        original_col_names = df_annot.columns

        for col_name in left_on:
            # merge to left side with appropriate suffix (forced)
            df_annot.columns = df_annot.columns.map(lambda x: f'{str(x)}_{col_name}' if suffix else str(x))
            df = pd.merge(df, df_annot, left_on=col_name, right_on=FD.ANNOT_FD_ID, how='left')
            df_annot.columns = original_col_names

        return df

    def _fetch_df(self, csv_name: str) -> pd.DataFrame:
        """
        Retrieve fastdup file from relevant workdir
        :param csv_name: basename of required file
        :return: requested csv
        """

        filename = csv_name if csv_name.startswith(self.work_dir) else self._work_dir / csv_name
        if not os.path.exists(filename):
            return None
        return pd.read_csv(filename)

    def _create_img_mapping(self):
        """
        Map a df to fastdup-img-id according to fname column (inner merge)
        The function can also create generate missing and filtered csv in the work-dir
        """

        # get mapping df from fastdup
        df_mapping = self._fetch_df(FD.MAPPING_CSV).reset_index()
        if  FD.MAP_INST_ID not in df_mapping.columns:
            df_mapping[FD.MAP_INST_ID] = df_mapping.index
        df_mapping = df_mapping.rename({
            FD.MAP_INST_ID: FD.ANNOT_FD_ID,
            FD.MAP_FILENAME: FD.ANNOT_FILENAME,
        }, axis=1)[[FD.ANNOT_FILENAME, FD.ANNOT_FD_ID]]

        if self._df_annot is None:
            self._df_annot = df_mapping
        else:
            self._df_annot = pd.merge(self._df_annot, df_mapping, on=FD.ANNOT_FILENAME, how='left')
            self._df_annot[FD.ANNOT_FD_ID] = self._df_annot[FD.ANNOT_FD_ID].astype(pd.UInt32Dtype())
            # read bad csv and add fastdup-id to bad files
            df_bad_files = self._fetch_df(FD.BAD_CSV)
            if df_bad_files is not None:
                mask = self._df_annot[FD.ANNOT_FILENAME].isin(df_bad_files[FD.BAD_FILENAME])
                self._df_annot[FD.ANNOT_FD_ID][mask] = df_bad_files[FD.BAD_FD_ID].tolist()

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

        # add crops to annotation df (available in case bounding box is not available=='face'/'yolov5s')
        if self._run_mode == FD.MODE_CROP:
            df_crops_annot = self._fetch_df(FD.CROPS_CSV).rename({'index': FD.ANNOT_FD_ID}, axis=1)
            self._df_annot = pd.merge(self._df_annot, df_crops_annot, on=FD.ANNOT_FD_ID, how='left')
            self._df_annot[FD.ANNOT_FILENAME] = self._df_annot[FD.ANNOT_FILENAME].apply(
                lambda fname: str(Path('crops') / fname))
            self._df_annot.rename({FD.ANNOT_FILENAME: FD.ANNOT_CROP_FILENAME,
                                   FD.MAP_FILENAME: FD.ANNOT_FILENAME,
                                   'col_x': FD.ANNOT_BBOX_X, 'row_y': FD.ANNOT_BBOX_Y,
                                   'height': FD.ANNOT_BBOX_H, 'width': FD.ANNOT_BBOX_W}, axis=1, inplace=True)

        # 1. collect subsets for error types analysis (MISSING_ANNOTATION, MISSING_IMAGE, FD-ERROR, VALID)
        df_bad_files = self._fetch_df(FD.BAD_CSV)
        df_mapping = self._fetch_df(FD.MAPPING_CSV)
        if FD.MAP_INST_ID not in df_mapping.columns:
            df_mapping[FD.MAP_INST_ID] = df_mapping.index
        df_mapping = df_mapping.rename({
            FD.MAP_INST_ID: FD.ANNOT_FD_ID,
            FD.MAP_FILENAME: FD.ANNOT_FILENAME,
        }, axis=1).reset_index()[[FD.ANNOT_FILENAME, FD.ANNOT_FD_ID]]
        seen_by_fastdup = set(df_bad_files[FD.BAD_FD_ID]).union(set(df_mapping[FD.ANNOT_FD_ID]))
        df_mapping_not_in_annot = df_mapping[~df_mapping[FD.ANNOT_FD_ID].isin(self._df_annot[FD.ANNOT_FD_ID])]
        df_annot_in_subset = self._df_annot[self._df_annot[FD.ANNOT_FD_ID].isin(seen_by_fastdup)]
        df_annot_not_in_subset = self._df_annot[~self._df_annot[FD.ANNOT_FD_ID].isin(seen_by_fastdup)]

        # 2. set NaN fastdup-id for instances that are not in df_mapping
        self._df_annot[FD.ANNOT_FD_ID] = self._df_annot[FD.ANNOT_FD_ID].mask(~self._df_annot[FD.ANNOT_FD_ID].
                                                                             isin(df_mapping[FD.ANNOT_FD_ID]))

        # 3. add not-in-annot instances (but in fastdup mapping) to df_annot
        self._df_annot = pd.concat([
            pd.merge(df_annot_in_subset, df_mapping,
                     on=FD.ANNOT_FD_ID, how='outer', suffixes=('', '_map')),
            df_annot_not_in_subset])
        self._df_annot[FD.ANNOT_FILENAME].fillna(self._df_annot[FD.ANNOT_FILENAME + '_map'], inplace=True)
        self._df_annot.drop(FD.ANNOT_FILENAME + '_map', axis=1, inplace=True)

        # 4. mark default error type as VALID
        self._df_annot[FD.ANNOT_ERROR] = 'VALID'

        # 5. mark error type of not-in-annot as  MISSING_ANNOTAION
        self._df_annot[FD.ANNOT_ERROR] = self._df_annot[FD.ANNOT_ERROR].mask(
            self._df_annot[FD.ANNOT_FD_ID].isin(df_mapping_not_in_annot[FD.ANNOT_FD_ID]), 'MISSING_ANNOTATION')

        # 6. mark error type of not-in-mapping as MISSING_IMAGE
        self._df_annot[FD.ANNOT_ERROR] = self._df_annot[FD.ANNOT_ERROR].mask(
            ~self._df_annot[FD.ANNOT_FD_ID].isin(seen_by_fastdup), 'MISSING_IMAGE')

        # 7. mark error types from bad-files
        if df_bad_files is not None:
            fd_errors = df_bad_files.set_index('index')[FD.ANNOT_ERROR]
            self._df_annot[FD.ANNOT_ERROR] = self._df_annot.apply(
                lambda row: fd_errors.get(row[FD.ANNOT_FD_ID], row[FD.ANNOT_ERROR]), axis=1)

        # 8. convert fastdup-id to UInt32Dtype and label to categorical (for memory efficiency)
        self._df_annot[FD.ANNOT_FD_ID] = self._df_annot[FD.ANNOT_FD_ID].astype(dtype=pd.UInt32Dtype())
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
        is_s3_input = False if input_dir is None else str(input_dir).startswith('s3://')
        if is_s3_input:
            return self._work_dir / "tmp" / str(input_dir).split("/", 3)[-1]
        else:
            return input_dir

    def _infer_dtype(self, requested_dtype: str, df_annot: pd.DataFrame = None) -> str:
        """
        Infer run-type - images/bboxes from.
        :param requested_dtype: users requested data type
        :param df_annot: data annotation
        :return: run-type
        """
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

        if requested_dtype == 'infer' and not isinstance(self._input_dir, list):
            return FD.BBOX if bbox_cols_available else FD.IMG

        assert requested_dtype != FD.BBOX or bbox_cols_available, f"missing df-annotations or bounding box columns"
        return requested_dtype

    def _verify_fastdup_run_args(self, input_dir, work_dir, df_annot, subset, data_type, pre_calc_features):
        """
        Verify constructor arguments and raise exception if invalid.
        This method is called by the constructor, and checks that the requested procedure is supported.
        the following checks are performed:
            1. arguments have valid values
                a. input_dir is None and is a string/pathlib.Path or a list of strings/pathlib.Path
                b. work_dir is not None and is a string/pathlib.Path
                c. data_type is one of FD.IMG, FD.BBOX, 'infer'
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
        assert isinstance(pre_calc_features, np.ndarray) or pre_calc_features is None, \
            'pre_calc_features must be a numpy array'
        assert pre_calc_features is not None or (isinstance(input_dir, list) or isinstance(input_dir, str) or
                                                 isinstance(input_dir, Path)), \
            'input_dir must be provided and be a string/pathlib.Path or a list of strings/pathlib.Path'
        assert work_dir is not None and (isinstance(work_dir, str) or isinstance(work_dir, Path)), \
            'work_dir must be provided and be a string or pathlib.Path'
        assert data_type in [FD.IMG, FD.BBOX, 'infer'], \
            f'invalid data_type, found: {data_type} supported: img, bbox, infer'
        assert df_annot is None or isinstance(df_annot, pd.DataFrame), 'df_annot must be a pandas DataFrame'

        if pre_calc_features is not None and df_annot is not None:
            assert pre_calc_features.shape[0] == df_annot.shape[0], \
                'pre_calc_features and df_annot must have the same number of rows'

        # verify arguments combinations
        assert any(
            # list of input dirs, without annotations, without subset - image data-type only
            [isinstance(input_dir, list) and df_annot is None and subset is None and data_type in ['infer', FD.IMG],
             # single input dir, without annotations, with/without subset - image data-type only
             (isinstance(input_dir, str) or isinstance(input_dir, Path)) and df_annot is None and (data_type in ['infer', FD.IMG] or self._run_mode == FD.MODE_CROP ),
             # single input dir, with annotations, with/without subset - image/bbox
             (isinstance(input_dir, str) or isinstance(input_dir, Path)) and df_annot is not None,
             # no input dir, with/without annotations, with/without subset - with-pre-calc-features
             input_dir is None and pre_calc_features is not None]
        ),\
            'invalid input (input-dir, annotation, subset, data-type) ' \
            '\nsupported options: ' \
            '\n1. single input dir, with annotations, with/without subset' \
            '\n2. single input dir, without annotations, with/without subset - image data-type only' \
            '\n3. list of input dirs, without annotations, without subset - image data-type only'

        # verify df_annot columns
        if df_annot is not None:
            assert FD.ANNOT_FILENAME in df_annot, \
                'when running on images df_annot must contain a column named "img_filename" ' \
                'if you wish to run on bounding boxes, make sure to provide the relevant columns'
            if data_type == FD.IMG:
                assert df_annot[FD.ANNOT_FILENAME].nunique() == len(df_annot[FD.ANNOT_FILENAME]), \
                    'df_annot must contain unique filenames'
            elif data_type == FD.BBOX:
                bbox_cols = {FD.ANNOT_FILENAME, FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H}
                rotated_bbox_cols = {FD.ANNOT_ROT_BBOX_X1, FD.ANNOT_ROT_BBOX_Y1, FD.ANNOT_ROT_BBOX_X2,
                                     FD.ANNOT_ROT_BBOX_Y2, FD.ANNOT_ROT_BBOX_X3, FD.ANNOT_ROT_BBOX_Y3,
                                     FD.ANNOT_ROT_BBOX_X4, FD.ANNOT_ROT_BBOX_Y4}
                assert bbox_cols.issubset(df_annot.columns) or rotated_bbox_cols.issubset(df_annot.columns), \
                    f'df_annot must contain columns named, {FD.ANNOT_FILENAME}, {FD.ANNOT_BBOX_X}, ' \
                    f'{FD.ANNOT_BBOX_Y}, {FD.ANNOT_BBOX_H}, {FD.ANNOT_BBOX_W}'

                if bbox_cols.issubset(df_annot.columns):
                    assert df_annot.groupby(list(bbox_cols)).size().max() == 1, \
                        'df_annot must contain unique bounding boxes'
                else:
                    assert df_annot.groupby(list(rotated_bbox_cols)).size().max() == 1, \
                        'df_annot must contain unique rotated bounding boxes'

        # verify input dir exists
        verify_input_dirs_or_s3_paths_exist(input_dir)


def verify_input_dirs_or_s3_paths_exist(input_dir_or_list):
    def verify_single_path(path):
        if path is None:
            pass
        elif str(path).startswith('s3'):
            assert s3_folder_exists_and_not_empty(path), f'Could not access S3 path {path}'
        else:
            assert os.path.isdir(path), f'input_dir {path} is not a directory'

    if isinstance(input_dir_or_list, list):
        for d in input_dir_or_list:
            verify_single_path(d)
    else:
        verify_single_path(input_dir_or_list)


def is_fastdup_dir(work_dir):
    return os.path.exists(Path(work_dir) / FD.MAPPING_CSV) and \
           os.path.exists(Path(work_dir) / FD.NNF_INDEX) and \
           os.path.exists(Path(work_dir) / FD.ANNOT_PKL)


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
        'min_input_image_width': {'map': None, 'default': 10}
    }

    for key, value in input_kwargs.items():
        if key not in fastdup_params and key not in turi_params:
            raise ValueError(f'invalid argument {key}')

    turi_kwargs = []
    for arg_name, param in turi_params.items():
        map_dict = param['map']
        map_func = lambda x: x if map_dict is None else map_dict[x]
        value = input_kwargs.get(arg_name, param['default'])
        turi_kwargs.append(f'{arg_name}={map_func(value)}')

    fastdup_kwargs = {key: value for key, value in input_kwargs.items() if key in fastdup_params}
    fastdup_kwargs['turi_param'] = ','.join(turi_kwargs)
    return fastdup_kwargs


def fastdup_convert_to_relpath(work_dir: Union[Path, str], input_dir: Union[Path, str]):
    """
    create mapping files, relative path to img-id/features/stats
    :param work_dir: location of files
    :param input_dir: base dir for images
    """
    work_dir, input_dir = Path(work_dir), Path(input_dir)
    fastdup_src_files = [FD.BAD_CSV, FD.MAPPING_CSV, FD.CROPS_CSV]

    input_dir = '' if input_dir is None else input_dir
    input_dir = input_dir if (str(input_dir).endswith("/") or input_dir == '') else f'{input_dir}/'

    def remove_working_dir(full_path: str):
        if full_path.startswith(input_dir) or str(Path.cwd() / full_path).startswith(input_dir):
            full_path = os.path.relpath(full_path, input_dir)
        return full_path

    # loop files, remove prefix and save files
    for src_file in fastdup_src_files:
        fname = work_dir / src_file
        if not fname.exists():
            continue
        try:
            df = pd.read_csv(fname)

        except EmptyDataError as e:
            print(f'{src_file} is empty')
            continue

        df['filename'] = df.filename.apply(remove_working_dir)
        df.to_csv(work_dir / src_file, index=False)


def get_input_dir(_input_dir):
    def pathlib_or_s3_cast(c):
        c = str(c)
        return c if c.startswith('s3://') else Path(c)
    return [pathlib_or_s3_cast(f) for f in _input_dir] if isinstance(_input_dir, list)\
        else pathlib_or_s3_cast(_input_dir)


def s3_folder_exists_and_not_empty(s3_path:str) -> bool:
    '''
    Folder should exist.
    Folder should not be empty.
    '''
    return True
    #def split_s3_path(s3_path):
    #    path_parts = s3_path.replace("s3://", "").split("/")
    #    bucket = path_parts.pop(0)
    #    key = "/".join(path_parts)
    #    return bucket, key
    #bucket, key = split_s3_path(s3_path)

    #s3 = boto3.client('s3')
    #if not key.endswith('/'):
    #    key = key+'/'
    #resp = s3.list_objects(Bucket=bucket, Prefix=key, Delimiter='/',MaxKeys=1)
    #return 'Contents' in resp
