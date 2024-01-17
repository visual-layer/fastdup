import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple

import fastdup
from fastdup.utils import shorten_path
import numpy as np
from fastdup.sentry import v1_sentry_handler, fastdup_capture_exception
from fastdup.fastdup_controller import FastdupController
from fastdup import create_outliers_gallery, create_duplicates_gallery, create_components_gallery, \
    create_similarity_gallery, find_top_components, create_stats_gallery
import fastdup.definitions as FD

from fastdup.galleries import load_and_merge_stats
import pandas as pd


class FastdupVisualizer:
    def __init__(self, controller: FastdupController, default_config=None):
        """
        Create galleries/plots from fastdup output.
        :param controller: FastdupController instance
        :param default_config: dict of default config for cv2, e.g. {'cv2_imread_flag': cv2.IMREAD_COLOR}
        """
        self._default_config = dict() if default_config is None else default_config
        self._controller = controller
        self._availavle_columns = None

    def _compute_label_col(self, label_col, load_crops):
        if label_col in FD.CAPTION_MODEL_NAMES or label_col == FD.VQA_MODEL1_NAME or label_col == FD.AGE_LABEL1_NAME:
            return label_col
        if self._controller._bbox == 'ocr':
            load_crops = True
        if self._controller._df_annot is not None and not self._controller._df_annot.empty and ('label' in self._controller._df_annot.columns):
            if load_crops and 'crop_filename' in self._controller._df_annot.columns:
                label_col = pd.Series(self._controller._df_annot["label"].values,
                                      index=self._controller._df_annot["crop_filename"]).to_dict()
            else:
                label_col = pd.Series(self._controller._df_annot["label"].values,index=self._controller._df_annot["filename"]).to_dict()
        return label_col

    def _get_available_columns(self):
        """
        Get available columns from annotations file.
        """
        if self._availavle_columns is None:
            assert self._controller.annotations() is not None, "Failed to find annotations()"
            self._availavle_columns = self._controller.annotations().columns
        return self._availavle_columns

    def _set_temp_save_path(self, sub_dirs: Union[str, Path]):
        """
        Get temporary save path for galleries. inside work_dir/galleries
        :param sub_dirs: subdirectories to create inside work_dir/galleries
        """
        save_path = str(Path(self._controller.work_dir) / 'galleries' / sub_dirs)
        return save_path

    def _infer_paths(self, users_input: Union[str, Path], temp_save_path: Union[str, Path],
                     out_html_fname: Union[str, Path], lazy_load: bool = False) -> Tuple[str, str, str]:
        """
        Infer save paths for galleries.
        :users_input: requested output path (if lazy_load - assume users_input is a directory)
        :temp_save_path: save path for temporary files
        :out_html_fname: save filename for html file
        :lazy_load: True if user requested to lazy load the html file
        """
        date_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        # return temporary directory and html path inside work_dir
        if users_input is None:
            html_src_path = Path(f'{self._set_temp_save_path(temp_save_path)}_{date_suffix}') / out_html_fname
            save_dir = html_src_path.parent
            html_dst_path = html_src_path if lazy_load else save_dir.parent / out_html_fname
        # if lazy_load - assume users_input is a directory
        elif lazy_load:
            if isinstance(users_input, str):
                users_input = shorten_path(users_input)
            save_dir = Path(users_input)
            html_src_path = save_dir / out_html_fname
            html_dst_path = save_dir / out_html_fname
        # if not lazy_load - assume users_input is an html file path
        else:
            if isinstance(users_input, str):
                users_input = shorten_path(users_input)
            html_dst_path = Path(users_input)
            save_dir = html_dst_path.parent / f'artifacts_{html_dst_path.name.split(".")[0]}_{date_suffix}'
            html_src_path = save_dir / out_html_fname

        os.makedirs(save_dir, exist_ok=True)
        assert os.path.exists(save_dir), "Failed to create save_dir " + str(save_dir)
        return str(save_dir), str(html_src_path), str(html_dst_path)

    @v1_sentry_handler
    def outliers_gallery(self, save_path: str = None, label_col: str = None, draw_bbox: bool = False,
                         num_images: int = 20, max_width: int = None, lazy_load: bool = False, how: str = 'one',
                         slice: Union[str, list] = None, ascending: bool = True,
                         save_artifacts: bool = False, show: bool = True, load_crops: bool = None, external_df: pd.DataFrame = None,
                         **kwargs):
        """
        Create gallery of outliers, i.e. images that are not similar to any other images in the dataset.
        :param save_path: path for saving the gallery. If None, saves to work_dir/galleries
        :param num_images: number of images to display
        :param lazy_load: if True, load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe (optional). fastdup beta: to generate automated
         image captions use either 'automatic', 'automatic2', or 'indoor_outdoor'
        :param how: (Optional) outlier selection method.
            - one = take the image that is far away from any one image
                    (but may have other images close to it).
            - all = take the image that is far away from all other images. Default is one.
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param draw_bbox: if True, draw bounding box on the images
        :param slice: (Optional) list/single label for filtering outliers
        :param ascending: (Optional) sort ascending or descending
        :param save_artifacts: save artifacts to disk
        :param show: show gallery in notebook
        :param load_crops: load crops instead of full images, on default is None which means load crop when they are present
        :param external_df: optional dataframe including data for report.
               Example usage:
               ```
               import fastdup
               fd = fastdup.create(input_dir='/my/input_dir', work_dir='/my/work_dir')
               fd.run()
               sim = fd.similarity()
               files = df.annotations()
               df = merge_with_filenames(sim, files)
               df['label'] = df['from'].apply(lambda x: os.path.basename(x)[:2])
               df['label2'] = df['to'].apply(lambda x: os.path.basename(x)[:2])
               fd.vis.outliers_gallery(load_crops=False, label_col='label', external_df=df)
               ```
        :param kwargs: additional parameters to pass to create_outliers_gallery
        """
        # get images to display

        out_fname = 'outliers.html'
        assert self._controller._fastdup_applied, "To generate a report, need to fastdup.run() at least once"

        if save_path is not None:
            html_dst_path = os.path.join(save_path, out_fname)
        else:
            html_dst_path = os.path.join(self._controller.work_dir, "galleries", out_fname)

        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        if draw_bbox:
            load_crops = True
        if load_crops is None:
            load_crops = self._controller._dtype == FD.BBOX

        label_col = self._compute_label_col(label_col, load_crops)
        if draw_bbox and external_df is None:
            external_df = self._controller.outliers(load_crops=load_crops)
            external_df = external_df.dropna()
        elif self._controller._dtype == FD.BBOX and external_df is None:
            external_df = self._controller.similarity(load_crops=load_crops)
            if load_crops and 'crop_filename_from' in external_df.columns and 'crop_filename_to' in external_df.columns:
                external_df['filename_from'] = external_df['crop_filename_from']
                external_df['filename_to'] = external_df['crop_filename_to']
            external_df = external_df.sort_values('distance', ascending=ascending)

        ret = create_outliers_gallery(self._controller.work_dir  if external_df is None else external_df,
                                      work_dir=self._controller.work_dir, save_path=html_dst_path,
                                get_bounding_box_func=self._get_bbox_func(draw_bbox),
                                get_label_func=label_col,
                                num_images=num_images,
                                lazy_load=lazy_load, how=how, max_width=max_width,
                                id_to_filename_func=self._get_filneme_func(load_crops),
                                get_display_filename_func=self._get_disp_filneme_func(),
                                save_artifacts=save_artifacts,
                                jupyter_html=jupyter_html, input_dir=self._controller.input_dir if not load_crops else self._controller.work_dir,
                                slice=slice, descending= not ascending, draw_bbox=draw_bbox, **kwargs)

        if ret != 0:
            return ret
        if show:
            self._controller.vl_datasets_ref_printout()
            self._disp_jupyter(html_dst_path)
        return 0

    @v1_sentry_handler
    def duplicates_gallery(self, save_path: str = None, label_col: str = None, draw_bbox: bool = False,
                           num_images: int = 20, max_width: int = None, lazy_load: bool = False,
                           slice: Union[str, list] = None, ascending: bool = False, threshold: float = None,
                           save_artifacts: bool = False, show: bool = True, load_crops: bool = None, external_df: pd.DataFrame = None,
                           **kwargs):
        """
        Generate gallery of duplicate images, i.e. images that are similar to other images in the dataset.
        :param save_path: path for saving the gallery. If None, saves to work_dir/galleries
        :param num_images: number of images to display
        :param descending: displays images with the highest similarity first
        :param lazy_load: load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe (optional). fastdup beta: to generate automated
         image captions use either 'automatic', 'automatic2', or 'indoor_outdoor'
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param draw_bbox: draw bounding box on the images
        :param slice: (Optional) list/single label for filtering outliers
        :param ascending: (Optional) sort ascending or descending
        :param threshold: (Optional) threshold to filter out images with similarity score below the threshold.
        :param save_artifacts: save artifacts to disk
        :param show: show gallery in notebook
        :param load_crops: load crops instead of full images, on default is None which means load crop when they are present
        :param external_df: optional dataframe including data for report.
        :param kwargs: additional parameters to pass to create_duplicates_gallery
        """

        assert self._controller._fastdup_applied, "To generate a report, need to fastdup.run() at least once"

        # get images to display
        out_fname = 'duplicates.html'
        if save_path is not None:
            html_dst_path = os.path.join(save_path, out_fname)
        else:
            html_dst_path = os.path.join(self._controller.work_dir, "galleries", out_fname)


        # create gallery
        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        if draw_bbox:
            load_crops = True
        label_col = self._compute_label_col(label_col, load_crops)
        if load_crops is None:
            load_crops = self._controller._dtype == FD.BBOX
        if draw_bbox and external_df is None:
            external_df = self._controller.similarity(load_crops=load_crops)
        elif self._controller._dtype == FD.BBOX and external_df is None:
            external_df = self._controller.similarity(load_crops=load_crops)
            if load_crops and 'crop_filename_from' in external_df.columns and 'crop_filename_to' in external_df.columns:
                external_df['filename_from'] = external_df['crop_filename_from']
                external_df['filename_to'] = external_df['crop_filename_to']
            external_df = external_df.sort_values('distance', ascending=ascending)

        ret = create_duplicates_gallery(self._controller.work_dir if external_df is None else external_df,
                                        work_dir=self._controller.work_dir, save_path=html_dst_path, lazy_load=lazy_load,
                                  get_bounding_box_func=self._get_bbox_func(draw_bbox),
                                  get_label_func=label_col,
                                  num_images=num_images, max_width=max_width, threshold=threshold,
                                  id_to_filename_func=self._get_filneme_func(load_crops), descending=not ascending,
                                  get_display_filename_func=self._get_disp_filneme_func(),
                                  save_artifacts=save_artifacts, input_dir=self._controller.input_dir,
                                  jupyter_html=jupyter_html, slice=slice, load_crops=load_crops, draw_bbox=draw_bbox, **kwargs)

        if ret != 0:
            return ret
        if show:
            self._controller.vl_datasets_ref_printout()
            self._disp_jupyter(html_dst_path)
        return 0

    @v1_sentry_handler
    def similarity_gallery(self, save_path: str = None, label_col: str = None, draw_bbox: bool = False,
                           num_images: int = 20, max_width: int = None, lazy_load: bool = False,
                           slice: Union[str, list] = None, ascending: bool = None, threshold: float = None,
                           show: bool = True, load_crops: bool = None, external_df: pd.DataFrame = None, **kwargs):
        """
        Generate gallery of similar images, i.e. images that are similar to other images in the dataset.
        :param save_path: path for saving the gallery. If None, saves to work_dir/galleries
        :param num_images: number of images to display
        :param ascending: display images with lowest similarity first
        :param lazy_load: load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe (optional). fastdup beta: to generate automated
         image captions use either 'automatic', 'automatic2', or 'indoor_outdoor'
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.

        :param max_width: max width of the gallery
        :param draw_bbox: draw bounding box on the images
        :param threshold: (Optional) threshold to filter out images with similarity score below the threshold.
        :param slice: (Optional) list/single label for filtering similarity dataframe. Reserved slice keywords are "diff" and "same" for finding similar images from different class label or same class label
        :param ascending: (Optional) sort ascending or descending
        :param show: show gallery in notebook
        :param load_crops: load crops instead of full images, on default is None which means load crop when they are present
        :param external_df: optional dataframe including data for report.
        :param kwargs: additional parameters to pass to create_duplicates_gallery
        """
        # get images to display
        assert self._controller._fastdup_applied, "To generate a report, need to fastdup.run() at least once"

        out_fname = 'similarity.html'
        #save_dir, html_src_path, html_dst_path = self._infer_paths(users_input=save_path, temp_save_path=temp,
        #                                                          out_html_fname=out_fname, lazy_load=lazy_load)
        if save_path is not None:
            html_dst_path = os.path.join(save_path, out_fname)
        else:
            html_dst_path = os.path.join(self._controller.work_dir, "galleries", out_fname)

        # create gallery
        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        if draw_bbox:
            load_crops = True
        if load_crops is None:
            load_crops = self._controller._dtype == FD.BBOX
        label_col = self._compute_label_col(label_col, load_crops)
        if ascending is None and slice is not None and isinstance(slice, str):
            if slice == "diff":
                ascending = True
                slice = "label_score"
            elif slice == "same":
                ascending = False
                slice = "label_score"
        elif ascending is None:
            ascending = True
        if draw_bbox and external_df is None:
            external_df = self._controller.similarity(load_crops=load_crops)
        elif self._controller._dtype == FD.BBOX and external_df is None:
            external_df = self._controller.similarity(load_crops=load_crops)
            if load_crops and 'crop_filename_from' in external_df.columns and 'crop_filename_to' in external_df.columns:
                external_df['filename_from'] = external_df['crop_filename_from']
                external_df['filename_to'] = external_df['crop_filename_to']


        ret = create_similarity_gallery(self._controller.work_dir if external_df is None else external_df,
                                  work_dir=self._controller.work_dir, save_path=html_dst_path, lazy_load=lazy_load,
                                  get_bounding_box_func=self._get_bbox_func(draw_bbox),
                                  get_label_func=label_col,
                                  num_images=num_images, max_width=max_width, threshold=threshold,
                                  id_to_filename_func=self._get_filneme_func(load_crops), descending=not ascending,
                                  get_display_filename_func=self._get_disp_filneme_func(),
                                  jupyter_html=jupyter_html, input_dir=self._controller.input_dir if not load_crops else self._controller.work_dir,
                                  slice=slice, draw_bbox=draw_bbox, **kwargs)

        if ret is None or not isinstance(ret, pd.DataFrame):
            return -1
        #self._clean_temp_dir(save_dir, html_src_path, html_dst_path, lazy_load=lazy_load)
        if show:
            self._controller.vl_datasets_ref_printout()
            self._disp_jupyter(html_dst_path)
        return ret


    def stats_gallery(self, save_path: str = None, metric: str = 'dark', draw_bbox: bool = False,
                      slice: Union[str, list] = None, ascending: bool = None,
                      label_col: str = None, lazy_load: bool = False, show: bool = True,
                      load_crops: bool = None, external_df: pd.DataFrame = None, **kwargs):
        """
        Generate gallery of images sorted by a specific metric.
        :param save_path: path for saving the gallery. If None, saves to work_dir/galleries
        :param metric: metric to sort images by (dark, bright, blur)
        :param slice: (Optional) list/single label for filtering stats dataframe
        :param ascending: (Optional) sort ascending or descending
        :param label_col: (Optional) column name of label in annotation dataframe. fastdup beta: to generate automated
         image captions use either 'automatic', 'automatic2', or 'indoor_outdoor'
        :param lazy_load: load images on demand, otherwise load all images into html. On default loads the crops in case they are present.
        :param load_crops: load crops instead of full images, on default is None which means load crop when they are present
        :param show: show gallery in notebook
        :param external_df: optional dataframe including data for report.

        """
        assert self._controller._fastdup_applied, "To generate a report, need to fastdup.run() at least once"

        auto_ascending = False
        if metric == 'dark':
            metric = 'mean'
            auto_ascending = True
        elif metric == 'bright':
            metric = 'mean'
            auto_ascending = False
        if metric == 'blur':
            auto_ascending = True
        ascending = auto_ascending if ascending is None else ascending
        # get images to display
        temp = 'stats'
        out_fname = f'{metric}.html'
        if save_path is not None:
            html_dst_path = os.path.join(save_path, out_fname)
        else:
            html_dst_path = os.path.join(self._controller.work_dir, "galleries", out_fname)

        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        label_col = self._compute_label_col(label_col, load_crops)

        if load_crops is None:
            load_crops = self._controller._dtype == FD.BBOX
        if draw_bbox:
            load_crops = True
        if (draw_bbox or self._controller._dtype == FD.BBOX) and external_df is None:
            external_df = self._controller.img_stats(load_crops=load_crops)
            external_df = external_df.sort_values(metric, ascending=ascending)

        ret = fastdup.create_stats_gallery(self._controller.work_dir if external_df is None else external_df,
                                     save_path=html_dst_path,
                                     get_label_func=label_col,
                                     metric=metric,
                                     input_dir=self._controller.input_dir if not load_crops else self._controller.work_dir,
                                     work_dir=self._controller.work_dir,
                                     lazy_load=lazy_load,
                                     descending=not ascending,
                                     id_to_filename_func=self._get_filneme_func(load_crops),
                                     jupyter_html=jupyter_html, slice=slice, **kwargs)

        if ret != 0:
            return ret
        if show:
            self._controller.vl_datasets_ref_printout()
            self._disp_jupyter(html_dst_path)
        return 0

    @v1_sentry_handler
    def component_gallery(self, save_path: str = None, label_col: str = None, draw_bbox: bool = False,
                          num_images: int = 20, max_width: int = None, lazy_load: bool = False,
                          slice: Union[str, list] = None, group_by: str = 'visual', min_items: int = None,
                          max_items: int = None, threshold: float = None, metric: str = None,
                          sort_by: str = 'comp_size', ascending: bool = False,
                          save_artifacts: bool = False, show: bool = True, load_crops: bool = None,
                          external_df: pd.DataFrame = None, **kwargs):
        """

        :param save_path: path for saving the gallery. If None, saves to work_dir/galleries
        :param num_images: number of images to display
        :param lazy_load: load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe (optional). fastdup beta: to generate automated
         image captions use either 'automatic', 'automatic2', or 'indoor_outdoor'
        :param group_by: [visual|label]. Group the report using the visual properties of the image or using the labels
            of the images. Default is visual.
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param min_items: threshold to filter out components with less than min_items
        :param max_items: max number of items to display for each component
        :param draw_bbox: draw bounding box on the images
        :param threshold: (Optional) threshold to filter out images with similarity score below the threshold.
        :param metric: (Optional) parameter to set the metric to use (like blur,min,max,mean,unique,size) for choosing the components. Default is None.
        :param slice: (Optional) list/single label for filtering  connected component
        :param sort_by: (Optional) 'comp_size'|ANY_COLUMN_IN_ANNOTATION
            column name to sort the connected component by
        :param ascending: (Optional) sort ascending or descending
        :param show: show gallery in notebook
        :param save_artifacts: save artifacts to disk
        :param load_crops: load crops instead of full images, on default is None which means load crop when they are present
        :param external_df: optional dataframe including data for report.
        :return:
        """

        # set save paths
        assert self._controller._fastdup_applied, "To generate a report, need to fastdup.run() at least once"
        out_fname = 'components.html'

        if save_path is not None:
            html_dst_path = os.path.join(save_path, out_fname)
        else:
            html_dst_path = os.path.join(self._controller.work_dir, "galleries", out_fname)

        # create gallery
        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        if draw_bbox:
            load_crops = True
        if load_crops is None:
            load_crops = self._controller._dtype == FD.BBOX
        label_col = self._compute_label_col(label_col, load_crops)
        if draw_bbox and external_df is None:
            kwargs2 = kwargs.copy()
            kwargs2['draw_bbox'] = draw_bbox
            external_df = self._controller.connected_components_grouped(sort_by, ascending, metric, load_crops, group_by, kwargs=kwargs2)

        ret = create_components_gallery(work_dir=self._controller.work_dir if external_df is None else external_df, save_path=html_dst_path, input_dir=self._controller.input_dir if not load_crops else self._controller.work_dir,
                                  get_bounding_box_func=self._get_bbox_func(draw_bbox),
                                  get_label_func=label_col, save_artifacts=save_artifacts,
                                  lazy_load=lazy_load, max_items=max_items, max_width=max_width, metric=metric,
                                  threshold=threshold, num_images=num_images, min_items=min_items, group_by=group_by,
                                  id_to_filename_func=self._get_filneme_func(load_crops),
                                  jupyter_html=jupyter_html, slice=slice, load_crops=load_crops, draw_bbox=draw_bbox, sort_by=sort_by,
                                  descending=not ascending, **kwargs)
        if ret != 0:
            return ret

        if show:
            self._controller.vl_datasets_ref_printout()
            self._disp_jupyter(html_dst_path)

        return 0

    def _get_filneme_func(self, load_crops: bool = False):
        if load_crops and FD.ANNOT_CROP_FILENAME in self._get_available_columns():
            filename_col = FD.ANNOT_CROP_FILENAME
        else:
            if load_crops:
                print(f'Warning: cant find {FD.ANNOT_CROP_FILENAME} in annotation dataframe. using {FD.ANNOT_FILENAME} instead.')
            filename_col = FD.ANNOT_FILENAME

        def get_filename_func(fastdup_id):
            return str(self._controller[fastdup_id][filename_col])
        return get_filename_func

    def _get_disp_filneme_func(self):
        def get_disp_filename_func(fastdup_id):
            if isinstance(fastdup_id, str):
                return fastdup_id.replace(self._controller.work_dir, '').replace(self._controller.input_dir, '')
            else:
                return str(self._controller[fastdup_id][FD.ANNOT_FILENAME])
        return get_disp_filename_func

    def _get_label_func(self, label_col):
        """
        Get a function that returns the label of an image based on the annotation dataframe
        :param df_annot: fastdup annotation dataframe
        :param dinput_dir: filenema base-dir to add to the filename in the annotation dataframe
        :param label_col: label column name in the annotation dataframe
        :return:
        """
        # get label column - default is the FD.ANNOT_LABEL column
        label_col = None if label_col is None or label_col is False else label_col
        if label_col is None or label_col not in self._get_available_columns():
            return None
        def label_func(fastdup_id):
            return self._controller[fastdup_id].get(label_col, None)
        return label_func

    def _get_bbox_func(self, draw_bbox=False):
        """
        Get a function that returns the bounding box of an image based on the annotation dataframe
        :param df_annot: fastdup annotation dataframe
        :param input_dir: filenema base-dir to add to the filename in the annotation dataframe
        :return:
        """
        if not draw_bbox:
            return None

        # check if the annotation dataframe has the bounding box columns
        bbox_cols = {FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H}
        if not bbox_cols.issubset(self._get_available_columns()):
            return None
        # create mapping from filename to bounding box
        # return  [x1, y1, x2, y2]
        def bbox_func(fastdup_id):
            if isinstance(fastdup_id, str):
                annot = self._controller._df_annot[self._controller._df_annot[FD.ANNOT_CROP_FILENAME] == fastdup_id]
                if len(annot) == 0:
                    return [[]]
                annot = annot.head(1).to_dict('records')[0]
            else:
                annot = self._controller[fastdup_id]
            return [[annot[FD.ANNOT_BBOX_X],
                    annot[FD.ANNOT_BBOX_Y],
                    annot[FD.ANNOT_BBOX_X] + annot[FD.ANNOT_BBOX_W],
                    annot[FD.ANNOT_BBOX_Y] + annot[FD.ANNOT_BBOX_H]]]
        # return a function that returns the bounding box of an image
        return bbox_func

    def _disp_jupyter(self, html_path):
        """
        Display in a jupyter notebook
        :param html_path: path to html file to display
        """
        # check if we are in a jupyter notebook and display the html file
        if 'JPY_PARENT_PID' in os.environ:
            from IPython.display import display, HTML
            display(HTML(open(html_path).read()))


