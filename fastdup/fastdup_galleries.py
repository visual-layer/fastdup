import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple

import fastdup
import numpy as np
from fastdup.sentry import v1_sentry_handler
from fastdup.fastdup_controller import FastdupController
from fastdup import create_outliers_gallery, create_duplicates_gallery, create_components_gallery, \
    create_similarity_gallery, find_top_components, create_stats_gallery
import fastdup.fastup_constants as FD
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

    def _get_available_columns(self):
        """
        Get available columns from annotations file.
        """
        if self._availavle_columns is None:
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
            save_dir = Path(users_input)
            html_src_path = save_dir / out_html_fname
            html_dst_path = save_dir / out_html_fname
        # if not lazy_load - assume users_input is an html file path
        else:
            html_dst_path = Path(users_input)
            save_dir = html_dst_path.parent / f'artifacts_{html_dst_path.name.split(".")[0]}_{date_suffix}'
            html_src_path = save_dir / out_html_fname

        os.makedirs(save_dir, exist_ok=True)
        return str(save_dir), str(html_src_path), str(html_dst_path)

    def _clean_temp_dir(self, save_dir, html_src_path, html_dst_path, lazy_load=False, save_artifacts=False):
        if not lazy_load:
            shutil.move(html_src_path, html_dst_path)
        if not save_artifacts and not lazy_load:
            shutil.rmtree(save_dir)

    @v1_sentry_handler
    def outliers_gallery(self, save_path: str = None, label_col: str = FD.ANNOT_LABEL, draw_bbox: bool = True,
                         num_images: int = 20, max_width: int = None, lazy_load: bool = False, how: str = 'one',
                         slice: Union[str, list] = None, sort_by: str = FD.OUT_SCORE, ascending: bool = True,
                         save_artifacts: bool = False, show: bool = True, **kwargs):
        """
        Create gallery of outliers, i.e. images that are not similar to any other images in the dataset.
        :param save_path: html file-name to save the gallery or directory if lazy_load is True,
            if None, save to fastdup work_dir
        :param num_images: number of images to display
        :param lazy_load: if True, load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe
        :param how: (Optional) outlier selection method.
            - one = take the image that is far away from any one image
                    (but may have other images close to it).
            - all = take the image that is far away from all other images. Default is one.
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param draw_bbox: if True, draw bounding box on the images
        :param slice: (Optional) list/single label for filtering outliers
        :param sort_by: (Optional) column name to sort the outliers by
        :param ascending: (Optional) sort ascending or descending
        :param save_artifacts: save artifacts to disk
        :param show: show gallery in notebook
        :param kwargs: additional parameters to pass to create_outliers_gallery
        """
        # get images to display
        temp = 'outliers'
        out_fname = 'outliers.html'
        save_dir, html_src_path, html_dst_path = self._infer_paths(users_input=save_path, temp_save_path=temp,
                                                                   out_html_fname=out_fname, lazy_load=lazy_load)
        df_outlier = self._controller.outliers(data=False).rename({'outlier': 'from', 'nearest': 'to'}, axis=1)
        if slice is not None and label_col is not None and label_col in df_outlier.columns:
            slice = [slice] if isinstance(slice, str) else slice
            df_outlier = df_outlier[df_outlier[label_col].isin(slice)]
        if sort_by in df_outlier.columns:
            df_outlier = df_outlier.sort_values(by=sort_by, ascending=ascending)

        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        create_outliers_gallery(df_outlier, work_dir=self._controller.work_dir, save_path=save_dir,
                                get_bounding_box_func=self._get_bbox_func() if draw_bbox else None,
                                get_label_func=self._get_label_func(label_col),
                                num_images=num_images,
                                lazy_load=lazy_load, how=how, max_width=max_width,
                                id_to_filename_func=self._get_filneme_func(),
                                get_display_filename_func=self._get_disp_filneme_func(),
                                get_extra_col_func=None, save_artifacts=save_artifacts,
                                jupyter_html=jupyter_html, **kwargs)
        self._clean_temp_dir(save_dir, html_src_path, html_dst_path, lazy_load=lazy_load, save_artifacts=save_artifacts)
        if show:
            self._disp_jupyter(html_dst_path)

    @v1_sentry_handler
    def duplicates_gallery(self, save_path: str = None, label_col: str = FD.ANNOT_LABEL, draw_bbox: bool = True,
                           num_images: int = 20, max_width: int = None, lazy_load: bool = False,
                           slice: Union[str, list] = None, ascending: bool = False, threshold: float = None,
                           save_artifacts: bool = False, show: bool = True, **kwargs):
        """
        Generate gallery of duplicate images, i.e. images that are similar to other images in the dataset.
        :param save_path: html file-name to save the gallery or directory if lazy_load is True,
            if None, save to fastdup work_dir
        :param num_images: number of images to display
        :param descending: display images with highest similarity first
        :param lazy_load: load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param draw_bbox: draw bounding box on the images
        :param slice: (Optional) list/single label for filtering outliers
        :param ascending: (Optional) sort ascending or descending
        :param threshold: (Optional) threshold to filter out images with similarity score below the threshold.
        :param save_artifacts: save artifacts to disk
        :param show: show gallery in notebook
        :param kwargs: additional parameters to pass to create_duplicates_gallery
        """
        # get images to display
        temp = 'duplicates'
        out_fname = 'duplicates.html'
        save_dir, html_src_path, html_dst_path = self._infer_paths(users_input=save_path, temp_save_path=temp,
                                                                   out_html_fname=out_fname, lazy_load=lazy_load)
        threshold = float(self._controller.config['fastdup_kwargs']['turi_param']['ccthreshold']) \
            if threshold is None else threshold
        df_sim = self._controller.similarity(data=False)
        if df_sim is None or df_sim.empty:
            print('No similar images found.')
            return
        df_sim = df_sim.query(f'{FD.SIM_SCORE}>={threshold}')
        if slice is not None and label_col is not None and label_col in df_sim.columns:
            slice = [slice] if isinstance(slice, str) else slice
            df_sim = df_sim[df_sim[label_col].isin(slice)]

        if df_sim is None or df_sim.empty:
            print('No duplicates found.')
            return
        # create gallery
        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        create_duplicates_gallery(df_sim, work_dir=self._controller.work_dir, save_path=save_dir, lazy_load=lazy_load,
                                  get_bounding_box_func=self._get_bbox_func() if draw_bbox else None,
                                  get_label_func=self._get_label_func(label_col),
                                  num_images=num_images, max_width=max_width, threshold=threshold,
                                  id_to_filename_func=self._get_filneme_func(), descending=not ascending,
                                  get_extra_col_func=None, save_artifacts=save_artifacts,
                                  jupyter_html=jupyter_html, **kwargs)
        self._clean_temp_dir(save_dir, html_src_path, html_dst_path, lazy_load=lazy_load, save_artifacts=save_artifacts)
        if show:
            self._disp_jupyter(html_dst_path)

    @v1_sentry_handler
    def similarity_gallery(self, save_path: str = None, label_col: str = FD.ANNOT_LABEL, draw_bbox: bool = True,
                           num_images: int = 20, max_width: int = None, lazy_load: bool = False,
                           slice: Union[str, list] = None, ascending: bool = False, threshold: float = None,
                           show: bool = True, **kwargs):
        """
        Generate gallery of similar images, i.e. images that are similar to other images in the dataset.
        :param save_path: html file-name to save the gallery or directory if lazy_load is True,
            if None, save to fastdup work_dir
        :param num_images: number of images to display
        :param descending: display images with highest similarity first
        :param lazy_load: load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param draw_bbox: draw bounding box on the images
        :param get_extra_col_func: (callable): Optional parameter to allow adding additional column to the report # TODO: add support
        :param threshold: (Optional) threshold to filter out images with similarity score below the threshold.
        :param slice: (Optional) list/single label for filtering similarity dataframe
        :param ascending: (Optional) sort ascending or descending
        :param show: show gallery in notebook
        :param kwargs: additional parameters to pass to create_duplicates_gallery
        """
        # get images to display
        temp = 'similarity'
        out_fname = 'similarity.html'
        save_dir, html_src_path, html_dst_path = self._infer_paths(users_input=save_path, temp_save_path=temp,
                                                                   out_html_fname=out_fname, lazy_load=lazy_load)
        df_sim = self._controller.similarity(data=False)
        if df_sim is None or df_sim.empty:
            print('No similar images found.')
            return
        if slice is not None and label_col is not None and label_col in df_sim.columns:
            slice = [slice] if isinstance(slice, str) else slice
            df_sim = df_sim[df_sim[label_col].isin(slice)]

        # create gallery
        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        create_similarity_gallery(df_sim, work_dir=self._controller.work_dir, save_path=save_dir, lazy_load=lazy_load,
                                  get_bounding_box_func=self._get_bbox_func() if draw_bbox else None,
                                  get_label_func=self._get_label_func(label_col),
                                  num_images=num_images, max_width=max_width, threshold=threshold,
                                  id_to_filename_func=self._get_filneme_func(), descending=not ascending,
                                  get_display_filename_func=self._get_disp_filneme_func(),
                                  get_extra_col_func=None, jupyter_html=jupyter_html, **kwargs)
        self._clean_temp_dir(save_dir, html_src_path, html_dst_path, lazy_load=lazy_load)
        if show:
            self._disp_jupyter(html_dst_path)

    def stats_gallery(self, save_path: str = None, metric: str = 'dark', slice: Union[str, list] = None,
                      label_col: str = FD.ANNOT_LABEL, lazy_load: bool = False, show: bool = True):
        """
        Generate gallery of images sorted by a specific metric.
        :param save_path: html file-name to save the gallery or directory if lazy_load is True,
            if None, save to fastdup work_dir
        :param metric: metric to sort images by (dark, bright, blur)
        :param slice: list/single label for filtering stats dataframe
        :param label_col: label column name
        :param lazy_load: load images on demand, otherwise load all images into html
        :param show: show gallery in notebook
        """
        ascending = False
        name = metric
        if metric == 'dark':
            metric = 'mean'
            ascending = True
        elif metric == 'bright':
            metric = 'mean'
            ascending = False
        if metric == 'blur':
            ascending = True
        # get images to display
        temp = 'stats'
        out_fname = f'{metric}.html'
        save_dir, html_src_path, html_dst_path = self._infer_paths(users_input=save_path, temp_save_path=temp,
                                                                   out_html_fname=out_fname, lazy_load=lazy_load)
        prefix, fname = Path(html_dst_path).parent, Path(html_dst_path).name
        html_dst_path = str(Path(prefix)/ fname.replace(metric, name))

        html_dst_path = str(Path(prefix)/ fname.replace(metric, name))
        df_stats = self._controller.img_stats()
        df_stats['filename'] = df_stats[FD.ANNOT_FD_ID].map(self._get_filneme_func())
        df_stats['index'] = df_stats[FD.ANNOT_FD_ID]
        if slice is not None and label_col is not None and label_col in df_stats.columns:
            slice = [slice] if isinstance(slice, str) else slice
            df_stats = df_stats[df_stats[label_col].isin(slice)]

        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        fastdup.create_stats_gallery(df_stats,
                                     save_path=save_dir,
                                     metric=metric, num_images=25, max_width=1000,
                                     input_dir=self._controller.input_dir,
                                     work_dir=self._controller.work_dir,
                                     lazy_load=lazy_load,
                                     descending=not ascending,
                                     jupyter_html=jupyter_html)
        self._clean_temp_dir(save_dir, html_src_path, html_dst_path, lazy_load=lazy_load)
        if show:
            self._disp_jupyter(html_dst_path)

    @v1_sentry_handler
    def component_gallery(self, save_path: str = None, label_col: str = FD.ANNOT_LABEL, draw_bbox: bool = True,
                          num_images: int = 20, max_width: int = None, lazy_load: bool = False,
                          slice: Union[str, list] = None, group_by: str = 'visual', min_items: int = None,
                          max_items: int = None, threshold: float = None, metric: str = None,
                          sort_by: str = 'comp_size', sort_by_reduction: str = None, ascending: bool = False,
                          save_artifacts: bool = False, show: bool = True, **kwargs):
        """

        :param save_path: html file-name to save the gallery
        :param num_images: number of images to display
        :param lazy_load: load images on demand, otherwise load all images into html
        :param label_col: column name of label in annotation dataframe
        :param group_by: [visual|label]. Group the report using the visual properties of the image or using the labels
            of the images. Default is visual.
        :param slice: (Optional) parameter to select a slice of the outliers file based on a specific label or a
            list of labels.
        :param max_width: max width of the gallery
        :param min_items: threshold to filter out components with less than min_items
        :param max_items: max number of items to display for each component
        :param draw_bbox: draw bounding box on the images
        :param get_extra_col_func: (callable): Optional parameter to allow adding additional column to the report # TODO: add support
        :param threshold: (Optional) threshold to filter out images with similarity score below the threshold.
        :param metric: (Optional) parameter to set the metric to use (like blur) for chose components. Default is None.
        :param slice: (Optional) list/single label for filtering  connected component
        :param sort_by: (Optional) 'area'|'comp_size'|ANY_COLUMN_IN_ANNOTATION
            column name to sort the connected component by
        :param sort_by_reduction: (Optional) 'mean'|'sum' reduction method to use for grouping connected components
        :param ascending: (Optional) sort ascending or descending
        :param show: show gallery in notebook
        :param save_artifacts: save artifacts to disk
        :param kwargs:
        :return:
        """

        # set save paths
        temp = 'components'
        out_fname = 'components.html'
        save_dir, html_src_path, html_dst_path = self._infer_paths(users_input=save_path, temp_save_path=temp,
                                                                   out_html_fname=out_fname, lazy_load=lazy_load)

        # get components and filter according to the user's input
        df_cc, _ = self._controller.connected_components(data=True)
        if df_cc is None or df_cc.empty:
            print('No components found.')
            return
        if slice is not None and label_col is not None and label_col in df_cc.columns:
            slice = [slice] if isinstance(slice, str) else slice
            df_cc = df_cc[df_cc[label_col].isin(slice)]

        # define aggregation & renaming functions
        rename_dict = {FD.ANNOT_FD_ID: 'files', 'mean_distance': 'distance', 'count': 'len'}
        agg_dict = {FD.ANNOT_FD_ID: list, 'mean_distance': 'mean', 'count': len}

        # add sort score to df_cc aggregation dict
        sort_by_reduction = 'mean' if sort_by_reduction is None else sort_by_reduction
        size_prefix = 'bbox' if self._controller.config['data_type'] == FD.BBOX else 'img'
        if sort_by == 'area' and {f"{size_prefix}_h", f"{size_prefix}_w"}.issubset(df_cc.columns):
            df_cc[sort_by] = df_cc[f"{size_prefix}_h"] * df_cc[f"{size_prefix}_w"]
        elif sort_by == 'comp_size':
            df_cc[sort_by] = 1
            sort_by_reduction = 'sum'
        elif sort_by not in df_cc.columns:
            sort_by = None
        if sort_by is not None:
            agg_dict[sort_by] = sort_by_reduction

        # set format of components dataframe to be used for the gallery
        group_by = label_col if group_by == 'label' and label_col is not None and label_col in df_cc.columns \
            else 'component_id'
        top_components = df_cc.groupby(group_by).agg(agg_dict).reset_index().rename(rename_dict, axis=1)
        top_components = top_components if sort_by is None else top_components.sort_values(sort_by, ascending=ascending)

        # filter and sort components
        min_items = 1 if min_items is None else min_items
        max_items = int(top_components['len'].max()) if max_items is None else max_items
        top_components = top_components.query(f'{min_items}<=len<={max_items}')
        if top_components.empty:
            print(f'No components found for {min_items}<=componentet_size<={max_items}')
            return

        # create gallery
        jupyter_html = 'JPY_PARENT_PID' in os.environ and show
        create_components_gallery(work_dir=top_components, save_path=save_dir, input_dir=self._controller.work_dir,
                                  get_bounding_box_func=self._get_bbox_func() if draw_bbox else None,
                                  get_label_func=self._get_label_func(label_col), save_artifacts=save_artifacts,
                                  lazy_load=lazy_load, max_items=max_items, max_width=max_width, metric=metric,
                                  threshold=threshold, num_images=num_images, min_items=min_items,
                                  get_extra_col_func=None, id_to_filename_func=self._get_filneme_func(),
                                  jupyter_html=jupyter_html, **kwargs)
        self._clean_temp_dir(save_dir, html_src_path, html_dst_path, lazy_load=lazy_load, save_artifacts=save_artifacts)
        if show:
            self._disp_jupyter(html_dst_path)

    def _get_filneme_func(self):
        def get_filename_func(fastdup_id):
            return str(Path(self._controller.input_dir) / self._controller[fastdup_id][FD.ANNOT_FILENAME])
        return get_filename_func

    def _get_disp_filneme_func(self):
        def get_disp_filename_func(fastdup_id):
            return str(self._controller[fastdup_id][FD.ANNOT_FILENAME])
        return get_disp_filename_func

    def _get_label_func(self, label_col):
        """
        Get a function that returns the label of an image based on the annotation dataframe
        :param df_annot: fastdup annotation dataframe
        :param input_dir: filenema base-dir to add to the filename in the annotation dataframe
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

    def _get_bbox_func(self):
        """
        Get a function that returns the bounding box of an image based on the annotation dataframe
        :param df_annot: fastdup annotation dataframe
        :param input_dir: filenema base-dir to add to the filename in the annotation dataframe
        :return:
        """
        # check if the annotation dataframe has the bounding box columns
        bbox_cols = {FD.ANNOT_BBOX_X, FD.ANNOT_BBOX_Y, FD.ANNOT_BBOX_W, FD.ANNOT_BBOX_H}
        if not bbox_cols.issubset(self._get_available_columns()):
            return None
        # create mapping from filename to bounding box
        # return  [x1, y1, x2, y2]
        def bbox_func(fastdup_id):
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

