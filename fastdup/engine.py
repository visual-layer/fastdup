import tempfile
from pathlib import Path
from typing import Union, Iterable
import pandas as pd
from fastdup.fastdup_controller import FastdupController
from fastdup.fastdup_galleries import FastdupVisualizer


class Fastdup(FastdupController):
    """
    This class provides all fastdup capabilities as a single class.
    Usage example
    =============


    # >>> from fastdup.engine.Fastdup
    # >>> import fastdup.fastdup_constants as FD
    # >>>
    # >>> annotation_csv = '/path/to/annotation.csv'
    # >>> data_dir = '/path/to/images/'
    # >>> output_dir = '/path/to/fastdup_analysis'
    # >>>
    # >>> fdp = FastDupProxy(work_dir=output_dir)
    # >>> fdp.run(input_dir=data_dir,
    # >>>         df_annot=pd.read_csv(annotation_csv))
    # >>>
    # >>>
    # >>> df_sim  = fdp.similarity(data=True)
    # >>> im1_id, im2_id, sim = df_sim.iloc[0]
    # >>> annot_im1, annot_im2 = fdp[im1_id], fdp[im2_id]
    # >>>
    # >>> df_cc, cc_info = fd.connected_components(data=True)
    # >>>
    """

    def __init__(self, work_dir: Union[str, Path], input_dir: Union[str, Path] = None):
        super().__init__(work_dir, input_dir=input_dir)
        self.vis = FastdupVisualizer(self)

    def run(self,
            input_dir: Union[str, Path, list] = None,
            annotations: pd.DataFrame = None,
            embeddings=None,
            subset: list = None,
            data_type: str = 'infer',
            overwrite: bool = True,
            model_path=None,
            distance='cosine',
            nearest_neighbors_k: int = 2,
            threshold: float = 0.9,
            outlier_percentile: float = 0.05,
            num_threads: int = None,
            num_images: int = None,
            verbose: bool = False,
            license: str = None,
            high_accuracy: bool = False,
            cc_threshold: float = 0.96,
            **kwargs):
        """
        :param input_dir: Location of the images/videos to analyze
                - A folder
                - A remote folder (s3 or minio starting with s3:// or minio://). When using minio append the minio
                  server name for example minio://google/visual_db/sku110k
                - A file containing absolute filenames each on its own row                                              TODO: add support for multiple folders
                - A file containing s3 full paths or minio paths each on its own row                                    TODO: add support for multiple folders
                - A python list with absolute filenames
                - A python list with absolute folders, all images and videos on those folders are added recursively
                - yolo-v5 yaml input file containing train and test folders (single folder supported for now)           TODO: add support for yolov5 yaml file
                - We support jpg, jpeg, tiff, tif, giff, heif, heic, bmp, png, mp4, avi.
                  In addition we support tar, tar.gz, tgz and zip files containing images

            If you have other image extensions that are readable by opencv imread() you can give them in a file
            (each image on its own row) and then we do not check for the known extensions and use opencv
            to read those formats

            Note: It is not possible to mix compressed (videos or tars/zips) and regular images.
                  Use the flag tar_only=True if you want to ignore images and run from compressed files
            Note: We assume image sizes should be larger or equal to 10x10 pixels.
                  Smaller images (either on width or on height) will be ignored with a warning shown
            Note: It is possible to skip small images also by defining minimum allowed file size using
                  min_file_size=1000 (in bytes)
            Note: For performance reasons it is always preferred to copy s3 images from s3 to local disk and then
                  run fastdup on local disk. Since copying images from s3 in a loop is very slow, Alternatively you can
                  use the flag sync_s3_to_local=True to copy ahead all images on the remote s3 bucket to disk
        :param annotations: Optional dataframe with annotations.
        :param subset: List of images to run on. If None, run on all the images/bboxes.
        :param data_type: Type of data to run on. Supported types: 'image', 'bbox'. Default is 'infer'.
        :param model_path: path to model for feature extraction. supported formats: onnx, ort.
            Make sure to update d parameter acordingly.
        :param distance: - distance metric for the Nearest Neighbors algorithm.
            The default is 'cosine' which works well in most cases. For nn_provider='nnf' the following distance metrics
            are supported. When using nnf_mode='Flat': 'cosine', 'euclidean', 'l1','linf','canberra',
            'braycurtis','jensenshannon' are supported. Otherwise 'cosine' and 'euclidean' are supported.,
        :param num_images: Number of images to run on. On default, run on all the images in the image_dir folder.
        :param nearest_neighbors_k:
        :param high_accuracy: Compute a more accurate model. Runtime is increased about 15% and feature vector storage
            size/ memory is increased about 60%. The upside is the model can distinguish better of minute details in
            images with many objects.
        :param outlier_percentile: Percentile of the outlier score to use as threshold. Default is 0.5 (50%).
        :param threshold: Threshold to use for the graph generation. Default is 0.9.
        :param cc_threshold: Threshold to use for the graph connected component. Default is 0.96.
        :param bounding_box: Optional bounding box to crop images, given as                                             TODO: internal bounding box or global bounding box?
            bounding_box='row_y=xx,col_x=xx,height=xx,width=xx'. This defines a global bounding box to be used
            for all images. Alternatively, it is possible to set bounding_box='face' to crop the face from the image
            (in case a face is present). For the face crop the margin around the face is defined by
            augmentation_horiz=0.2, augmentation_vert=0.2 where 0.2 mean 20% additional margin around the
            face relative to the width and height respectively. It is possible to change the margin,
            lower allower value is 0 (no margin) and upper allowed  value is 1. Default is 0.2.
        :param num_threads: Number of threads. By default, autoconfigured by the number of cores.
        :param license: Optional license key. If not provided, only free features are available.
        :param overwrite: Optional flag to overwrite existing fastdup results.
        :param verbose: Verbosity.
        :param kwargs: Additional parameters for fastdup.
        :return:
            -  d: Model Output dimension. Default is 576.
            -  min_offset: Optional min offset to start iterating on the full file list.
            - max_offset: Optional max offset to start iterating on the full file list.
            - nnf_mode: When nn_provider='nnf' selects the nnf model mode. default is HSNW32. More accurate is Flat
            - nnf_param: When nn_provider='nnf' selects assigns optional parameters.
                - num_em_iter=XX: number of KMeans EM iterations to run. Default is 20.
                - num_clusters=XX: number of KMeans clusters to use. Default is 100.
            - batch_size = None,
            - resume: Optional flag to resume tar extraction from a previous run.
            - run_cc = Run connected components on the resulting similarity graph. Default is True.
            - run_sentry = Default is True.,
            - delete_tar = Delete tar after download from s3/minio.
            - delete_img = Delete images after download from s3/minio.
            - tar_only = When working with tar files obtained from cloud storage delete the tar after download
            - run_stats = When working with images obtained from cloud storage delete the image after download
            - sync_s3_to_local = In case of using s3 bucket sync s3 to local folder to improve performance.
                Assumes there is enough local disk space to contain the dataDefault is False.
        """

        # TODO: Make sure work with s3 and minio is working
        input_dir = self._input_dir if input_dir is None else input_dir
        fastdup_func_params = dict(ccthreshold=cc_threshold,
                                   lower_threshold=outlier_percentile,
                                   distance=distance,
                                   nearest_neighbors_k=nearest_neighbors_k,
                                   threshold=threshold,
                                   num_threads=-1 if num_threads is None else num_threads,
                                   num_images=0 if num_images is None else num_images,
                                   verbose=verbose,
                                   license='' if license is None else license,
                                   high_accuracy=high_accuracy)
        if model_path is not None:
            assert 'd' in kwargs, 'Please provide d parameter to indicate the model output dimension'
            fastdup_func_params['model_path'] = model_path
        fastdup_func_params.update(kwargs)

        super().run(annotations=annotations, input_dir=input_dir, subset=subset, data_type=data_type,
                    overwrite=overwrite, embeddings=embeddings, **fastdup_func_params)
