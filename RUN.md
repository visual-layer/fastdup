
##### Table of Contents  

1. [Running the code](#run)  
2. [Input/Output](#input)
3. [Error handling](#error)
4. [Nearest neighbors](#nn)
5. [Visualization](#visualization)
6. [Resuming a stored run](#resume)
7. [Performing vector search](#external)
8. [Support for cloud storage](#s3)
9. [Working with tar/zip files as input](#tar)
10. [Working with video](#video)
11. [Debugging fastdup](#debug)

## Detailed Python API documentation <a name="run"/>
The main function of fastdup is `run`. It works by extracting short feature vectors from each image, clsutering the images together using a nearest neighbor model which computes similarities of pairs of images. Then a graph is formed to deduce the network structure of local similarities. The input/ outputs are described below in the section Input/Output. 

```
    Run fastdup tool for find duplicate, near duplicate images, outlier images and clusters of similar iamges in a corpus of images. 
    The only mandatory argument is image_dir. Given an image directory it will compare all pairs of images and store the most similar ones in the output file output_similarity.

    Parameters:
        input_dir (str): Location of the images/videos to analyze.
        - A local folder
        - A remote folder (s3 or minio starting with minio://)
        - A file containing absolute filenames each on its own row
        - A python list with absolute filenames
        - yolov5 yaml input file containing train and test folders (single folder supported for now)
        We support jpg, jpeg, tiff, tif, giff, png, mp4, avi. In addition we support tar, tar.gz, tgz and zip files containing images.
        If you have other image extensions that are readable by opencv imread() you can give them in a file and then we do not check for the
        known extnsions.
        Note: it is not possible to mix compressed (videos or tars/zips) and regular images. Use the flag turi_param='tar_only=1' if you want to ignore images and run from compressed files in case your folders are mixed.

        work_dir (str): Working directory for saving intermediate results and outputs. Default is local folder ('.').

	test_dir (str): Optional path for test data. When given similarity of train and test images is compared (vs. train/train or test/test which are not performed).

        compute (str): Compute type [cpu|gpu] default is cpu.

        verbose (boolean): Verbosity. Default is False.

        num_threads (int): Number of threads. Default is -1 to be auto configured by the number of cores.

        num_images (int): Number of images to run on. Default is -1 which means run on all the images in the image_dir folder.

        turi_param (str): Optional additional parameters for turi. Supported paramets are:
        - nnmodel=0|1|2 Nearest Neighbor model for clustering the features together, when using turi (has no effect when using nnf). Supported options are 0=brute_force (exact), 1=ball_tree and 2=lsh (both approximate). Default is brute_force.
        - ccthreshold=XX where XX in the range [0,1]. Construct similarities graph when the similarity > XX.
        - run_cc=0|1 Distable/enable connected components computation on the graph of similarities.
        - run_pagerank=0|1 Disable/enable pagerank computation on the graph of similarities.
        - run_degree=0|1 Distable/enable degree distribution computation on the graph of similarities
   	- store_int=0|1 store the similarity as string filenames or string index of the file id (to save space)
        - tar_only=0|1 run only on tar files and ignore images in folders. Default is 0.
        - delete_tar=0|1 when working with tar files obtained from cloud storage delete the tar after download
        - delete_img=0|1 when working with images obtained from cloud storage delete the image after download
	Example run: turi_param='nnmodel=0,ccthreshold=0.99'

        distance (str): Distance metric for the Nearest Neighbors algorithm. Default is cosine. Other distances are euclidean, squared_euclidean, manhattan.

        threshold (float): Similarity measure in the range 0->1, where 1 is totally identical, 0.98 and above is almost identical, and 0.85 and above is very similar. Default is 0.85 which means that only image pairs with similarity larger than 0.85 are stored.

        lower_threshold (float): Similarity measure to outline images that are far away (outliers) vs. the total distribution. Default value is 0.3.

        model_path(str): Optional location of ONNX model file, should not be used.

        version(bool): Print out the version number. This function takes no argument.

        nearest_neighbors_k (int): For each image, how many similar images to look for. Default is 2.

        run_mode (int): This software can run for either feature vector extraction and similarity measurement (0), or just feature vector extraction (1), or just similarity measure computation (2). 
 
   nn_provider (string): Provider of the nearest neighbor algorithm, allowed values are turi|nnf.

        min_offset (int): Optional min offset to start iterating on the full file list. Default is -1.

        max_offset (int): Optional max offset to start iterating on the full file list. Default is -1.

 	nnf_mode (str): When nn_provider='nnf' selects the nnf mode. Supported options are HNSW32 and any other nnf string.

   	nnf_param (str): When nn_provider='nnf' assigns optional nnf parameters. For example efSearch=175. Multiple params are supported - for example 'efSearch=175,nprobes=200'

	bounding_box (str): Optional bounding box for cropping images before the fastdup tool is applied. For example bounding_box='rows=100,cols=100,width=250,height=310'. Rows and cols gives the top left corner coordinates, and width and height the bounding box dimensions. (Row is the y axis and col is the x axis. The box is cropped in the range [rows:rows+height, cols:cols+width].

       
        
    Returns:
        Status code 0 = success, 1 = error.
```

## Input / output formats <a name="input"/>  

The input to fastdup tool is given in the command line argument: `input_dir`. There are a few options:
- Location of a local folder. In that case all images in this folder are searched recursively. Supported images are jpeg, jpg, tiff, tif, giff, png. 
- Location of an `s3` path. Again all images in the path will be used recursively.
- Location of a minio client path. Path should start with minio:// for example `minio://google/my_bucket/my_folder`.
- A file containing image locations (either local or full `s3` paths). Each image filename in its own row.
- In addition we support tar, tgz, tar.gz and zip files containing images. Those files could be either local, or on s3 bucket or minio clinet.

The intermediate outputs and final outputs are stored in the folder `work_dir`.

### Feature extraction related files:

Binary numpy array containing `n` rows (where `n` is the number of images) of 576 columns with the feature vectors. (Default filename is `features.dat`)
An additional csv file containing the full paths to the image names corresponding to the feature vectors (default filename is `features.dat.csv`). Both those files are linked to each other. The reason we save the list of filenaes is that theh order of extraction may change depends on the file system listing. In addition, in case of corrupted images, its feature vector is skipped and not generated. In that case an additional output file is provided ( `features.bad.csv`). This file lists all the bad or corrupted images that were skipped. 

Note: for using the binary features we provide the following function in Python:

```
def load_binary_feature(filename):

    Example Python function for loading the stored binary features and their matching filenames.

    Parameters:
        filename(str):The binary feature file location

    Returns:
        A list of with all image file names of length X.
        An np matrix of shape X rows x 576 cols. Each row conform to feature vector os a single image.

    Note: 
       When using tar, tgz, tar.gz or zip files, for each file a different features.dat and feature.dat.csv are created.

    Example:
        import fastdup
        file_list, mat_features = fastdup.load_binary('features.dat')

```


### Similarity pair list

The output of the fastdup tool is a similarity file (filename is `similarity.csv`) which is a csv file with 3 columns: `from`, `to`, `distance`. The file is sorted from the closest matching images to less similar images. Similarity scores are between 0 and 1 where 1 is identical.

Example similarity file:

```
$ head similarity.csv
from,to,distance
/mnt/data/sku110k/train_4690.jpg,/mnt/data/sku110k/train_1720.jpg,0.920244
/mnt/data/sku110k/val_202.jpg,/mnt/data/sku110k/train_4109.jpg,0.918458
```
Note that the number of image pairs inside the simiarity file is controlled by the `threshold` parameter. For example when `threshold=0.8` only image pairs with cosine similarity equal or larger than 0.8 are stored in the similarity file. 

A second output of the fastdup tool is outlier file (filename is `outliers.csv`) which is a csv file with 3 columns: `from`,`to`,`distance`. The file is sorted from least outliers on top to the most outliers in the bottom. 

Note that the number of image pairs inside the outlier file is controlled by the `lower_treshold` parameter which is the percentage. For example `lower_threshold=0.05` means that 5% outliers out of the total pairs of edges computed are stored in the outliers output file. When increasing `k` the number of nearest neighbor edges, the number of outliers increses as well since the outliers is from the number of pairs computed and not from the number of images.

Example outliers file:

```
$ head outliers.csv
from,to,distance
/mnt/data/sku110k/train_7010.jpg,/mnt/data/sku110k/train_8110.jpg,0.54312
/mnt/data/sku110k/train_7995.jpg,/mnt/data/sku110k/train_465.jpg,0.44212
```


### NNF index files

When using nnf an additional intermediate results file is created: `nnf.index`. This file is stored according to the nnf format. This file contains the trained nearest neighbor model. It is possible to resume a stored run and reading the trained nearest neighbor model from disk. See run_mode documentation.

### Graph computation

Following the image feature extraction and the contruction of the nearest neighbor model of paris of similar images, fastdup runs a connected component algorithm to cluster similar images together in groups. See [turi create](https://apple.github.io/turicreate/docs/api/generated/turicreate.connected_components.create.html). The connected components algorithm takes the pairs computed by fastdup in the `similarity.csv` file, builds a graph structure from them, and computes connected clusters of images.

- A file named `components_info.csv` is created with number of nodes (=images) per component (=cluster of images).

Example:

```
$ head component_info.csv
component_id,Count
0,1
1,1
2,1
3,1
```

- `component_id` is the integer component of the cluster of images. This is just an id, the number has no meaning.
- `Count` the number of images clusterd together on this cluster. This is just a stastics, the actual mapping between component id and all the images on that component id cluster is explained below.

- A file named `connected_components.csv` includes the output of [pagerank](https://apple.github.io/turicreate/docs/api/generated/turicreate.pagerank.create.html), degree distribution and connected component assignments. The first column is the index in the `features.dat.csv` file (the image list). This file is sorted according to the list.
Example:

```
$ head connected_components.csv
__id,component_id,pagerank,delta
0,7,0.15,0
1,16,0.15,0
2,19,0.15,0
3,14,0.15,0
4,19,0.15,0
```

- `__id` - the offset in the `features.dat.csv` image list file. Offset starts from 0.
- `component_id` - the id of the component, a positive integer. If two images have the same component_id it seems they have a graph connection
- `pagerank` - pagerank sccore of the image based on the similarity graph
- `delta` - pagerank related change

In the above example, both image 2 and 4 are part of component 19, images 0,1,3 have their own unique component, which means they are not connected to the other images in the graph. For experimenting with differnet component thresholds, change the parameter `ccthreshold` which is given inside `turi_param='ccthreshold=0.88'` flag. When the `ccthreshold` is higher, less images are grouped together and more components are created. When the ccthreshold is lower, more images are grouped together and less components are created.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/viz42.png)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/viz46.png)
*Exaple components obtained from the ImageNet dataset using ccthreshold=0.96*


## Error handling <a name="error"/>

When bad images are encountered, namely corrupted images that can not be read, an additional csv output file is generated called features.dat.bad. The bad images filenames are stored there. In addition there is a printout that states the number of good and bad images encountered. The good images filenames are stored in the file features.dat.csv file. Namely the bad images are excluded from the total images listing. The function fastdup.load_binary_features() reads the features corresponding to the good images and returns a list of all the good images, and a numpy array of all their corresponding features.
The output file similarity.csv with the list of all similar pairs does not include any of the bad images.



## Nearest neighbor search <a name="nn"/>

Once short feature vectors are generated per each image, we cluster them to find similarities using a nearest neighbor method. FastDup supports two families of algorithms (given using the nn_provider command line argument)
- turi
- nnf
Turi (nn_provider='turi') has the following methods inside
- nnmodel='brute_force' (exact method but may be slower)
- nnmodel='ball_tree' (approximate method)
- nnmodel='lsh'  (locality sensitive hashing, approximate method)
NNF (nn_provider='nnf') supports multiple methods
- nnf_mode='HSNW32' the default





Example command line:
```
> import fastdup
> fastdup.run('/path/to/folder', nn_provider='turi', nnmodel='brute_force')
> fastdup.run('/path/to/folder', nn_provider='nnf')
```


## Visualizing the outputs <a name="visualization"/>

The following command creates the html report:
```
def create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True, lazy_load=False, get_label_func=None, slice=None):

    Function to create and display a gallery of images computed by the similarity metrics

    Parameters:
        similarity_file (str): csv file with the computed similarities by the fastdup tool
        save_path (str): output folder location for the visuals
        num_images(int): Max number of images to display (deafult = 50)
        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.
	lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).
        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.
	slice (str): Optional parameter to select a slice of the duplicates based on a specific label.
```

The html generated had 3 images in each row. To the left and middle are the two similar images, and the third image to the right is a superposition of both images on top of each other to show the differnces more easy.

A second function create an html report gallery of outliers:
```
def create_outliers_gallery(similarity_file, save_path, num_images=20, descending=True, lazy_load=False, get_label_func=None, slice=None):

    Function to create and display a gallery of images computed by the outliers metrics

    Parameters:
        outliers_file (str): csv file with the computed outliers by the fastdup tool
        save_path (str): output folder location for the visuals
        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.
        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).

        get_label_func (callable): Optional parameter to allow adding more image information to the report like the image label. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.
	how (str): Optional outlier selection method. one = take the image that is far away from any one image (but may have other images close to it).
                                                      all = take the image that is far away from all other images. Default is one.
        slice (str): Optional parameter to select a slice of the outliers file based on a specific label.
```

Another function creates an html report gallery of components. Components are clusters of similar images (more than two).
```
def create_components_gallery(work_dir, save_path, num_images=20, lazy_load=False, get_label_func=None, group_by='visual', slice=None):

    Function to create and display a gallery of images for the largest graph components

    Parameters:
        work_dir (str): path to fastdup work_dir
        save_path (str): output folder location for the visuals
        num_images(int): Max number of images to display (default = 50). Be careful not to display too many images at once otherwise the notebook may go out of memory.

        lazy_load (boolean): If False, write all images inside html file using base64 encoding. Otherwise use lazy loading in the html to load images when mouse curser is above the image (reduced html file size).
        get_label_func (callable): optional label string, given a absolute path to an image return the label for the html report
        group_by (str): [visual|label]. Group the report using the visual properties of the image or using the labels of the images. Default is visual.
	 slice(str) : optional label to draw only a subset of the components conforming to this label.
```
     

Command line example for the html report generation:
```
import fastdup
fastdup.create_duplicates_gallery('/path/to/similarity.csv', save_path='/path/to/report/')
```

Note: the report should be generated on the same machine since we assume that the input folder for reading the images exists under the same location.

Another form of visualization is using TensorBoard Projector as shown in [this notebook](https://bit.ly/3ydvtVJ). 

The following function saves the data in a ready format for TensorBoard projector:

```
def export_to_tensorboard_projector(work_dir:str, log_dir:str, sample_size:int = 900,
                                    sample_method:str='random', with_images=True, get_label_func=None):
    Export feature vector embeddings to be visualized using tensorboard projector app.
    work_dir: work_dir where fastdup results are stored
    log_dir: output dir where tensorboard will read from
    sample_size: how many images to view
    sample_method: how to sample, currently 'random' is supported
    with_images:bool add images to the visualization (default True)
    get_label_func (callable): Optional parameter to allow adding class label name to the image. This is a function the user implements that gets the full file path and returns html string with the label or any other metadata desired.
    :return:
```

After storing the data you should run (in a Jupyter notebook)
```
%load_ext tensorboard
%tensorboard --logdir=<log_dir>
```

## Advanced topics: resuming a stored run <a name="resume"/>

There are several supported running modes:
- `run_mode=0` (the default) does the feature extraction and NN embedding to compute all pairs similarities.
It uses the `input_dir` command line argument for finding the directory to run on (or a list of files to run on). 
The features are extracted and saved into the `working_dir` path  (the default features out file nme is `features.dat`
in the same folder for storing the numpy features and `features.dat.csv` for storing the image file names corresponding to the numpy features).
For larger dataset it may be wise to split the run into two, to make sure intermediate results are stored in case you encounter an error. 
- `run_mode=1` computes the extracted features and stores them, does not compute the NN embedding. For large datasets, 
it is possible to run on a few computing nodes, to extract the features, in parallel. Use the `min_offset` and `max_offset` flags to allocate a subset of the images for each computing node. Offsets start from 0 to `n-1` where `n` is the number of images in the input_dir folder.
- `run_mode=2` reads a stored feature file and computes the NN embedding to provide similarities. The `input_dir` param is ignored, and the `work_dir` is used to point to the numpy feature file. (Give a full path and filename).
- `run_mode=3` Reads the NN model stored by `nnf.index` from the `work_dir` and computes all pairs similarity on all inages give by the `test_dir` parameter. `input_dir` should point to the location of the train data. This mode is used for scoring similarities on a new test dataset given a precomputed simiarity index on a train dataset.
- `run_mode=4` reads the NN model stored by `nnf.index` from the `work_dir` and computes all pairs similarity on pre extracted feature vectors computer by `run_mode=1`.  

## Advanced topics: vector search <a name="external"/>

It is possible to use fastdup for feature vector search. We assume you have feature vectors computed for your training data. Save your computed features using  `fastdup.save_binary_feature(save_path, filenames, np_array)`. Where `save_path` is the folder you like to run from, `filenames` is a list of aboslute paths of the images of length `n`, and `np_array` is a matrix of size `n x d` where `d` is the feature vector length. Note that the `np_array` should be of type `'float32'`.  

Next run fastdup with `run_mode=2` (which skips the image extraction phase and loads your stored features instead) and make sure to point `work_dir` to the `save_path` which is the location of your stored features. Don't forget to assign `d` to your feature length.

Example:
```python
import os
import fastdup 
import shutil
import numpy as np

filesnames = ['a.jpg', 'b.jpg', 'c.jpg']
d = 20 # feature length
train_n = 3 # number of images
work_dir = '/path/to/train_dir'  # temp working directory
train_array = np.random.rand(train_n,d) # replace this placeholder with your own features

#export the feature in fastdup readable format
fastdup.save_binary_feature(work_dir, filenames, train_array)
# build the NN model and store it to work_dir/nnf.index
fastdup.run(os.path.join(work_dir, 'atrain_features.dat.csv'), work_dir=work_dir, d=d, run_mode=2)
...
# Now search for test images using your precomputed features
test_filenames = ['d.jpg', 'e.jpg']
test_n = 2
test_dir = 'path/to/test_dir'
test_array = np.random.rand(test_n, d) # replace placeholder this with your own test features
fastdup.save_binary_feature(test_dir, test_filenames, test_array)
shutil.copy(os.path.join(work_dir, 'nnf.index'), test_dir)
fastdup.run(os.path.join(work_dir, 'atrain_features.dat.csv'), work_dir=test_dir, d=d ,run_mode=4)
fastdup.create_duplicates_gallery(os.path.join(test_dir, 'similarity.csv'))
```


## Support for s3 cloud/ google storage <a name="s3"/>

[Detailed instructions](CLOUD.md)

## Working with tar/tgz/zip files as input <a name="tar"/>

Some popular datasets like [LAOIN 400M](https://laion.ai/laion-400-open-dataset/) use webdataset compressed formats. Fastdup supports the following compressed file formats: `tar,tgz,tar.gz,zip`. Those compressed files can be located in a local folder or remote s3 or minio path.

For example, the LAOIN dataset contains the following tar files:

```
00000.tar containing:
000000000.jpg
000000001.jpg
000000002.jpg
...
```
Each tar file contains 10K images.

When working with compressed files you need to run with `run_mode=1` for performing the extraction of feature vectors first, since we do not know ahead how many files are in each tar when copied from s3. After the feature vectors are extracted, collect all the output files into the same folder and run again with `run_mode=2` to compute the NN model.

The compressed files are first copied locally into the `/tmp/<tarname>/` folder and then extracted. For each compressed tar file we generate two output files: `<tarname>features.dat` for the binary features and `<tarname>features.dat.csv` for the file list.

Example output file for the tar above (the path is given via the `work_dir` command line argument).

```
$ cat 00000.tarfeatures.dat.csv
filename
/tmp/00000.tar/000000000.jpg
/tmp/00000.tar/000000001.jpg
...
```

Note that it is possible to configure the behaviour regarding deletion of files. On default, both the downloaded tar files and the extracted images are deleted after the feature vectors are extracted. If you want to keep them locally (assuming there is large enough hard drive) you can run with : 
```
... turi_param='delete_tar=0,delete_img=0'
```
This keeps all the downloaded tars and images in the /tmp folder.

Running example. Assume you got to the full dataset downloaded into `s3://mybucket/myfolder`. In total there are 40,000 tar files. Further assume you want to run using 20 compute nodes to extract the feature in parallel. In this case you cam run:

```python
import fastdup
fastdup.run('s3://mybucket/myfolder', run_mode=1, work_dir='/path/to/work_dir',
            min_offset=0, max_offset=2000)

```
The first job runs on 2000 tars from 0 to 2000 not including. In parallel you can run with `min_offset=2000, max_offset=4000` on another node etc. We estimate the extraction speed in around 4M images per hour on a 32 core machine (no GPU).

Once all jobs are finished, collect all the output files from the `work_dir` into a single location and run:

```python
import fastdup
fastdup.run('', run_mode=2, work_dir='/path/to/work_dir')
```

For running on 50M images you will need an ubuntu machine with 32 cores and 256GB RAM. We are working on further scaling the implementation for the full dataset - stay tuned!



## Running video <a name="video"/>

fastdup supports video in mp4 and avi formats. For other formats please reach out. For running on video you need to install ffmpeg.

On Ubuntu
```bash
sudo apt install ffmpeg
```

On Mac
```bash
brew install ffmpeg
```

Note: on Mac 10.14 we encountered brew error, you can download statically compile ffmpeg [here](https://evermeet.cx/ffmpeg/).

Currently we extract frame 1 per sec, please reach out if you need other support. Our video tutorial is found here:
- [🔥Analyzing video of the MEVA dataset - Google Colab](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup_video.ipynb)


## Debugging fastdup <a name="debug"/>

To debug program execution the following is recommended
- Make sure you have upgraded fastdup to the latest version, we release versions a couple of times a week.
- It is recommneded to debug in a python shell (and not in a Jupyter notebook)
- Run with `verbose=1` to get additional traces
- Run with `num_images=10` to run on a small subset of your data before running on the full dataset.
- If the issue persist please join our [Slack Channel]("https://join.slack.com/t/visualdatabase/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA")
