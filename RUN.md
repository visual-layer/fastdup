
##### Table of Contents  

1. [Running the code](#run)  
2. [Input/Output](#input)
3. [Error handling](#error)
4. [Nearest neighbors](#nn)
5. [Visualization](#visualization)
6. [Clustering](#clustering)
7. [Resuming a stored run](#resume)
8. [Performing vector search](#external)
9. [Comparing train/test](#train_test)
10. [Support for cloud storage](#s3)
11. [Working with tar/zip files as input](#tar)
12. [Working with video](#video)
13. [Using your own onnx model as feature extractpr](#onnx)
14. [Extracting statistics about image dataset](#stats)
15. [Debugging fastdup](#debug)

## Detailed Python API documentation <a name="run"/>

[Documentation pages](https://visualdatabase.github.io/fastdup/)


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

Note: for using the binary features we provide the [following function](https://visualdatabase.github.io/fastdup/#fastdup.load_binary_feature).

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

[Documentation pages](https://visualdatabase.github.io/fastdup/#fastdup-visualization-of-results)


## Clustering <a name="clustering"/>

[Clustering images using kmeans and connected components algorithm](https://www.kaggle.com/graphlab/fastdup-kmeans)


## Exporting to tensorboard projects

Use the function `fastup.export_to_tensorboard_projector` descrbed [here](https://visualdatabase.github.io/fastdup/#fastdup.export_to_tensorboard_projector).


After storing the data you should run (in a Jupyter notebook)
```
%load_ext tensorboard
%tensorboard --logdir=<log_dir>
```

## Advanced topics: resuming a stored run <a name="resume"/>

[See run_mode=2 documentation](https://visualdatabase.github.io/fastdup/#fastdup.run)
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


## Comparing train/test datasets <a name="train_test"/>
It is possible to run fastdup to compare train and test datasets. In this case only similarities between the train and test are computed.
To run in this mode, point input_dir to the train folder, test_dir to the test folder, and work_dir is the intermediate artifacts and output of the results.\
The resulting similarity.csv file contains only relations between test images to train images (and not internal test images or internal train images).


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
train_array = np.zeros((train_n, d), dtype='float32')# replace this placeholder with your own features

#export the feature in fastdup readable format
fastdup.save_binary_feature(work_dir, filenames, train_array)
# build the NN model and store it to work_dir/nnf.index
fastdup.run(os.path.join(work_dir, 'atrain_features.dat.csv'), work_dir=work_dir, d=d, run_mode=2)
...
# Now search for test images using your precomputed features
test_filenames = ['d.jpg', 'e.jpg']
test_n = 2
test_dir = 'path/to/test_dir'
train_array = np.zeros((train_n, d), dtype='float32') # replace placeholder this with your own test features
fastdup.save_binary_feature(test_dir, test_filenames, test_array)
shutil.copy(os.path.join(work_dir, 'nnf.index'), test_dir)
fastdup.run(os.path.join(work_dir, 'atrain_features.dat.csv'), work_dir=test_dir, d=d ,run_mode=4)
fastdup.create_duplicates_gallery(os.path.join(test_dir, 'similarity.csv'))
```


## Support for s3 cloud/ google storage <a name="s3"/>

[Detailed instructions](CLOUD.md)

## Working with tar/tgz/zip files as input <a name="tar"/>

Some popular datasets like [LAION 400M](https://laion.ai/laion-400-open-dataset/) use webdataset compressed formats. Fastdup supports the following compressed file formats: `tar,tgz,tar.gz,zip`. Those compressed files can be located in a local folder or remote s3 or minio path.

For example, the LAION dataset contains the following tar files:

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
fastdup.run('s3://mybucket/myfolder', run_mode=2, work_dir='/path/to/work_dir')
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

## Using your own onnx model as feature extractor <a name="onnx">

It is possible to replace fastdup's default onnx model with your own. To support running your own onnx model you need two changes
- Use `model_path` parameter to point to the location of the onnx model
- Set `d` to be the feature vector output width.

Detailed example you can run is found [here](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup_model_support.ipynb).


## Extracting statistics about image dataset <a name="stats">

Fastdup now supports computing the following statistics about image dataset:
- bluriness
- width
- height
- size
- number of unique colors
- mean color value
- max color value
- min color value

You can use the function `fastdup.create_stats_gallery()` to visualize those propoerties and find issues with your data.
Detailed tutorial is found in [colab](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup_image_stats.ipynb).

## Debugging fastdup <a name="debug"/>

To debug program execution the following is recommended
- Make sure you have upgraded fastdup to the latest version, we release versions a couple of times a week. 
- In case of installation failure try to upgrade your pip version to the latest using `python3.XX -m pip install -U pip` where XX is your python version.
- It is recommneded to debug in a python shell (and not in a Jupyter notebook)
- Run with `verbose=1` to get additional traces
- Run with `num_images=10` to run on a small subset of your data before running on the full dataset.
- In a python shell run `import fastdup; help(fastdup)` to see function documentation.
- If the issue persist please join our [Slack Channel]("https://join.slack.com/t/visualdatabase/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA")
