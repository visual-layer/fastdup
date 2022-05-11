
# FastDup Manual

FastDup is a tool for fast detection of duplicate and near duplicate images.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/flower.png?raw=true)

# Installation
## Ubuntu 20.04 LTS Machine Setup
Required setup
- sudo apt update
- sudo apt -y install software-properties-common
- sudo add-apt-repository -y ppa:deadsnakes/ppa
- sudo apt update
- sudo apt -y install python3.8
- sudo apt -y install python3-pip
- pip install --upgrade pip



# Pip Package setup
Download the FastDup latest wheel from the following shared folder: `s3://visualdb`

Latest version: 0.25

## For pip (python 3.8) install using
```
pip install fastdup-<VERSION>-cp38-cp38-linux_x86_64.whl
```

## For conda (python 3.7.11) install using
```
conda install -y pandas tqdm opencv numpy
conda install fastdup-<VERSION>-py37_0.tar.bz
```


# Currently supported software/hardware

Operating system
- `Ubuntu 20.04 LTS`

Software versions
- `Python 3.8` (via pip) or `Python 3.7` (via pip or conda) or a `debian package` (Python is not required)

Hardware support
- CPU (GPU not needed!)


# Running the code
```
> python3
> import fastdup
> fastdup.__version__ # prints the version number
> fastdup.run(“/path/to/your/folder”) #main running function
```
  
Detailed Python API documentation

```
    Run fastdup tool for find duplicate and near duplicate images in a corpus of images. 
    The only mandatory argument is image_dir. Given an image directory it will compare all pairs of images and store the most similar ones in the output file output_similarity.

    Parameters:
        input_dir (str): Location of the images directory (or videos).
Alternatively, it is also possible to give a location of a file listing images full path, one image per row.

        work_dir (str): Working directory for saving intermediate results and outputs.

        compute (str): Compute type [cpu|gpu] default is cpu.

        verbose (boolean): Verbosity. Default is False.

        num_threads (int): Number of threads. Default is -1 to be auto configured by the number of cores.

        num_images (int): Number of images to run on. Default is -1 which means run on all the images in the image_dir folder.

        nnmodel (str): Nearest Neighbor model for clustering the features together, when using turi (has no effect when using faiss). Supported options are brute_force (exact), ball_tree and lsh (both approximate). Default is brute_force.

        distance (str): Distance metric for the Nearest Neighbors algorithm. Default is cosine. Other distances are euclidean, squared_euclidean, manhattan.

        threshold (float): Similarity measure in the range 0->1, where 1 is totally identical, 0.98 and above is almost identical, and 0.85 and above is very similar. Default is 0.85 which means that only image pairs with similarity larger than 0.85 are stored.

        lower_threshold (float): Similarity measure to outline images that are far away (outliers) vs. the total distribution. Default value is 0.3.

        model_path(str): Optional location of ONNX model file, should not be used.

        version(bool): Print out the version number. This function takes no argument.

        nearest_neighbors_k (int): For each image, how many similar images to look for. Default is 2.

        run_mode (int): This software can run for either feature vector extraction and similarity measurement (0), or just feature vector extraction (1), or just similarity measure computation (2). 
 
   nn_provider (string): Provider of the nearest neighbor algorithm, allowed values are turi|faiss.

        min_offset (int): Optional min offset to start iterating on the full file list. Default is -1.

        max_offset (int): Optional max offset to start iterating on the full file list. Default is -1.

 	   faiss_mode (str): When nn_provider='faiss' selects the faiss mode. Supported options are HNSW32 and any other faiss string.

   faiss_param (str): When nn_provider='faiss' assigns optional faiss parameters. For example efSearch=175. Multiple params are supported - for example 'efSearch=175,nprobes=200'


       
        
    Returns:
        Status code 0 = success, 1 = error.
```
  
## Input / output formats
The input to fastdup tool is given in the command line argument: data_dir. There are a few options:
Location of a local folder. In that case all images in this folder are searched recursively.
Location of an s3 path. Again all images in the path will be used recursively.
A file containing image locations (either local or full s3 paths). Each image in its own row.

The intermediate outputs and final outputs are stored in the folder work_dir.
Feature extraction related files:
Binary numpy array containing n rows of 576 columns with the feature vectors. (Default filename is features.dat)
An additional csv file containing the full paths to the image names corresponding to the feature vectors (default filename is features.dat.csv). This is needed from two reasons:
The order of extraction may change depends on the file system listing 
In case of corrupted images, its feature vector is skipped and not generated. In that case an additional output file is provided ( features.bad.csv) 

Similarity pair list
The output of the fastdup tool is a similarity file (filename is similarity.csv) which is a csv file with 3 columns: from, to, distance. The file is sorted from the closest matching images to less similar images.

Note: for exploiting the binary features we provide the following function in Python:

```
def load_binary_feature(filename):

    Example Python function for loading the stored binary features and their matching filenames.

    Parameters:
        filename(str):The binary feature file location

    Returns:
        A list of with all image file names of length X.
        An np matrix of shape X rows x 576 cols. Each row conform to feature vector os a single image.

    Example:
        import fastdup
        file_list, mat_features = fastdup.load_binary('features.dat')

```

Faiss index files
When using faiss an additional intermediate results file is created: faiss.index.
Support for cloud storage
FastDup supports two types of cloud storage:
Amazon s3 aws cli
Min.io cloud storage api

## Amazon s3 aws cli support
### Preliminaries:
- Install aws cli using the command
`sudo apt install awscli`
- Configure your aws using the command
`aws configure`
- Make sure you can access your bucket using
`aws s3 ls s3://<your bucket name>`

## How to run
There are two options to run.
In the input_dir command line argument put the full path your bucket for example: `s3://mybucket/myfolder/myother_folder/`
This option is useful for testing but it is not recommended for large corpouses of images as listing files in s3 is a slow operation. In this mode, all the images in the recursive subfolders of the given folders will be used.
Alternatively (and recommended) create a file with the list of all your images in the following format:
```
s3://mybucket/myfolder/myother_folder/image1.jpg
s3://mybucket/myfolder2/myother_folder4/image2.jpg
s3://mybucket/myfolder3/myother_folder5/image3.jpg
```
Assuming the filename is files.txt you can run with input_dir=’/path/to/files.txt’

Notes: 
Currently we support a single cloud provider and a single bucket.
It is OK to have images with the same name assuming they are nested in different subfolders.
In terms of performance, it is better to copy the full bucket to the local node first in case the local disk is hard enough. Then give the input_dir as the local folder location of the copied data. The explanation above is for the case the dataset is larger than the local disk (and potentially multiple nodes run in parallel).



## Min.io support
Preliminaries
Install the min.io client using the command
```
wget https://dl.min.io/client/mc/release/linux-amd64/mc
sudo mv mc /usr/bin/
chmod +x /usr/bin/mc
```
Configure the client to point to the cloud provider

```
mc alias set myminio/ http://MINIO-SERVER MYUSER MYPASSWORD
```
For example for google cloud:
```
/usr/bin/mc alias set google  https://storage.googleapis.com/ <access_key> <secret_key> 
```
Make sure the bucket is accessible using the command:
```
/usr/bin/mc ls google/mybucket/myfolder/myotherfolder/
```

How to run
There are two options to run.
In the input_dir command line argument put the full path your cloud storage provider as defined by the minio alias, for example: `minio://google/mybucket/myfolder/myother_folder/`
(Note that google is the alias set for google cloud, and the path has to start with `minio://` prefix).
This option is useful for testing but it is not recommended for large corpouses of images as listing files in s3 is a slow operation. In this mode, all the images in the recursive subfolders of the given folders will be used.
Alternatively (and recommended) create a file with the list of all your images in the following format:
```
minio://google/mybucket/myfolder/myother_folder/image1.jpg
minio://google/mybucket/myfolder/myother_folder/image2.jpg
minio://google/mybucket/myfolder/myother_folder/image3.jpg
```
Assuming the filename is `files.txt` you can run with `input_dir=’/path/to/files.txt’`


## Error handling
When bad images are encountered, namely corrupted images that can not be read, an additional csv output file is generated called features.dat.bad. The bad images filenames are stored there. In addition there is a printout that states the number of good and bad images encountered. The good images filenames are stored in the file features.dat.csv file. Namely the bad images are excluded from the total images listing. The function fastdup.load_binary_features() reads the features corresponding to the good images and returns a list of all the good images, and a numpy array of all their corresponding features.
The output file similarity.csv with the list of all similar pairs does not include any of the bad images.


## Speeding up the nearest neighbor search
Once short feature vectors are generated per each image, we cluster them to find similarities using a nearest neighbor method. FastDup supports two families of algorithms (given using the nn_provider command line argument)
- turi
- faiss
Turi (nn_provider=’turi’) has the following methods inside
- nnmodel=’brute_force’ (exact method but may be slower)
- nnmodel=’ball_tree’ (approximate method)
- nnmodel=’lsh’  (locality sensitive hashing, approximate method)
Faiss (nn_provider=’faiss’) supports multiple methods
- faiss_mode=’HSNW32’ the default





Example command line:
```
> import fastdup
> fastdup.run(“/path/to/folder”, nn_provider=”turi”, nnmodel=’brute_force’)
> fastdup.run(“/path/to/folder”, nn_provider=”faiss”, faiss_mode=’HSNW32’)
```


## Resuming a stored run
There are 3 supported running modes:
run_mode=0 (the default) does the feature extraction and NN embedding to provide similarities. It uses the input_dir command line argument for finding the directory to run on (or a list of files to run on). The features are extracted and saved into feature_out_file (the default features out file is features.dat in the same folder for storing the numpy features and features.dat.csv for storing the image file names corresponding to the numpy features).
For larger dataset it may be wise to split the run into two, to make sure intermediate results are stored in case you encounter an error.
run_mode=1 computes the extracted features and stores them, does not compute the NN embedding. For large datasets, it is possible to run on a few computing nodes, to extract the features, in parallel. Use the min_offset and max_offset flags to allocate a subset of the images for each computing node. Offsets start from 0 to n-1 where n is the number of images in the input_dir folder.
run_mode=2, reads a stored feature file and computes the NN embedding to provide similarities. The input_dir param is ignored, and the features_out_file is used to point to the numpy feature file. (Give a full path and filename).

## Visualizing the outputs
Once fastdup runs you can look at the results in an easy way using two options. When running from a jupyter notebook the code will produce a table gallery. Otherwise when running a from python shell an html report will be generated.

The following command creates the html report:
```
def create_duplicates_gallery(similarity_file, save_path, num_images=20, descending=True):

    Function to create and display a gallery of images computed by the similarity metrics

    Parameters:
        similarity_file (str): csv file with the computed similarities by the fastdup tool
        save_path (str): output folder location for the visuals
        num_images(int): Max number of images to display (deafult = 50)
        descending (boolean): If False, print the similarities from the least similar to the most similar. Default is True.
```

# Example of the html generated.

Example for the html report generation:
```
import fastdup
fastdup.generate_duplicates_gallery(‘/path/to/similarity.csv’, save_path=’/path/to/report/’)
```

Note: the report should be generated on the same machine since we assume that the input folder for reading the images exists under the same location.

Notes
This is an experimental version tested up to 13M images



