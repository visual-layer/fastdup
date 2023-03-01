# FastDup Installation Manual

FastDup is a tool for fast detection of duplicate and near duplicate images.
##Ubuntu 20.04 LTS Machine Setup

```
sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.8
sudo apt -y install python3-pip
pip install --upgrade pip
```

##Pip Package setup
Download the FastDup wheel  from the following shared folder: https://drive.google.com/drive/folders/1Pj5h6sUSPoRVO6Zwl0hfHIllRV004kib?usp=sharing

For pip (python 3.8) install using
`pip install fastdup-<VERSION>-cp38-cp38-linux_x86_64.whl`

For conda (python 3.7.11) install using
`conda install fastdup-<VERSION>-py37_0.tar.bz`
Running the code
> python3
> import fastdup
> fastdup.__version__ # prints the version number
> fastdup.run(“/path/to/your/folder”) #main running function

##Detailed Python API documentation

    Run fastdup tool for find duplicate and near duplicate images in a corpus of images. 
    The only mandatory argument is image_dir. Given an image directory it will compare all pairs of images and store the most similar ones in the output file output_similarity.

    Parameters:
        input_dir (str): Location of the images directory (or videos).
        Alternatively, it is also possible to give a location of a file listing images full path, one image per row.

        work_dir (str): Directory for saving intermediate files and results. 

        compute (str): Compute type [cpu|gpu] default is cpu. (Gpu not supported yet).

        verbose (boolean): Verbosity. Default is False.

        num_threads (int): Number of threads. Default is -1 to be auto configured by the number of cores.

        num_images (int): Number of images to run on. Default is -1 which means run on all the images in the image_dir folder.

        nnmodel (str): Nearest Neighbor model for clustering the features together. Supported options are brute_force (exact), ball_tree and lsh (both approximate). Default is brute_force.

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

        nnf_mode (str): When nn_provider='nnf' selects the nnf mode. Supported options are HNSW32.

        nnf_param (str): When nn_provider='nnf' assigns optional nnf parameters.

    Returns:
        Status code 0 = success, 1 = error.

    
#Support for cloud storage
FastDup supports two types of cloud storage:
Amazon s3 aws cli
Min.io cloud storage api

###Amazon s3 aws cli support
Preliminaries:
Install aws cli using the command
sudo apt install awscli
Configure your aws using the command
aws configure
Make sure you can access your bucket using
aws s3 ls s3://<your bucket name>

How to run
There are two options to run.
In the input_dir command line argument put the full path your bucket for example: s3://mybucket/myfolder/myother_folder/
This option is useful for testing but it is not recommended for large corpouses of images as listing files in s3 is a slow operation. In this mode, all the images in the recursive subfolders of the given folders will be used.
Alternatively (and recommended) create a file with the list of all your images in the following format:
s3://mybucket/myfolder/myother_folder/image1.jpg
s3://mybucket/myfolder2/myother_folder4/image2.jpg
s3://mybucket/myfolder3/myother_folder5/image3.jpg
Assuming the filename is files.txt you can run with input_dir=’/path/to/files.txt’

Notes: 
Currently we support a single cloud provider and a single bucket.
It is OK to have images with the same name assuming they are nested in different subfolders.
In terms of performance, it is better to copy the full bucket to the local node first in case the local disk is hard enough. Then give the input_dir as the local folder location of the copied data. The explanation above is for the case the dataset is larger than the local disk (and potentially multiple nodes run in parallel).



###Min.io support
Preliminaries
Install the min.io client using the command
```
wget https://dl.min.io/client/mc/release/linux-amd64/mc
sudo mv mc /usr/bin/
chmod +x /usr/bin/mc
```
Configure the client to point to the cloud provider
mc alias set myminio/ http://MINIO-SERVER MYUSER MYPASSWORD
For example for google cloud:
```
​​/usr/bin/mc alias set google  https://storage.googleapis.com/ <access_key> <secret_key> 
Make sure the bucket is accessible using the command:
/usr/bin/mc ls google/mybucket/myfolder/myotherfolder/
```
How to run
There are two options to run.
In the input_dir command line argument put the full path your cloud storage provider as defined by the minio alias, for example: minio://google/mybucket/myfolder/myother_folder/
(Note that google is the alias set for google cloud, and the path has to start with minio:// prefix).
This option is useful for testing but it is not recommended for large corpouses of images as listing files in s3 is a slow operation. In this mode, all the images in the recursive subfolders of the given folders will be used.
Alternatively (and recommended) create a file with the list of all your images in the following format:
```
minio://google/mybucket/myfolder/myother_folder/image1.jpg
minio://google/mybucket/myfolder/myother_folder/image2.jpg
minio://google/mybucket/myfolder/myother_folder/image3.jpg
```
Assuming the filename is `files.txt` you can run with input_dir=’/path/to/files.txt’


##Error handling
When bad images are encountered, namely corrupted images that can not be read, an additional csv output file is generated called features.dat.bad. The bad images filenames are stored there. In addition there is a printout that states the number of good and bad images encountered. The good images filenames are stored in the file features.dat.csv file. Namely the bad images are excluded from the total images listing. The function fastdup.load_binary_features() reads the features corresponding to the good images and returns a list of all the good images, and a numpy array of all their corresponding features.
The output file similarity.csv with the list of all similar pairs does not include any of the bad images.


##Speeding up the nearest neighbor search
Once short feature vectors are generated per each image, we cluster them to find similarities using a nearest neighbor method. FastDup supports two families of algorithms (given using the nn_provider command line argument)
- turi
- faiss

Turi has the following methods inside
- brute_force (exact method but may be slower)
- ball_tree (approximate method)
- lsh  (locality sensitive hashing, approximate method)

Faiss has the following methods
- FlatIndex (approximate)
- More to come soon
We suggest to use brute_force method for datasets up to 10K images. For larger datasets use approximate methods. 

Example command line:
```
import fastdup
fastdup.run(“/path/to/folder”, nn_provider=”turi”, nnmethod=’brute_force’)
fastdup.run(“/path/to/folder”, nn_provider=”faiss”)
```
			
##Notes
This is an experimental version tested up to 13M images, 
Look for the default output file `similarity.csv` to look at the most similar pairs found. It is generated in the same run folder.
Default installation location should be ~/.local/lib/python3.8/site-packages/fastdup
