  
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

