
# FastDup Manual

FastDup is a tool for fast detection of duplicate and near duplicate images.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/flower.png)

# FastDup is FAST

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/fastdup_performance.png)

We run on the full ImageNet dataset (11.5M images) to compare all pairs of images in less than 3 hours WITHOUT a GPU (with Google cloud cost of 5$).

# FastDup is ACCURATE

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/fastdup_duplicates.png)

FastDup identifies 1,200,000 duplicate images on the ImageNet dataset, a new unknown resut!


# Installing the code
For Python 3.7 and 3.8
```
pip install fastdup
```

[Install from stable release](INSTALL.md)


# Running the code

## Python
```
> python3
> import fastdup
> fastdup.__version__ # prints the version number
> fastdup.run(input_dir=“/path/to/your/folder”, work_dir="/path/to/your/folder") #main running function
```
  
## C++
```
/usr/bin/fastdup /path/to/your/folder --work_dir="/tmp/fastdup_files"

```

[Detailed running instructions](RUN.md)



# Support for s3 cloud/ google storage
[Detailed instructions](CLOUD.md)


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



