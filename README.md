
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
#> import fastdup
#> fastdup.run(input_dir=“/path/to/your/folder”, work_dir="/path/to/your/folder") #main running function
```
  
## C++
```
/usr/bin/fastdup /path/to/your/folder --work_dir="/tmp/fastdup_files"

```

[Detailed running instructions](RUN.md)



# Support for s3 cloud/ google storage
[Detailed instructions](CLOUD.md)


