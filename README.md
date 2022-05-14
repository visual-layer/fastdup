
# FastDup Manual

FastDup is a tool for fast detection of duplicate and near duplicate images.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/flower.png)

# FastDup is FAST

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/fastdup_performance.png)

|Dataset	        |Total Images	|Owner			|Image Res     |cost [$]|spot cost [$]|processing [sec]|throughput [1/sec]|
|-----------------------|---------------|-----------------------|--------------|--------|-------|-------|-----|
|imagenet21k-resized	|11,582,724	|alibaba/academy	|133x200	|4.98	|1.24	|11,561	|1,002|
|places365-standard	|2,168,460	|mit	                |256x256	|1.01	|0.25	|2,349	|923|
|landmark-recognition-2021|1,590,815	|kaggle	                |532x800	|0.96	|0.24	|2,236	|711|
|imdb-wiki	        |514,883	|eth zurich	        |684x1023	|0.65	|0.16	|1,509	|341|
|coco			|163,957	|academy	        |428x640	|0.09	|0.02	|218	|752|
|visualgenome		|108,079	|stanford	        |334x500	|0.05	|0.01	|124	|872|
|sku110k		|11,743	        |trax	                |4160x2340	|0.03	|0.01	|77	|153|
|herbarium-2022-fgvc9	|1,050,179	|kaggle	                |1000x772	|0.69	|0.17	|1,598	|657|
|iwildcam2021-fgvc9	|261,428	|kaggle	                |1536x2048	|0.29	|0.07	|682	|383|
|-----------------------|---------------|-----------------------|--------------|--------|-------|-------|-----|

We run on the full ImageNet dataset (11.5M images) to compare all pairs of images in less than 3 hours WITHOUT a GPU (with Google cloud cost of 5$).

# FastDup is ACCURATE

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/fastdup_duplicates.png)

FastDup identifies 1,200,000 duplicate images on the ImageNet dataset, a new unknown resut!


# Installing the code
For Python 3.7 and 3.8
```python
pip install fastdup
```

[Install from stable release](INSTALL.md)


# Running the code

## Python
```python
python3
import fastdup
fastdup.run(input_dir="/path/to/your/folder", work_dir="/path/to/your/folder") #main running function
```
  
## C++
```bash
/usr/bin/fastdup /path/to/your/folder --work_dir="/tmp/fastdup_files"
```

[Detailed running instructions](RUN.md)



# Support for s3 cloud/ google storage
[Detailed instructions](CLOUD.md)


