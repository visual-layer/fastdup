
# FastDup 

FastDup is a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images, clusters of similaritity, learn the normal behavior and temporal interactions between imsges. It can be used for smart subsampling of a higher quality dataset,  outlier removal, novelty detection of new information to be sent for tagging. FastDup  scales to millions of images running on CPU only.

<p align="center">
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/git_main-min.png)
Temporal relations between images identified by fastdup

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/wild_animal_outliers.png)
Outliers in a wildlife animal image dataset identified by fastdup
</p>


## Results on Key Datasets
We have thourougly tested fastdup across various famous visual dataset. Ranging from Academic datasets to Kaggle competitions. A key finding we have made using FastDup is that there are ~1.2M (!) duplicate images on the ImageNet21K dataset, a new unknown result! Full results are below.

### FastDup is FAST
|Dataset	        |Total Images	|Owner			|Image Res     |cost [$]|spot cost [$]|processing [sec]|throughput [1/sec]|
|-----------------------|---------------|-----------------------|--------------|--------|-------|-------|-----|
|[imagenet21k-resized](https://www.image-net.org/challenges/LSVRC/)	|11,582,724	|alibaba/academy	|133x200	|4.98	|1.24	|11,561	|1,002|
|[places365-standard](http://places2.csail.mit.edu/download.html)	|2,168,460	|mit	                |256x256	|1.01	|0.25	|2,349	|923|
|[herbarium-2022-fgvc9](https://www.kaggle.com/c/herbarium-2022-fgvc9)	|1,050,179	|kaggle	                |1000x772	|0.69	|0.17	|1,598	|657|
|[landmark-recognition-2021](https://www.kaggle.com/c/landmark-recognition-2021)|1,590,815	|kaggle	                |532x800	|0.96	|0.24	|2,236	|711|
|[imdb-wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)	        |514,883	|eth zurich	        |684x1023	|0.65	|0.16	|1,509	|341|
|[iwildcam2021-fgvc9](https://www.kaggle.com/c/iwildcam2022-fgvc9/)	|261,428	|kaggle	                |1536x2048	|0.29	|0.07	|682	|383|
|[coco](https://cocodataset.org/#home)			|163,957	|academy	        |428x640	|0.09	|0.02	|218	|752|
|[visualgenome](https://visualgenome.org/)		|108,079	|stanford	        |334x500	|0.05	|0.01	|124	|872|
|[sku110k](https://github.com/eg4000/SKU110K_CVPR19)		|11,743	        |trax	                |4160x2340	|0.03	|0.01	|77	|153|

* Experiments on a 32 core Google cloud machine, with 128GB RAM (no GPU required).

* We run on the full ImageNet dataset (11.5M images) to compare all pairs of images in less than 3 hours WITHOUT a GPU (with Google cloud cost of 5$).

### FastDup is ACCURATE
Dataset|	Identical Pairs|	Near-Identical Pairs
-------|----------------------|--------------------
[imagenet21k-resized](https://www.image-net.org/challenges/LSVRC/)	|1,194,059|	53,358
[geolifeclef-2022-lifeclef-2022-fgvc9](https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/data)	|93,109|	234,342
[places365-standard](http://places2.csail.mit.edu/download.html)	|33,115	|2,342
[landmark-recognition-2021](https://www.kaggle.com/c/landmark-recognition-2021)	|2,613	|2,484
[imdb-wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)		|187,985|	456
[coco](https://cocodataset.org/#home)		|54	|14
[visualgenome](https://visualgenome.org/)		|223	|315
[sku110k](https://github.com/eg4000/SKU110K_CVPR19)	|7	|110
[herbarium-2022-fgvc9](https://www.kaggle.com/c/herbarium-2022-fgvc9)		|8,599	|1,383
[iwildcam2021-fgvc9](https://www.kaggle.com/c/iwildcam2022-fgvc9/)	|120,991	|45,946
[sorghum-id-fgvc-9](https://www.kaggle.com/competitions/sorghum-id-fgvc-9/data)	|46	|212
[snakeclef2022-fgvc9](https://www.kaggle.com/competitions/snakeclef2022/data)	|6,953	|33,128
[fungiclef2022-fgvc9](https://www.kaggle.com/competitions/fungiclef2022/data)	|2,205	|75
[hotel-id-to-combat-human-trafficking-2022-fgvc9](https://www.kaggle.com/competitions/hotel-id-to-combat-human-trafficking-2022-fgvc9/data)|	3,544	|2,704

## Quick Installation (Only Ubuntu 20.04 supported for now!)
For Python 3.7 and 3.8
```python
python3.8 -m pip install fastdup
```

[Install from stable release](INSTALL.md)


## Running the code

### Python
```python
python3.8
import fastdup
fastdup.run(input_dir="/path/to/your/folder", work_dir="/path/to/your/folder") #main running function
```
  
### C++
```bash
/usr/bin/fastdup /path/to/your/folder --work_dir="/tmp/fastdup_files"
```

[Detailed running instructions](RUN.md)



### Support for s3 cloud/ google storage
[Detailed instructions](CLOUD.md)
