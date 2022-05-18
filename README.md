
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
|Dataset	        |Total Images	|cost [$]|spot cost [$]|processing [sec]|Identical pairs|Anomalies|
|-----------------------|---------------|--------|-------------|----------------|---------------|---------|
|[imagenet21k-resized](https://www.image-net.org/challenges/LSVRC/)	|11,582,724	|4.98	|1.24	|11,561	|[1,194,059](https://www.databasevisual.com/imagenet-21k-resized-leaks)|[View](https://www.databasevisual.com/imagenet-21k-anonalies)|
|[places365-standard](http://places2.csail.mit.edu/download.html)	|2,168,460	|1.01	|0.25	|2,349	|93,109|View|
|[herbarium-2022-fgvc9](https://www.kaggle.com/c/herbarium-2022-fgvc9)	|1,050,179	|0.69	|0.17	|1,598	|33,115|View|
|[landmark-recognition-2021](https://www.kaggle.com/c/landmark-recognition-2021)|1,590,815|0.96	|0.24	|2,236	|2,613|View|
|[imdb-wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)	|514,883	|0.65	|0.16	|1,509	|187,965|View|
|[iwildcam2021-fgvc9](https://www.kaggle.com/c/iwildcam2022-fgvc9/)	|261,428	|0.29	|0.07	|682	|54|View|
|[coco](https://cocodataset.org/#home)			|163,957	|0.09	|0.02	|218	|54|View|
|[visualgenome](https://visualgenome.org/)		|108,079	|0.05	|0.01	|124	|223|View|
|[sku110k](https://github.com/eg4000/SKU110K_CVPR19)		|11,743	|0.03	|0.01	|77	|7|View|

* Experiments on a 32 core Google cloud machine, with 128GB RAM (no GPU required).

* We run on the full ImageNet dataset (11.5M images) to compare all pairs of images in less than 3 hours WITHOUT a GPU (with Google cloud cost of 5$).

## Quick Installation (Only Ubuntu 20.04 supported for now!)
For Python 3.7 and 3.8
```python
python3.8 -m pip install fastdup
```

[Install from stable release and installation issues](INSTALL.md)


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


## Feature summary
|  | Free version | Enterprise Edition|
|--|--------------|-------------------|
|Operating Systems | Ubuntu 20.04 | Plus Amazon Linux, RedHat, Windows, Mac OS|
|Python Versions | Python 3.7+3.8+conda | Plus Python 3.6, 3.9, 3.10|
|Compute | CPU | GPU, TPU, Intel OpenVino|
|Instance | On demand | Support for spot instance|
|Numbr of images | Up to 1 million | Up to 1 billion|
|Execution | Single node | Cluster|
|Features | Outlier detection, duplicate detection | Plus novelty detection, wrong label detection, missing label detection, data summarization, connected components, train/test leaks, temporal sequence detection, advanced visual search, label quality analysis|
|Input | Images | Plus Video|






