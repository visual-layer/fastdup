
# FastDup 

FastDup is a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images, clusters of similarity, learn the normal behavior and temporal interactions between images. It can be used for smart subsampling of a higher quality dataset,  outlier removal, novelty detection of new information to be sent for tagging. FastDup scales to millions of images running on CPU only.

From the authors of [GraphLab](https://github.com/jegonzal/PowerGraph) and [Turi Create](https://github.com/apple/turicreate).

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/mscoco_duplicates-min.png)
*Duplicates and near duplicates identified in [MS-COCO](https://cocodataset.org/#home) and [Imagenet-21K](https://www.image-net.org) dataset*

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imdb_outliers-min.png)
*[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ ) outliers (data goal is for face detection, gender and age detection)*

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/landmark_outliers-min.png)
*Outliers in the [Google Landmark Recognition 2021 dataset](https://www.kaggle.com/competitions/landmark-recognition-2021) (dataset intention is to capture recognizable landmarks, like the empire state building etc.)*

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/red_wine.png)
*Cluster of wrong labels in the [Imagenet-21K](https://www.image-net.org) dataset. No human can tell those red wine flavors from their image.*

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imagenet21k_wrong_labels-min.png)
*Wrong labels in the [Imagenet-21K](https://www.image-net.org) dataset* Different labels to visaully similar daisy flower images.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/daisy.png)
*Cluster of wrong labels in the [Imagenet-21K](https://www.image-net.org) dataset.* Different labels to visually similar red-wine images.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imagenet21k_funny-min.png)
*Fun labels in the [Imagenet-21K](https://www.image-net.org) dataset*


## Results on Key Datasets
We have thoroughly tested fastdup across various famous visual datasets. Ranging from pilar Academic datasets to Kaggle competitions. A key finding we have made using FastDup is that there are ~1.2M (!) duplicate images on the ImageNet-21K dataset, out of which 104K pairs belong both to the train and to the val splits (this amounts to 20% of the validation set). This is a new unknown result! Full results are below. * train/val splits are taken from https://github.com/Alibaba-MIIL/ImageNet21 .

|Dataset	        |Total Images	|cost [$]|spot cost [$]|processing [sec]|Identical pairs|Anomalies|
|-----------------------|---------------|--------|-------------|----------------|---------------|---------|
|[imagenet21k-resized](https://www.image-net.org/challenges/LSVRC/)	|11,582,724	|4.98	|1.24	|11,561	|[1,194,059](https://www.databasevisual.com/imagenet-21k-resized-leaks)|[Anomalies](https://www.databasevisual.com/imagenet-21k-anonalies) [Wrong Labels](https://www.databasevisual.com/imagenet-21k-wrong-labels)||
|[imdb-wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)	|514,883	|0.65	|0.16	|1,509	|[187,965](https://www.databasevisual.com/imdb-wiki-leaks)|[View](https://www.databasevisual.com/imdb-wiki-anomalies)|
|[places365-standard](http://places2.csail.mit.edu/download.html)	|2,168,460	|1.01	|0.25	|2,349|[93,109](https://www.databasevisual.com/places-365-leaks)|[View](https://www.databasevisual.com/places-365-anomalies)|
|[herbarium-2022-fgvc9](https://www.kaggle.com/c/herbarium-2022-fgvc9)	|1,050,179	|0.69	|0.17	|1,598	|[33,115](https://www.databasevisual.com/herbarium-leaks)|[View](https://www.databasevisual.com/herbarium-2022-anomalies)|
|[landmark-recognition-2021](https://www.kaggle.com/c/landmark-recognition-2021)|1,590,815|0.96	|0.24	|2,236	|[2,613](https://www.databasevisual.com/landmarks-2021-leaks)|[View](https://www.databasevisual.com/landmark-anomalies)|
|[visualgenome](https://visualgenome.org/)		|108,079	|0.05	|0.01	|124	|223|View|
|[iwildcam2021-fgvc9](https://www.kaggle.com/c/iwildcam2022-fgvc9/)	|261,428	|0.29	|0.07	|682	|[54](https://www.databasevisual.com/iwildcam2022-leaks)|[View](https://www.databasevisual.com/iwildcam2022-anomalies)|
|[coco](https://cocodataset.org/#home)			|163,957	|0.09	|0.02	|218	|54|View|
|[sku110k](https://github.com/eg4000/SKU110K_CVPR19)		|11,743	|0.03	|0.01	|77	|[7](https://www.databasevisual.com/sku110k-leaks)|[View](https://www.databasevisual.com/sku110k-anomalies)|

* Experiments presented are on a 32 core Google cloud machine, with 128GB RAM (no GPU required).
* All experiments could be also reproduced on a 8 core, 32GB machine (excluding Imagenet-21K).
* We run on the full ImageNet-21K dataset (11.5M images) to compare all pairs of images in less than 3 hours WITHOUT a GPU (with Google cloud cost of 5$).

## Quick Installation (Ubuntu 20.04 or Ubuntu 18.04)
For Python 3.7 and 3.8
```python
python3.8 -m pip install fastdup
```

## Running the code

### Python
```python
import fastdup
fastdup.run(input_dir="/path/to/your/folder")                            #main running function
fastdup.create_duplicates_gallery('similarity.csv', save_path='.')       #create a visual gallery of found duplicates
fastdup.create_duplicates_gallery('outliers.csv',   save_path='.')       #create a visual gallery of anomalies
```

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/fastdup_clip_24s_crop.gif)
*Working on the Food-101 dataset. Detecting identical pairs, similar-pairs (search) and outliers (non-food images..)*

### Getting started examples
- [Getting started on a Kaggle dataset](https://github.com/visualdatabase/fastdup/blob/main/examples/getting_started_kaggle.ipynb)
- [Finding duplicates and outliers in the Food-101 datadset:](https://github.com/visualdatabase/fastdup/blob/main/examples/getting_started_food101.ipynb)

### Detailed instructions
[Detailed isntructions, install from stable release and installation issues](INSTALL.md)
[Detailed running instructions](RUN.md)


# Technology
We build upon several excellent open source tools. [Microsoft's ONNX Runtime](https://github.com/microsoft/onnxruntime), [Facebook's Faiss](https://github.com/facebookresearch/faiss), [Open CV](https://github.com/opencv/opencv), [Pillow Resize](https://github.com/zurutech/pillow-resize), [Apple's Turi Create](https://github.com/apple/turicreate), [Minio](https://github.com/minio/minio), [Amazon's awscli](https://github.com/aws/aws-cli).


# About Us
<a href="https://www.linkedin.com/in/dr-danny-bickson-835b32">Danny Bickson</a><br>
<a href="https://www.linkedin.com/in/amiralush">Amir Alush</a><br>
<a href="https://join.slack.com/t/visualdatabase/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA">Join our Slack channel</a>
