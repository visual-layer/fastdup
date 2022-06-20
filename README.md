<div>
   <a href="https://pepy.tech/project/fastdup"><img src="https://static.pepy.tech/personalized-badge/fastdup?period=total&units=none&left_color=blue&right_color=orange&left_text=Downloads"></a>
   <a href="https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 <a href="https://www.kaggle.com/graphlab/fastdup" rel="nofollow"><img src="https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667" alt="Open In Kaggle" data-canonical-src="https://kaggle.com/static/images/open-in-kaggle.svg" style="max-width: 100%;"></a>
<a href="https://mybinder.org/v2/gh/visualdatabase/fastdup/main?labpath=exmples%2Ffastdup.ipynb" rel="nofollow"><img src="ttps://mybinder.org/badge_logo.svg"></a>
<a href="https://join.slack.com/t/visualdatabase/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA" rel="nofollow"><img src="https://camo.githubusercontent.com/8df26cc38dabf1035cddfbed79714744bb93785bc8341cb883fef4cdc412572d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f536c61636b2d3441313534423f6c6f676f3d736c61636b266c6f676f436f6c6f723d7768697465" alt="Slack" data-canonical-src="https://img.shields.io/badge/Slack-4A154B?logo=slack&amp;logoColor=white" style="max-width: 100%;"></a>
<a href="https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22" rel="nofollow"><img src="https://camo.githubusercontent.com/771af957ebd52645704462209592c7a0a359feaec816337fee900e4478278219/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d656469756d2d3132313030453f6c6f676f3d6d656469756d266c6f676f436f6c6f723d7768697465" alt="Medium" data-canonical-src="https://img.shields.io/badge/Medium-12100E?logo=medium&amp;logoColor=white" style="max-width: 100%;"></a>
<a href="https://share-eu1.hsforms.com/1POrgIy-hTSyMaOTQzgjqhgfglt8" rel="nofollow"><img src="https://camo.githubusercontent.com/5042565e9cc3a40bff3d9be7b59955d984831f594d38297b6efecf804e41b8f7/687474703a2f2f6269742e6c792f324d643972784d" alt="Mailing list" data-canonical-src="http://bit.ly/2Md9rxM" style="max-width: 100%;"></a>
</div>

FastDup | A tool for gaining insights from a large image collection
===================================================================
> Large Image Datasets Today are a Mess | <a href="https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22">Blog Post</a> | <a href="https://youtu.be/vDqOlpZeeH8">Video Tutorial </a><br>

FastDup is a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images, clusters of similarity, learn the normal behavior and temporal interactions between images. It can be used for smart subsampling of a higher quality dataset,  outlier removal, novelty detection of new information to be sent for tagging. FastDup scales to millions of images running on CPU only.

From the authors of [GraphLab](https://github.com/jegonzal/PowerGraph) and [Turi Create](https://github.com/apple/turicreate).

## Identify duplicates
![alt text](./gallery/mscoco_duplicates-min.png)
*Duplicates and near duplicates identified in [MS-COCO](https://cocodataset.org/#home) and [Imagenet-21K](https://www.image-net.org) dataset*

## Find corrupted and broken images
![alt text](./gallery/imagenet21k_broken.png)
*Thousands of broken ImageNet images that have confusing labels of real objects.*

## Find outliers
![alt text](./gallery/imdb_outliers-min.png)
*[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ ) outliers (data goal is for face recognition, gender and age classification)*


## Find similar persons
![alt text](./gallery/viz0.png)
*Can you tell how many different persons?*


## Find wrong labels
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imagenet21k_wrong_labels-min.png)
*Wrong labels in the [Imagenet-21K](https://www.image-net.org) dataset*.

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/daisy.png)
*Cluster of wrong labels in the [Imagenet-21K](https://www.image-net.org) datas## Find image with contradicting labels

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/red_wine.png)
*Cluster of wrong labels in the [Imagenet-21K](https://www.image-net.org) dataset. No human can tell those red wines from their image.*

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imagenet21k_funny-min.png)
*Fun labels in the [Imagenet-21K](https://www.image-net.org) dataset*

## Coming soon: image graph search (please reach out if you like to beta test)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/viz23.png)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/viz2.png)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/tmp13_viz5.png)

*Upcoming new features: image graph search!*


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

# Quick Installation 
For Python 3.7 and 3.8 (Ubuntu 20.04 or Ubuntu 18.04 or Mac M1 or Mac Intel Mojave and up)
```python
python3.8 -m pip install fastdup
```

# Running the code

```python
import fastdup
fastdup.run(input_dir="/path/to/your/folder", work_dir='out')                            #main running function
fastdup.create_duplicates_gallery('out/similarity.csv', save_path='.')       #create a visual gallery of found duplicates
fastdup.create_outliers_gallery('out/outliers.csv',   save_path='.')       #create a visual gallery of anomalies
fastdup.create_components_gallery('out', save_path='.')                    #create visualiaiton of connected components
```

![alt text](./gallery/fastdup_clip_24s_crop.gif)
*Working on the Food-101 dataset. Detecting identical pairs, similar-pairs (search) and outliers (non-food images..)*

## Getting started examples
- [Getting started on a Kaggle dataset](https://github.com/visualdatabase/fastdup/blob/main/examples/getting_started_kaggle.ipynb)
- [Finding duplicates, outliers in the Food-101 datadset:](https://github.com/visualdatabase/fastdup/blob/main/examples/getting_started_food101.ipynb)
- [🔥🔥🔥 Finding duplicates, outliers and connected components in the Food-101 dataset, including Tensorboard Projector visualization - Google Colab](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup.ipynb)
- [🔥Analyzing video of the MEVA dataset - Google Colab](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup_video.ipynb)
- [Kaggle notebook - visualizing the pistachio dataset](https://www.kaggle.com/code/graphlab/fastdup/notebook)

![Tensorboard Projector integration is explained in our Colab notebook](./gallery/tensorboard_projector.png)

## Detailed instructions
- [Detailed instructions, install from stable release and installation issues](INSTALL.md)
- [Detailed running instructions](RUN.md)

# Support 
<a href="https://join.slack.com/t/visualdatabase/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA">Join our Slack channel</a>

# Technology
We build upon several excellent open source tools. [Microsoft's ONNX Runtime](https://github.com/microsoft/onnxruntime), [Facebook's Faiss](https://github.com/facebookresearch/faiss), [Open CV](https://github.com/opencv/opencv), [Pillow Resize](https://github.com/zurutech/pillow-resize), [Apple's Turi Create](https://github.com/apple/turicreate), [Minio](https://github.com/minio/minio), [Amazon's awscli](https://github.com/aws/aws-cli).

# About Us
<a href="https://www.linkedin.com/in/dr-danny-bickson-835b32">Danny Bickson</a>, <a href="https://www.linkedin.com/in/amiralush">Amir Alush</a><br>

