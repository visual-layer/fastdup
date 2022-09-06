<div>
   <img src="https://camo.githubusercontent.com/40d08d4012a37b9f33aa9515a916a7f6b17f6945300b9bc06656eb245462b3a4/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e37253230253743253230332e38253230253743253230332e392d626c75652e737667">
   <a href="https://pepy.tech/project/fastdup"><img src="https://static.pepy.tech/personalized-badge/fastdup?period=total&units=none&left_color=blue&right_color=orange&left_text=Downloads"></a>
   <a href="https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/graphlab/fastdup" rel="nofollow"><img src="https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667" alt="Open In Kaggle" data-canonical-src="https://kaggle.com/static/images/open-in-kaggle.svg" style="max-width: 100%;"></a>
  <a href="https://mybinder.org/v2/gh/visualdatabase/fastdup/main?labpath=exmples%2Ffastdup.ipynb" rel="nofollow"><img src="https://mybinder.org/badge_logo.svg"></a>
<a href="https://join.slack.com/t/visualdatabase/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA" rel="nofollow"><img src="https://camo.githubusercontent.com/8df26cc38dabf1035cddfbed79714744bb93785bc8341cb883fef4cdc412572d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f536c61636b2d3441313534423f6c6f676f3d736c61636b266c6f676f436f6c6f723d7768697465" alt="Slack" data-canonical-src="https://img.shields.io/badge/Slack-4A154B?logo=slack&amp;logoColor=white" style="max-width: 100%;"></a>
<a href="https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22" rel="nofollow"><img src="https://camo.githubusercontent.com/771af957ebd52645704462209592c7a0a359feaec816337fee900e4478278219/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d656469756d2d3132313030453f6c6f676f3d6d656469756d266c6f676f436f6c6f723d7768697465" alt="Medium" data-canonical-src="https://img.shields.io/badge/Medium-12100E?logo=medium&amp;logoColor=white" style="max-width: 100%;"></a>
<a href="https://share-eu1.hsforms.com/1POrgIy-hTSyMaOTQzgjqhgfglt8" rel="nofollow"><img src="https://camo.githubusercontent.com/5042565e9cc3a40bff3d9be7b59955d984831f594d38297b6efecf804e41b8f7/687474703a2f2f6269742e6c792f324d643972784d" alt="Mailing list" data-canonical-src="http://bit.ly/2Md9rxM" style="max-width: 100%;"></a>
</div>


FastDup | A tool for gaining insights from a large image collection
===================================================================
> Large Image Datasets Today are a Mess | <a href="https://bit.ly/3NJLxEe">Blog Post</a> | <a href="https://bit.ly/3P3hOHh">Video Tutorial </a><br>

FastDup is a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images, clusters of similarity, learn the normal behavior and temporal interactions between images. It can be used for smart subsampling of a higher quality dataset,  outlier removal, novelty detection of new information to be sent for tagging. FastDup scales to millions of images running on CPU only.

From the authors of [GraphLab](https://github.com/jegonzal/PowerGraph) and [Turi Create](https://github.com/apple/turicreate).

## [Compute Image Statistics](https://github.com/visualdatabase/fastdup/blob/main/examples/fastdup_image_stats.ipynb)
![alt text](./gallery/image_stats.jpg)
*[Compute image statistics and visualize the results](https://github.com/visualdatabase/fastdup/blob/main/examples/fastdup_image_stats.ipynb), using food-101 dataset*

## [Identify duplicates](https://bit.ly/3AkSUz1)
![alt text](./gallery/mscoco_duplicates-min.png)
*Duplicates and near duplicates identified in [MS-COCO](https://cocodataset.org/#home) and [Imagenet-21K](https://www.image-net.org) dataset*

## [Find corrupted and broken images](https://bit.ly/3nD3o5k)
![alt text](./gallery/imagenet21k_broken.png)
*Thousands of broken ImageNet images that have confusing labels of real objects.*

## [Find outliers](https://bit.ly/3bNiceX)
![alt text](./gallery/imdb_outliers-min.png)
*[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) outliers (data goal is for face recognition, gender and age classification)*


## [Find similar persons](https://bit.ly/3a9tud4)
![alt text](./gallery/viz0.png)
*Can you tell how many different persons?*


## [Find wrong labels](https://bit.ly/3bPbVPW)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imagenet21k_wrong_labels-min.png)
*Wrong labels in the [Imagenet-21K](https://www.image-net.org) dataset*.

## [Find confusing labels report](https://github.com/visualdatabase/fastdup/blob/main/examples/fastdup_wrong_labels.ipynb)
![alt test](./gallery/confusing_labels.jpg)
*Identify wrong / confusing labels using k-nearest neighbor visual classifier*


## [Find image with contradicting labels](https://bit.ly/3yE5kAI)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/red_wine.png)
*Cluster of wrong labels in the [Imagenet-21K](https://www.image-net.org) . No human can tell those red wines from their image.*

![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/imagenet21k_funny-min.png)
*Fun labels in the [Imagenet-21K](https://www.image-net.org) dataset*

## [Coming soon: image graph search (please reach out if you like to beta test)](https://bit.ly/3bJlIXC)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/viz23.png)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/viz2.png)
![alt text](https://github.com/visualdatabase/fastdup/blob/main/gallery/tmp13_viz5.png)

*Upcoming new features: image graph search!*


## Results on Key Datasets ([full results here](https://bit.ly/3nyQ3ef))
We have thoroughly tested fastdup across various famous visual datasets. Ranging from pillar Academic datasets to Kaggle competitions. A key finding we have made using FastDup is that there are ~1.2M (!) duplicate images on the ImageNet-21K dataset, out of which 104K pairs belong both to the train and to the val splits (this amounts to 20% of the validation set). This is a new unknown result! Full results are below. * train/val splits are taken from https://github.com/Alibaba-MIIL/ImageNet21 .

|Dataset	        |Total Images	|cost [$]|spot cost [$]|processing [sec]|Identical pairs|Anomalies|
|-----------------------|---------------|--------|-------------|----------------|---------------|---------|
|[imagenet21k-resized](https://www.image-net.org/challenges/LSVRC/)	|11,582,724	|4.98	|1.24	|11,561	|[1,194,059](https://www.databasevisual.com/imagenet-21k-resized-leaks)|[Anomalies](https://bit.ly/3OSJWx6) [Wrong Labels](https://bit.ly/3bMauSm)||
|[imdb-wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)	|514,883	|0.65	|0.16	|1,509	|[187,965](https://bit.ly/3OSK5AE)|[View](https://www.databasevisual.com/imdb-wiki-anomalies)|
|[places365-standard](http://places2.csail.mit.edu/download.html)	|2,168,460	|1.01	|0.25	|2,349|[93,109](https://www.databasevisual.com/places-365-leaks)|[View](https://bit.ly/3R4ceGH)|
|[herbarium-2022-fgvc9](https://www.kaggle.com/c/herbarium-2022-fgvc9)	|1,050,179	|0.69	|0.17	|1,598	|[33,115](https://www.databasevisual.com/herbarium-leaks)|[View](https://bit.ly/3OETEmT)|
|[landmark-recognition-2021](https://www.kaggle.com/c/landmark-recognition-2021)|1,590,815|0.96	|0.24	|2,236	|[2,613](https://www.databasevisual.com/landmarks-2021-leaks)|[View](https://bit.ly/3nwQbuN)|
|[visualgenome](https://visualgenome.org/)		|108,079	|0.05	|0.01	|124	|223|View|
|[iwildcam2021-fgvc9](https://www.kaggle.com/c/iwildcam2022-fgvc9/)	|261,428	|0.29	|0.07	|682	|[54](https://www.databasevisual.com/iwildcam2022-leaks)|[View](https://bit.ly/3Apwmx0)|
|[coco](https://cocodataset.org/#home)			|163,957	|0.09	|0.02	|218	|54|View|
|[sku110k](https://github.com/eg4000/SKU110K_CVPR19)		|11,743	|0.03	|0.01	|77	|[7](https://www.databasevisual.com/sku110k-leaks)|[View](https://bit.ly/3bLWOqa)|

* Experiments presented are on a 32 core Google cloud machine, with 128GB RAM (no GPU required).
* All experiments could be also reproduced on a 8 core, 32GB machine (excluding Imagenet-21K).
* We run on the full ImageNet-21K dataset (11.5M images) to compare all pairs of images in less than 3 hours WITHOUT a GPU (with Google cloud cost of 5$).

# Quick Installation 
For Python 3.7, 3.8, 3.9 (Ubuntu 20.04 or Ubuntu 18.04 or Debian 10 or Mac M1 or Mac Intel Mojave and up)
```python
# upgrade pip to its latest version
python3.XX -m pip install -U pip
# install fastdup
python3.XX -m pip install fastdup
```
Where XX is your python version.
For CentOS 7.X, RedHat 4.8 and other older Linux see our [Insallation instructions](./INSTALL.md).


# Running the code

```python
import fastdup
fastdup.run(input_dir="/path/to/your/folder", work_dir='out', nearest_neighbors_k=5, turi_param='ccthreshold=0.96')    #main running function.
fastdup.create_duplicates_gallery('out/similarity.csv', save_path='.')     #create a visual gallery of found duplicates
fastdup.create_outliers_gallery('out/outliers.csv',   save_path='.')       #create a visual gallery of anomalies
fastdup.create_components_gallery('out', save_path='.')                    #create visualiaiton of connected components
fastdup.create_stats_gallery('out', save_path='.', metric='blur')          #create visualization of images stastics (for example blur)
fastdup.create_similarity_gallery('out', save_path='.',get_label_func=lambda x: x.split('/')[-2])     #create visualization of top_k similar images assuming data have labels which are in the folder name
fastdup.create_aspect_ratio_gallery('out', save_path='.')                  #create aspect ratio gallery
```
[Full documentation is here](https://visualdatabase.github.io/fastdup/)


![alt text](./gallery/fastdup_clip_24s_crop.gif)
*Working on the Food-101 dataset. Detecting identical pairs, similar-pairs (search) and outliers (non-food images..)*

## Getting started examples
- [🔥 Finding duplicates, outliers and connected components in the Food-101 dataset, including Tensorboard Projector visualization - Google Colab](https://bit.ly/3ydvtVJ)
- [🔥🔥 Visualizing and understanding a new dataset, looking at dats outliers and label outliers, Training a baseline KNN classifier and getting to accuracy of 0.99 by removing confusing labels](https://www.kaggle.com/code/graphlab/horse-pork-meat-fastdup)  
- [Finding wrong lables via image similarity](./examples/fastdup_wrong_labels.ipynb)
- [Computing image statistics](./examples/fastdup_image_stats.ipynb)
- [Using your own onnx model for extraction](./examples/fastdup_model_support.ipynb)
- [Getting started on a Kaggle dataset](https://bit.ly/3OUqj7u)
- [Finding duplicates, outliers in the Food-101 datadset:](https://bit.ly/3yhoafo)
- [Analyzing video of the MEVA dataset - Google Colab](https://bit.ly/3yE29ZW)
- [Kaggle notebook - visualizing the pistachio dataset](https://bit.ly/3ygkg6C)

![Tensorboard Projector integration is explained in our Colab notebook](./gallery/tensorboard_projector.png)

## Detailed instructions
- [Detailed instructions, install from stable release and installation issues](https://bit.ly/3yDc2qw)
- [Detailed running instructions](https://bit.ly/3OFLlY5)

## User community contributions
![alt text](./gallery/anime.jpg)
*[FastDup based Anime Search Engine by Dorothy Walker](https://fbfa0c789b41e75.s3.amazonaws.com/visual-search-engine-anime.html)


# Support 
<a href="https://bit.ly/3OLojyT">Join our Slack channel</a>

# Technology
We build upon several excellent open source tools. [Microsoft's ONNX Runtime](https://github.com/microsoft/onnxruntime), [Facebook's Faiss](https://github.com/facebookresearch/faiss), [Open CV](https://github.com/opencv/opencv), [Pillow Resize](https://github.com/zurutech/pillow-resize), [Apple's Turi Create](https://github.com/apple/turicreate), [Minio](https://github.com/minio/minio), [Amazon's awscli](https://github.com/aws/aws-cli), [TensorBoard](https://github.com/tensorflow/tensorboard), [scikit-learn](https://github.com/scikit-learn/scikit-learn).

# About Us
<a href="https://www.linkedin.com/in/dr-danny-bickson-835b32">Danny Bickson</a>, <a href="https://www.linkedin.com/in/amiralush">Amir Alush</a><br>

