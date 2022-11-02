
<div align="center" style="display:flex;flex-direction:column;">
  <a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visualdatabase/fastdup/readme_v1/gallery/fastdup_logo.png" alt="fastdup" width="500">
  </a>
  <h1>Easily Manage, Clean & Curate Visual Data  at Scale</h1>
 </div>
 
 **fastdup** is a tool for gaining insights from a large image/video collection. It can find anomalies, duplicate and near duplicate images/videos, clusters of similarity, learn the normal behavior and temporal interactions between images/videos. It can be used for smart subsampling of a higher quality dataset,  outlier removal, novelty detection of new information to be sent for tagging. 
 

**fastdup** is:

 - <font size=10> **Unsupervised**:</font>  <font size=5> fits any dataset </font>
 - <font size=10> **Scalable** :</font> <font size=5> handles more than 400M images </font>
 - <font size=10> **Efficient**:</font> <font size=5> works on **CPU only** </font>
 - <font size=10> **Low Cost**: </font> <font size=5> can process 12M images on a $1 CPU machine </font>


 
 From the authors of [GraphLab](https://github.com/jegonzal/PowerGraph) and [Turi Create](https://github.com/apple/turicreate).

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


<a href="https://bit.ly/3NJLxEe">Large Image Datasets Today are a Mess Blog </a> | <a href="https://www.youtube.com/watch?v=s6qamoFzyis&t=2s">Processing LAION400m Video </a><br>

<br>
 <h2> Fastdup solves the following problems:</h2>
 <div align="center" style="display:flex;flex-direction:column;">
  <a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visualdatabase/fastdup/readme_v1/gallery/fastdup_features-min.png" alt="fastdup" width="900">
  </a>
 </div>

</br>
<h2> Just 2 lines of code to get you started:</h2>
<div align="center" style="display:flex;flex-direction:column;">
 <a href="https://www.youtube.com/watch?v=s6qamoFzyis&t=2s">
    <img src="https://raw.githubusercontent.com/visualdatabase/fastdup/readme_v1/gallery/fastdup_run.gif" alt="fastdup" width="700">
  </a>
  
  <div align="left" style="display:flex;flex-direction:column;">
  

# Quick installation 
- Python 3.7, 3.8, 3.9 
- Supported OS: Ubuntu 20.04, Ubuntu 18.04, Debian 10, Mac OSX M1,  Mac OSX Intel, Windows 10 Server.

```python
# upgrade pip to its latest version
python3.XX -m pip install -U pip
# install fastdup
python3.XX -m pip install fastdup
```
Where XX is your python version.
For Windows, CentOS 7.X, RedHat 4.8 and other older Linux see our [Insallation instructions](./INSTALL.md).

# Full documentation
[Full documentation is here](https://visualdatabase.github.io/fastdup/)


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

![alt text](./gallery/fastdup_clip_24s_crop.gif)
*Working on the Food-101 dataset. Detecting identical pairs, similar-pairs (search) and outliers (non-food images..)*

## Getting started examples
- [🔥 Finding duplicates, outliers and connected components in the Food-101 dataset, including Tensorboard Projector visualization - Google Colab](https://bit.ly/3ydvtVJ)
- [🔥🔥 Visualizing and understanding a new dataset, looking at dats outliers and label outliers, Training a baseline KNN classifier and getting to accuracy of 0.99 by removing confusing labels](https://www.kaggle.com/code/graphlab/horse-pork-meat-fastdup)  
- [Finding wrong lables via image similarity](./examples/fastdup_wrong_labels.ipynb)
- [Computing image statistics](./examples/fastdup_image_stats.ipynb)
- [Using your own onnx model for extraction](./examples/fastdup_model_support.ipynb)
- [Getting started on a Kaggle dataset](https://bit.ly/3OUqj7u)
- [Deduplication of videos - Google Colab](https://github.com/visualdatabase/fastdup/tree/main/examples/fastdup_video.ipynb)
- [Analyzing video of the MEVA dataset - Google Colab](https://bit.ly/3yE29ZW)
- [Working with multipe labels per image](https://github.com/visualdatabase/fastdup/blob/main/examples/fastdup_peta.ipynb)

## Detailed instructions
- [Detailed instructions, install from stable release and installation issues](https://bit.ly/3yDc2qw)
- [Detailed running instructions](https://bit.ly/3OFLlY5)

## User community contributions
[Stroke AIS Data](https://www.kaggle.com/code/mpwolke/stroke-ais-fastdup)
[Tire Data](https://www.kaggle.com/code/taranmarley/fastdup-image-insights)
[Butterfly Mimics](https://www.kaggle.com/code/mpwolke/butterfly-mimics-fastdup)
[Drugs and Vitamins](https://www.kaggle.com/code/mpwolke/drugs-and-vitamins-fastdup)
[Plastic Bottles](https://www.kaggle.com/code/mpwolke/plastic-bottles-fastdup)
[Micro Organisms](https://www.kaggle.com/code/mpwolke/micro-organism-fastdup)
[PCB Boards](https://www.kaggle.com/code/mpwolke/pcb-boards-fastdup)
[ZebraFish](https://www.kaggle.com/code/mpwolke/danio-rerio-zebrafish-fastdup)
[Whats the difference](https://www.kaggle.com/code/ovednagar/whats-the-difference)

# Support and feature requests 
<a href="https://bit.ly/3OLojyT">Join our Slack channel</a>

# fastdup enterprise edition
 <a href="https://www.visual-layer.com">Visual Layer</a>

# About us
<a href="https://www.linkedin.com/in/dr-danny-bickson-835b32">Danny Bickson</a>, <a href="https://www.linkedin.com/in/amiralush">Amir Alush</a><br>

</div>
