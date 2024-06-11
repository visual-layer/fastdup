<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![PyPi][pypi-shield]][pypi-url]
[![PyPi][pypiversion-shield]][pypi-url]
[![PyPi][downloads-shield]][downloads-url]
[![Contributors][contributors-shield]][contributors-url]
[![License][license-shield]][license-url]
[![OS][os-shield]][os-url]



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[pypi-shield]: https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-blue?style=for-the-badge
[pypi-url]: https://pypi.org/project/fastdup/
[pypiversion-shield]: https://img.shields.io/pypi/v/fastdup?style=for-the-badge&color=success
[downloads-shield]: https://img.shields.io/pypi/dm/fastdup?style=for-the-badge&color=lightblue
[downloads-url]: https://pypi.org/project/fastdup/
[contributors-shield]: https://img.shields.io/github/contributors/visual-layer/fastdup?style=for-the-badge&color=orange
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[license-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-purple.svg?style=for-the-badge
[license-url]: https://github.com/visual-layer/fastdup/blob/main/LICENSE
[os-shield]: https://img.shields.io/badge/Supported%20OS-macOS%20%7C%20Linux%20%7C%20Windows%20-yellow?style=for-the-badge&logo=windows&logoColor=white
[os-url]: https://visual-layer.readme.io/docs/installation

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://www.visual-layer.com" target="_blank" rel="noopener noreferrer" name="top">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./gallery/logo_dark_mode.png" width=400>
    <source media="(prefers-color-scheme: light)" srcset="./gallery/logo.png" width=400>
    <img alt="fastdup logo." src="./gallery/logo.png">
    </picture>
  </a>

<h3 align="center">Manage, Clean & Curate Visual Data - Fast and at Scale.</h3>
  <p align="center">
  An unsupervised and free tool for image and video dataset analysis.
<p>Founded by the authors of <a href="https://github.com/apache/tvm">XGBoost</a>, <a href="https://github.com/apache/tvm">Apache TVM</a> & <a href="https://github.com/apple/turicreate">Turi Create</a> - <a href="https://www.linkedin.com/in/dr-danny-bickson-835b32">Danny Bickson</a>, <a href="https://www.linkedin.com/in/carlos-guestrin-5352a869">Carlos Guestrin</a> and <a href="https://www.linkedin.com/in/amiralush">Amir Alush</a>.</p>
    <br />
    <br />
    <a href="https://visual-layer.readme.io/" target="_blank" rel="noopener noreferrer"><strong>Explore the docs »</strong></a>
    <br />
    <a href="#whats-included-in-fastdup" target="_blank" rel="noopener noreferrer">Features</a>
    ·
    <a href="https://github.com/visual-layer/fastdup/issues/new/choose" target="_blank" rel="noopener noreferrer">Report Bug</a>
    ·
    <a href="https://medium.com/visual-layer" target="_blank" rel="noopener noreferrer">Blog</a>
    ·
    <a href="https://visual-layer.readme.io/docs/getting-started" target="_blank" rel="noopener noreferrer">Quickstart</a>
    ·
    <a href="https://app.visual-layer.com/vl-datasets?utm_source=fastdup_readme" target="_blank" rel="noopener noreferrer">Visual Layer Cloud</a>
    ·
    <a href="https://visual-layer.com/about" target="_blank" rel="noopener noreferrer">About us</a>
    <br />
    <br /> 
    <a href="https://discord.gg/Dqw458EG" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/DISCORD%20COMMUNITY-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Logo">
    </a>
    <a href="https://visual-layer.readme.io/discuss" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/DISCUSSION%20FORUM-slateblue?style=for-the-badge&logo=discourse&logoWidth=20" alt="Logo">
    </a>
    <a href="https://www.linkedin.com/company/visual-layer/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Logo">
    </a>
    <a href="https://twitter.com/visual_layer" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/X%20(TWITTER)-000000?style=for-the-badge&logo=x&logoColor=white" alt="Logo">
    </a>
    <a href="https://www.youtube.com/@visual-layer" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/-YouTube-black.svg?style=for-the-badge&logo=youtube&colorB=red" alt="Logo">
    </a>
  </p>
  <br />
  <br />


<br />
</div>

## Getting Started

`pip` install fastdup from [PyPI](https://pypi.org/project/fastdup/):

```bash
pip install fastdup
```

More installation options are available [here](https://visual-layer.readme.io/docs/installation).

Initialize and run fastdup:
```python
import fastdup

fd = fastdup.create(input_dir="IMAGE_FOLDER/")
fd.run()
```

Explore the results in a interactive web UI:

```python
fd.explore()   
```


![run](./gallery/fastdup_install.gif)

Alternatively, visualize the result in a static gallery:

```python
fd.vis.duplicates_gallery()    # gallery of duplicates
fd.vis.outliers_gallery()      # gallery of outliers
fd.vis.component_gallery()     # gallery of connected components
fd.vis.stats_gallery()         # gallery of image statistics (e.g. blur, brightness, etc.)
fd.vis.similarity_gallery()    # gallery of similar images
```

![results](./gallery/gifl_fastdup_quickstart_V1_optimized.gif)


## Features & Advantages
fastdup handles labeled/unlabeled image/video datasets providing the following features:

<div align="center" style="display:flex;flex-direction:column;">
  <a href="https://www.visual-layer.com" target="_blank" rel="noopener noreferrer">
    <img src="./gallery/fastdup_features_new.png" alt="fastdup" width="1000">
  </a>
 </div>


What sets fastdup apart from other similar tools: 

+ 🎯 **Quality**: High-quality analysis to remove duplicates/near-duplicates, anomalies, mislabels, broken images, and poor-quality images.
+ 📊 **Scale**: Handles 400M images on a single CPU machine. Scales to billions of images.
+ 🚀 **Speed**: Highly optimized C++ engine runs efficiently even on low-resource CPU machines.
+ 🔒 **Privacy**: Runs locally or on your cloud infrastructure. Your data stays where it is.
+ 😊 **Ease of use**: Works on labeled or unlabeled datasets, images, or videos. Supported on major operating systems: MacOS, Linux and Windows.


## Learn from Examples
Learn the basics of fastdup through interactive examples. View the notebooks on GitHub or nbviewer. Even better, run them on Google Colab or Kaggle, for free.



<table>
   <tr>
      <td rowspan="4" width="160">
         <a href="https://visual-layer.readme.io/docs/getting-started">
         <img src="./gallery/cat_dog_thumbnail.jpg" width="200">
         </a>
      </td>
      <td rowspan="4">
         <b>⚡ Quickstart:</b> Learn how to install fastdup, load a dataset and analyze it for potential issues such as duplicates/near-duplicates, broken images, outliers, dark/bright/blurry images, and view visually similar image clusters. If you're new, start here!
         <br>
         <br>
         <b>📌 Dataset:</b> <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a>.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/visual-layer/fastdup/blob/main/examples/quick-dataset-analysis.ipynb">
         <img src="./gallery/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/visual-layer/fastdup/blob/main/examples/quick-dataset-analysis.ipynb">
         <img src="./gallery/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/visual-layer/fastdup/blob/main/examples/quick-dataset-analysis.ipynb">
         <img src="./gallery/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/visual-layer/fastdup/blob/main/examples/quick-dataset-analysis.ipynb">
         <img src="./gallery/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
   <tr>
      <td rowspan="4" width="160">
         <a href="https://visual-layer.readme.io/docs/cleaning-image-dataset">
         <img src="gallery/food_thumbnail.jpg" width="200">
         </a>
      </td>
      <td rowspan="4">
         <b>🧹 Clean Image Folder:</b> Learn how to analyze and clean a folder of images from potential issues and export a list of problematic files for further action. If you have an unorganized folder of images, this is a good place to start.
         <br>
         <br>
         <b>📌 Dataset:</b> <a href="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/">Food-101</a>.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/visual-layer/fastdup/blob/main/examples/cleaning-image-dataset.ipynb">
         <img src="./gallery/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/visual-layer/fastdup/blob/main/examples/cleaning-image-dataset.ipynb">
         <img src="./gallery/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/visual-layer/fastdup/blob/main/examples/cleaning-image-dataset.ipynb">
         <img src="./gallery/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/visual-layer/fastdup/blob/main/examples/cleaning-image-dataset.ipynb">
         <img src="./gallery/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
   <tr>
      <td rowspan="4" width="160">
         <a href="https://visual-layer.readme.io/docs/analyzing-labeled-images">
         <img src="./gallery/imagenette_thumbnail.jpg" width="200">
         </a>
      </td>
      <td rowspan="4">
         <b>🖼 Analyze Image Classification Dataset:</b> Learn how to load a labeled image classification dataset and analyze for potential issues. If you have labeled ImageNet-style folder structure, have a go!
         <br>
         <br>
         <b>📌 Dataset:</b> <a href="https://github.com/fastai/imagenette">Imagenette</a>.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/visual-layer/fastdup/blob/main/examples/analyzing-image-classification-dataset.ipynb">
         <img src="./gallery/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/visual-layer/fastdup/blob/main/examples/analyzing-image-classification-dataset.ipynb">
         <img src="./gallery/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/visual-layer/fastdup/blob/main/examples/analyzing-image-classification-dataset.ipynb">
         <img src="./gallery/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/visual-layer/fastdup/blob/main/examples/analysing-image-classification-dataset.ipynb">
         <img src="./gallery/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
   <tr>
      <td rowspan="4" width="160">
         <a href="https://visual-layer.readme.io/docs/objects-and-bounding-boxes">
         <img src="./gallery/coco_thumbnail.jpg" width="200">
         </a>
      </td>
      <td rowspan="4">
         <b>🎁 Analyze Object Detection Dataset:</b> Learn how to load bounding box annotations for object detection and analyze for potential issues. If you have a COCO-style labeled object detection dataset, give this example a try. 
         <br>
         <br>
         <b>📌 Dataset:</b> <a href="https://cocodataset.org/#home">COCO</a>.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/visual-layer/fastdup/blob/main/examples/analyzing-object-detection-dataset.ipynb">
         <img src="./gallery/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/visual-layer/fastdup/blob/main/examples/analyzing-object-detection-dataset.ipynb">
         <img src="./gallery/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/visual-layer/fastdup/blob/main/examples/analyzing-object-detection-dataset.ipynb">
         <img src="./gallery/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/visual-layer/fastdup/blob/main/examples/analyzing-object-detection-dataset.ipynb">
         <img src="./gallery/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
</table>

See more [examples](EXAMPLES.md).


## Join the Community

Get help from the fastdup team or community members via the following channels -
+ [Discord](https://discord.gg/Dqw458EG) chat.
+ GitHub [issues](https://github.com/visual-layer/fastdup/issues).
+ Discussion [forum](https://visual-layer.readme.io/discuss).


Community-contributed blog posts on fastdup:

<table>
  <tr>
    <td><img src="gallery/community_aws_lambda_docker.jpg" width="200"></td>
    <td>
      <a href="https://medium.com/@atahanbulus.w/deploying-aws-lambda-functions-with-docker-container-by-using-custom-base-image-2d110d307f9b">Deploying AWS Lambda functions with Docker Container by using Custom Base Image</a><br>
      🖋️ <a href="https://medium.com/@atahanbulus.w">atahan bulus</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 16 September 2023
    </td>
  </tr>
  <tr>
    <td><img src="gallery/community_cleaning_image_spotlight.jpg" width="200"></td>
    <td>
      <a href="https://medium.com/@daniel-klitzke/cleaning-image-classification-datasets-with-fastdup-and-renumics-spotlight-e68deb4730a3">Renumics: Cleaning Image Classification Datasets With fastdup and Renumics Spotlight</a><br>
      🖋️ <a href="https://medium.com/@daniel-klitzke">Daniel Klitzke</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 4 September 2023
    </td>
  </tr>
  <tr>
    <td><img src="gallery/community_reduce_dataset_thumbnail.jpg" width="200"></td>
    <td>
      <a href="https://blog.roboflow.com/how-to-reduce-dataset-size-computer-vision/">Roboflow: How to Reduce Dataset Size Without Losing Accuracy</a><br>
      🖋️ <a href="https://blog.roboflow.com/author/arty/">Arty Ariuntuya</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 9 August 2023
    </td>
  </tr>
  <tr>
    <td><img src="gallery/community_weighted_thumbnail.jpg" width="200"></td>
    <td>
      <a href="https://alexlanseedoo.medium.com/the-weighty-significance-of-data-cleanliness-eb03dce1d0f8">The weighty significance of data cleanliness — or as I like to call it, “cleanliness is next to model-ness” — cannot be overstated.</a><br>
      🖋️ <a href="https://alexlanseedoo.medium.com/">Alexander Lan</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 9 March 2023
    </td>
  </tr>
  <tr>
    <td><img src="gallery/community_cleanup_thumbnail.gif" width="200"></td>
    <td>
      <a href="https://dicksonneoh.com/blog/clean_up_your_digital_life/">Clean Up Your Digital Life: How I Found 1929 Fully Identical Images, Dark, Bright and Blurry Shots in Minutes, For Free.</a><br>
      🖋️ <a href="https://medium.com/@dickson.neoh">Dickson Neoh</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 23 February 2023
    </td>
  </tr>
  <tr>
    <td><img src="gallery/community_powerful_thumbnail.gif" width="200"></td>
    <td>
      <a href="https://dicksonneoh.com/portfolio/fastdup_manage_clean_curate/">fastdup: A Powerful Tool to Manage, Clean & Curate Visual Data at Scale on Your CPU - For Free.</a><br>
      🖋️ <a href="https://medium.com/@dickson.neoh">Dickson Neoh</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 3 January 2023
    </td>
  </tr>
  <tr>
    <td><img src="gallery/community_data_integration_thumbnail.jpg" width="200"></td>
    <td>
      <a href="https://towardsdatascience.com/master-data-integrity-to-clean-your-computer-vision-datasets-df432cf9e596">Master Data Integrity to Clean Your Computer Vision Datasets.</a><br>
      🖋️ <a href="https://pauliusztin.medium.com/">Paul lusztin</a> &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; 🗓 19 December 2022
    </td>
  </tr>
</table>


What our users say:

![feedback](./gallery/user_quotes.jpg)

![feedback2](./gallery/feedback2.png)

## License
fastdup is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/) Public License. 

> [!NOTE]
> Under this license, you are free to:
> + **Share** — copy and redistribute the material in any medium or format.
>
> Under the following terms:
> + **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
> + **NonCommercial** — You may not use the material for commercial purposes.
> + **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

For any more information or inquiries regarding the license, please contact us at info@visual-layer.com or see the [LICENSE](./LICENSE) file.

## Disclaimer
<details>
  <summary><b>Usage Tracking</b></summary>

We have added an experimental crash report collection, using [sentry.io](https://github.com/getsentry/). It does not collect user data  and it only logs fastdup library's own actions. We do NOT collect folder names, user names, image names, image content only aggregate performance statistics like total number of images, average runtime per image, total free memory, total free disk space, number of cores, etc. Collecting fastdup crashes will help us improve stability. 

The code for the data collection is found [here](./fastdup/sentry.py). On MAC we use [Google crashpad](https://chromium.googlesource.com/crashpad/crashpad) to report crashes.

It is always possible to opt out of the experimental crash report collection via either of the following two options:
- Define an environment variable called `SENTRY_OPT_OUT`
- or run() with `turi_param='run_sentry=0'`

</details>

## Visual Layer Cloud
Visual Layer offers commercial services for managing, cleaning, and curating visual data at scale. 

[Sign-up](https://app.visual-layer.com?utm_source=fastdup_readme) for free. 


https://github.com/visual-layer/fastdup/assets/6821286/57f13d77-0ac4-4c74-8031-07fae87c5b00

Not convinced? Interact with [Visual Layer Cloud](https://app.visual-layer.com/vl-datasets?utm_source=fastdup_readme) public dataset with no sign-up required.

## About Visual-Layer

<div align="center">
<a href="https://www.visual-layer.com" target="_blank" rel="noopener noreferrer">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./gallery/visual_layer_logo_dark_mode.png" width=250>
    <source media="(prefers-color-scheme: light)" srcset="./gallery/visual_layer_logo.png" width=250>
    <img alt="Visual Layer logo." src="./gallery/visual_layer_logo.png">
    </picture>
</a>
</div>

<div align="center">
   <a href="https://visual-layer.com/about" target="_blank" style="text-decoration: none;"> About Us </a> •
    <a href="https://medium.com/visual-layer" target="_blank" style="text-decoration: none;"> Blog </a> •
    <a href="https://visual-layer.readme.io/" target="_blank" style="text-decoration: none;"> Documentation </a>
    
</div>

<div align="center">
    <a href="https://discord.gg/Dqw458EG" target="_blank" style="text-decoration: none;"> Discord Community </a> •
    <a href="https://visual-layer.readme.io/discuss" target="_blank" style="text-decoration: none;"> Discussion Forum </a> •
    <a href="https://www.linkedin.com/company/visual-layer/" target="_blank" style="text-decoration: none;"> LinkedIn </a> •
    <a href="https://twitter.com/visual_layer" target="_blank" style="text-decoration: none;"> X (Twitter) </a>
</div>

<div align="right"><a href="#top">🔝 Back to Top</a></div>

