
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

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[pypi-shield]: https://img.shields.io/pypi/pyversions/fastdup?style=for-the-badge
[pypi-url]: https://pypi.org/project/fastdup/
[pypiversion-shield]: https://img.shields.io/pypi/v/fastdup?style=for-the-badge
[downloads-shield]: https://img.shields.io/pypi/dm/fastdup?style=for-the-badge
[downloads-url]: https://pypi.org/project/fastdup/
[contributors-shield]: https://img.shields.io/github/contributors/visual-layer/fastdup?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[license-shield]: https://img.shields.io/github/license/visual-layer/fastdup?style=for-the-badge
[license-url]: https://github.com/visual-layer/fastdup/blob/main/LICENSE

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visual-layer/fastdup/main/gallery/fastdup%20newlogo%20trasparent2.png" alt="fastdup" width="400">
  </a>

<h3 align="center">Manage, Clean & Curate Visual Data - Fast and at Scale</h3>

  <p align="center">
  An unsupervised and free tool for image and video dataset analysis.
    <br />
    <a href="https://visual-layer.readme.io/"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://visual-layer.readme.io/">Features</a>
    ·
    <a href="https://github.com/visual-layer/fastdup/issues">Report Bug</a>
    ·
    <a href="https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22">Read Blog</a>
    ·
    <a href="https://visual-layer.readme.io/docs/getting-started">Quickstart</a>
    ·
    <a href="https://visual-layer.com/">Enterprise Edition</a>
    ·
    <a href="https://visual-layer.com/">About us</a>
    <br />
    <br /> 
    <a href="https://visualdatabase.slack.com/join/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA#/shared-invite/email">
    <img src="https://img.shields.io/badge/JOIN US ON SLACK-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Logo">
    </a>
    <a href="https://www.linkedin.com/company/visual-layer/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Logo">
    </a>
    <a href="https://www.youtube.com/@visual-layer4035">
    <img src="https://img.shields.io/badge/-YouTube-black.svg?style=for-the-badge&logo=youtube&colorB=red" alt="Logo">
    </a>
  </p>
  <br />
    🔥 We've release V1.0! View the release notes
    <a href="./RELEASE_NOTES.md">here</a>.
    <br />
</div>

## 🏅 What's Included
fastdup let's you identify -
 <div align="center" style="display:flex;flex-direction:column;">
  <a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visual-layer/fastdup/main/gallery/issues.png" alt="fastdup" width="1000">
  </a>
 </div>

Additional features  -

<div align="center" style="display:flex;flex-direction:column;">
  <a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visual-layer/fastdup/main/gallery/features.png" alt="fastdup" width="1000">
  </a>
 </div>


## 📌 Why fastdup?

+ **Quality** - fastdup can assist you in reaching a high quality dataset by finding and removing anomalies and outliers from your datasets. Finding duplicate and near duplicate of images (&videos) and finding clusters of similarity at a large scale!

+ **Cost** - fastdup can also help you in reducing your data operations costs by facilitating the intelligent sampling of high-quality or novel datasets prior to labeling, as well as support the quality assessment of labeled data.

+ **Scale** - fastdup graph engine is written in C++ and is highly efficient and works in an incredible scale! Running locally on a CPU only machine and can handle up to 400M images on a single CPU machine!

 
## ⚙ Setting up

- **Supported Python**: 3.7, 3.8, 3.9, 3.10
- **Supported OS**: Windows 10, 11 and 2019 Server (Native), Windows WSL, Ubuntu (20.04, 18.04), Mac OSX 10+ (Intel and M1 CPUs), Amazon Linux 2, CentOS 7, RedHat 4.8.
- [Full installation instructions are here](https://visual-layer.readme.io/docs/installation)


```python
# upgrade pip to its latest version
pip install -U pip

# install fastdup
pip install fastdup
    
# Alternatively, use explicit python version (XX)
python3.XX -m pip install fastdup 
```


## ✅ Getting Started
```python
import fastdup

fd = fastdup.create(work_dir, images_dir)
fd.run(nearest_neighbors_k=5, cc_threshold=0.96)

fd.vis.duplicates_gallery()     # create a visual gallery of found duplicates
fd.vis.outliers_gallery()       # create a visual gallery of anomalies
fd.vis.component_gallery()     # create visualiaiton of connected components
fd.vis.stats_gallery()          # create visualization of images stastics (for example blur)
```

![alt text](https://github.com/visual-layer/fastdup/blob/main/gallery/gifl_fastdup_quickstart_V1.gif?raw=true)
*Working on the Oxford Pet Dataset. Detecting identical pairs, similar-pairs (search) and outliers*


## 💣 Learn from Examples

- [Quick dataset analysis](https://visual-layer.readme.io/docs/getting-started) <a target="_blank" href="https://colab.research.google.com/drive/18gbpq8A62KAjJolCuRnOAmJCRGT1Vu1J#scrollTo=pN6wiKBax7Pa">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Cleaning and preparing a dataset](https://visual-layer.readme.io/docs/abc)  <a target="_blank" href="https://colab.research.google.com/drive/1NBTD_Z5beSlumQOqDPdF2UzrhdEf0uxC">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Preparing an image dataset for training](https://visual-layer.readme.io/docs/analyzing-labeled-images)  <a target="_blank" href="https://colab.research.google.com/drive/1LMbwD5QcXqqk8HSGfHu8m5o5KvG7MfGc">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Preparing an object dataset for training](https://visual-layer.readme.io/docs/objects-and-bounding-boxes)  <a target="_blank" href="https://colab.research.google.com/drive/1MwxalEbILkSUt3NXZRhc9bWjfIGFbC6p">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## 🆘 Getting Help
Get help from the fastdup team or community members via the following channels -
+ [Slack](https://visualdatabase.slack.com/join/shared_invite/zt-19jaydbjn-lNDEDkgvSI1QwbTXSY6dlA#/shared-invite/email).
+ GitHub [issues](https://github.com/visual-layer/fastdup/issues).
+ Discussion [forum](https://visual-layer.readme.io/discuss).

## 💯 Community and Contributions


- [Master Data Integrity to Clean Your Computer Vision Datasets
](https://towardsdatascience.com/master-data-integrity-to-clean-your-computer-vision-datasets-df432cf9e596)
- [fastdup: A Powerful Tool to Manage, Clean & Curate Visual Data at Scale on Your CPU - For Free.](https://dicksonneoh.com/portfolio/fastdup_manage_clean_curate/)
- [Clean Up Your Digital Life: Simplify Your Photo Organization and Say Goodbye to Photo Clutter](https://dicksonneoh.com/blog/clean_up_your_digital_life/)

## 🎹 License
fastdup is licensed under Creative Commons 4.0 license.
See [LICENSE](./LICENSE).

For any queries, reach us at info@visual-layer.com 

## What our users think about fastdup: 
<div align="center" style="display:flex;flex-direction:column;">
  <a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visual-layer/fastdup/main/gallery/tweet.png" alt="fastdup" width="1000">
  </a>
<a href="https://www.visual-layer.com">
    <img src="https://raw.githubusercontent.com/visual-layer/fastdup/main/gallery/tweet2.png" alt="fastdup" width="1000">
  </a>
 </div>


## 🎷 Disclaimer
<details>
  <summary><b>Usage Tracking</b></summary>

We have added experimental crash report collection, using [sentry.io](https://github.com/getsentry/). It does not collect user data other than anonymized IP address data, and it only logs fastdup library's own actions. We do NOT collect folder name, user name, image names, image content only aggregate performance statistics like total number of images, average runtime per image, total free memory, total free disk space, number of cores etc. Collecting fastdup crashes will help us improve stability. 

The code for the data collection is found [here](./src/sentry.hpp). On MAC we use [Google crashpad](https://chromium.googlesource.com/crashpad/crashpad). 

It is always possible to opt out of the experimental crash report collection via either of the following two options:
- Define an environment variable called `SENTRY_OPT_OUT`
- or run() with `turi_param='run_sentry=0'`

</details>

## 🧭 About Visual-Layer
fastdup is founded by the authors of [XGBoost](https://github.com/apache/tvm), [Apache TVM](https://github.com/apache/tvm) & [Turi Create](https://github.com/apple/turicreate). 
<a href="https://www.linkedin.com/in/dr-danny-bickson-835b32">Danny Bickson</a>, <a href="https://www.linkedin.com/in/carlos-guestrin-5352a869/">Carlos Guestrin</a> & <a href="https://www.linkedin.com/in/amiralush">Amir Alush</a><br>

Learn more about Visual Layer [here](https://visual-layer.com).


