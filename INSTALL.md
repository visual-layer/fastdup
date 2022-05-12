# Installation
## Ubuntu 20.04 LTS Machine Setup
Required setup
- sudo apt update
- sudo apt -y install software-properties-common
- sudo add-apt-repository -y ppa:deadsnakes/ppa
- sudo apt update
- sudo apt -y install python3.8
- sudo apt -y install python3-pip
- pip install --upgrade pip



# Pip Package setup
Download the FastDup latest wheel from the following shared folder: `s3://visualdb`

Latest version: 0.25

## For pip (python 3.8) install using
```
pip install fastdup-<VERSION>-cp38-cp38-linux_x86_64.whl
```

## For conda (python 3.7.11) install using
```
conda install -y pandas tqdm opencv numpy
conda install fastdup-<VERSION>-py37_0.tar.bz
```


# Currently supported software/hardware

Operating system
- `Ubuntu 20.04 LTS`

Software versions
- `Python 3.8` (via pip) or `Python 3.7` (via pip or conda) or a `debian package` (Python is not required)

Hardware support
- CPU (GPU not needed!)



