# Installation
fastdup is currently only supported on Ubuntu 20.04 OS.


## Ubuntu 20.04 LTS Machine Setup
Required machine setup
```bash
sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.8
sudo apt -y install python3-pip
pip install --upgrade pip
```


# Pip Package setup
Download the FastDup latest wheel from the following shared folder: `s3://visualdb`

Latest version: 0.33

## For pip (python 3.8 or 3.7) install using
```bash
wget https://github.com/visualdatabase/fastdup/releases/download/v0.33/fastdup-0.33-cp37-cp37m-linux_x86_64.whl
or
wget https://github.com/visualdatabase/fastdup/releases/download/v0.33/fastdup-0.33-cp38-cp38-linux_x86_64.whl

python3.8 -m pip install *.whl
```

## For conda (python 3.7.11) install using
```bash
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


# Common installation issues and their solution

ERROR: fastdup-0.39-cp38-cp38-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
- Check that you are on ubuntu 20.04 (via the command `uname -a`). 
- Check that you are using the right python version (python3.8 and not python). 
- Make sure pip is up to date using `python3.8 -m pip install -U pip`). 
- Make sure you install using `python3.8 -m pip install..` and not just `pip install...`.
