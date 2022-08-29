# Installation
fastdup is currently supported on Ubuntu 20.04 or 18.04 OS, CentOS 7.9, Mac OS 10.X Intel chip, Mac OS 11.X M1 chip.


## Ubuntu 20.04/18.04 LTS Machine Setup
Required machine setup
```bash
sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.8
sudo apt -y install python3-pip
sudo apt -y install libgl1-mesa-glx
python3.8 -m pip install --upgrade pip
```

## Mac OS setup
```bash
brew install ffmpeg@4
```

## CentOS 7 Setup
```bash
sudo yum -y install epel-release
sudo yum -y update
sudo yum -y groupinstall "Development Tools"
sudo yum -y install openssl-devel bzip2-devel libffi-devel xz-devel
sudo yum -y install wget
sudo yum install redhat-lsb-core # for lsb_release
sudo yum install -y ffmpeg ffmpeg-devel # for video support
```

# Pip Package setup

## Using pypi
This is the recommended installation method for all Mac, Ubuntu 18-20, Debian 10.
Will not work for Centos 7.9, RedHat 4.8.

```bash
python3.8 -m pip install -U pip
python3.8 -m pip install fastdup
```

Note: you may need to upgrade your pip, using the command `python3.8 -m pip install -U pip`.

## Using stable release.
This is mandatory for CentOS 7.9 / RedHat 4.8 and similiar OS.

- download the latest wheel for your system from our [release page](https://github.com/visualdatabase/fastdup/releases). Assuming the wheel file is found in your working folder, run:

```bash
python3.8 -m pip install *.whl
```

Note: you may need to upgrade your pip, using the command `python3.8 -m pip install -U pip`.

# Conda setup (Python 3.7 only)
## Using Anaconda channels:
```bash
conda install -y pandas tqdm opencv numpy
conda install -c dbickson fastdup
```

## Using stable release
- download the latest bz2 for your system from our [release page](https://github.com/visualdatabase/fastdup/releases). Assuming the wheel file is found in your working folder, run:
```bash
conda install -y pandas tqdm opencv numpy
conda install fastdup-<VERSION>-py37_0.tar.bz
```

Note: don't forget to replace the <VERSION> with the latest version for example 0.45


# Debian package install
- download the latest deb for your system from our [release page](https://github.com/visualdatabase/fastdup/releases). Assuming the wheel file is found in your working folder, run:
```bash
sudo dpkg -i fastdup-<VERSION>-ubuntu-20.04.deb
```
Application name is fastdup.

# Docker

##Pull from docker hub the latest ubuntu

```bash
docker pull karpadoni/fastdup-ubuntu-20.04
```


## Build your own docker

```bash
docker build -f Dockerfile -t fastdup-ubuntu .
```




# Currently supported software/hardware

Operating system
- `Ubuntu 20.04 LTS`
- `Ubuntu 18.04 LTS`
- `Mac OSX M1 Chip` (tested on Big Sur)
- `Mac Intel Chip` (tested on Mojave)
- `CentOS 7.9` (via stable release)

Software versions
- `Python 3.7, 3.8, 3.9` (via pip) 

Hardware support
- CPU (GPU not needed!)




# Common installation issues and their solution

ERROR: fastdup-0.39-cp38-cp38-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
- Check that you are on ubuntu 20.04 or 18.04 (via the command `lsb_release -r`). Alternatively on Mac M1 Big Sur or Mac Intel Mojave (use the command `sw_vers`) 
- Check that you are using the right python version (python3.8 and not python) 
- Make sure pip is up to date using `python3.8 -m pip install -U pip`). 
- Make sure you install using `python3.8 -m pip install..` and not just `pip install...`.
- If that does not work, please open an issue with the otuput of `python3.8 -m pip debug --verbose` or join our slack channel.

ERROR on Ubuntu: `libGL.so.1: cannot open shared object file: No such file or directory`
- Need to install depedency: `sudo apt -y nstall libgl1-mesa-glx`

Error on Mac+conda: `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
zsh: abort      python3`
- Solution from [StackOverflow](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)
- You should install all packages without MKL support:
```
conda install nomkl
conda install numpy scipy pandas tensorflow
conda remove mkl mkl-service # may fail, don't worry
```

Error on Mac M1: `AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)`
Solution: downgrade your cv2 version to 4.5.5.64 using the command `python.XX -m pip install -U opencv-python==4.5.5.64` where XX is your python version.

Error on Mac M1: library not loaded when trying to import cv2 `ImportError: dlopen(/Users/mikasnoopy/homebrew/lib/python3.9/site-packages/cv2/python-3.9/cv2.cpython-39-darwin.so, 2): Library not loaded: /Users/mikasnoopy/homebrew/opt/dav1d/lib/libdav1d.5.dylib
  Referenced from: /Users/mikasnoopy/homebrew/opt/ffmpeg@4/lib/libavcodec.58.dylib
  Reason: image not found
` or any similar error.
Solution: Downgrade your ffmpg using `brew remove ffmpeg; brew install ffmpeg@4`


