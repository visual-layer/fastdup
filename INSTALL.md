# Installation

##### Table of Contents  

1. [Ubuntu 20.04/18.04 Preliminaries](#ubuntu)  
2. [Mac OSX Preliminaries](#macosx)
3. [CentOS 7 Preliminaries](#centos7)
4. [Amazon Linux 2](#amazon_linux)
5. [Windows Server 10 Preliminaries](#windows10)
6. [Pypi setup](#pypi)
7. [Preinstalled docker](#docker)
8. [Common installation errors](#common)

fastdup is currently supported on Ubuntu 20.04 or 18.04 OS, CentOS 7.9, Mac OS 10.X Intel chip, Mac OS 11.X M1 chip, Windows 10 Server (via WSL).


## Ubuntu 20.04/18.04 LTS Machine Setup <a name="ubuntu">
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

## Mac OS setup <a name="macosx">
```bash
brew install ffmpeg@4
```

## CentOS 7 Setup <a name="centos7">
```bash
sudo yum -y install epel-release
sudo yum -y update
sudo yum -y groupinstall "Development Tools"
sudo yum -y install openssl-devel bzip2-devel libffi-devel xz-devel
sudo yum -y install wget
sudo yum install redhat-lsb-core # for lsb_release
sudo yum install -y ffmpeg ffmpeg-devel # for video support
```
Download and istall CentOS 7 whl image from our [release page](https://github.com/visual-layer/fastdup/releases).
```
python3.7 -m pip install <path of the downloaded whl>
```

## Amazon Linux 2 Setup <a name="amazon_linux">
```bash
sudo yum install mesa-libGL -y
```
Download and isntall CentOS 7 whl image from our [release page](https://github.com/visual-layer/fastdup/releases).
```
python3.7 -m pip install <path of the downloaded whl>
```

## Windows 10 Server Setup <a name="windows10">

### Setting up WSL. The below instructions are for Windows Server 10. More detailed instructions are [here](https://learn.microsoft.com/en-us/windows/wsl/install-on-server).
For Windows 10+11 follow the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install).

- Enable WSL on your machine (Search -> powershell-> right click -> run as administrator)
```
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
```
- Reboot your machine
- Check that wsl is enabled using the command:
```
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
```

- Download ubuntu 18.04 from https://aka.ms/wslubuntu1804
- After the download, run powershell as admin, goto your download folder (for example c:\\users/danny_bickson/downloads/)
```
cd c:\\users\danny_bickson\Download # change to your download folders
Expand-Archive .\Ubuntu_1804.2019.522.0_x64.zip .\Ubuntu_1804
cd .\Ubuntu_1804\
 .\ubuntu1804.exe
```

This will take a few minutes, you will see an output of the kind:
```
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: danny_bickson # (chnage to your username)
Enter new UNIX password: *******
Retype new UNIX password: *******
passwd: password updated successfully
Installation successful!
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.
```

### Optional: Update WSL to version 2 (recommended, significant performance wins!)

- Download the installer from https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
- Run the installer and follow the instructions

### Once WSL and Ubuntu 18.04 are set up, continue with the below instructions

- Inside the Ubuntu shell, run the following installers (you will be asked for password you entered before).
```
sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.8
sudo apt -y install python3-pip
sudo apt -y install libgl1-mesa-glx
pip3 install --upgrade pip
python3.8 -m pip install fastdup
```

In case python fails to find fastdup, do the following:
- Go to the latest version in pypi (for example https://pypi.org/project/fastdup/0.143/#files)
- Download the file cp38-cp38-manylinux_2_27_x86_64.whl
- python3.8 -m pip install fastdup-0.143-cp38-cp38-manylinux_2_27_x86_64.whl 



# Pip Package setup

## Using pypi <a name="pypi">
This is the recommended installation method for all Mac, Ubuntu 18-20, Debian 10, Windows Server 10.
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


# Debian package install
- download the latest deb for your system from our [release page](https://github.com/visualdatabase/fastdup/releases). Assuming the wheel file is found in your working folder, run:
```bash
sudo dpkg -i fastdup-<VERSION>-ubuntu-20.04.deb
```
Application name is fastdup.

# Docker <a name="docker">

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
- `Windows 10 Server` (via WSL)

Software versions
- `Python 3.7, 3.8, 3.9` (via pip) 

Hardware support
- CPU (GPU not needed!)




# Common installation issues and their solution <a name="common">

ERROR: fastdup-0.39-cp38-cp38-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
- Check that you are on ubuntu 20.04 or 18.04 (via the command `lsb_release -r`). Alternatively on Mac M1 Big Sur or Mac Intel Mojave (use the command `sw_vers`) 
- Check that you are using the right python version (python3.8 and not python) 
- Make sure pip is up to date using `python3.8 -m pip install -U pip`). 
- Make sure you install using `python3.8 -m pip install..` and not just `pip install...`.
- If that does not work, please open an issue with the otuput of `python3.8 -m pip debug --verbose` or join our slack channel.

ERROR on Ubuntu: `libGL.so.1: cannot open shared object file: No such file or directory`
- Need to install depedency: `sudo apt -y install libgl1-mesa-glx`

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


