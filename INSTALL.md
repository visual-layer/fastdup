# Installation
fastdup is currently only supported on Ubuntu 20.04 or 18.04 OS.


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
pip install --upgrade pip
```


# Pip Package setup

## Using pypi

```bash
python3.8 -m pip install fastdup
```

## Using stable release

- download the latest wheel for your system from our [release page](https://github.com/visualdatabase/fastdup/releases). Assuming the wheel file is found in your working folder, run:

```bash
python3.8 -m pip install *.whl
```

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

# Currently supported software/hardware

Operating system
- `Ubuntu 20.04 LTS`
- `Ubuntu 18.04 LTS`

Software versions
- `Python 3.8` (via pip) or `Python 3.7` (via pip or conda) or a `debian package` (Python is not required)

Hardware support
- CPU (GPU not needed!)


# Common installation issues and their solution

ERROR: fastdup-0.39-cp38-cp38-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
- Check that you are on ubuntu 20.04 or 18.04 (via the command `lsb_release -r`). 
- Check that you are using the right python version (python3.8 and not python). 
- Make sure pip is up to date using `python3.8 -m pip install -U pip`). 
- Make sure you install using `python3.8 -m pip install..` and not just `pip install...`.
- If that does not work, please open an issue with the otuput of `python3.8 -m pip debug --verbose` 

ERROR: `libGL.so.1: cannot open shared object file: No such file or directory`
- Need to install depedency: `sudo apt -y nstall libgl1-mesa-glx`


