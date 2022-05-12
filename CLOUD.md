
# Support for cloud storage
FastDup supports two types of cloud storage:
- Amazon s3 aws cli
- Min.io cloud storage api

## Amazon s3 aws cli support
### Preliminaries:
- Install aws cli using the command
`sudo apt install awscli`
- Configure your aws using the command
`aws configure`
- Make sure you can access your bucket using
`aws s3 ls s3://<your bucket name>`

## How to run
There are two options to run.
In the input_dir command line argument put the full path your bucket for example: `s3://mybucket/myfolder/myother_folder/`
This option is useful for testing but it is not recommended for large corpouses of images as listing files in s3 is a slow operation. In this mode, all the images in the recursive subfolders of the given folders will be used.
Alternatively (and recommended) create a file with the list of all your images in the following format:
```
s3://mybucket/myfolder/myother_folder/image1.jpg
s3://mybucket/myfolder2/myother_folder4/image2.jpg
s3://mybucket/myfolder3/myother_folder5/image3.jpg
```
Assuming the filename is files.txt you can run with input_dir=’/path/to/files.txt’

Notes: 
Currently we support a single cloud provider and a single bucket.
It is OK to have images with the same name assuming they are nested in different subfolders.
In terms of performance, it is better to copy the full bucket to the local node first in case the local disk is hard enough. Then give the input_dir as the local folder location of the copied data. The explanation above is for the case the dataset is larger than the local disk (and potentially multiple nodes run in parallel).



## Min.io support
Preliminaries
Install the min.io client using the command
```
wget https://dl.min.io/client/mc/release/linux-amd64/mc
sudo mv mc /usr/bin/
chmod +x /usr/bin/mc
```
Configure the client to point to the cloud provider

```
mc alias set myminio/ http://MINIO-SERVER MYUSER MYPASSWORD
```
For example for google cloud:
```
/usr/bin/mc alias set google  https://storage.googleapis.com/ <access_key> <secret_key> 
```
Make sure the bucket is accessible using the command:
```
/usr/bin/mc ls google/mybucket/myfolder/myotherfolder/
```

How to run
There are two options to run.
In the input_dir command line argument put the full path your cloud storage provider as defined by the minio alias, for example: `minio://google/mybucket/myfolder/myother_folder/`
(Note that google is the alias set for google cloud, and the path has to start with `minio://` prefix).
This option is useful for testing but it is not recommended for large corpouses of images as listing files in s3 is a slow operation. In this mode, all the images in the recursive subfolders of the given folders will be used.
Alternatively (and recommended) create a file with the list of all your images in the following format:
```
minio://google/mybucket/myfolder/myother_folder/image1.jpg
minio://google/mybucket/myfolder/myother_folder/image2.jpg
minio://google/mybucket/myfolder/myother_folder/image3.jpg
```
Assuming the filename is `files.txt` you can run with `input_dir=’/path/to/files.txt’`


