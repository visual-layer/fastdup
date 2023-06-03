import os
import subprocess

def callsh(command):
  status = subprocess.run(command)
  status.check_returncode()
  print(status.stdout)

callsh(['wget', 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz', '-O', 'images.tar.gz'])
callsh(['tar', 'xf', 'images.tar.gz'])

import fastdup
print(f'fastdup version: {fastdup.__version__}')

fd = fastdup.create(work_dir="fastdup_work_dir/", input_dir="images/")
fd.run(num_images=1000)