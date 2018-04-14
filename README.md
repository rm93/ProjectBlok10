# Project blok 10

**Licence: GNU General Public License v3.0 (copy provided in directory)**<br />
<br />
Author: 
- Alex Staritsky
- Rick Medemblik
- William Sies
- Lisanne Dijksma

Contact:
- alexstaritsky@hotmail.nl
- rmedemblik93@gmail.com
- willysieswilly@gmail.com
- ldijksma@msn.com
         
### Description

With this application, X-rays can be used to identify photos of human lungs, whether a patient has a form of tuberculosis or not by using machine learning.

Here CUDA can be used to speed up the execution time of the application.

### Requirements

- [Cuda 8.0.61_375.26](https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run)
- [Cudnn 8.0-linux-x64-v6.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/cudnn-8.0-linux-x64-v6.0-tgz)
- H5py 2.7.1
- Keras 2.1.5
- Linux operating system. The software is developed on Linux Ubuntu 16.04<br />
**WARNING: Experiences when using different operating systems may vary.**
- Matplotlib 2.1.2
- Numpy 1.14.2
- Opencv-contrib-python 3.4.0.12
- Pandas 0.22.0
- Python 3.6
- Scipy 1.0.1
- Sklearn
- [Tensorflow 1.4.0](https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl)
- Tflearn 0.3.2
- Tqdm 4.19.8

### Preparations

The data can be downloaded via the following [link](https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities/downloads/ChinaSet_AllFiles.zip/1)

The data from the zip file must be unpacked in the data folder. The structure should look like this:

    ProjectBlok10
        data
            ClinicalReadings
            CXR_png

### Usage

To start the script you can use the terminal or an IDE (during the development [pycharm](https://www.jetbrains.com/pycharm/download/#section=linux) was used)

#### Terminal
- Make sure all requerements are installed on your computer or in a virtual environment
- Go to the folder where the script is.
- And run the script with the command: `python cnn.py`

#### IDE

- Make sure all requerements are installed on your computer or in a virtual environment
- Open the project in the IDE
- Look up the cnn.py file
- And run the file

### Output

After running the script several folders are created here the output map is the most interesting because here the created model, confusion_matrix and the results.csv are stored.
