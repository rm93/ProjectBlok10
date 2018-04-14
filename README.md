# Project blok 10

**Licence: GNU General Public License v3.0 (copy provided in directory)**<br />
<br />
Author: 
- Rick Medemblik
- Alex Staritsky
- William Sies
- Lisanne Dijksma

Contact:
- rmedemblik93@gmail.com
- alexstaritsky@hotmail.nl
- willysieswilly@gmail.com
- ldijksma@msn.com
         
### Description

Met deze applicatie kan met behulp van röntgen foto’s van menselijke longen geidentificeerd worden of een patiet een vorm van tuborcoluse heeft of niet dit kan door gebruik te maken van machine learning. 

Hier kan CUDA voor gebruikt worden om de executietijd van de applicatie te versnellen. 

### Requirements

- [Cuda 8.0.61_375.26](https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run)
- [Cudnn 8.0-linux-x64-v6.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/cudnn-8.0-linux-x64-v6.0-tgz)
- H5py 2.7.1
- Keras 2.1.5
- Linux operating system. This software is developed on Linux Ubuntu 16.04<br />
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

De data is te downloaden via de volgende [link](https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities/downloads/ChinaSet_AllFiles.zip/1)

Deze data dient uitgepakt in de data folder geplaatst te worden. De structuur dient er als volgt uit te zien:

    ProjectBlok10
        Data
            ClinicalReadings
            CXR_png

### Usage

Om het script te starten kan gebruikt worden gemaakt van de terminal of een IDE (tijdens de ontwikkeling is [pycharm](https://www.jetbrains.com/pycharm/download/#section=linux) gebruikt)

#### Terminal
- Zorg er voor dat alle Benodigheiden zijn geinstalleerd op je computer of in een virtual environment
- ga naar de folder waar het script en de data zijn.
- en voer het script uit met het commando: python cnn.py

#### IDE

- Zorg er voor dat alle Benodigheiden zijn geinstalleerd op je computer of in een virtual environment
- Open het project in de IDE
- Zoek het bestand cnn.py op
- En run het bestand

### Output

Na het runnen van het script worden er verschillende mappen aangemaakt hier van is de output map het meest interessant omdat hier het gemaakte model en de confusion_matrix worden opgeslagen.



