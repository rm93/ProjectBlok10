#Opencv-contrib-python, tqdm, pandas, sklearn, scipy, tensorflow, tflearn
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf

import model

def ResizeImages(path, new_path, ImgSize):
    #Crops, resizes, and stores all images from a directory in a new directory.

    if not os.path.exists(new_path):
        print("Resize the images to: {}px by {}px for further use.\n".format(ImgSize, ImgSize))
        os.makedirs(new_path)
        dirs = [l for l in os.listdir(path) if l != '.DS_Store']

        total = 1

        for item in tqdm(dirs):
            # Read in all images as grayscale
            img = cv2.imread(path + item, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (ImgSize, ImgSize), 1)
            cv2.imwrite(str(new_path + item), img)
        total += 1
    else:
        print("The images are already processed")

def CreateNewCsv(ImgSize):
    #Reads in the csv with al the info and edit the file to save only the needed info.
    if not os.path.exists('../ProjectBlok10/data/new_sample_labels.csv'):
        data = pd.read_csv("../ProjectBlok10/data/sample_labels.csv")
        sample = os.listdir('../ProjectBlok10/data/resized-'+str(ImgSize)+'/')

        sample = pd.DataFrame({'Image_Index': sample})

        sample = pd.merge(sample, data, how='left', on='Image_Index')

        sample.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up #', 'Patient ID',
                          'Patient Age', 'Patient Gender', 'View Position',
                          'OriginalImageWidth', 'OriginalImageHeight',
                          'OriginalImagePixelSpacing_x',
                          'OriginalImagePixelSpacing_y']

        sample['Finding_Labels'] = sample['Finding_Labels'].apply(lambda x: x.split('|')[0])

        sample.drop(['OriginalImagePixelSpacing_x', 'OriginalImagePixelSpacing_y'], axis=1, inplace=True)
        sample.drop(['OriginalImageWidth', 'OriginalImageHeight'], axis=1, inplace=True)

        print("Saving CSV")

        sample.to_csv('../ProjectBlok10/data/new_sample_labels.csv', index=False, header=True)
    else:
        print("The labels are already processed")

def CreateImageArray(ImgSize):
    #Converts each image to an array, and appends each array to a new NumPy array, based on the image column equaling the image file name.
    if not os.path.exists('../ProjectBlok10/data/X_sample_'+str(ImgSize)+'.npy'):
        print("Processing image array")
        labels = pd.read_csv("../ProjectBlok10/data/new_sample_labels.csv")
        lst_imgs = [l for l in labels['Image_Index']]
        X_train = np.array([np.array(cv2.imread('../ProjectBlok10/data/resized-'+str(ImgSize)+'/' + img, cv2.IMREAD_GRAYSCALE)) for img in tqdm(lst_imgs)])
        print("Saving image array")
        np.save('../ProjectBlok10/data/X_sample_'+str(ImgSize)+'.npy', X_train)
    else:
        print("The image array is already processed")

def SplitData():
    # Import data
    labels = pd.read_csv("../ProjectBlok10/data/sample_labels.csv")
    X = np.load("../ProjectBlok10/data/X_sample_"+str(ImgSize)+".npy")

    y = labels.Finding_Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.reshape(-1, 1)

    print("The data is split into train and test data sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

def plot(a):
    c = 0
    for i in a:
        c += 1
        if c <= 1:
            plt.imshow(i)
            print(i)
            plt.show()
            print(tf.cast(i, tf.float16))

def Predict():
    print("Ik kan niet voorspellen, zo slim ben ik nog niet.")
    return None

if __name__ == '__main__':
    ImgSize = 128  # 512

    ResizeImages(path='../ProjectBlok10/data/images/', new_path='../ProjectBlok10/data/resized-' + str(ImgSize) + '/',
                 ImgSize=ImgSize)
    CreateNewCsv(ImgSize)  #
    CreateImageArray(ImgSize)  # create image array of resized images
    X_train, y_train, X_test, y_test = SplitData()
    #plot(X_train)
    m = model.train_neural_network(X_train, y_train)
    # define and train model
    #Predict()  # test model
