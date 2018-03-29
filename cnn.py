#Opencv-contrib-python, tqdm, pandas, sklearn, scipy, tensorflow, tflearn, keras, numpy, h5py

import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    global path
    global nameInputCsv
    global namecsv
    global iArray

    ImgSize = 128
    batch_size = 100
    nb_classes = 8
    nb_epoch = 20
    channels = 1
    nb_filters = 32
    kernel_size = (2, 2)

    #Path data directory
    path = "../test_projectblok10/data/"
    #Name resized folder
    resizedName = 'resized-'+str(ImgSize)+'/'
    #Name input csv
    nameInputCsv = "reduced_sample_labels"
    #Name output csv
    namecsv = "new_sample_labels"
    #Name x image array
    iArray = "X_imageArray-"+str(ImgSize)
    #Name results csv
    nameResCsv = "testResults"
    #Name Model
    #MODEL_NAME = 'Model-{}-{}-{}.h5'.format(ImgSize, nb_epoch, batch_size)
    MODEL_NAME = 'Projectblok10-{}-{}.h5'.format(ImgSize, nb_epoch)

    ResizeImages(path=path+'/reduced_images/', new_path=path+resizedName, img_size=ImgSize)
    CreateNewCsv(ImgSize)
    CreateImageArray(ImgSize)
    X_train, y_train, X_test, y_test = SplitData(ImgSize, channels, nb_classes)
    model = cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, ImgSize, MODEL_NAME, path)
    y_test, y_pred = Predict(model, X_test, y_test)
    MakeCsv(y_test, y_pred, nameResCsv, path)


def ResizeImages(path, new_path, img_size):
    #Crops, resizes, and stores all images from a directory in a new directory.

    if not os.path.exists(new_path):
        print("Resize the images to: {}px by {}px for further use.".format(img_size, img_size))
        os.makedirs(new_path)
        dirs = [l for l in os.listdir(path) if l != '.DS_Store']

        total = 1

        for item in tqdm(dirs):
            # Read in all images as grayscale
            img = cv2.imread(path + item, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size), 1)
            cv2.imwrite(str(new_path + item), img)
        total += 1
    else:
        print("The images are already processed")

def CreateNewCsv(ImgSize):
    #Reads in the csv with al the info and edit the file to save only the needed info.
    if not os.path.exists(path+namecsv+'.csv'):
        data = pd.read_csv(path+nameInputCsv+".csv")
        sample = os.listdir(path+'resized-'+str(ImgSize)+'/')

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

        sample.to_csv(path+namecsv+'.csv', index=False, header=True)
    else:
        print("The labels are already processed")

def CreateImageArray(ImgSize):
    #Converts each image to an array, and appends each array to a new NumPy array, based on the image column equaling the image file name.
    if not os.path.exists(path+iArray+'.npy'):
        print("Processing image array")
        labels = pd.read_csv(path+namecsv+".csv")
        lst_imgs = [l for l in labels['Image_Index']]
        X_train = np.array([np.array(cv2.imread(path+'resized-'+str(ImgSize)+'/' + img, cv2.IMREAD_GRAYSCALE)) for img in tqdm(lst_imgs)])
        print("Saving image array")
        np.save(path+iArray+'.npy', X_train)
    else:
        print("The image array is already processed")

def SplitData(ImgSize, channels, nb_classes):
    # Import data
    labels = pd.read_csv(path+namecsv+".csv")
    X = np.load(path+iArray+".npy")

    y = labels.Finding_Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.reshape(-1, 1)

    print("The data is split into train and test data sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Reshaping Data")
    X_train = X_train.reshape(X_train.shape[0], ImgSize, ImgSize, channels)
    X_test = X_test.reshape(X_test.shape[0], ImgSize, ImgSize, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    return X_train, y_train, X_test, y_test

def cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, ImgSize, MODEL_NAME, path):

    if not os.path.exists(path+MODEL_NAME):
        model = Sequential()

        '''
        First set of three layers
        Image size: 256 x 256
        nb_filters = 32
        kernel_size = (2,2)
        '''

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                         padding='valid',
                         strides=1,
                         input_shape=(ImgSize, ImgSize, channels)))
        model.add(Activation('relu'))

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        '''
        Second set of three layers
        Image Size: 128 x 128
        nb_filters = 64
        kernel_size = 4,4
        '''

        nb_filters = 64
        kernel_size = (4, 4)

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        '''
        Third set of three layers
        Image Size: 64 x 64
        nb_filters = 128
        kernel_size = 8,8
        '''

        nb_filters = 128
        kernel_size = (8, 8)

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(12, 12)))

        model.add(Flatten())
        print("Model flattened out to: ", model.output_shape)

        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())

        stop = EarlyStopping(monitor='acc',
                             min_delta=0.001,
                             patience=2,
                             verbose=0,
                             mode='auto')

        tensor_board = TensorBoard(log_dir='./Log', histogram_freq=0, write_graph=True, write_images=True)

        model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=1,
                  validation_split=0.2,
                  class_weight='auto',
                  callbacks=[stop, tensor_board]
                  )

        model.save(path+MODEL_NAME)
        return model
    else:
        model = load_model(path+MODEL_NAME)
        print('Model loaded!')
        return model

def Predict(model, X_test, y_test):
    print("Predicting")

    y_pred = model.predict(X_test)

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("Accuracy: ", round(recall*100, 2), "%")

    return y_test, y_pred

def MakeCsv(y_test, y_pred, nameResCsv, path):
    with open(path+nameResCsv+".csv", "w") as f:
        f.write("Label\tPredicted label\n")

    print("Making results csv")
    with open(path+nameResCsv+".csv", "a") as f:
        for i in tqdm(range(len(y_test))):
            f.write("{}\t{}\n".format(y_test[i], y_pred[i]))

main()