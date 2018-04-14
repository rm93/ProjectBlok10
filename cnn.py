# Opencv-contrib-python, matplotlib, tqdm, pandas, sklearn, scipy, tensorflow, tflearn, keras, numpy, h5py

import cv2
import datetime
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class Constants:
    """
    Class used to store all constants of the program
    """
    # Path to data folder
    PATH_DATA = "data"
    PATH_PREPROCESSING = "preprocessed-data"
    PATH_OUTPUT = "output"
    PATH_LOG = "log"

    # Folder names
    LABEL_DATA = "ClinicalReadings"
    IMAGE_DATA = "CXR_png"

    # File names
    CSV_LABELS_FILENAME = "combined_labels.csv"

    # Prefixes for specific names
    RESIZE_FOLDER_PREFIX = "resized"
    IMAGE_ARRAY_PREFIX = "X_image_array"
    MODEL_PREFIX = "cnn_model"
    RESULTS_PREFIX = "results"
    CONFUSION_MATRIX_PREFIX = "confusion_matrix"

    # Parameters for the neural network
    # Default image size is 256
    IMAGE_SIZE = 256
    BATCH_SIZE = 10
    NUM_CLASSES = 2
    NUM_EPOCH = 100
    NUM_CHANNELS = 1
    NUM_FILTERS = 5
    # Default kernel size is (15,15)
    KERNEL_SIZE = (3, 3)

    # Combined constants
    LABEL_PATH = PATH_DATA + '/' + LABEL_DATA
    IMAGE_PATH = PATH_DATA + '/' + IMAGE_DATA
    RESIZE_PATH = PATH_PREPROCESSING + '/' + RESIZE_FOLDER_PREFIX + '-' + str(IMAGE_SIZE)
    CSV_LABELS_PATH = PATH_PREPROCESSING + '/' + CSV_LABELS_FILENAME
    IMAGE_ARRAY_PATH = PATH_PREPROCESSING + '/' + IMAGE_ARRAY_PREFIX + '-' + str(IMAGE_SIZE) + ".npy"
    MODEL_PATH = PATH_OUTPUT + '/' + MODEL_PREFIX + "-{}-{}-{}-{}.h5".format(NUM_EPOCH, BATCH_SIZE,
                                                                             KERNEL_SIZE[0], KERNEL_SIZE[1])
    RESULTS_PATH = PATH_OUTPUT + '/' + RESULTS_PREFIX + "-{}-{}-{}-{}.csv".format(NUM_EPOCH, BATCH_SIZE,
                                                                                   KERNEL_SIZE[0], KERNEL_SIZE[1])


def resize_images(image_path, new_path, image_size):
    """
    Function to resize all images to specific dimensions
    """
    # Crops, resizes, and stores all images from a directory in a new directory.
    if os.path.exists(new_path):
        print("The images are already pre-processed")
    else:
        print("Resizing the images to: {}px by {}px for further use".format(image_size, image_size))
        try:
            os.makedirs(new_path)
            files = [l for l in os.listdir(image_path) if not(l in ['.DS_Store', 'Thumbs.db'])]
            for filename in tqdm(files):
                # Read in all images as grayscale
                image = cv2.imread(image_path + '/' + filename, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (image_size, image_size), 1)
                cv2.imwrite(str(new_path + '/' + filename), image)
            print("Succesfully resized the images")
        except IOError or FileNotFoundError:
            os.rmdir(new_path)
            close_program("There was a problem while trying to open the images or the folder with images", 1)
        except cv2.error:
            os.rmdir(new_path)
            close_program("There was a problem while resizing the images in the dataset", 1)


def combine_labels(label_path, new_path):
    """
    Function used to combine the folder with multiple label files to one CSV file with all labels
    """
    if not os.path.exists(new_path):
        print("Processing the labels")
        # Write header for CSV file
        try:
            with open(new_path, "w") as csv_file:
                csv_file.write("Image_Index,Labels,Patient_Age,Patient_Gender\n")
                csv_file.close()
        except IOError or FileNotFoundError:
            close_program("There was an problem while creating a CSV file for the headers", 1)
        # Append label data to the CSV based on label files in the label data folder
        files = [l for l in os.listdir(label_path) if not(l in ['.DS_Store', 'Thumbs.db'])]
        for filename in tqdm(files):
            try:
                with open(label_path + '/' + filename, "r") as label_file:
                    lines = label_file.readlines()
                    label_file.close()
            except IOError or FileNotFoundError:
                os.remove(new_path)
                close_program("There was a problem while reading label data file: \"" + filename + "\"", 1)
            image_filename = filename[:-4]+".png"
            # Remove whitespaces and commas
            stripped_first_line = ''.join(lines[0].split()).replace(',', '')
            incorrect_format = True
            for i, c in enumerate(stripped_first_line):
                if c.isdigit():
                    incorrect_format = False
                    break
            # Incorrect format is True when there are no digits in the line (no age present)
            if incorrect_format:
                print("Warning: unknown label format found in line 1 of file: \"" + filename + "\" (skipping this file)")
                continue
            else:
                gender = stripped_first_line[:i]
                age = stripped_first_line[i:]
            # Check if "normal" is present in other lines -> "not sick" else -> "sick"
            other_lines = ''.join(lines[1:])
            if "normal" in other_lines:
                label = "not_sick"
            else:
                label = "sick"
            # Append file data to CSV file
            try:
                with open(new_path, "a") as csv_file:
                    csv_file.write("{0},{1},{2},{3}\n".format(image_filename, label, age, gender))
                    csv_file.close()
            except IOError or FileNotFoundError:
                os.remove(new_path)
                close_program("There was a problem while appending label data to the CSV file", 1)
        print("Successfully combined the labels")
    else:
        print("The labels are already pre-processed")


def create_image_array(path_image_array, path_csv, resized_images_path):
    """
    Converts each image to an array, and appends each array to a new NumPy array,
    based on the image column equaling the image file name.
    """
    if not os.path.exists(path_image_array):
        print("Processing image array")
        try:
            labels = pd.read_csv(path_csv)
        except IOError or FileNotFoundError:
            close_program("There was a problem while trying to read the CSV with labels", 1)
        lst_imgs = [l for l in labels['Image_Index']]
        try:
            X_train = np.array([np.array(cv2.imread(resized_images_path + '/' + img, cv2.IMREAD_GRAYSCALE))
                                for img in tqdm(lst_imgs)])
        except cv2.error:
            close_program("There was a problem while trying to read one of the resized images", 1)
        print("Saving image array")
        try:
            np.save(path_image_array, X_train)
        except IOError or FileNotFoundError:
            close_program("There was a problem while trying save the numpy image array", 1)
        print("The image array has succesfully been saved")
    else:
        print("The image array is already processed")


def split_data(image_size, channels, classes, path_csv, path_image_array):
    """
    Function to split the data in a trainings set and a test set
    """
    print("Reading label data")
    # Import data
    try:
        labels = pd.read_csv(path_csv)
    except IOError or FileNotFoundError:
        close_program("There was a problem while opening the CSV with labels", 1)
    try:
        X = np.load(path_image_array)
    except IOError or FileNotFoundError:
        close_program("There was a problem while opening the numpy image array", 1)
    y = labels.Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.reshape(-1, 1)

    print("The data is being split into train and test data sets")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError:
        close_program("The image array and the CSV with labels do not have the same length", 1)

    print("Reshaping data")
    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, channels)
    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, channels)

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    print("Normalizing data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)
    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)

    return X_train, y_train, X_test, y_test


def create_or_load_cnn_model(X_train, y_train, kernel_size, filters, channels,
                             epoch, batch_size, classes, img_size, path_model, path_output, log_path):
    """
    Function used to create a CNN model. If a model already exists, that model will be loaded instead
    """
    if not os.path.exists(path_model):
        print("Creating a model of the training data")
        model = Sequential()

        model.add(Conv2D(filters, (kernel_size[0], kernel_size[1]),
                         padding='valid',
                         strides=1,
                         input_shape=(img_size, img_size, channels)))
        model.add(Activation('relu'))

        model.add(Conv2D(filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(Conv2D(filters, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        print("Model flattened out to: ", model.output_shape)

        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(classes))
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

        tensor_board = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True)

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
                  verbose=1,
                  validation_split=0.2,
                  class_weight='auto',
                  callbacks=[stop, tensor_board]
                  )
        try:
            # Create output folder if it doesn't exist
            if not (os.path.exists(path_output)):
                os.makedirs(path_output)
            model.save(path_model)
        except:
            close_program("There was a problem while trying to save the CNN model!", 1)
    else:
        print("Model already exists\nOpening model")
        try:
            model = load_model(path_model)
        except:
            close_program("There was a problem while trying to load the CNN model!", 1)
        print('Model loaded')
    return model


def model_predict(model, X_test, y_test):
    """
    Function used to predict values for the training data
    """
    print("Predicting")

    y_pred = model.predict(X_test)

    y_given = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    recall = recall_score(y_given, y_pred, average='weighted')
    print("Accuracy: ", round(recall*100, 2), "%")

    return y_given, y_pred


def create_results_csv(y_given, y_pred, path_results):
    """
    Function to create and save a CSV with the results
    """
    dict_characters = {0: 'sick', 1: 'not_sick'}
    name_test = []
    name_pred = []

    for i in range(len(y_given)):
        name_test.append(dict_characters[y_given[i]])

    for i in range(len(y_pred)):
        name_pred.append(dict_characters[y_pred[i]])

    try:
        with open(path_results, "w") as f:
            f.write("Label,Name,Predicted label,Name\n")
    except IOError or FileNotFoundError:
        close_program("There was a problem while creating a header for the CSV file", 1)

    print("Making CSV with results")
    try:
        with open(path_results, "a") as f:
            for i in tqdm(range(len(y_given))):
                f.write("{},{},{},{}\n".format(y_given[i], name_test[i], y_pred[i], name_pred[i]))
    except IOError or FileNotFoundError:
        close_program("There was a problem while saving the CSV with the results", 1)
    print("Saved a CSV with results")


def create_confusion_matrix(y_given, y_pred, output_folder, prefix_figure, num_epoch, batch_size, kernel_size):
    """
    Function used to create and plot a confusion matrix
    """
    print("Creating and plotting a confusion matrix")
    cm = confusion_matrix(y_given, y_pred)
    plot_confusion_matrix(cm, output_folder, prefix_figure, ["sick", "not sick"],
                          num_epoch, batch_size, kernel_size, 'Confusion matrix')
    np.set_printoptions(precision=2)
    plt.figure()
    plt.show()


def plot_confusion_matrix(cm, output_folder, prefix_figure, classes, num_epoch, batch_size, kernel_size,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.BrBG):
    """
    Helper function used for creating a confusion matrix plot
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    try:
        plt.savefig(output_folder + '/' + prefix_figure +
                    "-{}-{}-{}-{}-{}.png".format(num_epoch, batch_size, kernel_size[0],
                                                 kernel_size[1], datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    except:
        close_program("There was a problem while trying to save the final plot", 1)


def start_program(message):
    """
    Helper function used to print a start message to the user and start the program
    """
    # Create an instance with all of the constants
    constants = Constants()
    print('-' * 50 + '\n' + message + '\n')
    return constants


def close_program(message, status):
    """
    Helper function used to print a final message to the user and close the program
    """
    print('\n' + message + '\n' + '-' * 50)
    exit(status)


def main():
    """
    Main function
    """
    # Give the user a message when the program started and load constants
    c = start_program("Program successfully started")
    # Resize image data
    resize_images(c.IMAGE_PATH, c.RESIZE_PATH, c.IMAGE_SIZE)
    # Combine the label data in a csv file
    combine_labels(c.LABEL_PATH, c.CSV_LABELS_PATH)
    # Create a numpy image array of the images
    create_image_array(c.IMAGE_ARRAY_PATH, c.CSV_LABELS_PATH, c.RESIZE_PATH)
    # Split the pre-processed data into a training- and testdataset
    X_train, y_train, X_test, y_test = split_data(c.IMAGE_SIZE, c.NUM_CHANNELS, c.NUM_CLASSES,
                                                  c.CSV_LABELS_PATH, c.IMAGE_ARRAY_PATH)
    # Create or load the CNN model
    model = create_or_load_cnn_model(X_train, y_train, c.KERNEL_SIZE, c.NUM_FILTERS, c.NUM_CHANNELS,c.NUM_EPOCH,
                                     c.BATCH_SIZE, c.NUM_CLASSES, c.IMAGE_SIZE, c.MODEL_PATH, c.PATH_OUTPUT, c.PATH_LOG)
    # Predict the labels for the test dataset
    y_given, y_pred = model_predict(model, X_test, y_test)
    # Create a CSV with the results of the prediction
    create_results_csv(y_given, y_pred, c.RESULTS_PATH)
    # Create and plot a confusion matrix of the results
    create_confusion_matrix(y_given, y_pred, c.PATH_OUTPUT, c.CONFUSION_MATRIX_PREFIX,
                            c.NUM_EPOCH, c.BATCH_SIZE, c.KERNEL_SIZE)
    # Tell the user the program successfully finished
    close_program("Program successfully finished", 0)


if __name__ == "__main__":
    main()
