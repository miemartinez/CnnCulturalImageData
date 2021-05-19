#!/usr/bin/env python

"""
This script builds a deep learning model using LeNet as the convolutional neural network architecture. This network is used to classify impressionist paintings by their artists. The output 

Parameters:
    path2train: str <path-to-train-data>, default = "../data/subset/training"
    path2test: str <path-to-test-data>, default = "../data/subset/validation"
    n_epochs: int <number-of-epochs>, default = 20
    batch_size: int <batch-size>, default = 32
Usage:
    cnn-artists.py -t <path-to-train> -c <path-to-test-data> -n <number-of-epochs> -b <batch-size>
Example:
    $ python3 cnn-artists.py -t ../data/training/training -te ../data/validation/validation -n 30 -b 40
    
## Task
- Make a convolutional neural network (CNN) and train on classifying paintings from 10 different painters.
- Save the model summary (as both txt and png), model history (accuracy and loss during training and testing), and the classification report in a folder called out (which will be created if it does not exist).
- The same outputs will be printed in the terminal.
- The user is able to specify filepaths for training and validation data as well as number of epochs for training the model and the batch size as command line arguments.    
"""

### DEPENDENCIES ###

# Data tools
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import argparse
from contextlib import redirect_stdout


# Sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


### ARGPARSE ###
    
# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: Path to training data
ap.add_argument("-t", "--path2train",
                type = str,
                required = False,
                help = "Path to the training data",
                default = "../data/subset/training")
    
# Argument 2: Path to test data
ap.add_argument("-te", "--path2test",
                type = str,
                required = False,
                help = "Path to the test/validation data",
                default = "../data/subset/validation")
    
# Argument 3: Number of epochs
ap.add_argument("-n", "--n_epochs",
                type = int,
                required = False,
                help = "The number of epochs to train the model on",
                default = 20)
    
# Argument 4: Batch size
ap.add_argument("-b", "--batch_size",
                type = int,
                required = False,
                help = "The size of the batch on which to train the model",
                default = 32)
    
# Parse arguments
args = vars(ap.parse_args())    

### MAIN FUNCTION ###

def main():
    
    # Save input parameters
    train_data = args["path2train"]
    test_data = args["path2test"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    
    # Create out directory if it doesn't exist in the data folder
    dirName = os.path.join("..", "out")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        # print that it has been created
        print("Directory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("Directory " , dirName ,  " already exists")

    # Start message to user
    print("\n[INFO] Initializing the construction of a LeNet convolutional neural network model...")
    
    # Create list of label names
    label_names = listdir_nohidden(train_data)
    
    # Find the optimal dimensions to resize the images 
    print("\n[INFO] Estimating the optimal image dimensions to resize images...")
    min_height, min_width = find_image_dimensions(train_data, test_data, label_names)
    print(f"\n[INFO] Input images are resized to dimensions of height = {min_height} and width = {min_width}...")
    
    # Create trainX and trainY
    print("\n[INFO] Resizing training images and creating training data, trainX, and labels, trainY...")
    trainX, trainY = create_trainX_trainY(train_data, min_height, min_width, label_names)
    
    # Create testX and testY
    print("\n[INFO] Resizing validation images and creating validation data, testX, and labels, testY...")
    testX, testY = create_testX_testY(test_data, min_height, min_width, label_names)
    
    # Normalize data and binarize labels
    print("\n[INFO] Normalize training and validation data and binarizing training and validation labels...")
    trainX, trainY, testX, testY = normalize_binarize(trainX, trainY, testX, testY)
    
    # Define model
    print("\n[INFO] Defining LeNet model architecture...")
    model = define_LeNet_model(min_width, min_height)
    
    # Train model
    print("\n[INFO] Training model...")
    H = train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size)
    
    # Plot loss/accuracy history of the model
    plot_history(H, n_epochs)
    
    # Evaluate model
    print("\n[INFO] Evaluating model... Below is the classification report. This can also be found in the out folder.\n")
    evaluate_model(model, testX, testY, batch_size, label_names)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained a convolutional neural network on impressionist paintings that can classify paintings by their artists\n")
    
    
    
    
    
    
### FUNCTIONS USED WITHIN MAIN FUNCTION ###

def listdir_nohidden(path):
    """
    Defining the label names by listing the names of the folders within training directory without listing hidden files. 
    """
    # Create empty list
    label_names = []
    
    # For every name in training directory
    for name in os.listdir(path):
        # If it does not start with . (which hidden files do)
        if not name.startswith('.'):
            label_names.append(name)
            
    return label_names

def find_image_dimensions(train_data, test_data, label_names):
    """
    Finding the minimum width and height in the train and test data. This will be used for as the image dimensions when resizing. 
    """
    # Create empty lists
    heights_train = []
    widths_train = []
    heights_test = []
    widths_test = []
    
    # Loop through directories for each painter
    for name in label_names:
        
        # Take images in train data
        train_images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # Loop through images in training data
        for image in train_images:
            # Load image
            loaded_img = cv2.imread(image)
            
            # Find dimensions of each image
            height, width, _ = loaded_img.shape
        
            # Append to lists
            heights_train.append(height)
            widths_train.append(width)
        
        # Take images in test data
        test_images = glob.glob(os.path.join(test_data, name, "*.jpg"))
        
        # Loop through images in test data
        for image in test_images:
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Find dimensions of each image
            height, width, _ = loaded_img.shape
        
            # Append to lists
            heights_test.append(height)
            widths_test.append(width)
            
    # Find the smallest image dimensions among all images 
    min_height = min(heights_train + heights_test + widths_train + widths_test)
    min_width = min(heights_train + heights_test + widths_train + widths_test)
    
    return min_height, min_width



def create_trainX_trainY(train_data, min_height, min_width, label_names):
    """
    Creating the trainX and trainY which contain the training data and its labels respectively. 
    """
    # Create empty array and list
    trainX = np.empty((0, min_height, min_width, 3))
    trainY = []
    
    # Loop through images in training data
    for name in label_names:
        images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image with the specified dimensions
            resized_img = cv2.resize(loaded_img, (min_width, min_height), interpolation = cv2.INTER_AREA)
        
            # Create array of image
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the trainX
            trainX = np.vstack((trainX, image_array))
            
            # Append the label name to the trainY list
            trainY.append(name)
        
    return trainX, trainY


def create_testX_testY(test_data, min_height, min_width, label_names):
    """
    Creating testX and testY which contain the test/validation data and its labels respectively. 
    """
    # Create empty array and list
    testX = np.empty((0, min_height, min_width, 3))
    testY = []
    
    # Loop through images in test data
    for name in label_names:
        images = glob.glob(os.path.join(test_data, name, "*.jpg"))
    
    # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image
            resized_img = cv2.resize(loaded_img, (min_width, min_height), interpolation = cv2.INTER_AREA)
        
            # Create array
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the testX
            testX = np.vstack((testX, image_array))
            # Append the label name to the testY list
            testY.append(name)
        
    return testX, testY


def normalize_binarize(trainX, trainY, testX, testY):
    """
    Normalizing the training and test data and binarizing the training and test labels so they can be used in the model.
    """
    
    # Normalize training and test data
    trainX_norm = trainX.astype("float") / 255.
    testX_norm = testX.astype("float") / 255.
    
    # Binarize training and test labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    return trainX_norm, trainY, testX_norm, testY


def define_LeNet_model(min_width, min_height):
    """
    Defining the LeNet model architecture and saving this as both a txt and png file in the out folder as well as returning it to be used globally.
    """
    
    # Define model
    model = Sequential()

    # Add first set of convolutional layer, ReLu activation function, and pooling layer
    # Convolutional layer
    model.add(Conv2D(32, (3, 3), 
                     padding="same", # padding with zeros
                     input_shape=(min_height, min_width, 3)))
    
    # Activation function
    model.add(Activation("relu"))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2))) # stride of 2 horizontal, 2 vertical
    
    # Add second set of convolutional layer, ReLu activation function, and pooling layer
    # Convolutional layer
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    
    # Activation function
    model.add(Activation("relu"))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    
    # Add fully-connected layer
    model.add(Flatten()) # flattening layer
    model.add(Dense(500)) # dense network with 500 nodes
    model.add(Activation("relu")) # activation function
    
    # Add output layer
    # softmax classifier
    model.add(Dense(10)) # dense layer of 10 nodes used to classify the images
    model.add(Activation("softmax"))

    # Define optimizer 
    opt = SGD(lr=0.01)
    
    # Compile model
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt, 
                  metrics=["accuracy"])
    
    # Model summary
    model_summary = model.summary()
    
    # name for saving model summary
    model_path = os.path.join("..", "out", "model_summary.txt")
    # Save model summary
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    

    # name for saving plot
    plot_path = os.path.join("..", "out", "LeNet_model.png")
    # Visualization of model
    plot_LeNet_model = plot_model(model,
                                  to_file = plot_path,
                                  show_shapes=True,
                                  show_layer_names=True)
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    return model


def train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size):
    """
    Training the LeNet model on the training data and validating it on the test data.
    """
    # Train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=batch_size, 
                  epochs=n_epochs, verbose=1)
    
    return H
    
    
def plot_history(H, n_epochs):
    """
    Plotting the loss/accuracy of the model during training and saving this as a png file in the out folder.
    """
    # name for saving output
    figure_path = os.path.join("..", "out", "model_history.png")
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    
    print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")
    
def evaluate_model(model, testX, testY, batch_size, label_names):
    """
    Evaluating the trained model and saving the classification report in the out folder. 
    """
    # Predictions
    predictions = model.predict(testX, batch_size=batch_size)
    
    # Classification report
    classification = classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names)
            
    # Print classification report
    print(classification)
    
    # name for saving report
    report_path = os.path.join("..", "out", "classification_report.txt")
    
    # Save classification report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names))
    
    print(f"\n[INFO] Classification report is saved as '{report_path}'.")

# Define behaviour when called from command line
if __name__=="__main__":
    main()