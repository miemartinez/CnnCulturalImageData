#!/usr/bin/env python

"""
This script builds a deep learning model using LeNet as the convolutional neural network architecture. This network is used to classify impressionist paintings by their artists. It saves the model summary, model history and the classification report.

Parameters:
    path2train: str <path-to-train-data>, default = "../data/subset/training"
    path2test: str <path-to-test-data>, default = "../data/subset/validation"
    n_epochs: int <number-of-epochs>, default = 20
    batch_size: int <batch-size>, default = 32
    optimizer: str <optimization-method>, default = sgd
Usage:
    cnn-artists.py -t <path-to-train> -c <path-to-test-data> -n <number-of-epochs> -b <batch-size> -o <optimizer>
Example:
    $ python3 cnn-artists-reworked.py -t ../data/training/training -te ../data/validation/validation -n 20 -b 32 -o adam
    
## Task
- Make a convolutional neural network (CNN) and train on classifying paintings from 10 different painters.
- Save the model summary (as both txt and png), model history (accuracy and loss during training and testing), and the classification report in a folder called out (which will be created if it does not exist).
- The same outputs will be printed in the terminal.
- The user is able to specify filepaths for training and validation data as well as number of epochs for training the model and the batch size as command line arguments.   
- Similarly, the user can choose between stochastic gradient descent (SGD) or Adam as optimizer.
"""
# Data tool libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import argparse
from contextlib import redirect_stdout
from math import floor


# Sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow tools
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K


# Argparse arguments
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

ap.add_argument("-o", "--optimizer",
                required = False,
                help = "Model optimizer, choose between: 'adam' and 'sgd'")               
                default = "sgd")
    
# Parse arguments
args = vars(ap.parse_args())  



def main(args):
    
    # Save input parameters
    train_data = args["path2train"]
    test_data = args["path2test"]
    epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    optimizer = args["optimizer"]
    
    # Create out directory if it doesn't exist in the data folder
    create_out_dir()

    # Start message to user
    print("\n[INFO] Initializing the construction of a LeNet convolutional neural network model...")
    
    cnn_artists = CNN_artists(train_data = train_data, 
                              test_data = test_data,
                              epochs = epochs,
                              batch_size = batch_size, 
                              optimizer = optimizer)
    
    # Create list of label names
    cnn_artists.make_labels()
    
    # Find the optimal dimensions to resize the images 
    cnn_artists.find_image_dimensions()
    
    # Create trainX and trainY
    cnn_artists.create_trainX_trainY()
    
    # Create testX and testY
    cnn_artists.create_testX_testY()
    
    # Normalize data and binarize labels
    cnn_artists.normalize_binarize()
    
    # Define model
    model = cnn_artists.define_model()
    
    # Train model
    H = cnn_artists.train_model(model)
    
    # Plot loss/accuracy history of the model
    cnn_artists.plot_history(H)
    
    # Evaluate model
    cnn_artists.evaluate_model(model)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained a convolutional neural network on impressionist paintings that can classify paintings by their artists\n")
    
    
    
def create_out_dir():
    '''
    Create out directory if it doesn't exist in the data folder
    '''
    dirName = os.path.join("..", "out")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        # print that it has been created
        print("Directory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("Directory " , dirName ,  " already exists")
        
        
        
def resize_image(img, size):
    '''
    Resizing the shorter size of the image to desired length and then crop the other length of the image to the same size to avoid distortion of the image.
    '''
    # getting the height and width of the image 
    height, width = img.shape[:2]
    
    # if the image is quadratic 
    if width == height:
        # resize on both dimensions
        dim = (size, size)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # if the width is bigger than the height
    elif height < width:
        # calculate the ratio between width and height
        ratio = float(width) / float(height)
        # use the ratio to scale the size of the width to preserve aspect ratio
        newwidth = ratio * size
        dim = (int(floor(newwidth)), size)
        # use new dimensions to resize
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # if the height is bigger than the width
    elif width < height:
        # calculate the ratio between height and width
        ratio = float(height) / float(width)
        # use the ratio to scale the size of the height to preserve aspect ratio
        newheight = ratio * size
        dim = (size, int(floor(newheight)))
        # use new dimensions to resize
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # crop on the longer side of the image to get quadratic image
    cropped = resized[0:size, 0:size]
    
    return cropped
        
    
    
# Make class object for the convolutional neural network for the artist data
class CNN_artists:
    
    def __init__(self, train_data, test_data, epochs, batch_size, optimizer):
        '''
        Constructing the Classification object
        '''
        # defining the self attributes 
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        
        

    def make_labels(self):
        """
        Defining the label names by listing the names of the folders within training directory without listing hidden files. 
        """
        # Create empty list
        self.label_names = []

        # For every name in training directory
        for name in os.listdir(self.train_data):
            # If it does not start with . (which hidden files do)
            if not name.startswith('.'):
                self.label_names.append(name)
                
                

    def find_image_dimensions(self):
        """
        Finding the minimum width and height in the train and test data. This will be used for as the image dimensions when resizing. 
        """
        
        print("\n[INFO] Estimating the optimal image dimensions to resize images...")
        
        # Create empty list
        dimension = []

        # Loop through directories for each painter
        for name in self.label_names:

            # Take images in train data
            train_images = glob.glob(os.path.join(self.train_data, name, "*.jpg"))

            # Loop through images in training data
            for image in train_images:
                # Load image
                loaded_img = cv2.imread(image)

                # Find dimensions of each image
                height, width, _ = loaded_img.shape

                # Append to lists
                dimension.append(height)
                dimension.append(width)

            # Take images in test data
            test_images = glob.glob(os.path.join(self.test_data, name, "*.jpg"))

            # Loop through images in test data
            for image in test_images:

                # Load image
                loaded_img = cv2.imread(image)

                # Find dimensions of each image
                height, width, _ = loaded_img.shape

                # Append to lists
                dimension.append(height)
                dimension.append(width)

        # Find the smallest image dimension among all images 
        self.min_value = min(dimension)

        print(f"\n[INFO] Input images are resized to dimensions of height = {self.min_value} and width = {self.min_value}...")
        
        
    def create_trainX_trainY(self):
        """
        Creating the trainX and trainY which contain the training data and its labels respectively. 
        """
        
        print("\n[INFO] Resizing training images and creating training data, trainX, and labels, trainY...")
        
        # Create empty array and list
        self.trainX = np.empty((0, self.min_value, self.min_value, 3))
        self.trainY = []

        # Loop through images in training data
        for name in self.label_names:
            images = glob.glob(os.path.join(self.train_data, name, "*.jpg"))

            # For each image
            for image in tqdm(images):

                # Load image
                loaded_img = cv2.imread(image)

                # Resize image with the specified dimensions
                resized_img = resize_image(loaded_img, self.min_value)

                # Create array of image
                image_array = np.array([np.array(resized_img)])

                # Append the image array to the trainX
                self.trainX = np.vstack((self.trainX, image_array))

                # Append the label name to the trainY list
                self.trainY.append(name)



    def create_testX_testY(self):
        """
        Creating testX and testY which contain the test/validation data and its labels respectively. 
        """
        
        print("\n[INFO] Resizing validation images and creating validation data, testX, and labels, testY...")
        
        # Create empty array and list
        self.testX = np.empty((0, self.min_value, self.min_value, 3))
        self.testY = []

        # Loop through images in test data
        for name in self.label_names:
            images = glob.glob(os.path.join(self.test_data, name, "*.jpg"))

        # For each image
            for image in tqdm(images):

                # Load image
                loaded_img = cv2.imread(image)

                # Resize image
                resized_img = resize_image(loaded_img, self.min_value)

                # Create array
                image_array = np.array([np.array(resized_img)])

                # Append the image array to the testX
                self.testX = np.vstack((self.testX, image_array))
                # Append the label name to the testY list
                self.testY.append(name)


    def normalize_binarize(self):
        """
        Normalizing the training and test data and binarizing the training and test labels so they can be used in the model.
        """
        
        print("\n[INFO] Normalize training and validation data and binarizing training and validation labels...")
        
        # Normalize training and test data
        self.trainX_norm = self.trainX.astype("float") / 255.
        self.testX_norm = self.testX.astype("float") / 255.

        # Binarize training and test labels
        lb = LabelBinarizer()
        self.trainY = lb.fit_transform(self.trainY)
        self.testY = lb.fit_transform(self.testY)



    def define_model(self):
        """
        Defining the model architecture and saving this as both a txt and png file in the out folder as well as returning it to be used globally.
        """
        
        print("\n[INFO] Defining LeNet model architecture...")
        
        # Defining the model architecture
        model = Sequential()
        
        # First convolutional layer: CONV => RELU => POOL
        model.add(layers.Conv2D(filters=32, 
                                kernel_size=(3, 3), 
                                padding="same", # padding with zeros
                                activation='relu', # relu activation function
                                input_shape=(self.min_value, self.min_value, 3)))

        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) # stride of 2 horizontal, 2 vertical

        
        # Second convolutional layer: CONV => RELU => POOL
        model.add(layers.Conv2D(filters=50, 
                                kernel_size=(5, 5), 
                                padding="same", 
                                activation='relu'))
        
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))# stride of 2 horizontal, 2 vertical
        
        # Flattening the input layer from the previous convolutional layer to make fully connected layers
        model.add(layers.Flatten())
        
        # Fully connected layer and relu activation function
        model.add(layers.Dense(units=512, activation='relu'))
        
        # Dropout layer randomly setting input units to 0 with a rate of 0.2
        model.add(layers.Dropout(0.2))
        
        # Fully connected layer and relu activation function
        model.add(layers.Dense(units=256, activation='relu'))
        
        # Output layer with softmax activation
        model.add(layers.Dense(units=10, activation = 'softmax'))
       
    

        # Defining optimizer as stochastic gradient descent with a learning rate of 0.01
        if self.optimizer = sgd:
            opt = SGD(lr=0.01)
        # or as adam with default setting of learning rate 0.0001
        elif self.optimizer = adam:
            opt = Adam
            
            
        # Compiling model
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
        plot_path = os.path.join("..", "out", "model_architecture.png")
        # Visualization of model
        plot_LeNet_model = plot_model(model,
                                      to_file = plot_path,
                                      show_shapes=True,
                                      show_layer_names=True)
        print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")

        return model


    def train_model(self, model):
        """
        Training the model on the training data and validating it on the test data.
        """
        
        print("\n[INFO] Training model...")
        
        # Train model
        H = model.fit(self.trainX, self.trainY, 
                      validation_data=(self.testX, self.testY), 
                      batch_size=self.batch_size, 
                      epochs=self.epochs, 
                      verbose=2)

        return H


    def plot_history(self, H):
        """
        Plotting the loss/accuracy of the model during training and saving this as a png file in the out folder.
        """
        # name for saving output
        figure_path = os.path.join("..", "out", "model_history.png")
        # Visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path)

        print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")
        

    def evaluate_model(self, model):
        """
        Evaluating the trained model and saving the classification report in the out folder. 
        """
        
        print("\n[INFO] Evaluating model and printing classification report... \n")
        
        # Predictions
        predictions = model.predict(self.testX, batch_size=self.batch_size)

        # Classification report
        classification = classification_report(self.testY.argmax(axis=1),
                                                      predictions.argmax(axis=1),
                                                      target_names=self.label_names)

        # Print classification report
        print(classification)

        # name for saving report
        report_path = os.path.join("..", "out", "classification_report.txt")

        # Save classification report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(classification_report(self.testY.argmax(axis=1),
                                                      predictions.argmax(axis=1),
                                                      target_names=self.label_names))

        print(f"\n[INFO] Classification report is saved as '{report_path}'.")

# Define behaviour when called from command line
if __name__=="__main__":
    main(args)
