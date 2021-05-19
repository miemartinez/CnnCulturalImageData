# CNNs on cultural image data
### Multi-class classification of impressionist painters
**This project was developed as part of the spring 2021 elective course Cultural Data Science - Visual Analytics at Aarhus University.**

__Task:__ This project aims at training a convolutional neural network to classify impressionist paintings by predicting the artist. 
As all the images are in different shapes and sizes, the first task is to resize all images, so they are homogeneous in shape. 
Similarly, data wrangling is needed for extracting labels for each artist to use for the classification.

In the data folder there is a subset folder of the data for training and validating the model. The folders contain 10 individual folders for each artist with jpg files of their paintings. 
The full data for this project can be found on Kaggle: https://www.kaggle.com/delayedkarma/impressionist-classifier-data <br>

The output of the python script is saved in the created out folder. This contains a txt file and a visualization saved as png of the model architecture. 
It also contains the development of loss and accuracy for the training and validation across epochs. Lastly, it contains the classification report that displays the model accuracy.

The reworked script cnn_artists.py is in the src and it can be run without specifying any parameters. 
However, the user is able to define the file paths to the training and validation data. Furthermore, the user can define the number of epochs to train over and the batch size. 
If nothing is chosen in the command line, defaults are set instead. <br>

I decided to also include the original script that I made as this performs better on the data than the one using resizing with preserved aspect ratio. 
This script is also in the src folder and is called cnn_LeNet_artists.py. 


__Dependencies:__ <br>
To ensure dependencies are in accordance with the ones used for the script, you can create the virtual environment ‘CNN_venv"’ from the command line by executing the bash script ‘create_cnn_venv.sh’.  
```
    $ bash ./create_cnn_venv.sh
```
This will install an interactive command-line terminal for Python and Jupyter as well as all packages specified in the ‘requirements.txt’ in a virtual environment. 
After creating the environment, it will have to be activated before running the CNN script.
```    
    $ source CNN_venv/bin/activate
```
After running these two lines of code, the user can commence running one of the scripts. <br>

### How to run cnn_artists.py <br>
__Parameters:__ <br>
```
    path2train: str <path-to-train-data>, default = "../data/subset/training"
    path2test: str <path-to-test-data>, default = "../data/subset/validation"
    n_epochs: int <number-of-epochs>, default = 20
    batch_size: int <batch-size>, default = 32
    optimizer: str <optimization-method>, default = sgd

```
    
__Usage:__ <br>
```
    cnn-artists.py -t <path-to-train> -c <path-to-test-data> -n <number-of-epochs> -b <batch-size> -o <optimizer>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 cnn-artists.py -t ../data/training/training -te ../data/validation/validation -n 20 -b 32 -o adam

```


The code has been developed in Jupyter Notebook and tested in the terminal on Jupyter Hub on worker02. I therefore recommend cloning the Github repository to worker02 and running the scripts from there. 

### Results:
The best result was obtained using the LeNet architecture with a batch size of 32 and 20 epochs and resizing without preserving aspect ratio. 
Here, a weighted average accuracy score of 42% was observed when running it in the group. However, when running it on my own computer the accuracy dropped to 33%. 
Surprisingly, running it with the cropped images worsened the model and I couldn’t get accuracy to exceed 10%, which is the equivalent of the model just labelling all paintings as being by the same artist. 
When resizing and cropping the images, I lose valuable data and I might be keeping uninteresting parts of the paintings. To gain better results one could consider slicing the images so to keep the middle of the image instead of the left-top corner as this might contain the most important part of the image motif. 




