import os
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from numpy.random import default_rng


devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

#channels
imageChannels= 1
resolution = 128 

def save_prediction(folder, ID, array1):
    global imageChannels
    if(imageChannels == 3):
        array1 = cv2.cvtColor(array1, cv2.COLOR_RGB2BGR)
    elif(imageChannels == 1):
        array1.reshape(resolution,resolution)
        array1 = array1/array1.max()
        array1 = array1*255
    cv2.imwrite(os.path.join(folder, str(ID).zfill(8)), array1)

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    plt.figure(figsize=(1, 2))
    
    ax = plt.subplot(2, 1, 1)
    plt.imshow(array1)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 1, 2)
    plt.imshow(array2)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()

def main(args):
    global resolution
    #channels
    imageChannels = args.imageChannels
    resolution = args.resolution
    params = {'dim': (args.resolution,args.resolution,imageChannels),
            'batch_size': 4,
            'shuffle': True}
    path = args.rootpath

    #read test data
    test_data_path = os.path.join(path, 'datasetreal\\distributions') 
    prediction_data_path = os.path.join(path, 'datasetreal\\predictions') 
    Path(prediction_data_path).mkdir(parents=True, exist_ok=True)
    test_files =[ i for i in  sorted(os.listdir(test_data_path)) if 'csv' not in i]

    #load model

    checkpoint_filepath = os.path.join(path,'checkpoint')  
    json_file = open(os.path.join(path,'autoencoder.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder.load_weights(checkpoint_filepath)

    autoencoder.summary()

    print("Predicting and Saving...")
    test_data = np.empty(shape=(len(test_files), *params['dim']))
    for c, i in enumerate(test_files):
        print(i)
        im = cv2.imread(os.path.join(test_data_path, test_files[c]))
        if(imageChannels == 3):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif(imageChannels == 1):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.expand_dims(im, axis=-1)
        test_data[c,] = im/255.0
        prediction = autoencoder.predict(test_data)
        #display(im, prediction[c])
        save_prediction(prediction_data_path, i, prediction[c]*255)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", type=dir_path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--imageChannels", type=int, required=True)
    args = parser.parse_args()
    main(args)
