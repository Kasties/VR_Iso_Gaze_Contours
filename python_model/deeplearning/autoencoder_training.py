import os
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse 
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from numpy.random import default_rng

imageChannels = 1


params = {'dim': (128,128,imageChannels),
            'batch_size': 4,
            'shuffle': True}

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, folder, dim=params['dim'], batch_size=4, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.folder = folder
        self.dim = dim
        self.shuffle = shuffle
        self.distributions_folder = os.path.join(folder, 'distributions')
        self.targets_folder = os.path.join(folder, 'targets')
        self.list_IDs = sorted(os.listdir(self.targets_folder))
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        D = np.empty((self.batch_size, *self.dim))
        T = np.empty((self.batch_size, *self.dim))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            fp_dist = os.path.join(self.distributions_folder, ID)
            im = cv2.imread(fp_dist)
            if(self.dim[2] == 3):
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            elif(self.dim[2] == 1):
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = np.expand_dims(im, axis=-1)
            #need to crop to small dimension
            #v = np.min(im.shape[:-1])
            #crop_img = im[0:v, 0:v]
            D[i,] = im/255.0
            
            fp_targ = os.path.join(self.targets_folder, ID)
            im = cv2.imread(fp_targ)
            if(self.dim[2] == 3):
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            elif(self.dim[2] == 1):
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = np.expand_dims(im, axis=-1)
            #v = np.min(im.shape[:-1])
            #crop_img = im[0:v, 0:v]
            T[i,] =  im/255.0
            
        return D, T

def main(args):

    #channels
    imageChannels = args.imageChannels
    params = {'dim': (args.resolution,args.resolution,imageChannels),
            'batch_size': 4,
            'shuffle': True}

    rootpath = args.rootpath

    training_generator = DataGenerator(os.path.join(rootpath,'dataset_training'), **params)
    validation_generator = DataGenerator(os.path.join(rootpath,'dataset_validation'), **params)

    input_dist = layers.Input(shape=params['dim'])

    channels = 32
    convsize = 3
    # Encoder
    x = layers.Conv2D(channels, (convsize, convsize), input_shape=params['dim'], activation="relu", padding="same")(input_dist)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(channels, (convsize*2, convsize*2), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(channels, (convsize*3, convsize*3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(channels, (convsize*4, convsize*4), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # x = layers.Reshape((16*16*channels,) )(x)
    # x = layers.Dense((2048),  activation="relu")(x)
    # x = layers.Dense((16*16*channels),  activation="relu")(x)
    # x = layers.Reshape((16, 16, channels) )(x)

    # Decoder
    x = layers.Conv2DTranspose(channels, (convsize*4, convsize*4), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(channels, (convsize*3, convsize*3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(channels, (convsize*2, convsize*2), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(channels, (convsize, convsize), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(imageChannels, (convsize, convsize), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(input_dist, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy") #test mse
    #autoencoder.compile(loss="mse", optimizer="adam")
    autoencoder.summary()

                        
    checkpoint_filepath = os.path.join(rootpath,'checkpoint')  
    Path(checkpoint_filepath).mkdir(parents=True, exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    model_json = autoencoder.to_json()
    with open(os.path.join(rootpath,"autoencoder.json"), "w") as json_file:
        json_file.write(model_json)

    autoencoder.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        shuffle=True,
                        epochs=12,
                        callbacks=[model_checkpoint_callback])

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--imageChannels", type=int, required=True)
    args = parser.parse_args()
    main(args)

 #--rootpath "dataset" --params --imageChannels 1

