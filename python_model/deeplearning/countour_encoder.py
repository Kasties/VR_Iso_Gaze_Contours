import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
from pandas.io import pickle

from tensorflow.python.keras.backend import relu
from tensorflow.keras.models import Model

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

def main(args):

  path = args.path
  outjsonpath = os.path.join(path,"countour_model.json")
  checkpoint_filepath = os.path.join(path,"checkpointSupervised")
  picklefile = os.path.join(path,"datasetreal_from_dl/predictions/parameters.pkl")

  #load data
  contour_train = pd.read_pickle(picklefile)
  contour_train.head()
  contour_features = contour_train.copy()
  contour_features.pop('anglesRange')
  contour_features.pop('magnitudeRangesmin')
  contour_features.pop('magnitudeRangesmax')
  contour_features.pop('samplesCount')

  #gaussian
  contour_G_labels = contour_features.pop('contoursG')
  contour_G_labels = np.array([i for i in contour_G_labels.to_list()]).reshape((-1,42))

  ccenter_G_labels = contour_features.pop('contoursGCenter')
  ccenter_G_labels = np.array([i for i in ccenter_G_labels.to_list()])

  icenter_G_labels = contour_features.pop('imageGCenter')
  icenter_G_labels = np.array([i for i in icenter_G_labels.to_list()])

  #prediction
  contour_P_labels = contour_features.pop('contoursP')
  contour_P_labels = np.array([i for i in contour_P_labels.to_list()]).reshape((-1,42))

  ccenter_P_labels = contour_features.pop('contoursPCenter')
  ccenter_P_labels = np.array([i for i in ccenter_P_labels.to_list()])

  icenter_P_labels = contour_features.pop('imagePCenter')
  icenter_P_labels = np.array([i for i in icenter_P_labels.to_list()])

  contour_features = np.array(contour_features)

  input_dist = layers.Input(shape=contour_features[0].shape)
  normalize = layers.Normalization()

  x = layers.Dense(2048,activation=relu)(input_dist)
  x = layers.Dense(2048,activation=relu)(x)
  x = layers.Dense(2048,activation=relu)(x)
  x = layers.Dense(2048,activation=relu)(x)
  out1 = layers.Dense(42,name='contourP')(x)
  out2 = layers.Dense(2,name='ccenterP')(x)
  out3 = layers.Dense(2,name='icenterP')(x)
  out4 = layers.Dense(42,name='contourG')(x)
  out5 = layers.Dense(2,name='ccenterG')(x)
  out6 = layers.Dense(2,name='icenterG')(x)

  # encoder
  contour_model = Model(input_dist, [out1, out2, out3, out4, out5, out6])
  contour_model.compile(loss={'contourP': tf.losses.MeanSquaredError(), 
                              'ccenterP': tf.losses.MeanSquaredError(), 
                              'icenterP': tf.losses.MeanSquaredError(),
                              'contourG': tf.losses.MeanSquaredError(), 
                              'ccenterG': tf.losses.MeanSquaredError(), 
                              'icenterG': tf.losses.MeanSquaredError()},
                        optimizer = tf.optimizers.Adam())
 
  contour_model.summary()

  Path(checkpoint_filepath).mkdir(parents=True, exist_ok=True)
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      checkpoint_filepath,
      monitor="loss",
      verbose=0,
      save_best_only=True,
      save_weights_only=False,
      mode="auto",
      save_freq="epoch",
      options=None,
  )

  model_json = contour_model.to_json()
  with open(outjsonpath, "w") as json_file:
      json_file.write(model_json)

  contour_model.fit(contour_features, 
  {'contourP': contour_P_labels,
   'ccenterP': ccenter_P_labels,
   'icenterP': icenter_P_labels,
   'contourG': contour_G_labels,
   'ccenterG': ccenter_G_labels,
   'icenterG': icenter_G_labels
   }, epochs=1000,callbacks=[model_checkpoint_callback])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
