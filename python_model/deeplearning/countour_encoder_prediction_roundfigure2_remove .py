import os
import sys
import inspect
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import model_from_json
import math
import argparse
from sklearn.metrics import mean_squared_error
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import time
import datetime
import matplotlib.cm as cm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from folders import *

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

cache = dict()
truncate = 2

def loadModel(path):
  
  #load model
  outjsonpath = os.path.join(path,"countour_model.json")
  checkpoint_filepath = os.path.join(path,"checkpointSupervised")
  json_file = open(outjsonpath, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  contour_model = model_from_json(loaded_model_json)


  # load weights into new model
  contour_model.load_weights(checkpoint_filepath)
  contour_model.summary()

  return contour_model

def roundtest(args):
  
  plt.figure(figsize=(5, 5))

  path = os.path.join(args.path)
  contour_model = loadModel(path)

  angles=[180,180,270,370]
  #angles_correct=[180,135,90,45,0,-45,-90,-135,-180]
  magnitude = [1,3,5,12]

  contour_features = np.stack((angles, magnitude), axis=-1)
  plt.xlim([-50, 50])
  plt.ylim([-50, 50])
  
  gaussian_contours=[]
  gaussian_contour_centers=[]

  for item in np.arange(0,len(contour_features)):

    plt.vlines(0,-50,50,lw=.1,linestyles='--',colors='k')
    plt.hlines(0,-50,50,lw=.1,linestyles='--',colors='k')

    x,y = np.cos(math.radians(contour_features[item][0])),np.sin(math.radians(contour_features[item][0]))

    ang = contour_features[item][0]
    mag = contour_features[item][1]
    perc = 0.7

    _, _, _,gaussian_contour, gaussian_contour_center, _ = predict(ang,mag,perc,contour_model)

    gaussian_contours.append(gaussian_contour)
    gaussian_contour_centers.append(gaussian_contour_center)
   
    # for g in gaussian_contour_centers:
    #   plt.scatter(g[:,0],g[:,1])
  # gaussian_contours2 = gaussian_contours[1:4]
  # gaussian_contours2 = [[ [i[0]*-1,i[1]] for i in g[0] ] for g in gaussian_contours2]
  # gaussian_contours2 = [np.array([ g ]) for g in gaussian_contours2]
  # gaussian_contours = gaussian_contours+gaussian_contours2

  for index,g in enumerate(gaussian_contours):
    g = g[0].tolist()
    g.append(g[0])    
    g = np.array(g)
    plt.plot(g[:,0],g[:,1],color=cm.jet(index/len(gaussian_contours)))
    # u=17*math.cos(math.radians(angles_correct[index]+90))
    # v=17*math.sin(math.radians(angles_correct[index]+90))
    #plt.quiver(0,0,u,v, color=cm.jet(index/len(gaussian_contours)),width=0.005)

  plt.legend([r"$\dot{\rho}=1$$^\circ$/s",
  r"$\dot{\rho}=3$$^\circ$/s",
  r"$\dot{\rho}=5$$^\circ$/s",
  r"$\dot{\rho}=12$$^\circ$/s",
], loc='upper left')


  plt.show()

def predict(angle,magnitude,percentile,contour_model):

  url = str(angle)+"_"+str(magnitude)+"_"+str(percentile)

  if url not in cache:

    features = np.expand_dims(np.array([angle,magnitude,percentile]),axis=0)

    prediction = contour_model.predict(features)

    p_contour= prediction[0].reshape(1,21,2)
    p_c_center = prediction[1]
    p_img_center = prediction[2]

    g_contour= prediction[3].reshape(1,21,2)
    g_c_center = prediction[4]
    g_img_center = prediction[5]
  
    cache[url] = [p_contour,p_c_center,p_img_center,g_contour,g_c_center,g_img_center]

  else: 
    p_contour,p_c_center,p_img_center,g_contour,g_c_center,g_img_center = cache[url]

  return p_contour,p_c_center,p_img_center,g_contour,g_c_center,g_img_center


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--GazeHeadFile", type=str, required=False)
    parser.add_argument("--evaluate", type=str, required=False)
    args = parser.parse_args()
    roundtest(args)
