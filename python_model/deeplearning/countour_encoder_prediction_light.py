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
  outjsonpath = os.path.join(path,"datasetreal\\countour_model_light.json")
  checkpoint_filepath = os.path.join(path,"datasetreal\\checkpointSupervised_light")
  json_file = open(outjsonpath, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  contour_model = model_from_json(loaded_model_json)

  # load weights into new model
  contour_model.load_weights(checkpoint_filepath)
  contour_model.summary()

  return contour_model

def roundtest(args):
  
  path = os.path.join(args.path)
  contour_model = loadModel(path)

  angles= np.arange(-180,180,2)
  magnitude = [7]*len(angles)

  contour_features = np.stack((angles, magnitude), axis=-1)

  for item in np.arange(0,len(contour_features)):

    plt.vlines(0,-50,50)
    plt.hlines(0,-50,50)

    x,y = np.cos(math.radians(contour_features[item][0])),np.sin(math.radians(contour_features[item][0]))

    plt.scatter(y*30,x*30)
    plt.axline((0, 0), (y*10,x*10))
    print(contour_features[item][0])

    ang = contour_features[item][0]
    mag = contour_features[item][1]
    perc = 0.7

    prediction_contour, prediction_contour_center, prediction_image_center,gaussian_contour, gaussian_contour_center, gaussian_image_center = predict(ang,mag,perc,contour_model)

    plt.scatter(prediction_contour_center[:,0],prediction_contour_center[:,1],c='orange')
    plt.scatter(prediction_image_center[:,0],prediction_image_center[:,1],c='orange')
    for contour in prediction_contour:
      plt.plot(contour[:,0],contour[:,1],c='orange')

    plt.scatter(gaussian_contour_center[:,0],gaussian_contour_center[:,1],c='green')
    plt.scatter(gaussian_image_center[:,0],gaussian_image_center[:,1],c='green')
    for contour in gaussian_contour:
      plt.plot(contour[:,0],contour[:,1],c='green')

    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.pause(0.1)
    plt.clf()

def evaluate(x,y,angle,mag,percentile,prediction,plot):

  p_contour,p_c_center,p_img_center,g_contour,g_c_center,g_img_center = prediction

  MeanX = np.mean(x)
  MeanY = np.mean(y)
  HeadDirectionMSE = error([0,0],x,y)
  

  Pcenter1MSE = error(p_c_center.flatten().tolist(),x,y)
  Pcenter2MSE = error(p_img_center.flatten().tolist(),x,y)

  Gcenter1MSE = error(g_c_center.flatten().tolist(),x,y)
  Gcenter2MSE = error(g_img_center.flatten().tolist(),x,y)

  percentile = str(int(percentile*10))
  afmcontour = np.load(os.path.join(folder_code,'afm','afm.'+percentile+'.pkl'),allow_pickle=True)


  polygonP = Polygon(p_contour[0])
  polygonG = Polygon(g_contour[0])
  afmpolygon = Polygon(afmcontour)
  point = Point(x,y)
  incountourP = polygonP.contains(point)
  incountourG = polygonG.contains(point)
  inafmcountour = afmpolygon.contains(point)


  polygonareaP = polygonP.area
  polygonareaG = polygonG.area
  afmpolygonarea = afmpolygon.area

  if (plot):
      plt.scatter(x,y,s=.01)
      plt.scatter(MeanX,MeanY,s=15,c='red')

      plt.scatter(imagePCenter[0],imagePCenter[1],s=10,c='orange')#predicted
      plt.scatter(contoursPCenter[0],contoursPCenter[1],s=20,c='orange')
      plt.plot(np.array(contourP)[:,0],np.array(contourP)[:,1],c='orange')

      plt.plot(np.array(afmcontour)[:,0],np.array(afmcontour)[:,1],c='magenta')
      plt.vlines(0,-50,50)
      plt.hlines(0,-50,50)
      plt.ylim(-50,50)
      plt.xlim(-50,50)
      plt.show()
      plt.clf()

  e = {
      'Angle': angle,
      'Magnitude': mag,
      'Head Dir MSE X': np.array(HeadDirectionMSE)[0],#head
      'Head Dir MSE Y': np.array(HeadDirectionMSE)[1],
      'Head Dir MSE': np.array(HeadDirectionMSE)[2],
      
      'P Center MSE X': np.array(Pcenter1MSE)[0],#predicted
      'P Center MSE Y': np.array(Pcenter1MSE)[1],
      'P Center MSE': np.array(Pcenter1MSE)[2],
      # 'P ImgCenter MSE X': np.array(Pcenter2MSE)[0],
      # 'P ImgCenter MSE Y': np.array(Pcenter2MSE)[1],
      # 'P ImgCenter MSE': np.array(Pcenter2MSE)[2],
 
      'G Center MSE X': np.array(Gcenter1MSE)[0],#gaussian
      'G Center MSE Y': np.array(Gcenter1MSE)[1],
      'G Center MSE': np.array(Gcenter1MSE)[2],
      # 'G ImgCenter MSE X': np.array(Gcenter2MSE)[0],
      # 'G ImgCenter MSE Y': np.array(Gcenter2MSE)[1],
      # 'G ImgCenter MSE': np.array(Gcenter2MSE)[2],
      
      'percentagecontainedP': int(incountourP),
      'percentagecontainedG': int(incountourG),
      'percentagecontainedafm': int(inafmcountour),

      'polygonareaP': polygonareaP,
      'polygonareaG': polygonareaG,
      'afmpolygonarea': afmpolygonarea,
      }

  evaluation = pd.DataFrame(data=e,index=[0])

  return evaluation

def error(pointer,gazex,gazey):

    gaze = [gazex,gazey]
    mse = mean_squared_error(pointer,gaze)
    mseX = mean_squared_error([pointer[0]], [gazex])
    mseY = mean_squared_error([pointer[1]], [gazey])

    return [mseX,mseY,mse]

def main(args):

  path = os.path.join(args.path,"datasetreal")
  contour_model = loadModel(args.path)
  pkl_data_path = os.path.join(args.path, args.GazeHeadFile)

  GazeandHeadVel = pd.read_pickle(pkl_data_path)
  GazeandHeadVelTraining  = GazeandHeadVel[GazeandHeadVel["usage"]==1]

  #remove unlinkely range of eye movements 
  GazeandHeadVelTraining  = GazeandHeadVelTraining[GazeandHeadVelTraining["GazeFoVDegreesX"]<50]
  GazeandHeadVelTraining  = GazeandHeadVelTraining[GazeandHeadVelTraining["GazeFoVDegreesX"]>-50]
  GazeandHeadVelTraining  = GazeandHeadVelTraining[GazeandHeadVelTraining["GazeFoVDegreesY"]<50]
  GazeandHeadVelTraining  = GazeandHeadVelTraining[GazeandHeadVelTraining["GazeFoVDegreesY"]>-50]

  #percentile = [np.percentile(GazeandHeadVelTraining["headVelMagnitude"].values,item) for item in np.arange(0,101,5)]
  percentile =  np.arange(0,12,1)
  
  evaluationall=[]

  for index,item in enumerate(percentile):
    if(index==len(percentile)-1): break
    minmag=percentile[index]
    maxmag=percentile[index+1]
    GazeandHeadVelTrainingCopy = GazeandHeadVelTraining[GazeandHeadVelTraining["headVelMagnitude"]>minmag]
    GazeandHeadVelTrainingCopy = GazeandHeadVelTrainingCopy[GazeandHeadVelTrainingCopy["headVelMagnitude"]<maxmag]
    GazeandHeadVelTrainingCopy = GazeandHeadVelTrainingCopy.dropna()

    maxsamples= 10000
    if len(GazeandHeadVelTrainingCopy)>maxsamples:
      remove_n = abs(len(GazeandHeadVelTrainingCopy)-maxsamples)
      drop_indices = np.random.choice(GazeandHeadVelTrainingCopy.index, remove_n, replace=False)
      GazeandHeadVelTrainingCopy = GazeandHeadVelTrainingCopy.drop(drop_indices)

    GazeandHeadVelTrainingCopy.reset_index(inplace = True)
    
    csv_data_out_path_eval = os.path.join(path, 'results_{0}_{1}.csv'.format(minmag,maxmag))

    columns = [
            'Angle',
            'Magnitude',
            'empty', 
            'Head Dir MSE X',
            'Head Dir MSE Y',
            'Head Dir MSE',
            'Head Dir MSE STD',
            'empty0',            
            'P Center MSE X',#predicted
            'P Center MSE Y',
            'P Center MSE',
            'P Center MSE STD',
            'G Center MSE X',#predicted
            'G Center MSE Y',
            'G Center MSE',
            'G Center MSE STD',
            'empty1',           
            'percentagecontainedP',
            'percentagecontainedG',
            'percentagecontainedafm',
            'empty3',
            'polygonareaP',
            'polygonareaG',
            'afmpolygonarea',
        ]

    evaluation = pd.DataFrame(columns=columns)
    average=[]

    for index, row in GazeandHeadVelTrainingCopy.iterrows():

      angle = round(row['headVelAng'], truncate)
      vel_head_x,vel_head_y = np.cos(math.radians(angle)),np.sin(math.radians(angle))
      mag = round(row['headVelMagnitude'], truncate)
      x = row['GazeFoVDegreesX']
      y = row['GazeFoVDegreesY']
      perc = 0.7

      start = time.time()
      prediction = predict(angle,mag,perc,contour_model)
      end = time.time()


      #e = evaluate(x,y,angle,mag,perc,prediction,False)
      #evaluation = evaluation.append(e, ignore_index=True)



      i = (index)/len(GazeandHeadVelTrainingCopy)
      average.append(end - start)
      expected = str(datetime.timedelta(seconds=round(np.mean(average),3) * (len(GazeandHeadVelTrainingCopy)-index)))
      sys.stdout.write("\rpercentage {0} expected time {1} average prediction time {2}ms".format(round(i,3),expected,np.mean(average)))
      sys.stdout.flush()
      if index>1000: average.pop(0)
        
       

def save(evaluation,csv_data_out_path_eval):

  anglerange=22.5
  anglesrangemin = np.arange(-180,181-anglerange,anglerange)
  anglesrangemmax = np.arange(-180+anglerange,181,anglerange)
  evaluationout = pd.DataFrame(columns=evaluation.columns)

  for index,item in enumerate(anglesrangemin):
    min = anglesrangemin[index]
    max = anglesrangemmax[index]
    range  = evaluation[evaluation['Angle']>min]
    range  = range[range['Angle']<max]
    
    if len(range)>0:
      e = {
          'Angle': np.mean(range['Angle'].values),
          'Magnitude': np.mean(range['Magnitude'].values),

          'Head Dir MSE X':  np.sqrt(np.mean(range['Head Dir MSE X'].values)),#head
          'Head Dir MSE Y':  np.sqrt(np.mean(range['Head Dir MSE Y'].values)),
          'Head Dir MSE': np.sqrt(np.mean(range['Head Dir MSE'].values)),
          'Head Dir MSE STD': np.std(np.sqrt(range['Head Dir MSE'].values)), 
          
          'P Center MSE X': np.sqrt(np.mean(range['P Center MSE X'].values)),#predicted
          'P Center MSE Y':  np.sqrt(np.mean(range['P Center MSE Y'].values)),
          'P Center MSE':  np.sqrt(np.mean(range['P Center MSE'].values)),
          'P Center MSE STD': np.std(np.sqrt(range['P Center MSE'].values)),

          'G Center MSE X':  np.sqrt(np.mean(range['G Center MSE X'].values)),#gaussian
          'G Center MSE Y':  np.sqrt(np.mean(range['G Center MSE Y'].values)),
          'G Center MSE':  np.sqrt(np.mean(range['G Center MSE'].values)),
          'G Center MSE STD': np.std(np.sqrt(range['G Center MSE'].values)),

          'percentagecontainedP': np.mean(range['percentagecontainedP'].values),
          'percentagecontainedG': np.mean(range['percentagecontainedG'].values),
          'percentagecontainedafm': np.mean(range['percentagecontainedafm'].values),

          'polygonareaP': np.mean(range['polygonareaP'].values),
          'polygonareaG': np.mean(range['polygonareaG'].values),
          'afmpolygonarea': np.mean(range['afmpolygonarea'].values),
          }
    else:
        e = {
          'Angle':np.mean([min,max]),
          'AngleStd': None,
          'Magnitude': np.mean(evaluation['Magnitude'].values),
          'MagnitudeSTD': None,

          'Head Dir MSE X': None,
          'Head Dir MSE Y': None,
          'Head Dir MSE': None,
          'Head Dir MSE STD': None,
          
          'P Center MSE X': None,
          'P Center MSE Y': None,
          'P Center MSE': None,
          'P Center MSE STD': None,

          'G Center MSE X': None,
          'G Center MSE Y': None,
          'G Center MSE': None,
          'G Center MSE STD': None,
          
          'percentagecontainedP': None,
          'percentagecontainedG': None,
          'percentagecontainedafm': None,

          'polygonareaP': None,
          'polygonareaG': None,
          'afmpolygonarea': None,
          }

    ev = pd.DataFrame(data=e,index=[0])
    evaluationout = evaluationout.append(ev, ignore_index=True)

  e = {
          'Angle': 0000000,
          'Magnitude': np.mean(evaluationout['Magnitude'].values),

          'Head Dir MSE X': np.mean(evaluationout['Head Dir MSE X'].values),#head
          'Head Dir MSE Y': np.mean(evaluationout['Head Dir MSE Y'].values),
          'Head Dir MSE': np.mean(evaluationout['Head Dir MSE'].values),
          'Head Dir MSE STD': np.mean(evaluationout['Head Dir MSE STD'].values),
          
          'P Center MSE X':np.mean(evaluationout['P Center MSE X'].values),#predicted
          'P Center MSE Y': np.mean(evaluationout['P Center MSE Y'].values),
          'P Center MSE': np.mean(evaluationout['P Center MSE'].values),
          'P Center MSE STD': np.std(np.sqrt(evaluationout['P Center MSE STD'].values)),

          'G Center MSE X': np.mean(evaluationout['G Center MSE X'].values),#gaussian
          'G Center MSE Y': np.mean(evaluationout['G Center MSE Y'].values),
          'G Center MSE': np.mean(evaluationout['G Center MSE'].values),
          'G Center MSE STD': np.std(np.sqrt(evaluationout['G Center MSE STD'].values)),

          'percentagecontainedP': np.mean(evaluationout['percentagecontainedP'].values),
          'percentagecontainedG': np.mean(evaluationout['percentagecontainedG'].values),
          'percentagecontainedafm': np.mean(evaluationout['percentagecontainedafm'].values),

          'polygonareaP': np.mean(evaluationout['polygonareaP'].values),
          'polygonareaG': np.mean(evaluationout['polygonareaG'].values),
          'afmpolygonarea': np.mean(evaluationout['afmpolygonarea'].values),
          }

  ev = pd.DataFrame(data=e,index=[0])
  evaluationout = evaluationout.append(e, ignore_index=True)

  evaluationout.sort_values(by=['Angle'])
  evaluationout.to_csv(csv_data_out_path_eval) 

  return evaluationout

def predict(angle,magnitude,percentile,contour_model):

  url = str(angle)+"_"+str(magnitude)+"_"+str(percentile)

  if url not in cache:

    features = np.expand_dims(np.array([angle,magnitude,percentile]),axis=0)

    prediction = contour_model.predict(features)

    p_contour= prediction[0].reshape(1,21,2)
    p_c_center = prediction[1]

    cache[url] = [p_contour,p_c_center]

  else: 
    p_contour,p_c_center = cache[url]

  return p_contour,p_c_center


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--GazeHeadFile", type=str, required=False)
    parser.add_argument("--evaluate", type=str, required=False)
    args = parser.parse_args()
    #main(args)
    roundtest(args)
