import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from folders import *


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def determineAngleCondition(df,angle,range):

    anglemax = angle+range
    anglemin = angle-range

    if(anglemax>180 or anglemin<-180):
        diff      = anglemax-180   if anglemax>180 else anglemin+180
        slice1min = anglemin       if anglemax>180 else -180
        slice1max = 180            if anglemax>180 else anglemax
        slice2min = -180           if anglemax>180 else 180 + diff
        slice2max = -180 + diff    if anglemax>180 else 180

        slice1topboundary = df["headVelAng"]>slice1min
        slice1bottomboundary = df["headVelAng"]<slice1max
        slice1 = np.logical_and(slice1topboundary,slice1bottomboundary)

        slice2topboundary = df["headVelAng"]>slice2min
        slice2bottomboundary = df["headVelAng"]<slice2max
        slice2 = np.logical_and(slice2topboundary,slice2bottomboundary)

        condition = np.logical_or(slice1,slice2)

    else:
        topboundary = df["headVelAng"]>angle-range
        bottomboundary = df["headVelAng"]<angle+range
        condition = np.logical_and(topboundary,bottomboundary)

    return condition

def conditionComputing(array):

    finalcondition = array[0]

    for index, condition in enumerate(array):

        if index>0:
            finalcondition =np.logical_and(finalcondition,condition)

    return finalcondition

def ConditionalFilter(df,conditions):

    if conditions != None:
        condition = conditionComputing(conditions)
        conditionalData = df[condition]
    else:
        conditionalData = df

    x = conditionalData['GazeFoVDegreesX'].values
    y = conditionalData['GazeFoVDegreesY'].values

    return  x,y

def evaluate(GazeandHeadVel,parameters,plot):

    angle = parameters['angles']
    anglesRange = parameters['anglesRange']
    magnitudeRangesmin = parameters['magnitudeRangesmin']
    magnitudeRangesmax = parameters['magnitudeRangesmax']
    
    contoursPCenter = parameters['contoursPCenter']
    imagePCenter = parameters['imagePCenter']
    contourP = parameters['contoursP']
    
    contoursGCenter = parameters['contoursGCenter']
    imageGCenter = parameters['imageGCenter']
    contourG = parameters['contoursG']

    print(angle,magnitudeRangesmin,magnitudeRangesmax)


    anglecondition = determineAngleCondition(GazeandHeadVel,angle,anglesRange)
    conditions=[
        anglecondition,
        GazeandHeadVel["headVelMagnitude"]>=magnitudeRangesmin,
        GazeandHeadVel["headVelMagnitude"]<magnitudeRangesmax
    ]

    x,y = ConditionalFilter(GazeandHeadVel,conditions)
    
    if(len(x)==0): 

            print("not enough data")

            e = {
            'Head Dir MSE X':None,
            'Head Dir MSE Y': None,
            'Head Dir MSE': None,
            'CCenter MSE X': None,
            'CCenter MSE Y': None,
            'CCenter MSE': None,
            'ImgCenter MSE X': None,
            'ImgCenter MSE Y': None,
            'ImgCenter MSE': None,
            'MeanCenter MSE X': None,
            'MeanCenter MSE Y': None,
            'MeanCenter MSE': None,
            'percentagecontained': None,
            'percentagecontainedafm': None,
            'polygonarea': None,
            'afmpolygonarea': None,
            }

    else: 
        
        print(len(x))

        MeanX = np.mean(x)
        MeanY = np.mean(y)
        HeadDirectionMSE = error([0,0],x,y)
        
        Pcenter1MSE = error(contoursPCenter,x,y)
        Pcenter2MSE = error(imagePCenter,x,y)
        
        Gcenter1MSE = error(contoursGCenter,x,y)
        Gcenter2MSE = error(imageGCenter,x,y)


        percentile = str(int(parameters['percentile']*10))
        afmcontour = np.load(os.path.join(folder_code,'afm','afm.'+percentile+'.pkl'),allow_pickle=True)
        
        pointscontainedincountourP=[]
        pointscontainedincountourG=[]
        pointscontainedinafmcountour=[]

        polygonP = Polygon(contourP)
        polygonG = Polygon(contourG)
        afmpolygon = Polygon(afmcontour)
        for i,px in enumerate(x):
            point = Point(x[i],y[i])
            pointscontainedincountourP.append(polygonP.contains(point))
            pointscontainedincountourG.append(polygonG.contains(point))
            pointscontainedinafmcountour.append(afmpolygon.contains(point))

        pointscontainedinafmcountour=np.array(pointscontainedinafmcountour)
        pointscontainedincountourP=np.array(pointscontainedincountourP)
        pointscontainedincountourG=np.array(pointscontainedincountourG)

        percentagecontainedP = len(pointscontainedincountourP[pointscontainedincountourP])/len(pointscontainedincountourP)
        percentagecontainedG = len(pointscontainedincountourG[pointscontainedincountourG])/len(pointscontainedincountourG)
        percentagecontainedafm = len(pointscontainedinafmcountour[pointscontainedinafmcountour])/len(pointscontainedinafmcountour)

        polygonareaP = polygonP.area
        polygonareaG = polygonG.area
        afmpolygonarea = afmpolygon.area

        if (plot):
            plt.scatter(x,y,s=.01)
            plt.scatter(MeanX,MeanY,s=15,c='red')

            plt.scatter(imagePCenter[0],imagePCenter[1],s=10,c='orange')#predicted
            plt.scatter(contoursPCenter[0],contoursPCenter[1],s=20,c='orange')
            plt.plot(np.array(contourP)[:,0],np.array(contourP)[:,1],c='orange')

            plt.scatter(imageGCenter[0],imageGCenter[1],s=10,c='green')#gaussian
            plt.scatter(contoursGCenter[0],contoursGCenter[1],s=20,c='green')
            plt.plot(np.array(contourG)[:,0],np.array(contourG)[:,1],c='green')

            plt.plot(np.array(afmcontour)[:,0],np.array(afmcontour)[:,1],c='magenta')
            plt.vlines(0,-50,50)
            plt.hlines(0,-50,50)
            plt.ylim(-50,50)
            plt.xlim(-50,50)
            plt.show()
            plt.clf()

        e = {
            'Head Dir MSE X': np.array(HeadDirectionMSE)[0],#head
            'Head Dir MSE Y': np.array(HeadDirectionMSE)[1],
            'Head Dir MSE': np.array(HeadDirectionMSE)[2],
            
            'CCPenter MSE X': np.array(Pcenter1MSE)[0],#predicted
            'CCPenter MSE Y': np.array(Pcenter1MSE)[1],
            'CCPenter MSE': np.array(Pcenter1MSE)[2],
            'ImgPCenter MSE X': np.array(Pcenter2MSE)[0],
            'ImgPCenter MSE Y': np.array(Pcenter2MSE)[1],
            'ImgPCenter MSE': np.array(Pcenter2MSE)[2],
            
            'CCGenter MSE X': np.array(Gcenter1MSE)[0],#gaussian
            'CCGenter MSE Y': np.array(Gcenter1MSE)[1],
            'CCGenter MSE': np.array(Gcenter1MSE)[2],
            'ImgGCenter MSE X': np.array(Gcenter2MSE)[0],
            'ImgGCenter MSE Y': np.array(Gcenter2MSE)[1],
            'ImgGCenter MSE': np.array(Gcenter2MSE)[2],

            'percentagecontainedP': percentagecontainedP,
            'percentagecontainedG': percentagecontainedG,
            'percentagecontainedafm': percentagecontainedafm,

            'polygonareaP': polygonareaP,
            'polygonareaG': polygonareaG,
            'afmpolygonarea': afmpolygonarea,
            }

    evaluation = pd.DataFrame(data=e,index=[0])

    return evaluation

def error(pointer,gazex,gazey):

    pointerlistX = [pointer[0]]*len(gazex)
    pointerlistY = [pointer[1]]*len(gazey)
    gaze = np.stack([gazex,gazey],axis=1)
    pointer = np.stack([pointerlistX,pointerlistY],axis=1)
    mse = mean_squared_error(pointer,gaze)
    mseX = mean_squared_error(pointerlistX, gazex)
    mseY = mean_squared_error(pointerlistY, gazey)

    return [mseX,mseY,mse]

def main(args):

    #read test data
    path = os.path.join(args.path,"datasetreal")
    data_path = os.path.join(path, 'distributions')
    prediction_data_path = os.path.join(path, 'predictions')
    gaussian_data_path = os.path.join(path, 'gaussians')
    pkl_data_path = os.path.join(args.path, args.GazeHeadFile)
    csv_data_path = os.path.join(data_path, 'parameters.csv')
    
    csv_data_out_path_eval = os.path.join(prediction_data_path, args.outEvaluate)
    conotours_pkl = os.path.join(prediction_data_path, 'parameters.pkl')
    
    contours = pd.read_pickle(conotours_pkl)
    GazeandHeadVel = pd.read_pickle(pkl_data_path)
    GazeandHeadVel = GazeandHeadVel[GazeandHeadVel["usage"]==1]

    columns = [
            'Head Dir MSE X',
            'Head Dir MSE Y',
            'Head Dir MSE',
            'empty0',            
            'CCPenter MSE X',#predicted
            'CCPenter MSE Y',
            'CCPenter MSE',
            'ImgPCenter MSE X',
            'ImgPCenter MSE Y',
            'ImgPCenter MSE',
            'empty1',           
            'CCGenter MSE X',#gaussian
            'CCGenter MSE Y',
            'CCGenter MSE',
            'ImgGCenter MSE X',
            'ImgGCenter MSE Y',
            'ImgGCenter MSE',
            'empty2',
            'percentagecontainedP',
            'percentagecontainedG',
            'percentagecontainedafm',
            'empty3',
            'polygonareaP',
            'polygonareaG',
            'afmpolygonarea',
        ]

    evaluation = pd.DataFrame(columns=columns)

    for index,item in enumerate(contours.values):

            parameters = contours.iloc[index]
 
            e = evaluate(GazeandHeadVel,parameters,False)

            evaluation = evaluation.append(e, ignore_index=True)
    
    evaluation.to_csv(csv_data_out_path_eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, required=True)
    parser.add_argument("--plot", type=bool, required=False, default=False)
    parser.add_argument("--GazeHeadFile", type=str, required=False, default=False)
    parser.add_argument("--outEvaluate", type=str, required=True)
    args = parser.parse_args()
    main(args)
