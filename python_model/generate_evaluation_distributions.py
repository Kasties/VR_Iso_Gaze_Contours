import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from folders import *
from heatmap import   generateheatmapFromDistributionrgba,generateheatmap,fitting2Dgaussian
import cv2

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



def main(args):

    #read test data
    path = os.path.join(args.path,"datasetreal")
    prediction_data_path = os.path.join(path, 'predictions')
    pkl_data_path = os.path.join(args.path, args.GazeHeadFile)
    conotours_pkl = os.path.join(prediction_data_path, 'parameters.pkl')
    contours = pd.read_pickle(conotours_pkl)
    GazeandHeadVel = pd.read_pickle(pkl_data_path)
    GazeandHeadVel = GazeandHeadVel[GazeandHeadVel["usage"]==1]
    distributionpathEval = os.path.join( path,"distributionsEval")
    resolution=128
    os.makedirs(distributionpathEval)

    for index,item in enumerate(contours.values):

            parameters = contours.iloc[index]
            angle = parameters['angles']
            anglesRange = parameters['anglesRange']
            magnitudeRangesmin = parameters['magnitudeRangesmin']
            magnitudeRangesmax = parameters['magnitudeRangesmax']

            print(angle,magnitudeRangesmin,magnitudeRangesmax)


            anglecondition = determineAngleCondition(GazeandHeadVel,angle,anglesRange)
            conditions=[
                anglecondition,
                GazeandHeadVel["headVelMagnitude"]>=magnitudeRangesmin,
                GazeandHeadVel["headVelMagnitude"]<magnitudeRangesmax
            ]

            x,y = ConditionalFilter(GazeandHeadVel,conditions)

            imgname =  str(index).zfill(8) +".png"

            rgb = generateheatmapFromDistributionrgba(x,y,None, resolution, min = -50,max= 50, isGray =True)
            cv2.imwrite(distributionpathEval+"\\"+imgname, rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, required=True)
    parser.add_argument("--GazeHeadFile", type=str, required=False, default=False)
    args = parser.parse_args()
    main(args)
