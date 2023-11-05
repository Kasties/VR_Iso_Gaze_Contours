import pandas as pd
import matplotlib.pyplot as plt
from heatmap import   generateheatmapFromDistributionrgba,generateheatmap,fitting2Dgaussian
import cv2
import numpy as np
import argparse
import os 

def Magnitudeslicing(df,condition,numeroPercentili,angle,anglerange):

    if condition != None: 
        condition = conditionComputing(condition)
        conditionalData = df[condition]
    else:
        conditionalData = df
    
    data = conditionalData["headVelMagnitude"].values
    
    percentileRange = np.arange(0,100+1,100/numeroPercentili)
    percentiles=[]
    conditions=[]
    datasets=[]
    sampleCount=[]
    means=[]
    rangesmin=[]
    rangesmax=[]
    angles=[angle]*numeroPercentili
    anglesranges=[anglerange]*numeroPercentili

    for pecentage in percentileRange:
        percentiles.append(np.percentile(data, pecentage))

    for index,percentile in enumerate(percentiles):
        if index+1 <len(percentiles):
            condition = np.logical_and(conditionalData["headVelMagnitude"]>=percentile,conditionalData["headVelMagnitude"]<percentiles[index+1])
            conditions.append(condition)
    
    for condition in conditions:
        datasets.append(conditionalData[condition])
        sampleCount.append(len(conditionalData[condition]["headVelMagnitude"]))

    for data in datasets:
        means.append(np.round(np.mean(data["headVelMagnitude"]), 3))
        rangesmin.append(np.round(np.min(data["headVelMagnitude"]),3))
        rangesmax.append(np.round(np.max(data["headVelMagnitude"]),3))

    d = {
    'angles':angles,
    'anglesRange':anglesranges,
    'datasets':datasets,
    'samplesCount': sampleCount,
    'magnitude':means,
    'magnitudeRangesmin':rangesmin,
    'magnitudeRangesmax':rangesmax}

    df = pd.DataFrame(data=d)

    return df

def ConditionalFilter(df,condition):

    if condition != None: 
        condition = conditionComputing(condition)
        conditionalData = df[condition]
    else:
        conditionalData = df

    x = conditionalData['GazeFoVDegreesX'].values
    y = conditionalData['GazeFoVDegreesY'].values 

    return  x,y

def conditionComputing(array):

    finalcondition = array[0]

    for index, condition in enumerate(array):

        if index>0:
            finalcondition =np.logical_and(finalcondition,condition)

    return finalcondition

def determineCondition(df,angle,range,tresh, numeropercentilimagnitudine):
    
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

        condition = [np.logical_or(slice1,slice2)]
 
    else:
        topboundary = df["headVelAng"]>angle-range
        bottomboundary = df["headVelAng"]<angle+range
        condition = [topboundary,bottomboundary]

    x,_ = ConditionalFilter(df,condition)
    count = len(x)
    if count>= tresh:
        return Magnitudeslicing(df,condition,numeropercentilimagnitudine,angle,np.round(range, 2))
    else:
        return determineCondition(df,angle,range+0.1,tresh, numeropercentilimagnitudine)

def file_path(string):
    print(string)
    if os.path.isfile(string):
        return string
    else:
        raise Exception(string + " is not a pkl file")

def dir_path_create(string):
    
    fullpath = os.path.abspath(string)
    print(fullpath)
    #NOTE I did a hack datsetreal_from_dl is what I dl I have no idea what the original idea was since there are no comments or documention and the paper says noting
    if os.path.isdir(fullpath):
        print(fullpath)
        raise Exception("directory already existing either delete directory manually or change output directory")
    else:
        os.makedirs(fullpath)
        return fullpath

def main(args):

    pklfilename = args.file 
    outputpath = args.outputpath
    distributionpath = os.path.join(outputpath,"distributions")
    gaussianpath = os.path.join(outputpath,"gaussians")

    os.makedirs(distributionpath)
    os.makedirs(gaussianpath)

    df = pd.read_pickle(pklfilename)

    conditions = []
    parameters = []
    resolution = args.resolution 
    angles = np.arange(-180,181,360/40)
    numberofanglesamples = 150000
    numeropercentilimagnitudine = 3
    numberofmagnitudesamples=numberofanglesamples/numeropercentilimagnitudine

    columns = [
        'angles',
        'anglesRange',
        'samplesCount',
        'magnitude',
        'magnitudeRangesmin',
        'magnitudeRangesmax']

    data= pd.DataFrame(columns=columns)

    imagenumber=0

    for angle in angles:
        range=1
        parameters.append(angle)
        dataAngle = determineCondition(df,angle,0.1,numberofanglesamples, numeropercentilimagnitudine)
        imgpaths=[]

        for index,item in enumerate(dataAngle.values):
            string =''
            
            for column in columns:
                string+=column+' '
                string+=str(dataAngle[column][index]) + " "

            print(string)

            #saveImage    
            x,y = ConditionalFilter(dataAngle['datasets'][index],None)
            imgname =  str(imagenumber).zfill(8) +".png"
            imgpaths.append(imgname)

            rgb = generateheatmapFromDistributionrgba(x,y,None, resolution, min = -50,max= 50, isGray =True)
            cv2.imwrite(distributionpath+"/"+imgname, rgb)

            rgbgauss = fitting2Dgaussian(x,y,None, resolution, min = -50,max= 50, isGray =True)     
            cv2.imwrite(gaussianpath+"/"+imgname, rgbgauss)

            imagenumber+=1
        
        dataAngle['imgpath']=imgpaths
        dataAngle = dataAngle.drop(['datasets'],axis=1)
        data= data.append(dataAngle, ignore_index=True)
        print(data.shape)

    data.to_csv(distributionpath+"/parameters.csv")

def saveafm(args):

    pklfilename = args.file 
    outputpath = args.outputpath
    distributionpath = os.path.join(outputpath,"distributions")
    resolution = resolution = args.resolution 

    df = pd.read_pickle(pklfilename)

    #saveImage    
    x,y = ConditionalFilter(df,None)
    rgb = generateheatmapFromDistributionrgba(x,y,None, resolution, min = -50,max= 50, isGray =True)
    rgbgauss = fitting2Dgaussian(x,y,None, resolution, min = -50,max= 50, isGray =True)
    imgname =  "afm.png"
    cv2.imwrite(distributionpath+"\\"+imgname, rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=file_path, required=True)
    parser.add_argument("--outputpath", type=dir_path_create, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
    #saveafm(args)

# --path 'AccVelarff.pkl' --type arff --outputpath dataset_real --resolution 128


