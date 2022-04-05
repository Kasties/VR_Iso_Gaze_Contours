import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
import math
import os
import pandas as pd
import argparse
from FixedArff import determineCondition
from sklearn.metrics import mean_squared_error
from math import sqrt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pickle as pkl
from folders import *

HeadDirectionMeanSquaredError =[]
center1MeanSquaredError = []
center2MeanSquaredError = []
centerMeanSquaredError = []
percentagecontained = []
percentagecontainedafm =[]
polygonarea=[]
afmpolygonarea=[]

def getImageXYZ(img):

    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(x, y)

    smooth = gaussian_filter(img, sigma=0)

    return X,Y,smooth

def imageplt(img,axs):

    imgplot = axs.imshow(img)

    #apply color to image
    cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
    imgplot.set_cmap(cmap)

def angleContour( c):


    contour = c[0][0]
    center = np.mean(contour, axis=0)
    keepangles =  np.arange(-180, 181, 360/20)
    angles=[]
    returnarray =[]
    x1, y1 = center

    for i in contour:
        x,y = i
        xc=x-center[0]
        yc=y-center[1]
        angles.append(Angle2D(xc,yc))

    for angle in keepangles:
        index,number = find_nearest(angle,angles)
        #print(str(angle)+" "+str(number))
        returnarray.append(contour[index])
        x2, y2 = contour[index]
        #plt.plot(x1, y1, x2, y2, marker = 'o')
        #plt.axline((x1, y1), (x2, y2))

    return returnarray

def Angle2D(x,y):

    zup=np.array([0,-1])

    return ((angle_between(np.array([x,y]), zup)/math.pi)*180)*np.sign(x)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def extract(imgpath,percentage, plot):

    fig, axs = plt.subplots(nrows=1, ncols=1)
    img = mpimg.imread(imgpath)
    imageplt(img,axs)

    X,Y,Z = getImageXYZ(img)
    xy = np.stack([X,Y], axis=2)
    center = np.array([np.ma.average(X, axis=(0,1), weights=Z), np.ma.average(Y, axis=(0,1), weights=Z)])
    plt.scatter(*center)
    print(center)

    level = [1-percentage,1]

    norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())

    cs = plt.contour(X, Y, Z, level, norm=norm, colors="k", linestyles="solid")
    p = cs.allsegs
    contour = angleContour( p)
    contour = np.array(contour)

    axs.plot(contour[:,0], contour[:,1])

    if(plot):
        plt.show()
    else:
        plt.close('all')

    # conversion pixel to angular coordinates
    resolution = img.shape[0]

    center = [center[0],abs(center[1]-resolution)]
    center = [((center[0]/resolution)*100)-50,((center[1]/resolution)*100)-50]

    contourcentered = [ [x,abs(y-resolution)] for x,y in contour.tolist() ] #invert y axes
    contourcentered = [ [((x/resolution)*100)-50,((y/resolution)*100)-50] for x,y in contourcentered ] #return to angle coordinates

    # fig,axes=plt.subplots(1,2)

    # axes[0].imshow(img)
    # axes[0].plot(np.array(contour)[:,0],np.array(contour)[:,1])
    # axes[0].set_box_aspect(1)

    # axes[1].scatter(center[0],center[1])
    # axes[1].plot(np.array(contourcentered)[:,0],np.array(contourcentered)[:,1])
    # axes[1].vlines(0,-50, 50)
    # axes[1].hlines(0,-50, 50)
    # axes[1].set_ylim(-50, 50)
    # axes[1].set_xlim(-50, 50)
    # axes[1].set_box_aspect(1)

    # plt.show()

    return contourcentered, center

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

def evaluate(GazeandHeadVel,parameters):

    angle = parameters['angles'].values[0]
    anglesRange = parameters['anglesRange'].values[0]
    magnitudeRangesmin = parameters['magnitudeRangesmin'].values[0]
    magnitudeRangesmax = parameters['magnitudeRangesmax'].values[0]
    contoursCenter = parameters['contoursPCenter'].values[0]
    imageCenter = parameters['imagePCenter'].values[0]
    contour = parameters['contoursP'].values[0]

    anglecondition = determineAngleCondition(GazeandHeadVel,angle,anglesRange)
    conditions=[
        anglecondition,
        GazeandHeadVel["headVelMagnitude"]>=magnitudeRangesmin,
        GazeandHeadVel["headVelMagnitude"]<magnitudeRangesmax
    ]

    x,y = ConditionalFilter(GazeandHeadVel,conditions)
    print(parameters['samplesCount'].values[0])
    print(len(x))


    MeanX = np.mean(x)
    MeanY = np.mean(y)
    HeadDirectionMeanSquaredError.append(error([0,0],x,y))
    center1MeanSquaredError.append(error(contoursCenter,x,y))
    center2MeanSquaredError.append(error(imageCenter,x,y))
    centerMeanSquaredError.append(error([MeanX,MeanY],x,y))



    percentile = str(int(parameters['percentile'].values[0]*10))
    afmcontour = np.load(os.path.join(folder_code,'afm','afm.'+percentile+'.pkl'),allow_pickle=True)
    pointscontainedincountour=[]
    pointscontainedinafmcountour=[]

    polygon = Polygon(contour)
    afmpolygon = Polygon(afmcontour)
    for i,px in enumerate(x):
        point = Point(x[i],y[i])
        pointscontainedincountour.append(polygon.contains(point))
        pointscontainedinafmcountour.append(afmpolygon.contains(point))

    pointscontainedinafmcountour=np.array(pointscontainedinafmcountour)
    pointscontainedincountour=np.array(pointscontainedincountour)

    percentagecontained.append(len(pointscontainedincountour[pointscontainedincountour])/len(pointscontainedincountour))
    percentagecontainedafm.append(len(pointscontainedinafmcountour[pointscontainedinafmcountour])/len(pointscontainedinafmcountour))

    polygonarea.append(polygon.area)
    afmpolygonarea.append(afmpolygon.area)

    plt.scatter(x,y,s=.01)
    plt.scatter(MeanX,MeanY,s=15,c='red')
    plt.scatter(imageCenter[0],imageCenter[1],s=15,c='green')
    plt.scatter(contoursCenter[0],contoursCenter[1],s=15,c='blue')
    plt.plot(np.array(contour)[:,0],np.array(contour)[:,1],c='orange')
    plt.plot(np.array(afmcontour)[:,0],np.array(afmcontour)[:,1],c='magenta')
    plt.vlines(0,-50,50)
    plt.hlines(0,-50,50)
    plt.ylim(-50,50)
    plt.xlim(-50,50)
    plt.show()
    plt.clf()

def error(pointer,gazex,gazey):

    pointerlistX = [pointer[0]]*len(gazex)
    pointerlistY = [pointer[1]]*len(gazey)
    mse = mean_squared_error(np.sqrt(np.square(pointerlistX)+np.square(pointerlistY)),np.sqrt(np.square(gazex)+np.square(gazey)))
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
    csv_data_out_path_csv = os.path.join(prediction_data_path, 'parameters.csv')
    csv_data_out_path_eval = os.path.join(prediction_data_path, 'evaluation.csv')
    csv_data_out_path_pkl = os.path.join(prediction_data_path, 'parameters.pkl')
    dfin = pd.read_csv(csv_data_path)
    
    GazeandHeadVel = pd.read_pickle(pkl_data_path)
    GazeandHeadVel = GazeandHeadVel[GazeandHeadVel["usage"]==0]

    columns = [
        'angles',
        'anglesRange',
        'magnitude',
        'magnitudeRangesmin',
        'magnitudeRangesmax',
        'percentile',
        'contoursP',
        'contoursPCenter',
        'imagePCenter',
        'contoursG',
        'contoursGCenter',
        'imageGCenter']

    dfout= pd.DataFrame(columns=columns)

    for index,item in enumerate(dfin.values):

        imgpathPredicted=os.path.join(prediction_data_path, dfin['imgpath'][index])
        imgpathGaussian=os.path.join(gaussian_data_path, dfin['imgpath'][index])
        percentileRange= np.arange(0.2,1,0.1)

        for percentile in percentileRange:
            print(percentile)
            contourPredicted,centerPredicted = extract(imgpathPredicted,percentile, args.plot)
            contourGaussian,centerGaussian = extract(imgpathGaussian,percentile, args.plot)

            if len(contourPredicted)!=21:
                continue

            d = {'angles': [dfin['angles'][index]],
            'anglesRange': [dfin['anglesRange'][index]],
            'samplesCount': [int(dfin['samplesCount'][index])],
            'magnitude': [dfin['magnitude'][index]],
            'magnitudeRangesmin':  [dfin['magnitudeRangesmin'][index]],
            'magnitudeRangesmax': [dfin['magnitudeRangesmax'][index]],
            'percentile':[percentile],
            
            'contoursP':[contourPredicted],
            'contoursPCenter':[np.mean(contourPredicted,axis=0)],
            'imagePCenter':[centerPredicted],

            'contoursG':[contourGaussian],
            'contoursGCenter':[np.mean(contourGaussian,axis=0)],
            'imageGCenter':[centerGaussian]}

            parameters = pd.DataFrame(data=d)

            if(args.evaluate):

                evaluate(GazeandHeadVel,parameters)

                e = {
                    'Head Dir MSE X': np.array(HeadDirectionMeanSquaredError)[:,0],
                    'Head Dir MSE Y': np.array(HeadDirectionMeanSquaredError)[:,1],
                    'Head Dir MSE': np.array(HeadDirectionMeanSquaredError)[:,2],
                    'CCenter MSE X': np.array(center1MeanSquaredError)[:,0],
                    'CCenter MSE Y': np.array(center1MeanSquaredError)[:,1],
                    'CCenter MSE': np.array(center1MeanSquaredError)[:,2],
                    'ImgCenter MSE X': np.array(center2MeanSquaredError)[:,0],
                    'ImgCenter MSE Y': np.array(center2MeanSquaredError)[:,1],
                    'ImgCenter MSE': np.array(center2MeanSquaredError)[:,2],
                    'MeanCenter MSE X': np.array(centerMeanSquaredError)[:,0],
                    'MeanCenter MSE Y': np.array(centerMeanSquaredError)[:,1],
                    'MeanCenter MSE': np.array(centerMeanSquaredError)[:,2],
                    'percentagecontained': percentagecontained,
                    'percentagecontainedafm': percentagecontainedafm,
                    'polygonarea': polygonarea,
                    'afmpolygonarea': afmpolygonarea,
                    }

                evaluation = pd.DataFrame(data=e)
                evaluation.to_csv(csv_data_out_path_eval)

            dfout= dfout.append(parameters, ignore_index=True)
            print(dfout.shape)

    dfout.to_csv(csv_data_out_path_csv)
    dfout.to_pickle(csv_data_out_path_pkl)

# contour,center = extract("C:\\Users\\Riccardo\\Documents\\GitHub\\ARFF_toolkit_new\\dataset2\\datasetreal\\afm\\afm_predicted.png",.9, True)
# with open('C:\\Users\\Riccardo\\Documents\\GitHub\\ARFF_toolkit_new\\dataset2\\datasetreal\\afm\\afm.9.pkl','wb') as f:
#     pkl.dump(contour, f)
# contour = np.load('C:\\Users\\Riccardo\\Documents\\GitHub\\ARFF_toolkit_new\\dataset2\\datasetreal\\afm\\afm.pkl',allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, required=True)
    parser.add_argument("--plot", type=bool, required=False, default=False)
    parser.add_argument("--GazeHeadFile", type=str, required=False, default=False)
    parser.add_argument("--evaluate", type=bool, required=False, default=False)
    args = parser.parse_args()
    main(args)
