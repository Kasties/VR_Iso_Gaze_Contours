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

def getImageXYZ(img):

    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(x, y)
    
    smooth = gaussian_filter(img, sigma=0)

    return X,Y,smooth

def imageplt(img,axs):
    
    imgplot = axs.imshow(img)
    cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
    imgplot.set_cmap(cmap)


def smoothimage(imgpath):

    fig, axs = plt.subplots(nrows=1, ncols=2)
    
    img = mpimg.imread(imgpath)
    img=img/img.max() 
    imageplt(img,axs[0])

    img = gaussian_filter(img, sigma=2)
    img = img*img/4
    img = gaussian_filter(img, sigma=1)
    img = img/img.max()
    imageplt(img,axs[1])

    plt.show()
    
    return img

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main(args):

    #read test data
    path = os.path.join(args.path,"datasetreal")
    data_path = os.path.join(path, 'distributions') 
    prediction_data_path = os.path.join(path, 'predictions')
    pkl_data_path = os.path.join(args.path, args.GazeHeadFile) 
    csv_data_path = os.path.join(data_path, 'parameters.csv') 
    csv_data_out_path_csv = os.path.join(prediction_data_path, 'parameters.csv') 
    csv_data_out_path_eval = os.path.join(prediction_data_path, 'evaluation.csv') 
    csv_data_out_path_pkl = os.path.join(prediction_data_path, 'parameters.pkl') 
    dfin = pd.read_csv(csv_data_path)
    GazeandHeadVel = pd.read_pickle(pkl_data_path)


    for index,item in enumerate(dfin.values):
    
        imgpath=os.path.join(data_path, dfin['imgpath'][index])
        smoothimage(imgpath)
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=dir_path, required=True)
    parser.add_argument("--plot", type=bool, required=False, default=False)
    parser.add_argument("--GazeHeadFile", type=str, required=False, default=False)
    parser.add_argument("--evaluate", type=bool, required=False, default=False)
    args = parser.parse_args()
    main(args)
