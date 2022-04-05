import os
from pathlib import Path
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

# importing cv2 
import cv2
  
from scipy.stats import gamma
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.stats import multivariate_normal

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from heatmap import  generateheatmapFromDistribution,generateheatmapFromDistributionrgba
# multivariate distributions with different types
# put in different functions
# take the full, draw heatmap
# take the full - 30% of data, draw heatmap
#OR
# distribution from curve is the full
# distribution from sampling is the one that need to be fixed

globalmask = 0

def augmentation(x,y):
    flip = random.choice(range(8)) # 0=no, 1 flip x, 2 flip y, 3 flip x,y # 4=swap, 5 flip x and swap, 6 flip y and swap, 7 flip x,y and swap
    if(flip == 0): return x,y
    elif(flip == 1): return np.negative(x),y
    elif(flip == 2): return x,np.negative(y)
    elif(flip == 3): return np.negative(x), np.negative(y)
    elif(flip == 4): return y,x
    elif(flip == 5): return y,np.negative(x)
    elif(flip == 6): return np.negative(y),x
    else: return np.negative(y), np.negative(x)

def augmentation(z):
    flip = random.choice(range(8)) # 0=no, 1 flip x, 2 flip y, 3 flip x,y # 4=swap, 5 flip x and swap, 6 flip y and swap, 7 flip x,y and swap
    if(flip == 0): return z
    elif(flip == 1): return np.rot90(z, k=1)
    elif(flip == 2): return np.rot90(z, k=2)
    elif(flip == 3): return np.rot90(z, k=3)
    elif(flip == 4): return np.flipud(z)
    elif(flip == 5): return np.rot90(np.flipud(z), k=1)
    elif(flip == 6): return np.rot90(np.flipud(z), k=2)
    else: return np.rot90(np.flipud(z), k=3)

#poissonCurve(0,1000,1, 3000,-2500)
def gammaCurveViz(a):
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    y = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = gamma.pdf(xx, a,  loc=-3, scale=1) * gamma.pdf(yy, a,  loc=-3, scale=1)
    h = plt.contourf(x, y, z)
    plt.axis('scaled')
    plt.show()
    return z

#gaussianCurve()
def gaussianCurveViz(mu=10, sigma=1):
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    y = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = norm.pdf(xx) * norm.pdf(yy)
    h = plt.contourf(x, y, z)
    plt.axis('scaled')
    plt.show()
    return z
    
#multivariateGaussianCurve()
def multivariateGaussianCurveViz(mu=2.5, cov=0.5):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    z = rv.pdf(pos)
    plt.contourf(x, y, z)
    plt.show()
    return z
    
def skewCurveViz(skewness):
    x = np.linspace(skewnorm.ppf(0.01, skewness), skewnorm.ppf(0.99, skewness), 100)
    y = np.linspace(skewnorm.ppf(0.01, skewness), skewnorm.ppf(0.99, skewness), 100)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = skewnorm.pdf(xx, skewness) * skewnorm.pdf(yy,skewness)
    h = plt.contourf(x, y, z)
    plt.axis('scaled')
    plt.show()
    return z

def addNoise(signal, params = {'type':'uniform', 'min':-2, 'max':2, 'loc':0, 'scale':3}, mask=False):
    if(params['type'] == 'uniform'):
        signal = signal + signal*np.random.uniform(params['min'], params['max'], signal.shape)
    if(params['type'] == 'normal'):
        signal = signal + signal*np.random.normal(params['loc'], params['var'], signal.shape)
    if(params['type'] == 'gamma'):
        signal = signal + signal*np.random.gamma(signal.shape, params['scale'], 30000)
    m = None
    if(mask):
        m = np.random.uniform(0,np.max(signal),signal.shape)
        signal = signal - m
    return signal

resolution = 128
px = 1/plt.rcParams['figure.dpi']

def saveImages(x,y,zOrig, zNoised, resolution, bounds, fileName, filenameNoise):
    
    fig,axes = plt.subplots(figsize=(resolution*px, (resolution+3)*px)) #trick to have a squared image
    
    axe = generateheatmapFromDistribution(x,y,zOrig,axes,1, resolution, bounds = bounds)
    extent = plt.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fileName, bbox_inches=extent)
    
    #fig.savefig(fileName)
    #plt.show()
    
    
    axe = generateheatmapFromDistribution(x,y,zNoised,axes,1, resolution, bounds = bounds)
    extent = plt.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filenameNoise, bbox_inches=extent)
    #fig.savefig(filenameNoise)
    plt.close('all')

def saveImagesrgba(x,y,zOrig, zNoised, resolution, bounds, fileName, filenameNoise):
    
    rgb = generateheatmapFromDistributionrgba(x,y,zOrig, resolution, isGray = True)

    # rgb = rgb/rgb.max()
    # rgb=rgb*rgb/3
    # rgb = rgb/rgb.max()
    # rgb=rgb*255

    cv2.imwrite(fileName, rgb)

    rgb = generateheatmapFromDistributionrgba(x,y,zNoised, resolution, isGray = True)

    
    cv2.imwrite(filenameNoise, rgb)

#MULTIVARIATE GAUSSIAN (WORKS)
def multivariateNormal(mean = [0.5, 0.5], cov = [[5, 3], [4, 5]], size = 50000, name = "img"):
    mx, my = np.random.multivariate_normal(mean, cov, size).T
    rvMN = multivariate_normal(mean, cov) #first array traslate, second is scaling
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    zRVMN = rvMN.pdf(pos)
    zRVMN = augmentation(zRVMN)
    zRVMNnoised = addNoise(zRVMN, params = {'type':'uniform', 'min':-2, 'max':2, 'loc':0, 'scale':3}, mask=np.random.binomial(1,globalmask))
    bounds = funbounds(zRVMN)
    
    #saveImages(x,y, zRVMN, zRVMNnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))
    saveImagesrgba(x,y, zRVMN, zRVMNnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))

# GAUSSIAN (WORKS)
def gaussian2D(locx = 0, scalex = 0.45, locy = 0, scaley = 0.45, resolution = 512, name = "img"):
    x = np.linspace(-norm.ppf(0.99), norm.ppf(0.99), resolution)
    y = np.linspace(-norm.ppf(0.99), norm.ppf(0.99), resolution)
    xx, yy = np.meshgrid(x, y, sparse=True)
    zRVG = norm.pdf(xx, loc=-locx, scale=scalex) * norm.pdf(yy, loc=locy, scale=scaley)
    zRVG = augmentation(zRVG)
    #bounds = np.linspace(0, np.max(zRVG), 8)
    bounds = funbounds(zRVG)
    #print(zRVG.shape, np.min(zRVG), np.max(zRVG))
    zRVGnoised = addNoise(zRVG, params = {'type':'uniform', 'min':-2, 'max':2, 'loc':0, 'scale':3}, mask=np.random.binomial(1,globalmask))
    
    #saveImages(x,y, zRVG, zRVGnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))
    saveImagesrgba(x,y, zRVG, zRVGnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))

# POISSON (WORKS)
def poisson2D(a=5, locx = -1, scalex = 0.25, locy = -1, scaley = 0.25, resolution = 512, name = "img"):
    x = np.linspace(-norm.ppf(0.99,a), norm.ppf(0.99,a), resolution)
    y = np.linspace(-norm.ppf(0.99,a), norm.ppf(0.99,a), resolution)
    xx, yy = np.meshgrid(x, y, sparse=True)
    zRVP = gamma.pdf(xx, a,  loc=locx, scale=scalex) * gamma.pdf(yy, a,  loc=locy, scale=scaley)
    zRVP = augmentation(zRVP)
    
    bounds = funbounds(zRVP)
    zRVPnoised = addNoise(zRVP, params = {'type':'uniform', 'min':-0.5, 'max':0.5, 'loc':0, 'scale':3}, mask=np.random.binomial(1,globalmask))
    
    #saveImages(x,y, zRVP, zRVPnoised,  resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))
    saveImagesrgba(x,y, zRVP, zRVPnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))

#SKEW NORM (Y)
def skewNorm2D(skewness=5, locx=-0.5, scalex=.9, locy=-0.5, scaley=.9, resolution=512, name = "img"):
    x = np.linspace(-skewnorm.ppf(0.99, skewness), skewnorm.ppf(0.99, skewness), resolution)
    y = np.linspace(-skewnorm.ppf(0.99, skewness), skewnorm.ppf(0.99, skewness), resolution)
    xx, yy = np.meshgrid(x, y, sparse=True)
    zRVS = skewnorm.pdf(xx, skewness, loc=locx, scale=scalex) * skewnorm.pdf(yy,skewness, loc=locy, scale=scaley)
    zRVS = augmentation(zRVS)
    
    bounds = funbounds(zRVS)
    zRVSnoised = addNoise(zRVS, params = {'type':'uniform', 'min':-0.5, 'max':0.5, 'loc':0, 'scale':3}, mask=np.random.binomial(1,globalmask))
    
    #aveImages(x,y, zRVS, zRVSnoised,  resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))
    saveImagesrgba(x,y, zRVS, zRVSnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))

def hybrid(d1,d2, skewness=5, a = 5, locx=-0.5, scalex=0.9, locy=-0.5, scaley=0.9, resolution=512, name = "img"):
    dist1 = None
    if(d1==1):
        dist1 = norm
        x = np.linspace(-dist1.ppf(0.99), dist1.ppf(0.99), resolution)
    elif(d1==2):
        dist1 = gamma
        x = np.linspace(-dist1.ppf(0.99,a), dist1.ppf(0.99,a), resolution)
    elif(d1==3):
        dist1 = skewnorm
        x = np.linspace(-dist1.ppf(0.99, skewness), dist1.ppf(0.99, skewness), resolution)
    dist2 = None
    if(d2==1):
        dist2 = norm
        y = np.linspace(-dist2.ppf(0.99), dist2.ppf(0.99), resolution)
    elif(d2==2):
        dist2 = gamma
        y = np.linspace(-dist2.ppf(0.99,a), dist2.ppf(0.99,a), resolution)
    elif(d2==3):
        dist2 = skewnorm
        y = np.linspace(-dist2.ppf(0.99, skewness), dist2.ppf(0.99, skewness), resolution)
    
    xx, yy = np.meshgrid(x, y, sparse=True)
    
    if(d1==1):
        pdfd1 = dist1.pdf(xx, loc=locx, scale=scalex)
    elif(d1==2):
        pdfd1 = dist1.pdf(xx, a, loc=locx, scale=scalex)
    elif(d1==3):
        pdfd1 = dist1.pdf(xx, skewness, loc=locx, scale=scalex)
    if(d2==1):
        pdfd2 = dist2.pdf(yy, loc=locx, scale=scalex)
    elif(d2==2):
        pdfd2 = dist2.pdf(yy, a, loc=locx, scale=scalex)
    elif(d2==3):
        pdfd2 = dist2.pdf(yy, skewness, loc=locx, scale=scalex)
    
    zRVH =  pdfd1 * pdfd2
    zRVH = augmentation(zRVH)
    
    bounds = funbounds(zRVH)
    
    zRVHnoised = addNoise(zRVH, params = {'type':'uniform', 'min':-0.5, 'max':0.5, 'loc':0, 'scale':3}, mask=np.random.binomial(1,0.5))
    
    #saveImages(x,y, zRVH, zRVHnoised,  resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))
    saveImagesrgba(x,y, zRVH, zRVHnoised, resolution, bounds, os.path.join(file_path_targets,name), os.path.join(file_path_distributions,name))    

def funbounds(val):
    counters = 50000.0
    mi, ma = np.min(val), np.max(val)
    val = (val * counters).astype(int) # this emulates the number of samples
    val = val[val!=0.0]
    #cond25 = np.logical_and(val != 0.0,val <= np.percentile(val[val != 0.0], 25))
    #cond50 = np.logical_and(val != 0.0,val <= np.percentile(val[val != 0.0], 50))
    #cond75 = np.logical_and(val != 0.0,val <= np.percentile(val[val != 0.0], 75))
    #cond95 = np.logical_and(val != 0.0,val <= np.percentile(val[val != 0.0], 95))
    bounds = [np.percentile(val, 10), np.percentile(val, 20), np.percentile(val, 30), np.percentile(val, 40), 
              np.percentile(val, 50), np.percentile(val, 60), np.percentile(val, 70), np.percentile(val, 80),np.percentile(val, 90), np.percentile(val, 100)]
    bounds = [i/counters for i in bounds]    
    print(bounds)
    #print(np.max(val), len(val[val != 0.0]), len(val[cond]), bounds)
    
    return bounds

def create(rootFolder, datasetSize):

    global file_path_targets, file_path_distributions

    file_path_distributions = os.path.join(rootFolder,"distributions")
    Path(file_path_distributions).mkdir(parents=True, exist_ok=True)
    file_path_targets = os.path.join(rootFolder,"targets")
    Path(file_path_targets).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(rootFolder, "filelist.txt") , "w") as f:
        for i in range(datasetSize):
            #random to select what type of distribution
            distributionSelection = random.randint(1, 4)
            name = str(i).zfill(8) + ".png"
            if(distributionSelection == 0):#multivariateNormal # contains bug!!!!
                mean = np.random.uniform(-0.5, 0.5,2) 
                A = np.random.uniform(0, 5, [2,2])
                cov = np.dot(A, A.transpose())/5
                size = random.randint(40000, 80000)
                print(mean,cov,size)
                multivariateNormal(mean=mean, cov = cov, size = size, name=name)
                f.write(name + " " + str(distributionSelection) + " " + str(mean) + " " + str(cov) + " " + str(size) +"\n" )
            elif(distributionSelection == 1): #gaussian2D 
                #loc = np.random.uniform(-0.2, 0.2, 2)
                loc = [0,0]
                scale = np.random.uniform(0.3, 1, 2)
                gaussian2D(locx = loc[0], scalex = scale[0], locy = loc[1], scaley = scale[1], resolution = resolution, name=name)
                f.write(name + " " + str(distributionSelection) + " " + str(loc[0]) + " " + str(scale[0]) + " " + str(loc[1])+ " " + str(scale[1]) +"\n" )
            elif(distributionSelection == 2): #poisson2D
                a = random.randint(2, 5)
                #loc = np.random.uniform(-2, 2, 2) 
                #scale = np.random.uniform(0, 3, 2)
                loc = [-2, -2] 
                #scale = [0.3918627496021445, 0.5225246990500866]
                scale = np.random.uniform(0.5, 1, 2)
                poisson2D(a=a, locx = loc[0], scalex = scale[0], locy = loc[1], scaley = scale[1], resolution = resolution, name=name)
                f.write(name + " " + str(distributionSelection)+ " " + str(a) + " " + str(loc[0]) + " " + str(scale[0]) + " " + str(loc[1])+ " " + str(scale[1]) +"\n" )
            elif(distributionSelection == 3): #skewNorm2D
                skewness = random.randint(2, 5)
                loc = np.random.uniform(0, 0, 2) 
                scale = np.random.uniform(0.5, 1, 2)
                skewNorm2D(skewness=skewness, locx = loc[0], scalex = scale[0], locy = loc[1], scaley = scale[1], resolution=resolution, name=name)
                f.write(name + " " + str(distributionSelection)+ " " + str(skewness) + " " + str(loc[0]) + " " + str(scale[0]) + " " + str(loc[1])+ " " + str(scale[1]) +"\n" )
            elif(distributionSelection == 4): #hybrid2D
                a = random.randint(2, 5)
                skewness = random.randint(2, 5)
                loc = np.random.uniform(0, 0, 2) 
                scale = np.random.uniform(0.5, 1, 2)
                distributionSelection1 = random.randint(1, 3)
                distributionSelection2 = random.randint(1, 3)
                hybrid(distributionSelection1,distributionSelection2,skewness=skewness, a=a, locx = loc[0], scalex = scale[0], locy = loc[1], scaley = scale[1], resolution=resolution, name=name)
                f.write(name + " " + str(distributionSelection) + " " + str(distributionSelection1) + " " + str(distributionSelection2) + " " + str(skewness) + " " + str(a) + " " + str(loc[0]) + " " + str(scale[0]) + " " + str(loc[1])+ " " + str(scale[1]) +"\n" )
            else:
                print(distributionSelection)
                pass #hybrid?
            #hybrid ???

def main(args):

    global resolution

    path = args.path 
    resolution = args.resolution 

    training = os.path.join(path,"dataset_training")
    validation = os.path.join(path,"dataset_validation")
    test = os.path.join(path,"dataset_test")

    datasetSize = 12000
    #create dataset folders
    rootFolder, datasetSize = (training,int(datasetSize*0.8)) 
    create(rootFolder, datasetSize)

    rootFolder, datasetSize = (validation,int(datasetSize*0.2))  #0 for validation, 1 for training
    create(rootFolder, datasetSize)

    #uncomment below for test dataset
    rootFolder, datasetSize = (test,int(datasetSize*0.3))
    create(rootFolder, datasetSize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()
    main(args)
