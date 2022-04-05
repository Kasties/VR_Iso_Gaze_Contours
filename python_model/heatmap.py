import pandas as pd
import numpy as np 
import matplotlib
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
# importing cv2 
import cv2
import scipy.optimize as opt

def surface3D(x,y):

    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    resolution = 256
        
    def toGrid(x,y, resolution):
        gridx = np.linspace(-1, 1, resolution+1)
        gridy = np.linspace(-1, 1, resolution+1)
        X, Y = np.meshgrid(gridx[:-1], gridy[:-1])
        grid, _, _ = np.histogram2d(x, y, bins=[gridx, gridy])
        grid = gaussian_filter(grid, sigma=1)
        return grid/np.max(grid), X, Y, grid
        
    grid, gridx, gridy, sumgrid  = toGrid(x,y, resolution)
    
    mingrid = np.min(sumgrid)
    maxgrid = np.max(sumgrid)
    total = np.sum(sumgrid)
    print(mingrid, maxgrid, total)
    bounds = [np.percentile(grid[grid != 0.0], 40), np.percentile(grid[grid != 0.0], 50),np.percentile(grid[grid != 0.0], 60),np.percentile(grid[grid != 0.0],75)]
    print(bounds)
    

    #uncomment to see surface
    #surf = ax.plot_surface(gridx, gridy, grid, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')
    
    
    blurred = gaussian_filter(grid, sigma=9)
    
    percentage = 1 - (0.7) 
    epsilon = 0.01
    
    current = 0
    value = np.mean(sumgrid)
    i = mingrid
    j = maxgrid
    while True:
        partial = sumgrid[sumgrid <= value]
        current = np.sum(partial)/total
        
        #input([current, value, np.sum(partial)])
        if (current > percentage + epsilon):
            j = value
            value = (i + value)*0.5
            
            
            print("to dec:", value)
        elif (current < percentage - epsilon):
            i = value
            value = (j + value)*0.5
            
            print("to inc:", value)
        else:
            break
    
    print(value/maxgrid)
    
    cs = ax.contour(gridx, gridy, blurred, [value/maxgrid], lw=3, colors="k", linestyles="solid")
    p = cs.allsegs
    print(p,  dir(p))
    plt.show()

def generateheatmapOld(x,y,axe, cf, resolution = 512, min = -1.2,max= 1.2 ):
    #@@todo cf not used, needs to be removed




    #colormaps
    cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
    grid, gridx, gridy = coordinatestoGrid(x,y, resolution, min , max) 
    blurred = gaussian_filter(grid, sigma=10)
    blurred = blurred*blurred/1.5
    mingrid = np.min(blurred)
    maxgrid = np.max(blurred)
    total = np.sum(blurred)

    bounds = np.linspace(0, 1, 10)
    print(bounds)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N) #discrete 
    blurred = blurred/maxgrid
    

    percentage = 1 - (0.99) 
    epsilon = 0.1
    
    current = 0
    value = np.mean(blurred)
    i = mingrid
    j = maxgrid
    while True:
        partial = blurred[blurred <= value]
        current = np.sum(partial)/total
        
        #input([current, value, np.sum(partial)])
        if (current > percentage + epsilon):
            j = value
            value = (i + value)*0.5
            
            
            print("to dec:", value)
        elif (current < percentage - epsilon):
            i = value
            value = (j + value)*0.5
            
            print("to inc:", value)
        else:
            break
    
    
 
    #axe.contour(gridx, gridy, blurred, [value/maxgrid], lw=1, colors="k", linestyles="solid")
    #plt.clabel(cs,fmt='%2.1f', inline=True, fontsize=6)
    #p = cs.allsegs

    
    axe.set_xlim([min,max])
    axe.set_ylim([min,max])
    c = axe.pcolormesh(gridx, gridy, grid, cmap=cmap, norm=norm, shading='gouraud') # discrete 
    #axe.contourf(gridx, gridy, blurred,[0,.005,.01, 0.02, .03, .04, .05, .07, .1], cmap=cmap)
    axe.contour(gridx, gridy, blurred,[.07], colors='white',linewidths=1)
    #axe.imshow(grid, extent=[0, 5, 0, 5], origin='lower', cmap=cmap)
    
    #axe.axvline(0, linewidth=0.2,color='white')
    #axe.axhline(0, linewidth=.2,color='white')
    

    return c

def generateheatmap(x,y,axe, cf, resolution = 512, min = -1.2,max= 1.2 ):

    #colormaps
    # cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)

    # rgb = generateheatmapFromDistributionrgba(x,y,None, resolution = resolution, min=min, max=max, isGray=False)
    # axe.imshow(rgb,cmap=cmap)

    
    
    # #colormaps
    cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
    grid, gridx, gridy = coordinatestoGrid(x,y, resolution, min , max) 
    maxgrid = np.max(grid)

    bounds = funbounds(grid/maxgrid)
    print(bounds)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N) #discrete 
 
    axe.set_xlim([min,max])
    axe.set_ylim([min,max])
    axe.pcolormesh(gridx, gridy, grid, cmap=cmap, norm=norm, shading='gouraud',zorder=0) # discrete 
    

    return axe
        
def generateheatmapFromDistribution(x,y,z,axe, cf, resolution = 512, min = -1.2,max= 1.2, bounds = np.linspace(0, 0.1, 10) ):
    #@@todo cf not used, needs to be removed



    fig = plt.figure()
    plt.scatter(x,y ,s=0.1)
    plt.show()


    #colormaps
    cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)
    grid, gridx, gridy = z,x,y
    blurred = gaussian_filter(grid, sigma=0)
    mingrid = np.min(blurred)
    maxgrid = np.max(blurred)
    total = np.sum(blurred)

    
    print(bounds)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N) #discrete 
    blurred = blurred/maxgrid
    

    percentage = 1 - (0.99) 
    epsilon = 0.1
    
    current = 0
    value = np.mean(blurred)
    i = mingrid
    j = maxgrid
    maxIter = 5
    c = 0
    while True:
        partial = blurred[blurred <= value]
        current = np.sum(partial)/total
        
        #input([current, value, np.sum(partial)])
        if (current > percentage + epsilon):
            j = value
            value = (i + value)*0.5
            
            
            print("to dec:", value)
        elif (current < percentage - epsilon):
            i = value
            value = (j + value)*0.5
            
            print("to inc:", value)
        else:
            break
        c+=1
        if(c > maxIter): break
    
 
    #cs = axe.contour(gridx, gridy, blurred, [value/maxgrid], lw=1, colors="k", linestyles="solid")
    #plt.clabel(cs,fmt='%2.1f', inline=True, fontsize=6)
    #p = cs.allsegs

    
    axe.set_xlim([min,max])
    axe.set_ylim([min,max])
    axe.pcolormesh(gridx, gridy, grid, cmap=cmap, norm=norm, shading='gouraud') # discrete 
    #axe.contourf(gridx, gridy, blurred,[0,.005,.01, 0.02, .03, .04, .05, .07, .1], cmap=cmap)
    #axe.contour(gridx, gridy, blurred,[0,.005,.01, 0.02, .03, .04, .05, .06, .07], colors='black',linewidths=.3)
    #axe.imshow(grid, extent=[0, 5, 0, 5], origin='lower', cmap=cmap)
    
    #axe.axvline(0, linewidth=0.2,color='white')
    #axe.axhline(0, linewidth=.2,color='white')
  
    return axe
   
def generateheatmapFromDistributionrgba(x,y,z, resolution = 512, min = -1.2,max= 1.2, isGray = False ):
    


    # fig = plt.figure()
    # plt.scatter(x,y ,s=0.1)
    # plt.xlim([-50,50])
    # plt.ylim([-50,50])
    # plt.show()

    #colormaps
    if(isGray == False):
        cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    elif(isGray == True):
        cmap = matplotlib.cm.gray #plt.cm.jet  # define the colormap
        
        
    if(z is None):
        grid, _, _ = coordinatestoGrid(x,y, resolution, min , max) 
        # bounds = np.linspace(0, 0.5, 20)
        # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N) #discrete 
        # grid = norm(grid)
        #maxgrid = np.max(grid)
        #grid = grid/maxgrid
        powernorm = matplotlib.colors.PowerNorm(0.9)
        grid = powernorm(grid)
        grid = grid/np.max(grid)
    else: 
        grid, _, _ = z,x,y
        maxgrid = np.max(grid)
        grid = grid/maxgrid

    rgba = cmap(grid)
    rgb = np.array(rgba[:,:,0:3]*255, dtype=np.uint8)
    

    if(isGray == False):
        rgb=255-rgb
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif(isGray == True):
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pass

        
    # fig = plt.figure()
    # plt.imshow(rgb)
    # plt.show()

    return np.flipud(rgb)
   
def funbounds(val):
    
    counters = 50000.0

    val = (val * counters).astype(int) # this emulates the number of samples
    val = val[val!=0.0]
   
    bounds = [np.percentile(val, 10), np.percentile(val, 20), np.percentile(val, 30), np.percentile(val, 40), 
              np.percentile(val, 50), np.percentile(val, 60), np.percentile(val, 70), np.percentile(val, 80),np.percentile(val, 90), np.percentile(val, 100)]
    
    bounds = [i/counters for i in bounds]    
 
    return bounds   
   
def coordinatestoGrid(x,y, resolution,min,max):
        gridx = np.linspace(min, max, resolution+1)
        gridy = np.linspace(min, max, resolution+1)
        grid, _, _ = np.histogram2d(x, y, bins=[gridx, gridy])
        #grid = gaussian_filter(grid, sigma=0)
        grid = grid.transpose()
        #grid = np.flipud(grid)
        return grid/np.max(grid), gridx[:-1], gridy[:-1]

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def areaContour(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

def fitting2Dgaussian(x,y,z, resolution = 512, min = -1.2,max= 1.2,  isGray = False):


    #colormaps
    if(isGray == False):
        cmap = matplotlib.cm.coolwarm #plt.cm.jet  # define the colormap
    elif(isGray == True):
        cmap = matplotlib.cm.gray #plt.cm.jet  # define the colormap
        
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 'Custom cmap', cmaplist, cmap.N)

    grid, gridx, gridy = coordinatestoGrid(x,y, resolution, min , max) 
    maxgrid = np.max(grid)
    grid = grid/maxgrid

    # Create x and y indices
    x = np.linspace(min, max, resolution)
    y = np.linspace(min, max, resolution)
    xy = np.meshgrid(x, y)

    # add some noise to the data and try to fit the data generated beforehand
    initial_guess = (3,0,0,20,40,0,50)

    popt, pcov = opt.curve_fit(twoD_Gaussian, xy, grid.ravel(), p0=initial_guess, maxfev=1000000)

    data_fitted = twoD_Gaussian(xy, *popt)

    grid = np.reshape(data_fitted,(resolution,resolution))
    
    mingrid = grid.min()
    grid = grid-mingrid

    maxgrid = np.max(grid)
    grid = grid/maxgrid

    rgba = cmap(grid)
    rgb = np.array(rgba[:,:,0:3]*255, dtype=np.uint8)
    
    if(isGray == False):
        rgb=255-rgb
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif(isGray == True):
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pass

    rgb = np.flipud(rgb)

    #####
    # fig, ax = plt.subplots(1, 2)
    # ax[1].imshow(rgb)
    # ax[0].pcolormesh(gridx, gridy, grid, cmap=cmap,  shading='gouraud')
    # ax[0].set_xlim([min,max])
    # ax[0].set_ylim([min,max])
    # ax[0].contour(x, y, data_fitted.reshape(resolution, resolution), 10, colors='w')
    # plt.show()
    #######

    return rgb


#define model function and pass independant variables x and y as a list
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y=xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


