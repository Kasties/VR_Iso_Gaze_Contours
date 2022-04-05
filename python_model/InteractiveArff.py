import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from heatmap import  generateheatmap, generateheatmapFromDistributionrgba,generateheatmapOld
import cv2
from sklearn.metrics import mean_squared_error

#gui
from matplotlib.widgets import Slider, Button, RadioButtons

VelocityAngle = None
VelocityAngleRange = None
fig2 = None
axes =None
saveIMG = False
p2 = None

df = pd.read_pickle("dataset9/360_VR_gaze.pkl")

def update(val):
   
    # recalculate
    conditions = recalculateConditions()

    displayConditions(df,conditions)
    fig2.canvas.draw_idle()

def saveImg(): 
    global saveIMG
    saveIMG=True 
    
def displayConditions(df,conditions):
    global fig2,axes,VelocityAngle,VelocityAngleRange,saveIMG,p2

    r ,c= getRowandCol(conditions)
    if(fig2 is None):
        fig2,axes = plt.subplots(r,c)
        fig2.set_size_inches(6,7)
        axcolor = 'lightgoldenrodyellow'

        axampVelocityAngle = plt.axes([0.2, 0.01, 0.65, 0.02], facecolor=axcolor)
        VelocityAngle = Slider(axampVelocityAngle, 'VelocityAngle', -180, 180, valinit=0)

        axampVelocityAngleRange= plt.axes([0.2, 0.08, 0.65, 0.02], facecolor=axcolor)
        VelocityAngleRange = Slider(axampVelocityAngleRange, 'VelocityAngleRange', 0.0, 180, valinit=45)

        VelocityAngleRange.on_changed(update)
        VelocityAngle.on_changed(update)
   
        # axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        # bprev = Button(axprev, 'Previous')
        # bprev.on_clicked(saveImg())

    for i,condition in enumerate(conditions):
        if(p2 is not None ): 
            p2.remove()
            axes.clear()
        axe = getAxes(axes,i,r,c)
        x,y = ConditionalFilter(df,condition)
        generateheatmap(x,y,axe,c, 256, min = -50,max= 50)
        center = [np.mean(x),np.mean(y)]#-1
        p2 = axe.scatter(center[0],center[1],c='green',s=5, zorder=2)
        msex,msey,hmsex,hmsey = evaluate(center,x,y )
        axe.vlines(0,-50,50, colors='white')
        axe.hlines(0,-50,50,colors='white')
        axe.quiver(0,0, math.sin(math.radians(VelocityAngle.val))*20, math.cos(math.radians(VelocityAngle.val))*20)
        axe.set_title("MSE X,Y:"+str(msex)[:5]+"  "+str(msey)[:5]+" Head MSE X,Y:"+str(hmsex)[:5]+"  "+str(hmsey)[:5])
        #axe.scatter(x,y,s=.1,zorder=2)
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        
def evaluate(center,x,y ):

        msex = mean_squared_error([center[0]]*len(x), x)
        msey = mean_squared_error([center[1]]*len(y), y)#-1
        hmsex = mean_squared_error([0]*len(x), x)
        hmsey = mean_squared_error([0]*len(y), y)#-1

        return msex,msey,hmsex,hmsey

def recalculateConditions():

    global fig2,VelocityAngle,VelocityAngleRange,Resolution,AccAngle, AccAngleRange,VelMag,VelMagRange

    print(VelocityAngle.val-VelocityAngleRange.val)
    print(VelocityAngle.val+VelocityAngleRange.val)

    # recalculate
    conditions = determineCondition(df,VelocityAngle.val,VelocityAngleRange.val)
    return conditions

def determineCondition(df,angle,range):
    
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

        condition = [[np.logical_or(slice1,slice2)]]
 
    else:
        topboundary = df["headVelAng"]>angle-range
        bottomboundary = df["headVelAng"]<angle+range
        condition = [[topboundary,bottomboundary]]

    return condition

def getRowandCol(columns):

    if len(columns) ==1:
        r=1
        c=1
    if len(columns) ==2:
        r=1
        c=2
    if len(columns) ==3:
        r=1
        c=3
    elif len(columns) ==4:
        r=2
        c=2
    elif len(columns) ==6:
        r=3
        c=2

    return r,c

def getAxes(axes,i,r,c):

    tot = r*c

    if r==1 and c==1:
        return axes
    elif r==1 :
        return axes[i]
    elif c==2  and math.floor(i/2) ==0 :
        return axes[math.floor(i/2),i]
    elif c==2  and math.floor(i/2) ==1 :
        return axes[math.floor(i/2),i-2]
    elif c==2  and math.floor(i/2) ==2 :
        return axes[math.floor(i/2),i-4]

def ConditionalFilter(df,condition):

    if condition != None: 
        condition = conditionComputing(condition)
        conditionalData = df[condition]
    else:
        conditionalData = df

    # x = conditionalData['GazeXProjected'].values
    # y = conditionalData['GazeYProjected'].values

    x = conditionalData['GazeFoVDegreesX'].values
    y = conditionalData['GazeFoVDegreesY'].values 

    return  x,y

def conditionComputing(array):

    finalcondition = array[0]

    for index, condition in enumerate(array):

        if index>0:
            finalcondition =np.logical_and(finalcondition,condition)

    return finalcondition

conditions = [[df["headVelX"]<0]]

displayConditions(df,conditions)

plt.show()


