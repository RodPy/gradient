# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:17:48 2022

@author: to_reilly
"""

import numpy as np
import matplotlib.pyplot as plt

inputFile           = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0.txt' 
outputFile          = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App.txt'

#######################################

appendStart         = True
appendEnd           = True

zOffset_start       = -2   #offset along the bore in mm
shift_start         = 7.43  #offset from the center line in mm, along the direction of curvature 
curveShape_start    = 1     #diagonal if 1, flattened off if 2
numPoints_start     = 72    #number of points along curve

zOffset_end         = 9.5    #offset along the bore in mm
shift_end           = 12.6    #offset from the center line in mm, along the direction of curvature 
curveShape_end      = 1     #diagonal if 1, flattened off if 2
numPoints_end       = 72    #number of points along curve

centerAngle         = np.pi/2


####################################### flatten the curve

rawCurve        = np.genfromtxt(inputFile, delimiter = ',')

coilRadius      = np.mean(np.sqrt(np.square(rawCurve[:,0])+np.square(rawCurve[:,1])))
initialPoint    = (coilRadius*np.cos(centerAngle), coilRadius*np.sin(centerAngle))
delta           = np.sqrt(np.sum(np.square(rawCurve[:,:2]- initialPoint), axis = -1))*np.sign(rawCurve[:,1]- initialPoint[1])/2

theta           = 2*np.arcsin(delta/np.mean(coilRadius))
flatCurve       = np.stack((theta, rawCurve[:,2]), axis = -1)

plt.figure()
plt.plot(theta, rawCurve[:,2])

####################################### append the curve

# if np.abs(rawCurve[0,1]) < np.abs(rawCurve[-1,1]):
#     print("Curve starts at the center of the bore, the inside is therefor on the end")
# else:
#     print("Curve starts at the edge of the bore, the inside is therefor on the start, flipping curve to have inside at end")
#     insideOut = np.flip(flatCurve, axis = 0)

if appendStart:
    curveDirection  = np.sign(flatCurve[0,0] - flatCurve[1,0])
    zDirection      = np.sign(flatCurve[0,1] - flatCurve[-1,1])
    
    angles          = np.linspace(-np.pi,(curveShape_start-1)*np.pi, numPoints_start +1 )[1:] #chop off the first point to avoid overlapping existing curve
        
    zShape          = np.tanh(angles)+1     #define the curve behaviour
    zShape          /= np.max(zShape)   #normalise to account for curveShape
    zPoints         = flatCurve[0,1] + zDirection*zOffset_start*zShape
    
    span_start      = shift_start/coilRadius #convert from mm to radians, is an approximation which is valid if shiftStart << coilRadius
    
    angularPoints   = np.linspace(0,1,numPoints_start +1)[1:] 
    curve_angles    = flatCurve[0,0] + curveDirection*span_start*angularPoints
    
    plt.plot(curve_angles, zPoints, color = "C1")
    
    appendage       = np.stack((curve_angles, zPoints), axis = -1)
    appendage       = np.flip(appendage, axis = 0)
    flatCurve       = np.vstack((appendage,flatCurve))

       
if appendEnd:
    curveDirection = np.sign(flatCurve[-1,0] - flatCurve[-2,0])
    zDirection      = np.sign(flatCurve[-1,1] - flatCurve[0,1])
    
    angles          = np.linspace(-np.pi,(curveShape_end-1)*np.pi, numPoints_end +1 )[1:] #chop off the first point to avoid overlapping existing curve
        
    zShape          = np.tanh(angles)+1     #define the curve behaviour
    zShape          /= np.max(zShape)   #normalise to account for curveShape
    zPoints         = flatCurve[-1,1] + zDirection*zOffset_end*zShape
    
    span_end      = shift_end/coilRadius #convert from mm to radians, is an approximation which is valid if shiftStart << coilRadius
    
    angularPoints   = np.linspace(0,1,numPoints_end +1)[1:] 
    curve_angles    = flatCurve[-1,0] + curveDirection*span_end*angularPoints
    
    plt.plot(curve_angles, zPoints, color = "C1")    

    appendage       = np.stack((curve_angles, zPoints), axis = -1)
    flatCurve       = np.vstack((flatCurve, appendage))
    
    
outputCurve3D = np.stack((coilRadius*np.cos(flatCurve[:,0]-centerAngle), np.round(np.cos(2*centerAngle))*coilRadius*np.sin(flatCurve[:,0]-centerAngle),flatCurve[:,1]), axis = 1)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot3D(outputCurve3D[:,0], outputCurve3D[:,1], outputCurve3D[:,2], color = 'C1')




np.savetxt(outputFile, outputCurve3D, delimiter=",", fmt='%f')

