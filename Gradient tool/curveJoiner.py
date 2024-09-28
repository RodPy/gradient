# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:17:48 2022

@author: to_reilly
"""

import numpy as np
import matplotlib.pyplot as plt

inputCurve_1_File   = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients/wire paths/Xgrad/X_rot+00979_1.txt'
inputCurve_2_File   = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients/wire paths/Xgrad/X_rot+00979_2.txt'

outputFile          = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients/wire paths/Xgrad/X_rot+00979_1-2_join.txt'


#################### Input parameters

numPoints           = 400 #number of points along curve
initialAngle        = np.pi/2 # set to np.pi/2 if there is weird wrapping going on
# initialAngle        = 0 # set to np.pi/2 if there is weird wrapping going on

#################### Flatten curves

rawCurve1       = np.genfromtxt(inputCurve_1_File, delimiter = ',')

coilRadius      = np.mean(np.sqrt(np.square(rawCurve1[:,0])+np.square(rawCurve1[:,1])))
initialPoint    = (coilRadius*np.cos(initialAngle), coilRadius*np.sin(initialAngle))
delta           = np.sqrt(np.sum(np.square(rawCurve1[:,:2]- initialPoint), axis = -1))*np.sign(rawCurve1[:,1]- initialPoint[1])/2

theta           = 2*np.arcsin(delta/np.mean(coilRadius))
flatCurve1      = np.stack((theta, rawCurve1[:,2]), axis = -1)

rawCurve2       = np.genfromtxt(inputCurve_2_File, delimiter = ',')

coilRadius      = np.mean(np.sqrt(np.square(rawCurve2[:,0])+np.square(rawCurve2[:,1])))
initialPoint    = (coilRadius*np.cos(initialAngle), coilRadius*np.sin(initialAngle))
delta           = np.sqrt(np.sum(np.square(rawCurve2[:,:2]- initialPoint), axis = -1))*np.sign(rawCurve2[:,1]- initialPoint[1])/2

theta           = 2*np.arcsin(delta/np.mean(coilRadius))
flatCurve2      = np.stack((theta, rawCurve2[:,2]), axis = -1)


################# Checek orientations

#flip so the ends of one lines up with the start of the other
if np.abs(flatCurve1[0,1]) > np.abs(flatCurve1[-1,1]):
    if np.abs(flatCurve2[0,1]) > np.abs(flatCurve2[-1,1]):
        flatCurve2 = np.flip(flatCurve2, axis = 0)

startJoinerFlat1 = 0
startJoinerFlat2 = 0

if np.abs(flatCurve1[0,1]) > np.abs(flatCurve1[-1,1]):
    startJoinerFlat1 = np.size(flatCurve1, axis = 0)-1
if np.abs(flatCurve2[0,1]) > np.abs(flatCurve2[-1,1]):
    startJoinerFlat2 = np.size(flatCurve2, axis = 0)-1

#define the center of the connector
centerAngle     = (flatCurve1[startJoinerFlat1,0] + flatCurve2[startJoinerFlat2,0])/2
angles          = np.linspace(-np.pi, np.pi, numPoints +2)[1:-1] #chop off the first and last point to avoid overlapping with original curves

spanAngle       = np.min((np.abs(flatCurve1[-1,:]-centerAngle),np.abs(flatCurve2[0,:]-centerAngle)))
angleToPos      = centerAngle  + spanAngle*angles/np.pi

changeZ         = flatCurve2[0,1] - flatCurve1[-1,1]
zPoints         = flatCurve1[-1,1] + changeZ*(np.tanh(angles)+1)/2
  
connector       = np.stack((angleToPos, zPoints), axis = -1)

plt.figure()
plt.plot(flatCurve1[:,0],flatCurve1[:,1], color = 'C0')
plt.plot(flatCurve2[:,0],flatCurve2[:,1], color = 'C0')
plt.plot(connector[:,0], connector[:,1], color = 'C1')

################ Output curve

outputCurve = np.vstack((flatCurve1, connector))
outputCurve = np.vstack((outputCurve, flatCurve2))

outputCurve3D = np.stack((coilRadius*np.cos(outputCurve[:,0]-initialAngle), np.round(np.cos(2*initialAngle))*coilRadius*np.sin(outputCurve[:,0]-initialAngle),outputCurve[:,1]), axis = 1)

np.savetxt(outputFile, outputCurve3D, delimiter=",", fmt='%f')


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot3D(outputCurve3D[:,0], outputCurve3D[:,1], outputCurve3D[:,2], color = 'C1')


