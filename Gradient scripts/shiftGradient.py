# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:40:14 2020

@author: Tom
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


filename            = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App_osR+1_17_Dwn_10.txt'
outputFilename      = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App_osR+1_17_Dwn_10_shftZ3_3.txt'
contourData         = np.genfromtxt(filename, delimiter = ',')


shiftAxis           = 2   #axis to shift, 0 = x, 1, = y, 2 = z
shiftDistance       = 3.3   #shift distance in mm

#generate the shifting vector
shiftVec            = np.zeros(np.size(contourData, axis = -1))
shiftVec[shiftAxis] = shiftDistance

shiftedContour      = contourData - shiftVec


#plot the contour, first check if it's 2D
zeroColumns = np.where(~contourData.any(axis=0))[0]

if np.size(zeroColumns) == 0:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(contourData[:,0], contourData[:,1], contourData[:,2], label = "Initial")
    ax.plot3D(shiftedContour[:,0], shiftedContour[:,1], shiftedContour[:,2], label = "shifted")
    ax.legend()
else:
    flatContour = np.delete(contourData,zeroColumns[0],1)
    flatShift   = np.delete(shiftedContour,zeroColumns[0],1)
    
    plt.figure()
    plt.plot(flatContour[:,0], flatContour[:,1], label = "Initial")
    plt.plot(flatShift[:,0], flatShift[:,1], label = "Shifted")
    plt.legend()

np.savetxt(outputFilename, shiftedContour, delimiter=",", fmt='%f')

print("Min Z: %.2f mm, max Z: %.2f mm"%(np.min(shiftedContour[:,2]), np.max(shiftedContour[:,2])))

