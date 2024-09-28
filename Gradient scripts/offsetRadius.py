# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:40:14 2020

@author: Tom
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


filename = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App.txt'

contourData3D   = np.genfromtxt(filename, delimiter = ',')
offset          = 1.17

#calculate the coil radius
rad = np.sqrt(np.square(contourData3D[:,0])+np.square(contourData3D[:,1]))

initialPoint = (np.mean(rad), 0)

delta = np.sqrt(np.sum(np.square(contourData3D[:,:2]- initialPoint), axis = -1))*np.sign(contourData3D[:,1]- initialPoint[1])/2
theta = 2*np.arcsin(delta/np.mean(rad))

xPos = np.mean(rad + offset)*np.cos(theta)
yPos = np.mean(rad + offset)*np.sin(theta)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot3D(contourData3D[:,0], contourData3D[:,1], contourData3D[:,2])


temp = np.stack((xPos, yPos, contourData3D[:,2]), axis = -1)
np.savetxt(r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App_osR+1_17.txt', temp, delimiter=",", fmt='%f')

print("Radius: %.2f mm"%np.mean(rad))
print("Min Z: %.2f mm, max Z: %.2f mm"%(np.min(contourData3D[:,2]), np.max(contourData3D[:,2])))
# print("Center point: %.2f x, %.2f Y, %2f. Z"%(np.mean(horizontalPos),0, np.mean(contourData3D[:,2])))
