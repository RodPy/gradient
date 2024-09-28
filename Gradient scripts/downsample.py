# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:40:14 2020

@author: Tom
"""

import numpy as np


filename = r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App_osR+1_17.txt'

contourData3D = np.genfromtxt(filename, delimiter = ',')

#Reduce the number of points to increase speed of curve manipulations in solidworks
underSampleFactor = int(10)

np.savetxt(r'P:/WT shared/Halbach V3/Active designs/Version 1.0b - Dev/_Workshop Wouter/Mk_III_Gradients_CNC_Heff/X_16_8_480_26_0_App_osR+1_17_Dwn_10.txt', contourData3D[::underSampleFactor,:], delimiter=",", fmt='%f')


