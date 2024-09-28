#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 08:26:34 2021

@author: tom
"""

import numpy as np

# filename = r'/Users/tom/Dropbox/Low field data/Lukas gradients/Y gradient -0.txt'
filename = r'P:\WT shared\Halbach V3\Active designs\Version 1.0b - Dev\_Workshop Wouter\Mk_III_Gradients\wire paths\flat_157_11_10_18_11_0_with_offset.txt'

rawPoints = np.genfromtxt(filename, delimiter = ',')

#Reduce the number of points to increase speed of curve manipulations in solidworks
#underSampleFactor = 4

lengthCutoff = 0.1

#rawPoints = rawPoints[::underSampleFactor,:]

vectors = rawPoints[:-1,:] - rawPoints[1:,:]

magnitude = np.sqrt(np.sum(np.square(vectors), axis =-1))

print("Number of points removed: %i"%(np.sum(magnitude < lengthCutoff)))

outputFilename = r'P:\WT shared\Halbach V3\Active designs\Version 1.0b - Dev\_Workshop Wouter\Mk_III_Gradients\wire paths\clean_flat_157_11_10_18_11_0_with_offset.txt'

with open(outputFilename, 'w') as file:
    file.write("%f, %f, %f\n"%(rawPoints[0,0], rawPoints[0,1], rawPoints[0,2]))
    for idx, magVec in enumerate(magnitude):
        if magVec > lengthCutoff:
            file.write("%f, %f, %f\n"%(rawPoints[idx+1,0], rawPoints[idx+1,1], rawPoints[idx+1,2]))

