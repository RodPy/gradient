# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:30:27 2021

@author: bdevos
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import os
import SumofSinesCalculation_v1_2 as gradCalc

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure
from matplotlib.pyplot import close as closePlots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.pyplot import yscale

class PopUpTextMessage(tk.Toplevel):
    def __init__(self, parent, textmsg):
        super(PopUpTextMessage, self).__init__(parent)

        self.wm_title('!')
        label = ttk.Label(self, text=textmsg)
        label.pack(side='top', fill='x', pady=10, padx=10)

        b1 = ttk.Button(self, text='Ok', command=self.cleanup)
        b1.pack(pady=10, padx=10)

    def cleanup(self):
        self.destroy()

class GDT_GUI(tk.Tk):
    
    def __init__ (self, *args, **kwargs):
        super(GDT_GUI, self).__init__(*args, **kwargs)
        # set name and icon
        # super().iconbitmap(self, default=None)
        super().wm_title('Gradient design tool Sum of Sines')

        inputFrame = ttk.Frame(self,borderwidth=1)
        inputFrame.grid(row=0,column=0,columnspan=1, sticky=tk.W, padx=10, pady=10)
        
        inputDescriptionLabel = ttk.Label(inputFrame, text='Design parameters', font='Helvetica 12 bold')
        inputDescriptionLabel.grid(row=0,column=0, columnspan = 2, sticky=tk.W)
        
        #create variables for storing inputs
        self.modesN             = tk.DoubleVar(inputFrame, value = 8)
        self.modesM             = tk.DoubleVar(inputFrame, value = 4)
        self.coilRadius         = tk.DoubleVar(inputFrame, value = 150)
        self.coilLength         = tk.DoubleVar(inputFrame, value = 400)
        self.numWires           = tk.IntVar(inputFrame, value = 15)
        self.linearityError     = tk.DoubleVar(inputFrame, value = 20)
        self.wireDiameter       = tk.DoubleVar(inputFrame, value = 1.5)
        self.DSV                = tk.DoubleVar(inputFrame, value = 200)
        
        
        directionLabel = ttk.Label(inputFrame, text='Gradient direction')
        directionLabel.grid(row=1,column=0, pady=5,sticky=tk.W)
        self.directionInput = ttk.Combobox(inputFrame, values=["X", "Y", "Z"], width = 7, justify=tk.RIGHT, state = 'readonly')
        self.directionInput.current(0)
        self.directionInput.grid(row=1,column=1, padx = 20)
    
        modesNLable = ttk.Label(inputFrame, text='Number of modes N')
        modesNLable.grid(row=2,column=0, pady=5,sticky=tk.W)
        modesNInput = ttk.Entry(inputFrame, textvariable=self.modesN, width = 10, justify=tk.RIGHT)
        modesNInput.grid(row=2,column=1,padx = 20)
    
        modesMLable = ttk.Label(inputFrame, text='Number of modes M')
        modesMLable.grid(row=3,column=0, pady=5,sticky=tk.W)
        modesMInput = ttk.Entry(inputFrame, textvariable=self.modesM, width = 10, justify=tk.RIGHT)
        modesMInput.grid(row=3,column=1,padx = 20)
    
        coilRadiusLabel = ttk.Label(inputFrame, text='Coil radius (mm)')
        coilRadiusLabel.grid(row=4,column=0, pady=5,sticky=tk.W)
        coilRadiusInput = ttk.Entry(inputFrame, textvariable=self.coilRadius, width = 10, justify=tk.RIGHT)
        coilRadiusInput.grid(row=4,column=1,padx = 20)
        
        coilLengthLabel = ttk.Label(inputFrame, text='Coil length (mm)')
        coilLengthLabel.grid(row=5,column=0, pady=5,sticky=tk.W)
        coilLengthInput = ttk.Entry(inputFrame, textvariable=self.coilLength, width = 10, justify=tk.RIGHT)
        coilLengthInput.grid(row=5,column=1,padx = 20)
        
        numWiresLabel = ttk.Label(inputFrame, text='Turns per quadrant')
        numWiresLabel.grid(row=6,column=0, pady=5,sticky=tk.W)
        numWiresInput = ttk.Entry(inputFrame, textvariable=self.numWires, width = 10, justify=tk.RIGHT)
        numWiresInput.grid(row=6,column=1,padx = 20)
        
        linErrorLabel = ttk.Label(inputFrame, text='Maximum Linearity Error')
        linErrorLabel.grid(row=7,column=0, pady=5,sticky=tk.W)
        linErrorInput = ttk.Entry(inputFrame, textvariable=self.linearityError, width = 10, justify=tk.RIGHT)
        linErrorInput.grid(row=7,column=1,padx = 20)
        
        wireDiameterLabel = ttk.Label(inputFrame, text='Wire diameter (mm)')
        wireDiameterLabel.grid(row=8,column=0, pady=5,sticky=tk.W)
        wireDiameterInput = ttk.Entry(inputFrame, textvariable=self.wireDiameter, width = 10, justify=tk.RIGHT)
        wireDiameterInput.grid(row=8,column=1,padx = 20)

        DSVLabel = ttk.Label(inputFrame, text='Simulation DSV (mm)')
        DSVLabel.grid(row=9,column=0, pady=5,sticky=tk.W)
        DSVInput = ttk.Entry(inputFrame, textvariable=self.DSV, width = 10, justify=tk.RIGHT)
        DSVInput.grid(row=9,column=1,padx = 20)

        calculateWireButton = ttk.Button(inputFrame, text="Calculate wire pattern", command=self.calculateGradientWires)
        calculateWireButton.grid(row=10, column=0, pady=5, sticky=tk.W)
        
        ''' OUTPUT FRAME'''
        outputFrame = ttk.Frame(self)
        outputFrame.grid(row=1,column=0,columnspan=1, sticky=tk.NW, padx=10, pady=20)
        
        outputDescriptionLabel = ttk.Label(outputFrame, text='Simulation output', font='Helvetica 12 bold')
        outputDescriptionLabel.grid(row=0,column=0, columnspan = 1, sticky=tk.W)

        self.gradEfficiencyString   = tk.StringVar(outputFrame, value = "-")
        self.gradErrorString        = tk.StringVar(outputFrame, value = "-")
        self.resistanceString       = tk.StringVar(outputFrame, value = "-")
        self.inductanceString       = tk.StringVar(outputFrame, value = "-")
        self.wireLengthString       = tk.StringVar(outputFrame, value = "-")
        self.zRangeString           = tk.StringVar(outputFrame, value = "-")
        self.spacingString          = tk.StringVar(outputFrame, value = "-")
        self.exportConjoined        = tk.BooleanVar(outputFrame, value = True)
        
        efficiencyTextLabel = ttk.Label(outputFrame, text='Gradient efficiency:')
        efficiencyTextLabel.grid(row=1,column=0, pady=5,sticky=tk.W)
        efficiencyValueLabel = ttk.Label(outputFrame, textvariable=self.gradEfficiencyString, justify = tk.RIGHT)
        efficiencyValueLabel.grid(row=1,column=1, padx=10,sticky=tk.E)
        
        self.linearityTextString = tk.StringVar(outputFrame, value = "Error over %.0f mm DSV:"%(self.DSV.get()))
        linearityTextLabel = ttk.Label(outputFrame, textvariable=self.linearityTextString)
        linearityTextLabel.grid(row=2,column=0, pady=5,sticky=tk.W)
        linearityValueLabel = ttk.Label(outputFrame, textvariable=self.gradErrorString, justify = tk.RIGHT)
        linearityValueLabel.grid(row=2,column=1, padx=10,sticky=tk.E)

        wireLengthTextLabel = ttk.Label(outputFrame, text='Wire length:')
        wireLengthTextLabel.grid(row=3,column=0, pady=5,sticky=tk.W)
        wireLengthValueLabel = ttk.Label(outputFrame, textvariable=self.wireLengthString, justify = tk.RIGHT)
        wireLengthValueLabel.grid(row=3,column=1, padx=10,sticky=tk.E)

        resistanceTextLabel = ttk.Label(outputFrame, text='Coil resistance:')
        resistanceTextLabel.grid(row=4,column=0, pady=5,sticky=tk.W)
        resistanceValueLabel = ttk.Label(outputFrame, textvariable=self.resistanceString, justify = tk.RIGHT)
        resistanceValueLabel.grid(row=4,column=1, padx=10,sticky=tk.E)
        
        zRangeStringTextLabel = ttk.Label(outputFrame, text='Min/Max X: ')
        zRangeStringTextLabel.grid(row=5,column=0, pady=5,sticky=tk.W)
        zRangeStringValueLabel = ttk.Label(outputFrame, textvariable=self.zRangeString, justify = tk.RIGHT)
        zRangeStringValueLabel.grid(row=5,column=1, padx=10,sticky=tk.E)
        
        zSpacingStringTextLabel = ttk.Label(outputFrame, text='Min wire spacing: ')
        zSpacingStringTextLabel.grid(row=6,column=0, pady=5,sticky=tk.W)
        zSpacingStringValueLabel = ttk.Label(outputFrame, textvariable=self.spacingString, justify = tk.RIGHT)
        zSpacingStringValueLabel.grid(row=6,column=1, padx=10,sticky=tk.E)

        self.saveBfieldButton = ttk.Button(outputFrame, text="Export B-field",state=tk.DISABLED, command=self.exportBfield)
        self.saveBfieldButton.grid(row=7, column=0, pady=5, sticky=tk.SW)
        
        saveWireButton = ttk.Button(outputFrame, text="Export wire to CSV", command=self.exportWireCSV)
        saveWireButton.grid(row=7, column=1, pady = 5, padx=20,sticky=tk.SE)
        
        conjoinedExport = ttk.Checkbutton(outputFrame, text="Join wires", variable = self.exportConjoined)
        conjoinedExport.grid(row=8, column=1, pady = 5, padx=20,sticky=tk.SE)
        
        '''Position plots'''
        
        self.contourfig = Figure(figsize=(4,4))
        self.contourax = self.contourfig.add_subplot(111, projection='3d')
        self.contourfig.set_tight_layout(True)
        
        self.contourCanvas = FigureCanvasTkAgg(self.contourfig, master = self)
        self.contourCanvas.draw()
        self.contourCanvas.get_tk_widget().grid(row=0,column=1)
        self.contourCanvas.mpl_connect('button_press_event', self.contourfig.gca()._button_press)
        self.contourCanvas.mpl_connect('button_release_event', self.contourfig.gca()._button_release)
        self.contourCanvas.mpl_connect('motion_notify_event', self.contourfig.gca()._on_move)
        
        self.linearityFig = Figure(figsize=(4,4))
        self.linearityFig.set_tight_layout(True)
        
        self.linearityCanvas = FigureCanvasTkAgg(self.linearityFig, master = self)
        self.linearityCanvas.draw()
        self.linearityCanvas.get_tk_widget().grid(row=1,column=1)
        
        # self.calculateGradientWires()
        
    
    def calculateGradientWires(self):
        
        #calculate the wire patters
        if (self.directionInput.current() == 0):
            self.contourData, linError, optGradEff, regularisation, err, gradEff = gradCalc.SoSXgradient(modes_N = self.modesN.get(), modes_M = self.modesM.get(), radius = self.coilRadius.get(), length = self.coilLength.get(), numContours = self.numWires.get(), MaxLinError = self.linearityError.get(), DSV = self.DSV.get(), gradDirection='X')  
                                      
        elif (self.directionInput.current() == 1):
            self.contourData, linError, optGradEff, regularisation, err, gradEff = gradCalc.SoSXgradient(modes_N = self.modesN.get(), modes_M = self.modesM.get(), radius = self.coilRadius.get(), length = self.coilLength.get(), numContours = self.numWires.get(), MaxLinError = self.linearityError.get(), DSV = self.DSV.get(), gradDirection='Y') 

        elif (self.directionInput.current() == 2):
            self.contourData, linError, optGradEff, regularisation, err, gradEff = gradCalc.SoSXgradient(modes_N = self.modesN.get(), modes_M = self.modesM.get(), radius = self.coilRadius.get(), length = self.coilLength.get(), numContours = self.numWires.get(), MaxLinError = self.linearityError.get(), DSV = self.DSV.get(), gradDirection='Z') 
        
        #extract the wire levels from the contour data set
        wireLevels = self.contourData.allsegs
        
        self.linearityTextString.set("Error over %.0f mm DSV:"%(self.DSV.get()))
        self.gradErrorString.set("%.2f %%"%linError)

        self.gradEfficiencyString.set("%.4f mT/m/A"%optGradEff)

        self.contourfig.gca().clear()
        self.contourfig.gca().set_xlabel('Z (mm)')
        self.contourfig.gca().set_ylabel('Y (mm)')
        self.contourfig.gca().set_zlabel('X (mm)')
        
        #initialise the length
        self.length = 0
        
        #array for storing the maximum z position for each of the coil segments
        zPos = []
        #plot the wires and calculate length
        for idx, wireLevel in enumerate(wireLevels):
            #there will typically be two wires with the same current (4 with the same current magnitude)
            currentLevel = self.contourData.levels[idx]
            for wire in range (np.size(wireLevel,0)):
                wirePath = wireLevel[wire]
                #wire data is stored on a 2D cylindrical surface, first direction is theta, second is length, convert to 3d (cartesian) coordinates
                wirePath3D = np.stack((np.cos(wirePath[:,0])*self.coilRadius.get(), np.sin(wirePath[:,0])*self.coilRadius.get(),wirePath[:,1]*1e3),axis=1)
                #change the plotting current depending on the direction
                # print("Max distance from center: %.2f mm"%(wire[np.argmax(np.abs(wire[:,1])),1]*1e3))
                zPos.append(wirePath[np.argmax(np.abs(wirePath[:,1])),1]*1e3)
                if currentLevel>0:
                    self.contourfig.gca().plot3D(wirePath3D[:,0],wirePath3D[:,1],wirePath3D[:,2],'red')
                else:
                    self.contourfig.gca().plot3D(wirePath3D[:,0],wirePath3D[:,1],wirePath3D[:,2],'blue')
                
                #CHECK - this cuts out contour points that are extremely close together, avoids bugs
                temp, indices = np.unique(wirePath3D.round(decimals=6), axis = 0, return_index = True)
                wirePath3D = wirePath3D[sorted(indices)]
                
                #calculate the length of each wire segment
                delta = wirePath3D[1:,:] - wirePath3D[:-1,:]
                segLength = np.sqrt(np.sum(np.square(delta), axis = 1))
                self.length += np.sum(segLength)# + np.sum(np.square(wirePath3D[0,:] - wirePath3D[-1,:]))
        
                    
        zPos = np.array(zPos) 
        
        self.zRangeString.set("%.2f / %.2f mm"%(min(zPos), max(zPos)))
        
        #split to deal with positive and negative sides
        zPos_pos = zPos[zPos>0]
        zPos_neg = zPos[zPos<0]
        
        #cut in half due to symmetry issues
        zPos_pos = zPos_pos[:int(np.size(zPos_pos)/2)]
        zPos_neg = zPos_neg[:int(np.size(zPos_neg)/2)]
        
        zPos_pos_distance = np.abs(zPos_pos[1:]-zPos_pos[:-1])
        zPos_neg_distance = np.abs(zPos_neg[1:]-zPos_neg[:-1])
        
        minSpacing = np.min([zPos_pos_distance,zPos_neg_distance])
        self.spacingString.set("%.2f mm"%(minSpacing))
        
        self.contourfig.gca().mouse_init()
        self.wireLengthString.set("%.2f meters"%(self.length*1e-3))
        self.contourCanvas.draw()
        self.contourCanvas.flush_events()
        
        self.linearityFig.gca().clear()
        self.linearityFig.gca().set_xlabel('Gradient Efficiency [mT/m/A]')
        self.linearityFig.gca().set_ylabel('Linearity Error [%]')
        self.linearityFig.gca().grid(True, which="both", ls="-")
        
        self.linearityFig.gca().plot(gradEff,err)
        self.linearityFig.gca().plot(optGradEff,linError,marker="D")
        
        self.linearityCanvas.draw()
        self.linearityCanvas.flush_events()
             
        #self.wireDict = gradCalc.exportWires(self.contourData,  self.coilRadius.get(), self.directionInput.current(), self.exportConjoined.get())
        
        self.calculateResistance()

    def calculateResistance(self):
        copperResistivity = 1.68*1e-8
        resistance = copperResistivity*self.length*1e-3/(np.pi*(self.wireDiameter.get()*1e-3 /2)**2)
        self.resistanceString.set("%.4f Ohms"%(resistance,))
    
    def exportBfield(self):
        bFieldOutputFile = tk.filedialog.asksaveasfilename(defaultextension = '.csv', filetypes=(("CSV file","*.csv"), ("Text file","*.txt")))
        if bFieldOutputFile == None:
            return
        header = 'X (mm),\tY (mm),\tZ (mm),\tBz (mT)'
        delimiter  = ',\t'
        outputArray = np.zeros((np.size(self.bField),np.size(self.coords,0)+1))
        for idx in range(np.size(self.coords,0)):
            outputArray[:,idx] = np.ravel(self.coords[idx])
        outputArray[:,-1] = np.ravel(self.bField)
        np.savetxt(bFieldOutputFile.name , outputArray,delimiter  = delimiter ,header = header )
        bFieldOutputFile.close()
    
    def exportWireCSV(self):
        wireOutputFile = tk.filedialog.asksaveasfilename(defaultextension = '.txt', filetypes=(("CSV file","*.csv"), ("Text file","*.txt")))
        if wireOutputFile == '':
            return
        folder, filename = os.path.split(wireOutputFile)
        file, extension = os.path.splitext(filename)
        
        self.wireDict = gradCalc.exportWires(self.contourData,  self.coilRadius.get(), self.directionInput.current(), self.exportConjoined.get())
        
        for key in self.wireDict:
            filename = os.path.join(folder,file+key+extension)
            np.savetxt(filename,self.wireDict[key],delimiter=",", fmt='%f')
        
        
app = GDT_GUI()
app.mainloop()
closePlots('all')