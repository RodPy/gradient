# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:17:02 2021
Sum of sines method 
@author: bdevos
"""



import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

def calculateContour(streamF, numWires, phi, z):
    
    phi2D, z2D= np.meshgrid(phi,z)
    levels = np.linspace(np.min(streamF), np.max(streamF), 2*(numWires + 2))
    levels = levels[1:-1]

    # Wire should be laid along contours between the isolines, calculate midpoint between isolines
    midPointLevels = [(levels[i]+levels[i+1])/2 for i in range(np.size(levels)-1)]
    midPointLevels = np.array(midPointLevels)[np.abs(midPointLevels) >= 1e-6] #remove zeros, account for floating point error
    
    plt.ioff()
    plt.figure(1)
    contWires = plt.contour(phi2D,z2D,streamF,levels = midPointLevels)

    return contWires


def dx_ZonalCylinder(Tx,Ty,Tz,a,phi,z): 
    return ((Tx[:,np.newaxis, np.newaxis]-a*np.cos(phi))**2+(Ty[:, np.newaxis, np.newaxis]-a*np.sin(phi))**2+(Tz[:,np.newaxis, np.newaxis]-z)**2)**1.5

def Es_z(n,m,phi,z,L,a):
    return 2*m*L/(n*np.pi*a)*np.multiply(np.sin(n*np.pi*(z+L)/(2*L)),np.cos(m*phi))
   
def Es_y(n,m,phi,z,L):
    return np.multiply(np.cos(phi)*np.sin(m*phi), np.cos(n*np.pi*(z+L)/(2*L)))

def Ds_z(n,m,phi,z,L,a):
    return 2*m*L/(np.pi*n*a)*np.multiply(np.sin(n*np.pi*(z+L)/(2*L)),np.sin(m*phi))

def Ds_y(n,m,phi,z,L):
    return np.multiply(np.cos(phi)*np.cos(m*phi), np.cos(n*np.pi*(z+L)/(2*L)))

def Beta_x(n,m,phi,L,dphi,dz,z,a, dx, Ty_sinPhi, Tz_z):  
    return np.sum((Ds_y(n,m,phi,z,L)*Tz_z - Ds_z(n,m,phi,z,L,a)*Ty_sinPhi)/dx, axis = (1,2))

def Gamma_x(n,m,phi,L,dphi,dz,z,a, dx, Ty_sinPhi, Tz_z):
    return np.sum((Es_y(n,m,phi,z,L)*Tz_z + Es_z(n,m,phi,z,L,a)*Ty_sinPhi)/dx, axis = (1,2))

def dx_ZonalCylinder_vec(Tx,Ty,Tz,a,phi,z): 
    return ((Tx[:,np.newaxis]-a*np.cos(phi))**2+(Ty[:, np.newaxis]-a*np.sin(phi))**2+(Tz[:,np.newaxis]-z)**2)**1.5

def Es_z_vec(n,m,phi,z,L,a):
    return (2*m*L/(n*np.pi*a))[:,np.newaxis]*np.multiply(np.sin(np.outer(n*np.pi,(z+L)/(2*L))),np.cos(np.outer(m,phi)))
   
def Es_y_vec(n,m,phi,z,L):
    return np.multiply(np.cos(phi)[np.newaxis,:]*np.sin(np.outer(m,phi)), np.cos(np.outer(n*np.pi,(z+L)/(2*L))))

def Ds_z_vec(n,m,phi,z,L,a):
    return (2*m*L/(np.pi*n*a))[:,np.newaxis]*np.multiply(np.sin(np.outer(n*np.pi,(z+L)/(2*L))),np.sin(np.outer(m,phi)))

def Ds_y_vec(n,m,phi,z,L):
    return np.multiply(np.cos(phi)[np.newaxis,:]*np.cos(np.outer(m,phi)), np.cos(np.outer(n*np.pi,(z+L)/(2*L))))

def Beta_x_vec(n,m,phi,L,dphi,dz,z,a, dx, Ty_sinPhi, Tz_z):  
    return np.sum((Ds_y_vec(n,m,phi,z,L)[np.newaxis,:,:]*Tz_z[:,np.newaxis,:] - Ds_z_vec(n,m,phi,z,L,a)[np.newaxis,:,:]*Ty_sinPhi[:,np.newaxis,:])/dx[:,np.newaxis,:], axis = -1)

def Gamma_x_vec(n,m,phi,L,dphi,dz,z,a, dx, Ty_sinPhi, Tz_z):
    return np.sum((Es_y_vec(n,m,phi,z,L)[np.newaxis,:,:]*Tz_z[:,np.newaxis,:] + Es_z_vec(n,m,phi,z,L,a)[np.newaxis,:,:]*Ty_sinPhi[:,np.newaxis,:])/dx[:,np.newaxis,:], axis = -1)


def SoSXgradient(modes_N = 8, modes_M = 4, radius = 150, length = 450, numContours = 15, MaxLinError = 20, DSV=200, gradDirection='X' ):

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%              Input Parameters                            %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    #conert units from mm to meter
    radius  = radius*1e-3
    length  = length*1e-3
    DSV     = DSV*1e-3
    
    mu0     = 4*np.pi*1e-7
   
    modes_N = int(modes_N)
    modes_M = int(modes_M)  

    if gradDirection == 'X':
        print("Only even modes used for N")
        print("Only odd modes used for M")
        modes_N_list = np.arange(2, modes_N+1, 2)
        modes_M_list = np.arange(1, modes_M+1, 2)
    elif gradDirection == 'Y':
        print("Only odd modes used for N")
        print("Only even modes used for M")
        modes_N_list = np.arange(1, modes_N+1, 2)
        modes_M_list = np.arange(2, modes_M+1, 2)
        # modes_M_list = np.arange(0, modes_M, 1)
    elif gradDirection == 'Z':
        print("Only odd modes used for N")
        print("Only even modes used for M")
        modes_N_list = np.arange(1, modes_N+1, 2)
        modes_M_list = np.arange(2, modes_M+1, 2)
    
    #tile tthe data 
    nVec                = np.repeat(modes_N_list, np.size(modes_M_list))
    mVec                = np.tile(modes_M_list,np.size(modes_N_list))         
       
    Nsamp_TF            = 40        
    gradstrength        = 10e-3
    # TF_length_factor    = 0.9
    # TF_radius_factor    = 0.8
    
    Nsamp = 200 # 200 = quick fix for straighter loop connections. orig = 100 (WT 20220120)
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%              Surface Current Coordinates                 %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    z = np.linspace(-length/2, length/2, num=Nsamp)
    z = np.vstack(z)
    
    if gradDirection == 'X':
        phi = np.linspace(0.5*np.pi, 2.5*np.pi, num = Nsamp)
    elif gradDirection == 'Y':
        phi = np.linspace(0, 2*np.pi, num = Nsamp)
    elif gradDirection == 'Z':
        phi = np.linspace(0.25*np.pi, 2.25*np.pi, num = Nsamp)
    
    PHI,Z = np.meshgrid(phi , z)
    
    # dphi = max(phi)/Nsamp
    # dz = (length)/Nsamp
    
    dphi = (max(phi) - min(phi))/(Nsamp-1)
    dz = (length)/(Nsamp-1)
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% Target Field Coordinates: Cartesian grid -> sphere       %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    xT = np.linspace(-DSV/2, DSV/2, num=Nsamp_TF)
    yT = np.linspace(-DSV/2, DSV/2, num=Nsamp_TF)
    zT = np.linspace(-DSV/2, DSV/2, num=Nsamp_TF)
    
    TX_3D,TY_3D,TZ_3D   = np.meshgrid(xT, yT, zT, indexing='xy')
    
    #calculate a mask in which the target field should be calculated, spherical in this case
    rho                 = np.sqrt(np.square(TX_3D) + np.square(TY_3D) + np.square(TZ_3D))
    fieldMask           = (rho<=(DSV/2))
    
    #Create a spherical shell mask to consider only data points on the surface of the sphere
    import scipy.ndimage as cp
    erodedMask = cp.binary_erosion(fieldMask)                    # remove the outer surface of the initial spherical mask
    fieldMask = np.array(fieldMask^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
    # fieldMask[shellMask == 0] = np.nan                          # set points outside mask to 'NaN', works better than setting it to zero for calculating mean fields etc.

    #remove the origin from the mask to avoid a divide by zero
    fieldMask[rho == 0] = 0 #
    
    Tx_vec              = TX_3D[fieldMask==1]
    Ty_vec              = TY_3D[fieldMask==1]
    Tz_vec              = TZ_3D[fieldMask==1]
    
    #define target field as a linear field
    if gradDirection == 'X':
        #add two points at the end, at the center of the DSV, for calculating efficiency, lazy solution
        Tx_vec      = np.append(Tx_vec, (0,0))
        Ty_vec      = np.append(Ty_vec, (0,0))
        Tz_vec      = np.append(Tz_vec, (-DSV/(2*Nsamp_TF),DSV/(2*Nsamp_TF)))
        #target field is a linear field in the direction of the gradient
        targetField = np.multiply(Tz_vec,gradstrength)
    elif gradDirection == 'Y':
        Tx_vec      = np.append(Tx_vec, (0,0))
        Ty_vec      = np.append(Ty_vec, (-DSV/(2*Nsamp_TF),DSV/(2*Nsamp_TF)))
        Tz_vec      = np.append(Tz_vec, (0,0))
        targetField = np.multiply(Ty_vec,gradstrength)
    elif gradDirection == 'Z':
        Tx_vec      = np.append(Tx_vec, (-DSV/(2*Nsamp_TF),DSV/(2*Nsamp_TF)))
        Ty_vec      = np.append(Ty_vec, (0,0))
        Tz_vec      = np.append(Tz_vec, (0,0))
        targetField = np.multiply(Tx_vec,gradstrength)
    
    targetField_norm= targetField/np.nanmax(targetField)

    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%               Creating the system matrix                  %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    beta    = np.empty((len(Tx_vec),np.size(mVec)), dtype=float)
    gamma   = np.empty((len(Tx_vec),np.size(mVec)), dtype=float)
    
    import time
    start = time.time()

    dx          = dx_ZonalCylinder(Tx_vec,Ty_vec,Tz_vec,radius,phi,z)
    Tz_z        = Tz_vec[:, np.newaxis, np.newaxis] - z
    Ty_sinPhi   = Ty_vec[:, np.newaxis, np.newaxis] - radius*np.sin(phi)
    
    for idx in range(np.size(nVec)):  
        beta[:,idx]     = Beta_x(nVec[idx],mVec[idx],phi,length/2,dphi,dz,z,radius, dx, Ty_sinPhi, Tz_z)
        gamma[:,idx]    = Gamma_x(nVec[idx],mVec[idx],phi,length/2,dphi,dz,z,radius, dx, Ty_sinPhi, Tz_z)
        
    print("Execution: %.2f sec"%(time.time()-start))

    A       = np.hstack([beta,gamma])*dphi*dz*radius*mu0/(4*np.pi)

    """
    %%%%%%%%%%%%%%%%%%% Regularised least squares %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    kappa   = radius*np.pi*length/2*(1+((2*mVec*length/2)/(nVec*np.pi*radius))**2)
          
    G       = np.diag(np.hstack([kappa,kappa]))

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%               Regularisation sweep                       %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """ 
    
    #The above code only needs to be computed once. 

    regularisation      = np.logspace(-12,-9,80)
    
    err                 = np.zeros(len(regularisation))
    linearity_bench     = np.zeros(len(regularisation))
    gradEff             = np.zeros(len(regularisation))
    gradCurrent         = np.zeros(len(regularisation))


    for ll in range(len(regularisation)):
        
        x=np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)+regularisation[ll]*G),np.matmul(np.transpose(A),targetField))
        
        S=0
        for idx in range(np.size(mVec)):
            S += np.multiply(length/(nVec[idx]*np.pi)*x[idx]*np.sin((nVec[idx])*np.pi*(Z+length/2)/length),np.cos(mVec[idx]*PHI))
            S += np.multiply(length/(nVec[idx]*np.pi)*x[np.size(mVec) + idx]*np.sin(nVec[idx]*np.pi*(Z+length)/length),np.sin(mVec[idx]*PHI))

        #determine the field using Ax=b 
        B_forward=np.dot(A,x)    
        norm_forward=B_forward/np.max(B_forward)
    
        residual=np.abs(norm_forward-targetField_norm)/(targetField_norm)
        #pdb.set_trace()
    
        
        err[ll]=np.nanmax(residual)*100
        
        if gradDirection == 'X':
            currentLevels = np.linspace(np.nanmin(S), np.nanmax(S), numContours*2 + 4, endpoint = True)
            gradCurrent[ll] = currentLevels[1] - currentLevels[0]
            linearity_bench[ll]=B_forward[-1] - B_forward[-2]
        elif gradDirection == 'Y':
            print("Efficiency calculation for transverse gradients need to be checked")
            currentLevels = np.linspace(np.nanmin(S), np.nanmax(S), (numContours+1)*2 + 3, endpoint = True)
            gradCurrent[ll] = currentLevels[1] - currentLevels[0]
            linearity_bench[ll]=B_forward[-1] - B_forward[-2]
        elif gradDirection == 'Z':
            print("Efficiency calculation for transverse gradients need to be checked")
            currentLevels = np.linspace(np.nanmin(S), np.nanmax(S), numContours*2 + 3, endpoint = True)
            gradCurrent[ll] = currentLevels[1] - currentLevels[0]
            linearity_bench[ll]==B_forward[-1] - B_forward[-2]
    
    
        gradEff[ll]=(linearity_bench[ll]/(DSV/Nsamp_TF))*1000/gradCurrent[ll]
        print("Reg param: %f, error: %.2f %%, efficiency: %f"%(np.log10(regularisation[ll]),err[ll],gradEff[ll]))
        
   
    
    for ii in range(0,len(regularisation)):
        if err[-1-ii]<=MaxLinError:
            error=err[-1-ii]
            regularisation_opt=regularisation[-1-ii]
            optimalGradEff=gradEff[-1-ii]
            break
        else: 
            error=min(err)
            idx=min(range(len(err)), key=err.__getitem__)
            regularisation_opt=regularisation[idx]
            optimalGradEff=gradEff[idx]

  
    """
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%             Display Output optimal lambda                %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

  
    x=np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)+regularisation_opt*G),np.matmul(np.transpose(A),targetField))
  
    
    S=0
    for idx in range(np.size(mVec)):
        S += np.multiply(length/(nVec[idx]*np.pi)*x[idx]*np.sin(nVec[idx]*np.pi*(Z+length/2)/length),np.cos(mVec[idx]*PHI))
        S += np.multiply(length/(nVec[idx]*np.pi)*x[np.size(mVec) + idx]*np.sin(nVec[idx]*np.pi*(Z+length/2)/length),np.sin(mVec[idx]*PHI))
        
        # print("N: %i, M: %i, Weight: %.5f and %.5f"%(nVec[idx], mVec[idx], x[idx]/np.max(np.abs(x)), x[np.size(mVec) + idx]/np.max(np.abs(x))))

    return calculateContour(S, numContours, phi, z), error, optimalGradEff, regularisation, err, gradEff



def exportWires(contours, coilRad, direction, conjoined):
    
    wireNum = 0
    contourDict = {}
    wireLevels = contours.allsegs
    
    if ((direction == 0) and conjoined):        #for the X gradient the center of the smallest contour is needed for joining the wires
        minLength = np.inf
        for wireLevel in wireLevels:
            for wire in wireLevel:
                if(np.size(wire,0) < minLength):
                    centerHeight = np.abs(np.mean(wire[:,1])*1e3)
    for idx, wireLevel in enumerate(wireLevels):
           for wire in wireLevel:
            wirePath3D = np.stack((np.cos(wire[:,0])*coilRad,np.sin(wire[:,0])*coilRad,wire[:,1]*1e3),axis=1)
            if(conjoined):
                gapSize = 8 #gap in which the sections are joined
                gapAngle = gapSize/coilRad      
                centerAngle = np.mean(wire[:,0])
                
                if(direction == 0):
                    #mask = (np.abs(wire[:,0] - centerAngle) > gapAngle) | (np.abs(wirePath3D[:,2]) < centerHeight)
                    mask = (np.abs(wire[:,0] - centerAngle) > gapAngle) | (np.abs(wirePath3D[:,2]) > centerHeight)
                else:
                    mask = (np.abs(wire[:,0] - centerAngle) > gapAngle) | (wirePath3D[:,2] < 0)
                
                while mask[0]:
                    mask = np.roll(mask,1)
                    wirePath3D = np.roll(wirePath3D, 1, axis = 0)
        
                contourDict[str(wireNum)] = np.stack((wirePath3D[mask, 0],wirePath3D[mask, 1],wirePath3D[mask, 2]),axis=1)
            else:
                contourDict[str(wireNum)] = wirePath3D
            wireNum += 1
            

    if(not conjoined):
        return contourDict
    else:
        
        #############################################
        # Join the wires with a gap in to one array #
        #############################################
        
        numCoilSegments = 4             #Number of quadrants

        joinedContour = {}
        joinedContour[str(0)] = contourDict[str(0)]
        joinedContour[str(1)] = contourDict[str(1)]
        joinedContour[str(2)] = contourDict[str(int(2*wireNum/numCoilSegments))]
        joinedContour[str(3)] = contourDict[str(int(2*wireNum/numCoilSegments)+1)]
        
        for idx in range(1,int(wireNum/numCoilSegments)):
            joinedContour[str(0)] = np.append(joinedContour[str(0)], contourDict[str(2*idx)], axis = 0)
            joinedContour[str(1)] = np.append(joinedContour[str(1)], contourDict[str(2*idx+1)], axis = 0)
            joinedContour[str(2)] = np.append(joinedContour[str(2)], contourDict[str(int(2*wireNum/numCoilSegments) + idx*2 )], axis = 0)
            joinedContour[str(3)] = np.append(joinedContour[str(3)], contourDict[str(int(2*wireNum/numCoilSegments) + idx*2 +1)], axis = 0)
        
        
        ############################################
        # Check for consecutive identical elements #
        ############################################
        tol = 1e-5
        for key in joinedContour:
            delta = joinedContour[key][1:,:] - joinedContour[key][:-1,:]
            delta = np.sum(np.square(delta), axis = 1)
            zeroElements = delta < tol
            joinedContour[key] = np.delete(joinedContour[key],np.nonzero(zeroElements), axis = 0)
            
        return joinedContour
        

    