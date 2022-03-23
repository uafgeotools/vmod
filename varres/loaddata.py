###############################################################
# Translated from load_igram.m and load_S.m                   #
# Added utility to approximate LOS.
# Written by Piyush Agram                                     #
# Date: Jan 2, 2012                                           #
###############################################################

import numpy as np
import logmgr
import sys

logger = logmgr.logger('varres')

#######Load_igram
def load_igram(I_name,nx,ny,scale_by_wvl):
    fin = open(I_name,'rb'); #Open in Binary format
    phs = np.fromfile(fin, dtype=np.float32, count= 2*nx*ny)
    fin.close()

    phs = np.reshape(phs,(ny,2*nx))
    phs = phs[:,nx:]

    (badi,badj) = np.where(phs == 0)
    (goodi,goodj) = np.where(phs!=0)

    ngood = len(goodi)
    temp = phs[goodi,goodj]
    dc = np.median(temp)
    phs = phs - dc
    del temp
    del goodi
    del goodj
    phs[badi,badj] = np.nan
    phs = phs * scale_by_wvl
    del badi
    del badj
    return phs,ngood

#####End of load_igram

######Loading S_file ############
def load_S(Sname,nx,ny):
    import sys
    import os.path

    #Amplitude -> Look Angle (theta)
    #Phase-> Azimuth of satellite to ground (phi) (NOT heading)
    fin = open(Sname,'rb'); #Open in Binary format
    phs = np.fromfile(file=fin, dtype=np.float32, count=2*nx*ny)
    fin.close()

    phs = np.reshape(phs,(ny,2*nx))
    
    theta = phs[:,:nx]*np.pi/180.0
    phi = phs[:,nx:]*np.pi/180.0

    (badi,badj) = np.where(theta == 0)
    theta[badi,badj] = np.nan
    phi[badi,badj] = np.nan
    phi = phi + np.pi
    del phs
 
    Se = np.sin(theta) * np.sin(phi)
    Sn = np.sin(theta) * np.cos(phi)
    Su = np.cos(theta)
    return Se,Sn,Su


######End of load_S

#######get_los#######################################################
# Computes the approx LOS vectors from the information in geo_unw.rsc
# Errors in LOS vectors are about 10%.
#####################################################################
def get_los(rdict):
    rgrid = np.zeros(4)
    lgrid = np.zeros(4)
    
    for i in xrange(4):
        lgrid[i] = np.float(rdict['LOOK_REF%d'%(i+1)])
        rgrid[i] = np.float(rdict['RGE_REF%d'%(i+1)])


    phi = np.float(rdict['HEADING_DEG'])
    logger.info('HEADING   : %f'%(phi))
    phi = 1.5*np.pi + phi*np.pi/180.0

    look = np.sum(lgrid)*0.25	
    rng  = np.sum(rgrid)*0.25*1000
    hgt  = np.float(rdict['HEIGHT'])
    re   = np.float(rdict['EARTH_RADIUS'])
    
    beta = (re*re + rng*rng - (re+hgt)*(re+hgt))/(2*re*rng)
    theta = np.pi-np.arccos(beta)
    logger.info('LOOK ANGLE: %f'%(theta*180/np.pi))

    Se = np.sin(theta) * np.sin(phi)
    Sn = np.sin(theta) * np.cos(phi)
    Su = np.cos(theta)
    return Se,Sn,Su

#######get_azi#######################################################
# Computes the approx heading vectors from the information in geo_unw.rsc
# Errors in LOS vectors are about 10%.
#####################################################################
def get_azi(rdict):
    phi = np.float(rdict['HEADING_DEG'])
    logger.info('HEADING : %f'%(phi))
    phi = phi*np.pi/180.0     #Convert to radians
    
    Se = np.sin(phi)
    Sn = np.cos(phi)
    Su = 0.0          #Ignoring Verticals.
    return Se,Sn,Su 

############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
