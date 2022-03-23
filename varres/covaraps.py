#####################################################################
#  Approximating Atmospheric covariance function                    #
#  with an exponential.						    #
#  Translated from variable_res.m                                   #
#  Written by Piyush Agram.					    #
#  Date: Jan 2, 2012                                                #
#####################################################################

import numpy as np
import utils
import matplotlib.pyplot as plt
import scipy.optimize as sp
import logmgr

logger=logmgr.logger('varres')

#Structure function
def model_fn(t,sig,lam):
    return sig*(1-np.exp(-t/lam))


def ramp_fn(t,a,b):
    dx = t[:,0]
    dy = t[:,1]
    d = np.sqrt(dx*dx+dy*dy)
    return a*dx*dx + b*dy*dy + 2*a*b*dx*dy

#Estimation of sigma and lambda
# covariance of atmosphere = sigma^2 * exp(-dist/lambda)
#### Could definitely implement this better.
#### Plotting and histograms can be improved.
def aps_param(phs,frac,scale,plotflag):
    [ii,jj] = np.where(np.isnan(phs) == False)
    val = phs[ii,jj]
    num = len(ii)
    Nsamp = np.floor(frac*num)
    samp = np.random.random_integers(0,num-1,size=(Nsamp,2))
    s1 = samp[:,0]
    s2 = samp[:,1]
    dx = scale*(ii[s1]-ii[s2])
    dy = scale*(jj[s1]-jj[s2])
    dist = np.sqrt(dx*dx+dy*dy)
    ind = dist.nonzero()
    dist = dist[ind]
    dx = dx[ind]
    dy = dy[ind]
    Nsamp = len(dist)
    dv = val[s1]-val[s2]
    dv = dv*dv
    dv = dv[ind]

    mask = (dist < 1)
    dist = dist[mask]
    dx = dx[mask]
    dy = dy[mask]
    dv = dv[mask]
    Nsamp = len(dist)

    Amat = np.zeros((Nsamp,2))
    Amat[:,0] = dx
    Amat[:,1] = dy
    opt_pars, pars_cov = sp.curve_fit(ramp_fn,Amat,dv)
    a = opt_pars[0]
    b = opt_pars[1]
    logger.info('RAMP: %f %f'%(a,b))
    dv = dv - ramp_fn(Amat,a,b)

    opt_pars, pars_cov = sp.curve_fit(model_fn,dist,dv)
    sig = opt_pars[0]
    lam = opt_pars[1]

    logger.info('SIGMA   : %f'%(sig))
    logger.info('LAMBDA  : %f'%(lam))

    if plotflag:
        plt.figure('Structure')
        plt.hold(True)
        plt.scatter(dist,dv,s=1,c='k')
        x = np.arange(100)*0.1*dist.max()/100.0
        y = model_fn(x,sig,lam)
        plt.plot(x,y)
        plt.xlabel('Normlized Distance')
        plt.ylabel('log(phase var)')
        plt.show()
    return sig,lam


############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
