#!/usr/bin/env python

###################Python version of var_res ######################
# Adapted from the matlab version of var_res originally written by#
# Mark Simons and Yuri Fialko.                                    #
# Python version written by Piyush Agram                          #
# Date: Jan 2, 2012                                               #
###################################################################

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys
import os.path
import reader
import tree
import covaraps
import argparse
import logmgr

#######Logger for varrespy
logger = logmgr.logger('varres')

######For parsing command line options
def parse():
    '''Command line parser for varres. Use -h option to get complete list of options.'''
    parser = argparse.ArgumentParser(description='Variable resolution resampler for unwrapped interferograms.')
    parser.add_argument('-i', action='store', required=True, dest="I_name", help="ROI-PAC geocoded unwrapped file with .rsc in the same directory", type=str)
    parser.add_argument('-g', action='store', dest = "S_name", default=None, help="2 component (angles) or 3 component (LOS vector) file with same dimensions as geocoded unwrapped file", type=str)
    parser.add_argument('-o', action='store', dest = "out_name", default="varres", help='The output file prefix', type=str)
    parser.add_argument('-az', action='store_true', dest="useaz", default=False, help="Data represents azimuth offsets / MAI interferogram. Default: False")
    parser.add_argument('-default', action='store_true', dest="usedefgeom", default=False, help="Use default geometry from .rsc file. Default: False")
    parser.add_argument('-t', action='store', required=True, dest='thresh', help='Threshold for resampling in cm',type=float)
    parser.add_argument('-var', action='store_true', dest='usevar', default=False, help='Use variance instead of curvature. Default: False')
    parser.add_argument('-nseg', action='store', dest='nseg', default=1, help='Number of vertical segments for stitched frames. Default: 1', type=int)
    parser.add_argument('-noplot', action='store_false',dest='plot', default=True, help='To turn off plotting. Default always plots.')
    parser.add_argument('-covar', action='store_true', dest='covar', default=False, help='Covariance function computation. Default: False')
    parser.add_argument('-minsize', action='store', dest='minsize', default=2, help='Minimum size of the resampling boxes. Default: 2', type=int)
    parser.add_argument('-maxsize', action='store', dest='maxsize', default=np.inf, help='Maximum size of the resampling boxes. Default: Infinite', type=int)
    parser.add_argument('-noscale', action='store_false', dest='scale', default=True, help='To stop scaling of data to cm before resampling. Default: Always scales to cm.')
    parser.add_argument('-mult', action='store', dest='mult', default=1.0, help='Multiplier before resampling. If data in m, use -mult 100 to convert to cm. Default: 1.0', type=float)
    parser.add_argument('-minres', action='store', dest='minres', default=2, help='Minimum resolution level. Default: 2', type=int)
    parser.add_argument('-vflip', action='store_true', dest='flipvert', default=False, help='Flip the image in vertical direction befor resampling. Default: False')
    parser.add_argument('-nfrac', action='store', dest='nfrac', default=0.1, help='Fraction of randomly selected pixels for covariance computation. Default: 0.1', type=float)
    parser.add_argument('-dscale', action='store', dest='dscale', default=0.001, help='Distance scaling for covariance function computation. Default: 0.001', type=float)
    parser.add_argument('-rsp', action='store_true', dest='rsp', default=False, help='Store a resampling map for use with other data sets. Default: False')

    inps = parser.parse_args()

    #########Check input options
    if np.sum(np.array([inps.useaz,inps.usedefgeom,(inps.S_name is not None)]))>1:
        logger.error('More than one geometry options set.')
        sys.exit(1)

    ########Check threshold
    if (inps.thresh <= 0):
        logger.error('Threshold should be a positive number.')
        sys.exit(1)

    ########Check segments
    if (inps.nseg < 1):
        logger.error('Nseg: %d should be one or greater. '%(inps.nseg))
        sys.exit(1)

    ########Check maxsize
    if (inps.maxsize <= 4):
        logger.error('Maxsize should be atleast 4.')
        sys.exit(1)

    ########Check Nfrac
    if ((inps.nfrac<=0) | (inps.nfrac>=1)):
        logger.error('Nfrac should be between 0 and 1.')
        sys.exit(1)

    ########Check dscale
    if (inps.dscale <= 0):
        logger.error('dscale should be a positive number.')
        sys.exit(1)

    return inps


##########Parsing program inputs##########
inps = parse()

indata = reader.reader(inps.I_name, inps.S_name)
indata.read_igram(scale=inps.scale, flip=inps.flipvert, mult = inps.mult)
indata.read_geom(az = inps.useaz, defgeom = inps.usedefgeom, flip = inps.flipvert)

sampler = tree.tree(inps.minsize, inps.maxsize, inps.thresh, inps.minres, method=inps.usevar)
sampler.resample(indata)
sampler.write(inps.out_name, rsp=inps.rsp)
npts = len(sampler.xi)


########Plot data 
if inps.plot:
    plt.figure('Decomposition')
    plt.jet()
    orgax = plt.subplot(221)
    plt.imshow(indata.phs)
    tks = np.linspace(np.nanmin(indata.phs), np.nanmax(indata.phs), num=5)
    orgax.set_ylabel('Pixels')
    orgax.set_xticklabels([])
    plt.title('Original IFG,cm')
    plt.colorbar(aspect=8,shrink=0.6,ticks=np.round(tks))

    subax = plt.subplot(222)
    plt.scatter(sampler.xi-1,sampler.yi-1,s=8,c='k')
    plt.axis('equal')
    plt.xlim((0,indata.nx-1))
    plt.ylim((0,indata.ny-1))
    subax.set_ylim([indata.phs.shape[0],0])
    subax.set_xlim([0,indata.phs.shape[1]])
    subax.yaxis.tick_right()
    plt.axis('tight')
    plt.title('%d to %d Samples'%(np.isfinite(indata.phs).sum(),sampler.xi.size))

    resax = plt.subplot(223)
    x=np.arange(indata.nx)+1
    y=np.arange(indata.ny)+1
    newz = griddata((sampler.xi,sampler.yi),sampler.zi,(x[None,:],y[:,None]),method='cubic')
    (badi,badj) = np.where(np.isnan(indata.phs) == True)
    newz[badi,badj]=np.nan
    plt.imshow(newz,clim=[tks[0],tks[-1]])
    resax.set_ylabel('Pixels')
    resax.set_ylim([indata.phs.shape[0],0])
    resax.set_xlim([0,indata.phs.shape[1]])
    plt.colorbar(aspect=8,shrink=0.6,ticks=np.round(tks))
    plt.title('Interpolated Data')

    err = indata.phs-newz
    tks = np.linspace(np.nanmin(err), np.nanmax(err),num=5)
    errax = plt.subplot(224)
    plt.imshow(err)
    errax.set_ylim([indata.phs.shape[0],0])
    errax.set_xlim([0,indata.phs.shape[1]])
    plt.colorbar(aspect=8,shrink=0.6, ticks=np.round(tks))
    errax.set_yticklabels([])
    plt.title('Difference')
    plt.savefig('originalvsresample.png',dpi=300)
else:
    logger.info('No plotting requested.')


########Compute covariance matrix.
if inps.covar:
    (sigma,lam) = covaraps.aps_param(indata.phs, inps.nfrac, inps.dscale, inps.plot)
    cvar = np.zeros((npts,npts))
    # Atmospheric phase screen component
    for m in xrange(npts):
        for n in xrange(npts):
            dx = indata.x[sampler.xi[m]-1] - indata.x[sampler.xi[n]-1]
            dy = indata.y[sampler.yi[m]-1] - indata.y[sampler.yi[n]-1]
            dist = inps.dscale*np.sqrt(dx*dx+dy*dy)
            cvar[m,n] = covaraps.model_fn(dist,sigma,lam)

    cvar = sigma - cvar
    
    #Error component due to resampling
    for m in xrange(npts):
        cvar[m,m] += sampler.zerr[m]*sampler.zerr[m]

    #Writing the covariance matrix to file
    fout = open('%s.cov'%(inps.out_name),'wb')
    covf = cvar.astype(np.float32)
    covf.tofile(fout)
    fout.close()
    
    if inps.plot:
        plt.figure('Covariance')
        plt.imshow(cvar)
        plt.colorbar(aspect = 8 , shrink = 0.6)
        plt.title('Covariance (cm^2)')
        plt.savefig('covariance.png')


if inps.plot:
    plt.show()

############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
