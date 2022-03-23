import numpy as np
import os
import sys
import logmgr
import loaddata 
import utils

logger = logmgr.logger('varres')

class reader:
    def __init__(self, iname, sname):
        '''Class for reading in the geometry data and setting it up.'''
        
        if(os.path.isfile(iname) == False):
            logger.error('IFG file: %s not found.'%(iname))
            sys.exit(1)

        if(os.path.isfile('%s.rsc'%(iname)) == False):
            logger.error('IFG rsc file: %s not found.'%(iname))
            sys.exit(1)

        if sname is not None:
            if(os.path.isfile(sname) == False):
                logger.error('Geometry file: %s not found.'%(sname))
                sys.exit(1)

        self.rdict = utils.read_rsc('%s.rsc'%(iname))
        self.nx = np.int(self.rdict['WIDTH'])
        self.ny = np.int(self.rdict['FILE_LENGTH'])
        self.dx = np.float(self.rdict['X_STEP'])
        self.dy = np.float(self.rdict['Y_STEP'])
        self.TL_east = np.float(self.rdict['X_FIRST'])
        self.TL_north = np.float(self.rdict['Y_FIRST'])
        self.wvl = 100.0 * np.float(self.rdict['WAVELENGTH'])
       
        self.iname = iname
        self.sname = sname
        self.phs = None
        self.geom = []
        self.x = None
        self.y = None

    def read_igram(self, scale=True, flip=False, mult=1.0):
        fact = np.choose(scale,[1.0,self.wvl/(4*np.pi)])
        fact = fact*mult

        (phs,ngood) = loaddata.load_igram(self.iname,self.nx,self.ny,fact)
        if flip:
            self.phs = np.flipud(phs)
        else:
            self.phs = phs

        logger.info('Original number of data points: %d'%(ngood))

    def read_geom(self, az= False, defgeom=False, flip=False):
        self.x = self.TL_east + np.arange(self.nx) * self.dx
        self.y = self.TL_north + np.arange(self.ny) * self.dy
        if flip:
            self.y = self.y[::-1]

        if defgeom:
            self.geom = loaddata.get_los(self.rdict)

        elif az:
            self.geom = loaddata.get_azi(self.rdict)

        elif self.sname is not None:
            self.geom = loaddata.load_S(self.sname,self.nx,self.ny)
            if flip:
                for k in xrange(3):
                    self.geom[k] = np.flipud(self.geom[k])

        else:
            for k in xrange(3):
                self.geom[k] = 0





############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
