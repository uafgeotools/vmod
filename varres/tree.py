import numpy as np
import logmgr
import utils
import sys

logger = logmgr.logger('varres')

class tree:
    def __init__(self, minsize, maxsize, thresh, minres, method=False):
        '''Create a tree from the data in the reader.'''
        ######Data holders
        self.xi = []
        self.yi = []
        self.zi = []
        self.ulx = []
        self.uly = []
        self.drx = []
        self.dry = []
        self.zerr = []
        self.wgt = []

        ######Algorithm params
        self.minsize = minsize
        self.maxsize = maxsize
        self.thresh = thresh
        self.minres = minres
        self.ptscounter = 0
        self.reslev = 1
        self.usevar = method

        ######Geometry
        self.ecoord = None
        self.ncoord = None
        self.geom = []
        self.counter = None

    def resample(self, reader, nseg=1):
        '''Resample the data contained in reader.'''
        sys.setrecursionlimit(500)
        logger.info('Resampling data')
        self.counter = utils.LineCounter('Samples')
        for iseg in range(nseg):
            dyseg = np.floor(reader.ny/(1.0*nseg))
            ystart = iseg*dyseg+1
            if (iseg==nseg):
                yend = ny
            else:
                yend = ystart + dyseg
            print(reader.nx, ystart, yend, reader)
            self.iter_res(1, reader.nx, ystart, yend, reader)

        self.counter.close()
        npts = len(self.xi)
        self.xi = np.array(self.xi,dtype=np.int)
        self.yi = np.array(self.yi,dtype=np.int)
        self.zi = np.array(self.zi,dtype=np.float)

        logger.info('Preparing geometry products')
        self.ecoord = reader.x[self.xi-1]
        self.ncoord = reader.y[self.yi-1]

        if np.isscalar(reader.geom[0]):
            for k in range(3):
                self.geom.append(np.ones(npts)*reader.geom[k])

        else:
            for k in range(3):
                temp = reader.geom[k]
                self.geom.append(temp[self.yi-1,self.xi-1])

    def write(self, prefix, rsp=False):
        '''Write outputs to prefix.txt and prefix.rsp.'''

        logger.info('Output   File - %s.txt'%(prefix))
        fout = open('%s.txt'%(prefix),'w')
        fout.write('Number xind yind east north data err wgt Elos Nlos Ulos\n')
        fout.write('********************************************************\n')

        if rsp:
            logger.info('Location File - %s.rsp'%(prefix))
            fout2 = open('%s.rsp'%(prefix),'w')
            fout2.write('xind yind UpperLeft-x,y DownRight-x,y\n')
            fout2.write('********************************************************\n')

        npts = len(self.xi)

        for i in range(npts):
            str = "%4d %4d %4d %3.5f %3.6f %3.6f %2.5f %6d %2.5f %2.5f %2.5f\n" \
               % (int(i+1),int(self.xi[i]),int(self.yi[i]),self.ecoord[i],self.ncoord[i],self.zi[i], \
            self.zerr[i],self.wgt[i],self.geom[0][i],self.geom[1][i],self.geom[2][i])
            fout.write(str)
            if rsp:
                str = "%4d %4d %4d %4d %4d %4d\n" \
                    %(int(self.xi[i]),int(self.yi[i]),int(self.ulx[i]),int(self.uly[i]),int(self.drx[i]),int(self.dry[i]))
                fout2.write(str)

        fout.close()
        if rsp:
            fout2.close()
            

    def iter_res(self, xmin0, xmax0, ymin0, ymax0, reader):
        '''Iterative resampler.'''
        xmid = np.floor((xmax0+xmin0)*0.5)
        ymid = np.floor((ymax0+ymin0)*0.5)

        xc = np.zeros((4,2))
        xc[0,:] = [xmin0,xmid]
        xc[1,:] = [xmin0,xmid]
        xc[2,:] = [xmid+1,xmax0]
        xc[3,:] = [xmid+1,xmax0]

        yc = np.zeros((4,2))
        yc[0,:] = [ymin0,ymid]
        yc[1,:] = [ymid+1,ymax0]
        yc[2,:] = [ymid+1,ymax0]
        yc[3,:] = [ymin0,ymid]

        for iframe in range(4):
            xmin = int(xc[iframe,0])
            xmax = int(xc[iframe,1])
            nx = xmax-xmin+1
            ymin = int(yc[iframe,0])
            ymax = int(yc[iframe,1])
            ny = ymax-ymin+1
            npix = nx * ny
            temp = reader.phs[ymin-1:ymax,:]
            tframe = temp[:,xmin-1:xmax]
            [yg,xg] = np.where(np.isnan(tframe) == False)
            ng = yg.size
            gframe = tframe[yg,xg]
            if (ng == 0):
                tstd = np.nan;
            elif ((npix < 100) & ((ng/npix) < 0.9)):
                tstd = np.nan;
            elif (self.reslev < self.minres):
                tstd = 2*self.thresh
            elif (ng <= 10):
                tstd = np.std(gframe)
            else:
                if self.usevar:
                    tstd = np.std(gframe)
                else:
                    A = np.zeros((ng,4))
                    A[:,0] = 1
                    A[:,1] = xg
                    A[:,2] = yg
                    A[:,3] = xg*yg
                    cffs = np.linalg.lstsq(A,gframe,rcond=None)
                    gframe2 = gframe - np.dot(A,cffs[0])
                    tstd = np.std(gframe2)

            if(np.isnan(tstd)):
                pass
            elif (((tstd < self.thresh) & (ny <= self.maxsize) & (nx <=self.maxsize)) | (nx <= self.minsize) | (ny <= self.minsize)):
                xg = xg+xmin
                yg = yg+ymin
                xgood = np.floor(0.5*(xmin+xmax))
                ygood = np.floor(0.5*(ymin+ymax))
                dist_to_good = np.sqrt((xg - xgood)*(xg-xgood) + (yg-ygood)*(yg-ygood))
                mdist = dist_to_good.min()
                target_idx = np.where(dist_to_good == mdist)
                idx = target_idx[0][0]
                xgood = xg[idx]
                ygood = yg[idx]
                zgood = np.median(gframe)

                self.xi.append(xgood)
                self.yi.append(ygood)
                self.zi.append(zgood)

                self.counter.increment()

                self.ulx.append(xmin) 
                self.uly.append(ymin)
                self.drx.append(xmax) 
                self.dry.append(ymax)

                self.zerr.append(tstd)
                self.wgt.append(ng)
                self.ptscounter += 1
            else:
                self.reslev = self.reslev+1
                self.iter_res(xmin, xmax, ymin, ymax, reader)
                self.reslev = self.reslev-1 

        return


    def load_rsp(self,fname):
        '''Loads a predefined resampling grid from an RSP file.'''
        fin = open(fname,'r')
        lin = fin.readline()   ######Reading 2 lines of header
        lin = fin.readline()
       
        for lin in fin:
            vals = lin.split()
            self.xi.append(np.int(vals[0]))
            self.yi.append(np.int(vals[1]))
            self.ulx.append(np.int(vals[2]))
            self.uly.append(np.int(vals[3]))
            self.drx.append(np.int(vals[4]))
            self.dry.append(np.int(vals[5]))

        fin.close()

        self.xi = np.array(self.xi,dtype=np.int)
        self.yi = np.array(self.yi,dtype=np.int)
        npts = len(self.xi)
        logger.info('Number of samples = %d'%(npts))


    def resamplewithrsp(self,reader):
        '''Resamples data in reader with a predefined map.'''

        npts = len(self.xi)
        for kk in range(npts):
            tframe = reader.phs[self.uly[kk]-1:self.dry[kk],self.ulx[kk]-1:self.drx[kk]]
            [yg,xg] = np.where(np.isnan(tframe)==False)
            ng = yg.size

            if ng >0:
                gframe = tframe[yg,xg]
                z = np.median(gframe)
                if self.usevar:
                    err = np.std(gframe)
                else:
                    A = np.zeros((ng,4))
                    A[:,0] = 1
                    A[:,1] = xg
                    A[:,2] = yg
                    A[:,3] = xg*yg

                    cffs = np.linalg.lstsq(A,gframe)
                    gframe2 = gframe - np.dot(A,cffs[0])
                    err = np.std(gframe2)

                self.zi.append(z)
                self.zerr.append(err)
                self.wgt.append(ng)

            else:
                self.zi.append(np.nan)
                self.zerr.append(np.nan)

        self.zi = np.array(self.zi,dtype=np.float)
        self.zerr = np.array(self.zerr, dtype=np.float)
            
        logger.info('Preparing geometry products')
        self.ecoord = reader.x[self.xi-1]
        self.ncoord = reader.y[self.yi-1]

        if np.isscalar(reader.geom[0]):
            for k in range(3):
                self.geom.append(np.ones(npts)*reader.geom[k])

        else:
            for k in range(3):
                temp = reader.geom[k]
                self.geom.append(temp[self.yi-1,self.xi-1])


############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
