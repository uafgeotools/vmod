import numpy as np
from .. import util
from . import Source
from .okada import Okada

class Regsill(Source):
    """
    Class to represent a regularized sill using dislocation patches from Okada (1985) model.

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    typ : str
        type of dislocation, open for tensile and slip for fault
    ln : 
    """
    def __init__(self, data, typ=None, ln=None, wn=None):
        if typ==None or typ=='open':
            self.typ='open'
        else:
            self.typ='slip'
        if ln==None:
            self.ln=1
        else:
            self.ln=ln
        if wn==None:
            self.wn=1
        else:
            self.wn=wn
        
        super().__init__(data)
        
    def time_dependent(self):
        return False
    
    def set_parnames(self):
        self.parameters=("xcen","ycen","depth","length","width","strike","dip","slips/openings")
        
    def rotate_xyz(self,xcen,ycen,depth,length,width,strike,dip):
    # this function calculates the coordinates of four corners of a triangle
    # (in this RegSill class, the triangle represents a fault plane)
    # inputs: parameters defining the triangle
    #   - xcen, ycen, depth [m]: location of fault center
    #   - strike, dip [deg]: fault orientation parameters
    #   - length [m]: fault length (along strike line)
    #   - width [m]: fault width (along dip line)
    # output: coordinates of top-left, bottom-left, bottom=right, and top-right corners

        zcen  = -depth
        srad  = math.radians(strike)
        drad  = math.radians(dip)
        
        # top-left corner
        x1 = xcen - length/2.0*math.sin(srad) - width/2.0*math.cos(drad)*math.cos(srad)
        y1 = ycen - length/2.0*math.cos(srad) + width/2.0*math.cos(drad)*math.sin(srad)
        z1 = zcen + width/2.0*math.sin(drad)
        
        # bottom-left corner
        x2 = xcen - length/2.0*math.sin(srad) + width/2.0*math.cos(drad)*math.cos(srad)
        y2 = ycen - length/2.0*math.cos(srad) - width/2.0*math.cos(drad)*math.sin(srad)
        z2 = zcen - width/2.0*math.sin(drad)
        
        # bottom-right corner
        x3 = xcen + length/2.0*math.sin(srad) + width/2.0*math.cos(drad)*math.cos(srad)
        y3 = ycen + length/2.0*math.cos(srad) - width/2.0*math.cos(drad)*math.sin(srad)
        z3 = zcen - width/2.0*math.sin(drad)
        
        # top-right corner
        x4 = xcen + length/2.0*math.sin(srad) - width/2.0*math.cos(drad)*math.cos(srad)
        y4 = ycen + length/2.0*math.cos(srad) + width/2.0*math.cos(drad)*math.sin(srad)
        z4 = zcen + width/2.0*math.sin(drad)
        
        return [x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]
        
    def get_centers(self,xcen,ycen,depth,length,width,strike,dip):
    # this function calculates the center coordinates of all fault patches
    # the order of patches to be calculated is from left to right along length, from top to bottom along width
    # for example, for a nl=3,nw=2 fault, the order of calculation is
    #     [1] [2] [3]
    #     [4] [5] [6]
    #
    # calculation is based on the following formula
    # x_{i,j} = x_{top-left} + i/nl*( x_{top-right} - x_{top-left} ) + j/nw*( x_{bottom-left} - x_{top-left} )
    
        ln = self.ln
        wn = self.wn
        [x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4] = self.rotate_xyz(xcen,ycen,depth,length,width,strike,dip)
        
        xcenters,ycenters,zcenters=[],[],[]
        for j in range(wn):
            for i in range(ln):
                xcenters.append( x1 + (x4-x1)/ln*(i + 0.5) + ((x2-x1)/wn)*(j+0.5) )
                ycenters.append( y1 + (y4-y1)/ln*(i + 0.5) + ((y2-y1)/wn)*(j+0.5) )
                zcenters.append( z1 + (z4-z1)/ln*(i + 0.5) + ((z2-z1)/wn)*(j+0.5) )
        return xcenters,ycenters,zcenters
    
    def get_laplacian(self,xcen,ycen,depth,length,width,strike,dip,ln,wn):
        xcs,ycs,zcs=self.get_centers(xcen,ycen,depth,length,width,strike,dip,ln,wn)
        L=np.zeros((ln*wn,ln*wn))
        for i in range(len(xcs)):
            dist=(np.array(xcs)-xcs[i])**2+(np.array(ycs)-ycs[i])**2+(np.array(zcs)-zcs[i])**2
            pos=np.argsort(dist)
            #print(dist[pos[0:5]])
            if dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]] and dist[pos[1]]==dist[pos[4]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
                L[i,pos[4]]=1
            elif dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
            elif dist[pos[1]]==dist[pos[2]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
        return L
        
    def get_reg_sill(self,xcen,ycen,depth,length,width,strike,dip,ops):
        ln=self.ln
        wn=self.wn
        xs,ys,zs=self.get_centers(xcen,ycen,depth,length,width,strike,dip)
        oks=[]
        params=[]
        ln=self.ln
        wn=self.wn
        dat=self.data
        for i in range(len(xs)):
            oki = Okada(dat)
            oki.set_type('open')
            #Initial parameters [xcen,ycen,depth,length,width,opening,strike,dip]
            oki.set_bounds(low_bounds = [0, 0, 1e3, 1e3, 1e3,10.0,1.0,1.0], high_bounds = [0, 0, 1e3, 1e3, 1e3,10.0,1.0,1.0])
            oks.append(oki)
            params+=[xs[i],ys[i],-zs[i],length/ln,width/wn,ops[i],strike,dip]

        return oks,params
    
    def get_greens(self,xcen,ycen,depth,length,width,strike,dip):
        ln=self.ln
        wn=self.wn
        x=self.data.xs
        y=self.data.ys
        
        oki=Okada(self.data)
        #print('Tipo',self.typ)
        oki.set_type(self.typ)
        
        xcs,ycs,zcs=self.get_centers(xcen,ycen,depth,length,width,strike,dip)
        xo=[xcen,ycen,depth,length,width,1,strike,dip]
        
        defo=oki.forward(xo)
        
        slength=length/ln
        swidth=width/wn
        
        if self.typ=='open':
            op=1
            sl=0
        else:
            op=0
            sl=1
        G=np.zeros((len(defo),ln*wn))
        for i in range(len(xcs)):
            if self.typ=='open':
                xp=[xcs[i],ycs[i],-zcs[i],slength,swidth,op,strike,dip]
            else:
                xp=[xcs[i],ycs[i],-zcs[i],slength,swidth,sl,strike,dip]
            defo=oki.forward(xp)
            G[:,i]=defo
        return G
    
    def get_laplacian(self,xcen,ycen,depth,length,width,strike,dip):
        ln=self.ln
        wn=self.wn
        xcs,ycs,zcs=self.get_centers(xcen,ycen,depth,length,width,strike,dip)
        L=np.zeros((ln*wn,ln*wn))
        for i in range(len(xcs)):
            dist=(np.array(xcs)-xcs[i])**2+(np.array(ycs)-ycs[i])**2+(np.array(zcs)-zcs[i])**2
            pos=np.argsort(dist)
            if dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]] and dist[pos[1]]==dist[pos[4]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
                L[i,pos[4]]=1
            elif dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
            elif dist[pos[1]]==dist[pos[2]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
        return L
    
    def model(self,x,y,xcen,ycen,depth,length,width,strike,dip,ops):
        G=self.get_greens(xcen,ycen,depth,length,width,strike,dip)
        data=G@model
        
        ux=data[0:len(x)]
        uy=data[len(x):2*len(x)]
        uz=data[2*len(x):3*len(x)]
        
        return ux,uy,uz
        
    def plot_patches(self,length,width,ops):
    # this function plot the 3D fault into a 2D plane
    # input parameter ops is a 1D array for openings. its order is described in self.get_centers()
        import matplotlib
        import matplotlib.pyplot as plt
        
        ln = self.ln
        wn = self.wn

        ok1 = Okada(self.data)
        ok1.set_type('open')
        # lay the fault flat for plotting, length is on east-west direction, and width in north-south
        xs,ys,zs = self.get_centers(0,0,0,length,width,90,0)

        patches = []
        for i in range(len(xs)):
            # rect takes the bottom-left coordinates and length/width for a patch 
            rect = matplotlib.patches.Rectangle((xs[i]-length/(2*ln),ys[i]-width/(2*wn)), length/ln, width/wn)
            patches.append(rect)
        values = ops
        
        # viridis for all-positive fault, bwr for bi-symbol opennings
        if (values.min()*values.max())<0:
            cmap  = 'bwr'
            maxop = np.maximum( np.abs(values.min()) , values.max() )
            norm  = plt.Normalize(-maxop, maxop)
        else:
            cmap = 'viridis'
            norm = plt.Normalize(values.min(), values.max())

        coll = matplotlib.collections.PatchCollection(patches, cmap=cmap,
                                                      norm=norm, match_original = True)
        coll.set_array(values)
        
        fig, ax = plt.subplots()
        polys   = ax.add_collection(coll)
        cbar    = fig.colorbar(polys, label='Opening (m)', orientation='horizontal')
        
        plt.gca().set_aspect('equal')
        plt.xlim(-length/2,length/2)
        plt.ylim(-width/2,width/2)

        plt.show()
        print( 'plotting length (m): '+str(length) )
        print( 'plotting width (m):'+str(width) )
