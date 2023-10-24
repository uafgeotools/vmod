import numpy as np
from .. import util
from . import Source
from .okada import Okada
from ..data import Gnss

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
        cx=xcen
        cy=ycen
        cz=-depth
        wp=width*np.cos(np.radians(dip))
        wr=width*np.sin(np.radians(dip))
        l=length
        phi=strike
        x1 = cx + wp/2 * np.cos(np.radians(phi)) - l/2 * np.sin(np.radians(phi))
        y1 = cy + wp/2 * np.sin(np.radians(phi)) + l/2 * np.cos(np.radians(phi))
        z1 = cz - wr/2
        x2 = cx - wp/2 * np.cos(np.radians(phi)) - l/2 * np.sin(np.radians(phi))
        y2 = cy - wp/2 * np.sin(np.radians(phi)) + l/2 * np.cos(np.radians(phi))
        z2 = cz + wr/2
        x3 = cx - wp/2 * np.cos(np.radians(phi)) + l/2 * np.sin(np.radians(phi))
        y3 = cy - wp/2 * np.sin(np.radians(phi)) - l/2 * np.cos(np.radians(phi))
        z3 = cz + wr/2
        x4 = cx + wp/2 * np.cos(np.radians(phi)) + l/2 * np.sin(np.radians(phi))
        y4 = cy + wp/2 * np.sin(np.radians(phi)) - l/2 * np.cos(np.radians(phi))
        z4 = cz - wr/2
        
        return [x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]
        
    def get_centers(self,xcen,ycen,depth,length,width,strike,dip):
        xc=xcen
        yc=ycen
        zc=-depth
        ln=self.ln
        wn=self.wn
        lslice=length/ln
        wslice=width/wn
        fwc=xcen-width/2+width/(2*wn)
        flc=ycen-length/2+length/(2*ln)
        #print(fwc,flc)
        xcs,ycs,zcs=[],[],[]
        if wn%2==0:
            wi=wn/2
        else:
            wi=(wn-1)/2
            
        if ln%2==0:
            li=ln/2
        else:
            li=(ln-1)/2
            
        for i in range(int(wi)):
            wfake=2*np.abs(fwc-xcen+float(i)*wslice)
            for j in range(int(li)):
                lfake=2*np.abs(flc-ycen+float(j)*lslice)
                xs,ys,zs=self.rotate_xyz(xcen,ycen,depth,lfake,wfake,strike,dip)
                for x in xs:
                    xcs.append(x)
                for y in ys:
                    ycs.append(y)
                for z in zs:
                    zcs.append(z)
        print('Puntos 1',len(xcs),wn%2,ln%2)
        if not ln%2==0:
            for j in range(int(li)):
                wfake=0
                lfake=2*np.abs(flc-ycen+float(j)*lslice)
                xs,ys,zs=self.rotate_xyz(xcen,ycen,depth,lfake,wfake,strike,dip)
                for x in xs[1:3]:
                    xcs.append(x)
                for y in ys[1:3]:
                    ycs.append(y)
                for z in zs[1:3]:
                    zcs.append(z)
        print('Puntos 2',len(xcs))
        if not wn%2==0:
            for i in range(int(wi)):
                wfake=2*np.abs(fwc-xcen+float(i)*wslice)
                lfake=0
                xs,ys,zs=self.rotate_xyz(xcen,ycen,depth,lfake,wfake,strike,dip)
                for x in xs[0:2]:
                    xcs.append(x)
                for y in ys[0:2]:
                    ycs.append(y)
                for z in zs[0:2]:
                    zcs.append(z)
        print('Puntos 3',len(xcs))
        if (not wn%2==0) and (not ln%2==0):
            print('Ninguno')
            xcs.append(xcen)
            ycs.append(ycen)
            zcs.append(-depth)
        print('Puntos 4',len(xcs))
        return xcs,ycs,zcs
    
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
        
        dat1 = Gnss()
        dat1.add_xs(xs)
        dat1.add_ys(ys)
        
        dat1.add_data(xs*0,xs*0,xs*0)
        
        oki=Okada(dat1)
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
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        
        ln=self.ln
        wn=self.wn

        ok1 = Okada(self.data)
        ok1.set_type('open')
        xs,ys,zs=self.get_centers(0,0,0,length,width,0,0)

        patches=[]
        fig, ax = plt.subplots()
        for i in range(len(xs)):
            rect = matplotlib.patches.Rectangle((xs[i]-length/(2*ln),ys[i]-width/(2*wn)), length/ln, width/wn)
            patches.append(rect)

        # values as numpy array
        values = ops

        # define the norm 
        norm = plt.Normalize(values.min(), values.max())
        coll = matplotlib.collections.PatchCollection(patches, cmap='viridis',
                                                      norm=norm, match_original = True)

        coll.set_array(values)
        polys = ax.add_collection(coll)
        fig.colorbar(polys, label='Opening(m)')

        plt.xlim(-width/2,width/2)
        plt.ylim(-length/2,length/2)

        plt.show()
