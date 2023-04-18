import numpy as np
from .. import util
from . import Source
from .okada import Okada

class CRFault(Source):
    def __init__(self, data, ori=None, segs=None, lw=None):
        if lw==None:
            self.lw=1
        else:
            self.lw=lw
            
        if ori==None:
            self.ori=1
        else:
            self.ori=ori
            
        if segs==None:
            self.segs=6
        else:
            self.segs=segs
        
        super().__init__(data)
    
    def get_source_id(self):
        return "CFRing"
    
    def time_dependent(self):
        return False

    def print_model(self, x):
        print("CFRing")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\ts= %f" % x[3])
        print("\to= %f" % x[4])
        print("\ta= %f" % x[5])
        print("\tb= %f" % x[6])
        print("\tW= %f" % x[7])
        print("\tstrike= %f" % x[8])
        
    def set_parnames(self):
        self.parameters=("xcen","ycen","depth","slip","opening","smajor","sminor","width","strike")

    # =====================
    # Forward Models
    # =====================
    def get_parameters(self,xcen,ycen,a,b,strike):
        angles=np.linspace(0,360,self.segs+1)
        params=np.ones((len(angles)-1,8))
        for i,angle in enumerate(angles[1::]):
            pangle=angles[i]
            print(pangle,angle)
            xc1 = a*np.cos(np.radians(pangle))*np.cos(np.radians(90-strike)) -\
                 b*np.sin(np.radians(pangle))*np.sin(np.radians(90-strike)) + xcen
            yc1 = a*np.cos(np.radians(pangle))*np.sin(np.radians(90-strike)) +\
                 b*np.sin(np.radians(pangle))*np.cos(np.radians(90-strike)) + ycen
            xc2 = a*np.cos(np.radians(angle))*np.cos(np.radians(90-strike)) -\
                 b*np.sin(np.radians(angle))*np.sin(np.radians(90-strike)) + xcen
            yc2 = a*np.cos(np.radians(angle))*np.sin(np.radians(90-strike)) +\
                 b*np.sin(np.radians(angle))*np.cos(np.radians(90-strike)) + ycen
            sxc = np.mean([xc1,xc2])
            syc = np.mean([yc1,yc2])
            slength=np.sqrt((xc2-xc1)**2+(yc2-yc1)**2)
            sstrike=90-np.degrees(np.arctan2(yc2-yc1,xc2-xc1))
            if sstrike<0:
                sstrike=360+sstrike
            params[i,0]=xc1
            params[i,1]=yc1
            params[i,2]=xc2
            params[i,3]=yc2
            params[i,4]=sxc
            params[i,5]=syc
            params[i,6]=slength
            params[i,7]=sstrike
        return params
    
    def get_greens(self,x,y, xcen, ycen, depth, a, b, width, strike):
        if self.ori==1:
            rake=90.0
        else:
            rake=-90.0
        
        ok = Okada(None)
        G=np.zeros((len(x)*3,self.segs*self.lw*2))
        params=self.get_parameters(xcen,ycen,a,b,strike)
        for i in range(params.shape[0]):
            xc1=params[i,0]
            yc1=params[i,1]
            xc2=params[i,2]
            yc2=params[i,3]
            sxc=params[i,4]
            syc=params[i,5]
            slength=params[i,6]
            sstrike=params[i,7]
            
            ux,uy,uz=ok.model(x,y, sxc, syc, depth, slength, width, 1.0, 0.0, sstrike, 90.0, rake)
            G[0:len(x),i]=ux
            G[len(x):2*len(x),i]=uy
            G[2*len(x):3*len(x),i]=uz
            
            ux,uy,uz=ok.model(x,y, sxc, syc, depth, slength, width, 0.0, 1.0, sstrike,90.0, 0.0)
            G[0:len(x),self.segs*self.lw+i]=ux
            G[len(x):2*len(x),self.segs*self.lw+i]=uy
            G[2*len(x):3*len(x),self.segs*self.lw+i]=uz
        
        return G #returns tuple
        
    def model(self,x,y, xcen, ycen, depth, slip, opening, a, b, width, strike):       
        """
        Calculates surface deformation based on point pressure source
        References: Mogi 1958, Segall 2010 p.203

        Args:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)

        Kwargs:
        -----------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to point (m)
        dV: change in volume (m^3)
        nu: poisson's ratio for medium

        Returns:
        -------
        (ux, uy, uz)


        Examples:
        --------

        """
        if isinstance(slip,(list,np.ndarray)):
            if not len(slip)==self.segs*self.lw:
                print('The slip elements do not agree with patches. Taking the mean...')
                slip=np.ones((len(self.segs*self.lw),))*np.mean(slip)
        else:
            slip=np.ones((self.segs*self.lw,))*slip
                
        if isinstance(opening,(list,np.ndarray)):
            if not len(opening)==self.segs*self.lw:
                print('The opening elements do not agree with patches. Taking the mean...')
                opening=np.ones((len(self.segs*self.lw),))*np.mean(opening)
        else:
            opening=np.ones((self.segs*self.lw,))*opening
        
        G=self.get_greens(x,y, xcen, ycen, depth, a, b, width, strike)
        model=np.concatenate((slip,opening))
        
        data=G@model
        
        ux=data[0:len(x)]
        uy=data[len(x):2*len(x)]
        uz=data[2*len(x):3*len(x)]
        
        return ux,uy,uz
