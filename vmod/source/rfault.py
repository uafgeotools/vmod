import numpy as np
from .. import util
from . import Source
from .okada import Okada

class RFault(Source):
    """
    A class used to represent a Ring Fault model

    ...

    Attributes
    ----------
    data : Data
        Data object used as input
    parameters : array
        names for the parameters in the model
    ori : int
        orientation for the slip patches in the caldera ring fault, ori=1 for caldera uplift, ori=-1 for caldera collapse (default 1)
    lw : int
        number of patches in the vertical orientation
    segs : int
        number of horizontal segments (default 6)
    """
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
        """
        The function defining the name for the model.
          
        Returns:
            str: Name of the model.
        """
        return "RFault"

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("RFault")
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
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","slip","opening","smajor","sminor","width","strike")

    # =====================
    # Forward Models
    # =====================
    def get_parameters(self,xcen,ycen,a,b,strike):
        """
        Function defining parameters for the patches that compose the ring fault.
        
        Parameters:
           xcen (float) : x coordinate for the center of the caldera in meters.
           ycen (float) : y coordinate for the center of the caldera in meters.
           a (float) : semimajor axis for the caldera.
           b (float) : semiminor axis for the caldera.
           strike (deg) : azimuth for the caldera clockwise is positive from North.
        
        Returns:
            params (array) : Matrix where each column has the parameters for an Okada patch.
        """
        angles=np.linspace(0,360,self.segs+1)
        params=np.ones((len(angles)-1,8))
        for i,angle in enumerate(angles[1::]):
            pangle=angles[i]
            #print(pangle,angle)
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
        """
        Green functions for the ring fault.
        
        Parameters:
           xcen (float) : x coordinate for the center of the caldera in meters.
           ycen (float) : y coordinate for the center of the caldera in meters.
           depth (float) : depth for the Okada patches.
           width (float) : vertical width for the Okada patches.
           a (float) : semimajor axis for the caldera.
           b (float) : semiminor axis for the caldera.
           strike (deg) : azimuth for the caldera clockwise is positive from North.
        
        Returns:
            G (array) : G-Matrix containing the green functions.
        """
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
            
            ok.set_type('slip')
            ux,uy,uz=ok.model(x,y, sxc, syc, depth, slength, width, 1.0, sstrike, 90.0, rake)
            G[0:len(x),i]=ux
            G[len(x):2*len(x),i]=uy
            G[2*len(x):3*len(x),i]=uz
            
            ok.set_type('open')
            ux,uy,uz=ok.model(x,y, sxc, syc, depth, slength, width, 1.0, sstrike,90.0)
            G[0:len(x),self.segs*self.lw+i]=ux
            G[len(x):2*len(x),self.segs*self.lw+i]=uy
            G[2*len(x):3*len(x),self.segs*self.lw+i]=uz
        
        return G #returns tuple
        
    def model(self,x,y, xcen, ycen, depth, slip, opening, a, b, width, strike):       
        """
        Green functions for the ring fault.
        
        Parameters:
           xcen (float) : x coordinate for the center of the caldera in meters.
           ycen (float) : y coordinate for the center of the caldera in meters.
           depth (float) : depth for the Okada patches.
           slip (float) : slip in the Okada patches in meters.
           opening (float) : opening in the Okada patches in meters.
           a (float) : semimajor axis for the caldera.
           b (float) : semiminor axis for the caldera.
           width (float) : vertical width for the Okada patches.
           strike (deg) : azimuth for the caldera clockwise is positive from North.
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
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
