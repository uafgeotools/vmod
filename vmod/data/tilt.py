import numpy as np
from . import Data
import scipy

class Tilt(Data):
    """
    Class that represents the tilt geodetic datatype
    
    Attributes:
        dx: inclination in the x-axis (radians)
        dy: inclination in the y-axis (radians)
        errx: uncertainty in the inclination in the x-axis (radians)
        erry: uncertainty in the inclination in the y-axis (radians)
        azx: azimuth for the x-axis, the angle is positive clockwise starting from the north
        delta: Step to calculate the derivate with finite differences
    """
    def __init__(self):
        self.names=None
        self.dx=None
        self.dy=None
        self.errx=None
        self.erry=None
        self.azx=None
        self.delta=1e-6
        super().__init__()

    def set_delta(self,delta):
        """
        Set step to calculate the derivative using finite differences
        
        Parameters:
            delta (float): step
        """
        self.delta=delta
    
    def add_azx(self,azx):
        """
        Adds the azimuth angles for the x-axes
        
        Parameters:
            azx (array): azimuth angles clockwise from north
        """
        self.assert_size(azx,'azx')
        self.azx=azx
        
    def add_dx(self,dx):
        """
        Adds the inclination in the x-axis
        
        Parameters:
            dx (array): inclination in the x-axis
        """
        self.add_comp(dx,'dx')
        self.dx=dx
    
    def add_errx(self,errx):
        """
        Adds the uncertainties in the inclination for the x-axis
        
        Parameters:
            errx (array): uncertainties for the inclination in the x-axis
        """
        self.add_errcomp(errx,'dx')
        self.errx=errx
    
    def add_dy(self,dy):
        """
        Adds the inclination in the y-axis
        
        Parameters:
            dy (array): inclination in the y-axis
        """
        self.add_comp(dy,'dy')
        self.dy=dy
        
    def add_erry(self,erry):
        """
        Adds the uncertainties in the inclination for the y-axis
        
        Parameters:
            erry (array): uncertainties for the inclination in the y-axis
        """
        self.add_errcomp(erry,'dy')
        self.erry=erry
        
    def get_index(self,name):
        """
        Defines the order for the components in the data array
        
        Returns:
            index (int): component's index in the data array
        """
        if name=='dx':
            return 0
        elif name=='dy':
            return 1
        
    def ref_possible(self):
        """
        Overrides the function to specify that the tilt dataset cannot take references
        """
        return False
    
    def add_data(self,dx,dy):
        """
        Adds the two components at the same time
        
        Parameters:
            dx (array): inclination in the x-axis
            dy (array): inclination in the y-axis
        """
        self.add_ux(dx)
        self.add_uy(dy)
        super().add_data(np.concatenate((dx,dy)))
    
    def add_err(self,errx,erry):
        """
        Adds the uncertainties for the two components at the same time
        
        Parameters:
            errx (array): uncertainty for the inclination in the x-axis
            erry (array): uncertainty for the inclination in the y-axis
        """
        self.add_errx(errx)
        self.add_erry(erry)
        super().add_err(np.concatenate((errx,erry)))
    
    def from_model3d(self,func,unravel=True):
        """
        Uses the function from the forward model to compute the inclination in the tiltmeters
        
        Parameters:
            func: tilt displacements forward model function
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
            
        Returns:
            model (array): array containing the components
        """
        #print(func.__name__)
        if 'tilt' in func.__name__:
            dx,dy=func(self.xs,self.ys)
        else:
            uzx= lambda xpos: func(xpos,self.ys)[2]
            uzy= lambda ypos: func(self.xs,ypos)[2]

            dx=-scipy.misc.derivative(uzx,self.xs,dx=self.delta)
            dy=-scipy.misc.derivative(uzy,self.ys,dx=self.delta)
        
        model=()
        if isinstance(self.dx,(list,np.ndarray)):
            model=(*model,dx)
        if isinstance(self.dy,(list,np.ndarray)):
            model=(*model,dy)
        
        return model