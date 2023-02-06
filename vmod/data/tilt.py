"""
Class to implement tilt geodetic data

Author: Mario Angarita
Date: 9/23/2022
"""

import numpy as np
from . import Data
import scipy

class Tilt(Data):
    def __init__(self):
        self.names=None
        self.dx=None
        self.dy=None
        self.errx=None
        self.erry=None
        self.azx=None
        super().__init__()
    
    def add_azx(self,azx):
        self.assert_size(azx,'azx')
        self.azx=azx
        
    def add_dx(self,dx):
        self.add_comp(dx,'dx')
        self.dx=dx
    
    def add_errx(self,errx):
        self.add_errcomp(errx,'dx')
        self.errx=errx
    
    def add_dy(self,dy):
        self.add_comp(dy,'dy')
        self.dy=dy
        
    def add_erry(self,erry):
        self.add_errcomp(erry,'dy')
        self.erry=erry
        
    def get_index(self,name):
        if name=='dx':
            return 0
        elif name=='dy':
            return 1
        
    def ref_possible(self):
        return False
    
    def add_data(self,dx,dy):
        self.add_ux(dx)
        self.add_uy(dy)
        super().add_data(self,np.concatenate((dx,dy)))
    
    def add_err(self,errx,erry):
        self.add_errx(errx)
        self.add_erry(erry)
        super().add_err(self,np.concatenate((errx,erry)))
    
    def from_model3d(self,func,unravel=True):
        if 'tilt' in func.__name__:
            dx,dy=func(self.xs,self.ys)
        else:
            uzx= lambda xpos: func(xpos,self.ys)[2]
            uzy= lambda ypos: func(self.xs,ypos)[2]

            dx=-scipy.misc.derivative(uzx,self.xs,dx=1e-6)
            dy=-scipy.misc.derivative(uzy,self.ys,dx=1e-6)
        
        model=()
        if isinstance(self.dx,(list,np.ndarray)):
            model=(*model,dx)
        if isinstance(self.dy,(list,np.ndarray)):
            model=(*model,dy)
        
        return model