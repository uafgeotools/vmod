"""
Class to implement GNSS geodetic data

Author: Mario Angarita
Date: 9/23/2022
"""

import numpy as np
from . import Data
from vmod import util

class Gnss(Data):
    def __init__(self):
        self.ux=None
        self.uy=None
        self.uz=None
        self.errx=None
        self.erry=None
        self.errz=None
        super().__init__()    
        
    def add_ux(self,ux):
        self.add_comp(ux,'ux')
        self.ux=ux
    
    def add_errx(self,errx):
        self.add_errcomp(errx,'ux')
        self.errx=errx
    
    def add_uy(self,uy):
        self.add_comp(uy,'uy')
        self.uy=uy
        
    def add_erry(self,erry):
        self.add_errcomp(erry,'uy')
        self.erry=erry
        
    def add_uz(self,uz):
        self.add_comp(uz,'uz')
        self.uz=uz
        
    def add_errz(self,errz):
        self.add_errcomp(errz,'uz')
        self.errz=errz
        
    def get_index(self,name):
        if name=='ux':
            return 0
        elif name=='uy':
            return 1
        elif name=='uz':
            return 2
    
    def add_data(self,ux,uy,uz):
        self.add_ux(ux)
        self.add_uy(uy)
        self.add_uz(uz)
        super().add_data(self,np.concatenate((ux,uy,uz)))
    
    def add_err(self,errx,erry,errz):
        self.add_errx(errx)
        self.add_erry(erry)
        self.add_errz(errz)
        super().add_err(self,np.concatenate((errx,erry,errz)))
    
    def importcsv(self,csvfile,ori=None):
        names,lons,lats,uxs,uys,uzs,euxs,euys,euzs=util.read_gnss_csv(csvfile)
        
        self.add_names(np.array(names))
        
        self.add_lls(lons,lats,ori)
        
        self.add_ux(uxs)
        self.add_uy(uys)
        self.add_uz(uzs)
        
        
        self.add_errx(euxs)
        self.add_erry(euys)
        self.add_errz(euzs)
    
    def from_model3d(self,func,unravel=True):
        
        ux,uy,uz=func(self.xs,self.ys)
            
        model=()
        
        if isinstance(self.ux,(list,np.ndarray)):
            model=(*model,ux)
        if isinstance(self.uy,(list,np.ndarray)):
            model=(*model,uy)
        if isinstance(self.uz,(list,np.ndarray)):
            model=(*model,uz)
        
        return model