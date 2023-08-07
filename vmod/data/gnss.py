import numpy as np
from . import Data
from vmod import util

class Gnss(Data):
    """
    Class that represents the GNSS geodetic datatype
    
    Attributes:
        ux: deformation in the east (m)
        uy: deformation in the north (m)
        uz: deformation in the vertical (m)
        errx: uncertainty in the east deformation (m)
        erry: uncertainty in the north deformation (m)
        errz: uncertainty in the vertical deformation (m)
    """
    def __init__(self):
        self.ux=None
        self.uy=None
        self.uz=None
        self.errx=None
        self.erry=None
        self.errz=None
        super().__init__()    
        
    def add_ux(self,ux):
        """
        Adds the deformation in the east component
        
        Parameters:
            ux (array): deformation in the east component
        """
        self.add_comp(ux,'ux')
        self.ux=ux
    
    def add_errx(self,errx):
        """
        Adds the uncertainty in the deformation for the east component
        
        Parameters:
            errx (array): uncertainty in the east component
        """
        self.add_errcomp(errx,'ux')
        self.errx=errx
    
    def add_uy(self,uy):
        """
        Adds the deformation in the north component
        
        Parameters:
            uy (array): deformation in the north component
        """
        self.add_comp(uy,'uy')
        self.uy=uy
        
    def add_erry(self,erry):
        """
        Adds the uncertainty in the deformation for the north component
        
        Parameters:
            erry (array): uncertainty in the north component
        """
        self.add_errcomp(erry,'uy')
        self.erry=erry
        
    def add_uz(self,uz):
        """
        Adds the deformation in the vertical component
        
        Parameters:
            uz (array): deformation in the vertical component
        """
        self.add_comp(uz,'uz')
        self.uz=uz
        
    def add_errz(self,errz):
        """
        Adds the uncertainty in the deformation for the vertical component
        
        Parameters:
            errz (array): uncertainty in the vertical component
        """
        self.add_errcomp(errz,'uz')
        self.errz=errz
        
    def get_index(self,name):
        """
        Defines the order for the components in the data array
        
        Returns:
            index (int): component's index in the data array
        """
        if name=='ux':
            return 0
        elif name=='uy':
            return 1
        elif name=='uz':
            return 2
    
    def add_data(self,ux,uy,uz):
        """
        Adds the three components at the same time
        
        Returns:
            ux (array): deformation in the east component
            uy (array): deformation in the north component
            uz (array): deformation in the vertical component
        """
        self.add_ux(ux)
        self.add_uy(uy)
        self.add_uz(uz)
        super().add_data(np.concatenate((ux,uy,uz)))
    
    def add_err(self,errx,erry,errz):
        """
        Adds uncertainties for the three components at the same time
        
        Returns:
            errx (array): uncertainty in the east component
            erry (array): uncertainty in the north component
            errz (array): uncertainty in the vertical component
        """
        self.add_errx(errx)
        self.add_erry(erry)
        self.add_errz(errz)
        super().add_err(np.concatenate((errx,erry,errz)))
    
    def importcsv(self,csvfile,ori=None):
        """
        Imports csv file into a gnss object
        
        Parameters:
            csvfile (str): Path to the csv file
            ori (array): lon/lat for the origin coordinate in the projection
        """
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
        """
        Uses the function from the forward model to compute the deformation in the three components
        
        Parameters:
            func: 3d displacements forward model function
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
            
        Returns:
            model (array): array containing the components
        """
        ux,uy,uz=func(self.xs,self.ys)
            
        model=()
        
        if isinstance(self.ux,(list,np.ndarray)):
            model=(*model,ux)
        if isinstance(self.uy,(list,np.ndarray)):
            model=(*model,uy)
        if isinstance(self.uz,(list,np.ndarray)):
            model=(*model,uz)
        
        return model