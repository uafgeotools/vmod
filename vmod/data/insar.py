"""
Class to implement interface to geodetic data types used in source inversions

Author: Mario Angarita
Date: 9/23/2022
"""

from . import Data
from vmod import util
import numpy as np

class Insar(Data):
    def __init__(self):
        self.az=None
        self.inc=None
        self.los=None
        super().__init__()

    def add_az(self,az):
        self.assert_size(az,'az')
        self.az=az
    
    def add_inc(self,inc):
        self.assert_size(inc,'inc')
        self.inc=inc
        
    def add_vecs(self,az,inc):
        self.add_az(az)
        self.add_inc(inc)
        
    def add_los(self,los):
        self.add_comp(los,'los')
        self.los=los
        
    def add_ref(self,ref):
        if self.xs is None or self.ys is None or self.los is None:
            raise Exception('You cannot add a reference to an incomplete dataset')
        else:
            names=np.array([str(self.xs[i])+','+str(self.ys[i]) for i in range(len(self.xs))])
            posmin=np.argmin((self.xs-ref[0])**2+(self.ys-ref[1])**2)
            self.utmz=[ref[2],ref[3],ref[4]]
            if self.names is None:
                self.add_names(names)
            super().add_ref(names[posmin])
    
    def importcsv(self,csvfile):
        xs,ys,azs,lks,los,elos,ref=util.read_insar_csv(csvfile,trans=True)
        self.add_vecs(azs,lks)
        self.add_xs(xs)
        self.add_ys(ys)
        self.add_los(los)
        self.add_err(elos)
        self.add_ref(ref)
        
    def from_model3d(self,func,unravel=True):
        ux,uy,uz=func(self.xs,self.ys)
        assert isinstance(self.inc,(list,np.ndarray)) and isinstance(self.az,(list,np.ndarray)), 'The look angles have not been defined'
        los=ux*np.sin(self.inc)*np.cos(self.az)-uy*np.sin(self.inc)*np.sin(self.az)-uz*np.cos(self.inc)
        los_ref=self.reference_dataset(los,unravel)
        return (-los_ref,)
        