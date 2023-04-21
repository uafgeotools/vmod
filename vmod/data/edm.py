"""
Class to implement interface to geodetic data types used in source inversions

Author: Mario Angarita
Date: 9/23/2022
"""

from . import Data
import numpy as np

class Edm(Data):
    def __init__(self):
        self.delta=None
        self.lonsori=None
        self.xorigins=None
        self.lonsend=None
        self.xends=None
        self.latsori=None
        self.yorigins=None
        self.latsend=None
        self.yends=None
        self.zorigins=None
        self.zends=None
        super().__init__()
        
    def add_deltas(self,delta):
        self.add_comp(delta,'delta')
        self.delta=delta
        
    def add_coords(self,coords,obj,origin=True):
        #The assert checks that coords has the same length as obj
        self.assert_origins_ends(coords)
        if obj is None:
            obj=coords
        elif origin:
            if len(obj)==len(coords):
                obj=coords+obj
            else:
                obj[0:len(coords)]=coords
        else:
            if len(obj)==len(coords):
                obj=obj+coords
            else:
                obj[len(coords):2*len(coords)]=coords
        return obj
    
    def add_xorigins(self,origins):
        self.assert_origins_ends(origins)
        self.xorigins=origins
        newcoords=self.add_coords(self.xorigins,self.xs,origin=True)
        self.add_xs(newcoords)
        
    def add_xends(self,ends):
        self.assert_origins_ends(ends)
        self.xends=ends
        newcoords=self.add_coords(self.xends,self.xs,origin=False)
        self.add_xs(newcoords)
        
    def add_yorigins(self,origins):
        self.assert_origins_ends(origins)
        self.yorigins=origins
        newcoords=self.add_coords(self.yorigins,self.ys,origin=True)
        self.add_ys(newcoords)
        
    def add_yends(self,ends):
        self.assert_origins_ends(ends)
        self.yends=ends
        newcoords=self.add_coords(self.yends,self.ys,origin=False)
        self.add_ys(newcoords)
        
    def add_lls_lines(self,lons,lats,origin=True,ori=None):
        if origin==True:
            self.lonsori=lons
            self.latsori=lats
            if not self.lonsend is None:
                lons=np.concatenate((self.lonsori,self.lonsend))
                lats=np.concatenate((self.latsori,self.latsend))
                self.add_lls(lons,lats,ori=ori)
        else:
            self.lonsend=lons
            self.latsend=lats
            if not self.lonsori is None:
                lons=np.concatenate((self.lonsori,self.lonsend))
                lats=np.concatenate((self.latsori,self.latsend))
                self.add_lls(lons,lats,ori=ori)
    
    def add_zorigins(self,origins):
        self.assert_origins_ends(origins)
        self.zorigins=origins
        newcoords=self.add_coords(self.zorigins,self.zs,origin=True)
        self.add_zs(newcoords)
        
    def add_zends(self,ends):
        self.assert_origins_ends(ends)
        self.zends=ends
        newcoords=self.add_coords(self.zends,self.zs,origin=False)
        self.add_zs(newcoords)
        
    def get_num_points(self):
        params=[self.xorigins,self.yorigins,self.zorigins,self.xends,self.yends,self.zends,self.ts,self.data]
        for i,l in enumerate(params):
            if isinstance(l,(list,np.ndarray)):
                if i==len(params)-1:
                    return len(l)/self.get_size_per_point()
                else:
                    return len(l)
        return None
        
    def assert_origins_ends(self,arr):
        assert isinstance(arr,(list,np.ndarray)),'It should be an array'
        assert len(arr)==self.get_num_points() or self.get_num_points()==None,'The size does not correspond to the postions size'
        
    def from_model3d(self,func,unravel=True):
        ux1,uy1,uz1=func(self.xorigins,self.yorigins)
        ux2,uy2,uz2=func(self.xends,self.yends)
        
        old_dist=np.sqrt((self.xends-self.xorigins)**2+(self.yends-self.yorigins)**2+(self.zends-self.zorigins)**2)
        new_dist=np.sqrt((self.xends-self.xorigins+(ux2-ux1))**2+(self.yends-self.yorigins+(uy2-uy1))**2+(self.zends-self.zorigins+(uz2-uz1))**2)
        
        deltas=new_dist-old_dist
        return (deltas,)