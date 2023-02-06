"""
Class to implement interface to geodetic data types used in source inversions

Author: Mario Angarita
Date: 9/23/2022
"""

from . import Data

class Edm(Data):
    def __init__(self):
        self.delta=None
        self.xorigins=None
        self.xends=None
        self.yorigins=None
        self.yends=None
        self.zorigins=None
        self.zends=None
        super().__init__()
        
    def add_deltas(self,delta):
        self.add_comp(delta,'delta')
        self.delta=delta
    
    def add_coords(self,coords,obj,origin=True):
        self.assert_origins_ends(coords)
        if obj==None:
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
    
    def add_xorigins(self,origins):
        assert_origins_ends(origins)
        self.xorigins=origins
        self.add_coords(self.xs,origin=True)
        
    def add_xends(self,ends):
        assert_origins_ends(ends)
        self.xends=ends
        self.add_coords(self.xs,origin=False)
        
    def add_yorigins(self,origins):
        assert_origins_ends(origins)
        self.yorigins=origins
        self.add_coords(self.ys,origin=True)
        
    def add_yends(self,ends):
        assert_origins_ends(ends)
        self.yends=ends
        self.add_coords(self.ys,origin=False)

    def add_zorigins(self,origins):
        assert_origins_ends(origins)
        self.zorigins=origins
        self.add_coords(self.zs,origin=True)
        
    def add_zends(self,ends):
        assert_origins_ends(ends)
        self.zends=ends
        self.add_coords(self.zs,origin=False)    
        
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
        assert isinstance(arr,list,np.ndarray),'It should be an array'
        assert len(arr)==self.get_num_points() or self.get_num_points()==None,'The size does not correspond to the postions size'
        
    def from_model3d(self,func,unravel=True):
        ux1,uy1,uz1=func(self.xorigins,self.yorigins)
        ux2,uy2,uz2=func(self.xorigins,self.yorigins)
        deltas=np.sqrt((ux2-ux1)**2+(uy2-uy1)**2+(uz2-uz1)**2)
        return (deltas,)