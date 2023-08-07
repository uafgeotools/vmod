from . import Data
import numpy as np

class Edm(Data):
    """
    Class that represents the Electronic distance measurements datatype
    
    Attributes:
        delta: deformation in the lines
        lonsori: longitudes for the lines origins
        xorigins: x-coordinates for the line origins
        lonsends: longitudes for the lines target
        xends: x-coordinates for the line ends
        latsori: latitudes for the lines origins
        yorigins: y-coordinates for the line origins
        latsends: latitudes for the lines target
        yends: y-coordinates for the line ends
        zorigins: z-coordinates for the line origins
        zends: z-coordinates for the line ends
    """
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
        
    def add_ts(self,ts):
        """
        Add time steps for the datapoints
        
        Parameters:
            ts: time steps (has to be the same size as the lines)
        """
        self.assert_size(np.concatenate((ts,ts)),'ts')
        self.ts=ts
        
    def add_deltas(self,delta):
        """
        Add changes in the EDM lines
        
        Parameters:
            delta: changes in the lines (m)
        """
        delta=delta.ravel()
        self.assert_origins_ends(delta)
        self.data=delta
        self.err=delta*0+1
        self.comps=[delta]
        self.delta=delta
    
    def add_xorigins(self,origins):
        """
        Add x-coordinates for the line origins
        
        Parameters:
            origins: x-coordinates for the line origins
        """
        self.assert_origins_ends(origins)
        self.xorigins=origins
        if not self.xends is None:
            self.add_xs(np.concatenate((self.xorigins,self.xends)))
        
    def add_xends(self,ends):
        """
        Add x-coordinates for the line ends
        
        Parameters:
            ends: x-coordinates for the line ends
        """
        self.assert_origins_ends(ends)
        self.xends=ends
        if not self.xorigins is None:
            self.add_xs(np.concatenate((self.xorigins,self.xends)))
        
    def add_yorigins(self,origins):
        """
        Add y-coordinates for the line origins
        
        Parameters:
            origins: y-coordinates for the line origins
        """
        self.assert_origins_ends(origins)
        self.yorigins=origins
        if not self.yends is None:
            self.add_ys(np.concatenate((self.yorigins,self.yends)))
        
    def add_yends(self,ends):
        """
        Add y-coordinates for the line ends
        
        Parameters:
            ends: y-coordinates for the line ends
        """
        self.assert_origins_ends(ends)
        self.yends=ends
        if not self.yorigins is None:
            self.add_ys(np.concatenate((self.yorigins,self.yends)))
        
    def add_lls_lines(self,lons,lats,origin=True,ori=None):
        """
        Assign spatial positions using longitudes and latitudes
        
        Parameters:
            lons (array): longitudes for the stations or datapoints
            lats (array): latitudes for the stations or datapoints
            origin (boolean): if True the lons and lats refer to the line origins, 
            if False they refer to the line ends
            ori (array): lon/lat origin for the projection of longitudes and latitudes,
            if None the mean position is the origin
        """
        if origin==True:
            self.lonsori=lons
            self.latsori=lats
        else:
            self.lonsend=lons
            self.latsend=lats
            
        if (not self.lonsori is None) and (not self.lonsend is None):
            lons=np.concatenate((self.lonsori,self.lonsend))
            lats=np.concatenate((self.latsori,self.latsend))
            self.add_lls(lons,lats,ori=ori)
            self.xorigins=self.xs[0:int(len(lons)/2)]
            self.xends=self.xs[int(len(lons)/2):len(lons)]
            self.yorigins=self.ys[0:int(len(lons)/2)]
            self.yends=self.ys[int(len(lons)/2):len(lons)]
    
    def add_zorigins(self,origins):
        """
        Add z-coordinates for the line origins
        
        Parameters:
            origins: z-coordinates for the line origins
        """
        self.assert_origins_ends(origins)
        self.zorigins=origins
        if not self.zends is None:
            self.add_zs(np.concatenate((self.zorigins,self.zends)))
        
    def add_zends(self,ends):
        """
        Add z-coordinates for the line ends
        
        Parameters:
            ends: z-coordinates for the line ends
        """
        self.assert_origins_ends(ends)
        self.zends=ends
        if not self.zorigins is None:
            self.add_zs(np.concatenate((self.zorigins,self.zends)))
        
    def get_num_lines(self):
        """
        Gives the number of lines that compose the EDM dataset
        
        Returns:
            num (int): number of lines
        """
        params=[self.xorigins,self.yorigins,self.zorigins,self.xends,self.yends,self.zends,self.ts,self.data]
        for i,l in enumerate(params):
            if isinstance(l,(list,np.ndarray)):
                if i in [len(params)-2,len(params)-1]:
                    return len(l)/self.get_size_per_point()
                else:
                    return len(l)
        return None
        
    def assert_origins_ends(self,arr):
        """
        Checks if an input array (origins or ends) has the right size
        """
        assert isinstance(arr,(list,np.ndarray)),'It should be an array'
        assert len(arr)==self.get_num_lines() or self.get_num_lines()==None,'The size does not correspond to the postions size'
        
    def from_model3d(self,func,unravel=True):
        """
        Uses the function from the forward model to compute the changes in the lines
        
        Parameters:
            func: 3d displacements forward model function
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
            
        Returns:
            deltas (array): changes (contraction or dilation) of lines (m)
        """
        ux1,uy1,uz1=func(self.xorigins,self.yorigins)
        ux2,uy2,uz2=func(self.xends,self.yends)
        
        old_dist=np.sqrt((self.xends-self.xorigins)**2+(self.yends-self.yorigins)**2+(self.zends-self.zorigins)**2)
        new_dist=np.sqrt((self.xends-self.xorigins+(ux2-ux1))**2+(self.yends-self.yorigins+(uy2-uy1))**2+(self.zends-self.zorigins+(uz2-uz1))**2)
        
        deltas=new_dist-old_dist
        return (deltas,)