"""
Functions for forward volcano-geodesy analytic models

Author: Scott Henderson
Date: 8/31/2012



TODO:
-benchmark codes against paper results
-test for georeferenced coordinate grids?
-add function to convert to los
-add sphinx docstrings
"""
import numpy as np
import util
import pandas as pd
import copy
from data import Data

class Source:
    def __init__(self, data):
        self.data  = data
        self.model = None

    def get_obs(self):
        return self.data.get_obs()

    def get_xs(self):
        return self.data.get_xs()

    def get_ys(self):
        return self.data.get_ys()

    def get_zs(self):
        return self.data.get_zs()

    def res_norm(self):
        ux,uy,uz = self.forward_mod()
        return np.linalg.norm(self.get_obs()*1000-np.concatenate([ux, uy, uz])*1000)

    ##inversion methods
    def invert(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun, x0, bounds=bounds))
        return self.model

    def invert_dipole(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun_dipole, x0, bounds=bounds))
        return self.model

    def invert_bh(self, x0):
        from scipy.optimize import basinhopping
        return basinhopping(self.fun, x0)

    ##output writers
    def write_forward_gmt(self, prefix):
        if self.model is not None:
            ux,uy,uz = self.forward_mod()

            dat = np.zeros(self.data.data['id'].to_numpy().size, 
                dtype=[ ('lon', float), ('lat', float), ('east', float), ('north', float), 
                        ('esig', float), ('nsig', float), ('corr', float), ('id', 'U6')] )

            dat['lon']   = self.data.data['lon'].to_numpy()
            dat['lat']   = self.data.data['lat'].to_numpy()
            dat['east']  = ux*1000
            dat['north'] = uy*1000
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.data.data['id'].to_numpy()

            print(dat)

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz*1000
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )

#    def make_map(self, region):
#        import pygmt
#
#        fig = pygmt.Figure()
#        df = self.data.data(['lon','lat','ux','uy','sx','sy','id'])
#
#        df = pd.DataFrame(
#            data={
#            "x": self.get_xs()
#            "y": self.get_ys()
#            "east_velocity": [0, 3, 4, 6, -6, 6],
#            "north_velocity": [0, 3, 6, 4, 4, -4],
#            "east_sigma": [4, 0, 4, 6, 6, 6],
#            "north_sigma": [6, 0, 6, 4, 4, 4],
#            "correlation_EN": [0.5, 0.5, 0.5, 0.5, -0.5, -0.5],
#            "SITE": ["0x0", "3x3", "4x6", "6x4", "-6x4", "6x-4"],
#        }
#        )
#        fig.velo(
#            data=df,
#            region=[-10, 8, -10, 6],
#            pen="0.6p,red",
#            uncertaintycolor="lightblue1",
#            line=True,
#            spec="e0.2/0.39/18",
#            frame=["WSne", "2g2f"],
#            projection="x0.8c",
#            vector="0.3c+p1p+e+gred",
#        )#
#
#        fig.show()        
   

