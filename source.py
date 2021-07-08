"""
Base class for analytical magmatic source models. Implements
common functions required from all child classes.

Author: Ronni Grapenthin, UAF
Date: 6/23/2021


TODO:
-add sphinx docstrings
"""
import numpy as np
import util
import pandas as pd
import copy
from data import Data

class Source:
    def __init__(self, data):
        self.data        = data
        self.x0          = None
        self.low_bounds  = []
        self.high_bounds = []

    def set_x0(self, x0):
        self.x0 = x0

    def set_bounds(self, low_bounds, high_bounds):
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

    def get_obs(self):
        return self.data.get_obs()

    def get_xs(self):
        return self.data.get_xs()

    def get_ys(self):
        return self.data.get_ys()

    def get_zs(self):
        return self.data.get_zs()

    def get_site_ids(self):
        return self.data.get_site_ids()

    def get_lats(self):
        return self.data.get_lats()

    def get_lons(self):
       return self.data.get_lons()

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

            dat['lon']   = self.get_lons()
            dat['lat']   = self.get_lats()
            dat['east']  = ux*1000
            dat['north'] = uy*1000
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.get_site_ids()

            print(dat)

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz*1000
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )

    def make_map(self, west, east, south, north, proj):
        import pygmt

        fig = pygmt.Figure()

        if self.model is not None:
            ux,uy,uz = self.forward_mod()
            df_mod_hori = pd.DataFrame(
                            data={
                                "x": self.get_xs(),
                                "y": self.get_ys(),
                                "east_velocity":  ux,
                                "north_velocity": uy,
                                "east_sigma": ux*0,
                                "north_sigma": uy*0,
                                "correlation_EN": ux*0,
                                "SITE": self.get_site_ids(),
                                }
                    )
        fig.velo(
            data=df_mod_hori,
            region=[west, east, south, north],
            pen="0.6p,red",
            uncertaintycolor="lightblue1",
            line=True,
            spec="e0.2/0.39/18",
            frame=["WSne", "2g2f"],
            projection=proj, 
            vector="0.3c+p1p+e+gred",
        )

        fig.show()        
   

