"""
Class to implement inverse methods operating on one or more analytical volcanic source models

Author: Ronni Grapenthin
Date: 6/23/2021


TODO:
- docstrings
"""

import copy
import numpy as np

class Inverse:
    def __init__(self, obs):
        self.sources = []       #simple list that will contain instances of class Source
        self.obs     = obs      #instance of Data
        self.model   = None
    
    #add a new source to the geometry
    def register_source(self, source):
        self.sources.append(source)

    #interface to scipy bounded nonlinear least squares implementation
    def nlsq(self):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun, self.get_x0(), bounds=self.get_bounds()))
        return self.model

    #initial guess of source model characteristics, defined when creating the source model
    def get_x0(self):
        x0 = []

        for s in self.sources:
            x0 = np.concatenate((x0, s.x0))

        return x0

    #high and low bounds for the parameters
    def get_bounds(self):
        low_b  = []
        high_b = []

        for s in self.sources:
            low_b  = np.concatenate((low_b,  s.low_bounds))
            high_b = np.concatenate((high_b, s.high_bounds))

        return (low_b, high_b)

    #least_squares residual function for dipole
    def fun(self, x):
        ux_m, uy_m, uz_m = self.forward(x)

        diff = np.concatenate((ux_m,uy_m,uz_m))-self.obs.get_obs()

        return diff

    #call all forward models for registered sources to calculate 
    #total synthetic displacements
    def forward(self, x):
        param_cnt = 0

        ux_m = None
        uy_m = None
        uz_m = None

        for s in self.sources:
            m_x, m_y, m_z = s.forward_mod(x[param_cnt:param_cnt+s.get_num_params()])
            param_cnt += s.get_num_params()
            
            if ux_m is None:
                ux_m = m_x
                uy_m = m_y
                uz_m = m_z
            else:
                ux_m += m_x
                uy_m += m_y
                uz_m += m_z

        return ux_m, uy_m, uz_m

    ##output writers
    def print_model(self):
        param_cnt = 0

        for s in self.sources:
            s.print_model(self.model.x[param_cnt:param_cnt+s.get_num_params()])
            param_cnt += s.get_num_params()

    ##writes gmt files for horizontal and vertical deformation, each, to use with gmt velo.
    def write_forward_gmt(self, prefix):
        if self.model is not None:

            ux,uy,uz = self.forward(self.model.x)

            dat = np.zeros(self.obs.data['id'].to_numpy().size, 
                dtype=[ ('lon', float), ('lat', float), ('east', float), ('north', float), 
                        ('esig', float), ('nsig', float), ('corr', float), ('id', 'U6')] )

            dat['lon']   = self.obs.data['lon'].to_numpy()
            dat['lat']   = self.obs.data['lat'].to_numpy()
            dat['east']  = ux*1000
            dat['north'] = uy*1000
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.obs.data['id'].to_numpy()

            print(dat)

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz*1000
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )
        else:
            print("No model, nothing to write")



