"""
Functions for forward volcano-geodesy analytic models

Author: Scott Henderson
Date: 8/31/2012

Turned in object oriented code 22-jun-2021, Ronni Grapenthin

TODO:
-benchmark codes against paper results
-test for georeferenced coordinate grids?
-add function to convert to los
-add sphinx docstrings
"""

import numpy as np
import util
from source import Source

class ViscoShell(Source):

    def get_num_params(self):
        return 9

    def get_source_id(self):
        return "ViscoShell"

    def print_model(self, x):
        print("ViscoShell:")
        print("\tt  = %f (s)" % x[0])
        print("\tx  = %f (m)" % x[1])
        print("\ty  = %f (m)" % x[2])
        print("\td  = %f (m)" % x[3])
        print("\ta  = %f (m)" % x[4])
        print("\tb  = %f (m)" % x[5])
        print("\tdP = %f" % x[6])
        print("\tmu = %f" % x[7])
        print("\teta= %f" % x[8])

    ##residual function for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(t=x[0], xcen=x[1], ycen=x[2], d=x[3], a=x[4], b=x[5], dP=x[6], mu=x[7], eta=x[8])
        diff = np.concatenate((ux,uy,uz))-self.get_obs()
        return diff

    # =====================
    # Forward Models
    # =====================
    def forward_mod(self, x):
        return self.forward(t=x[0], xcen=x[1], ycen=x[2], d=x[3], a=x[4], b=x[5], dP=x[6], mu=x[7], eta=x[8])

    def forward(self, t, xcen=0,ycen=0,d=4e3,a=1000.0,b=1200.0,dP=100e6,
                        mu=30e9, eta=2e16, nu=0.25):
       
        """
        Spherical Source surronded by a viscoelastic shell in an elastic halfspace
        Derivation of equations 7.105 in Segall Ch.7 p245
        NOTE: good approximation if duration of intrusion << relaxation time tR

        Required arguments:
        ------------------
        x: x-coordinate grid (m) 
        y: y-coordinate grid (m)
        t: time (s)

        Keyword arguments:
        -----------------
        Same as forward_dp() plus:
        a: inner chamber radius
        b: extent of viscoelastic region
        eta: viscosity

        Returns:
        -------
        (ux, uy, uz)

        """

        # characteristic relaxation time
        tR = (3*eta*(1-nu)*b**3) / (mu*(1+nu)*a**3)

        #avoid ZeroDivisionError
        if tR == 0:
            scale = 1
        else:
            scale = (np.exp(-t/tR) + ((b/a)**3)*(1.0 - np.exp(-t/tR)))

        # Center coordinate grid on point source
        x = self.get_xs() - xcen
        y = self.get_ys() - ycen

        # Convert to surface cylindrical coordinates
        th, rho = util.cart2pol(x,y) # surface angle and radial distance
        rho = rho/d                  # radial distance from center of source normalized by source depth

        uz = (((1-nu)*dP*a**3) / (mu*d**2)) * scale * (1 + rho**2)**(-3/2.0)
        ur = rho * uz

        #convert to cartesian coordinates
        ux, uy = util.pol2cart(th, ur)

        #return ux, uy, uz #returns tuple
        return np.array([ux,uy,uz])


