"""
Functions for forward volcano-geodesy analytic models

Author: Scott Henderson, Mario Angarita, Ronni Grapenthin
Date: 9/30/2021

Turned in object oriented code 22-jun-2021, Ronni Grapenthin

TODO:
-benchmark codes against paper results
-test for georeferenced coordinate grids?
-add function to convert to los
-add sphinx docstrings
"""

import numpy as np
import util
import scipy
from source import Source

class Mctigue(Source):

    def get_num_params(self):
        return 5

    def get_source_id(self):
        return "Mctigue"

    def print_model(self, x):
        print("Mctigue")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius = %f" % x[3])
        print("\tdV= %f" % x[4])
        
    def get_parnames(self):
        return "xcen","ycen","depth","radius","dV"
    
    ##residual function for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0], ycen=x[1], d=x[2], rad=x[3], dV=x[4])
        diff =np.concatenate((ux,uy,uz))-self.get_obs()
        return diff

    # =====================
    # Forward Models
    # =====================
    def forward_gps(self, x):
        return self.gps(x[0],x[1],x[2],x[3],x[4])

    def forward_tilt(self, x):
        return self.tilt(x[0],x[1],x[2],x[3],x[4])
    
    def gps(self,xcen,ycen,d,rad,dV):
        x=self.get_xs()
        y=self.get_ys()
        return self.model(x,y,xcen,ycen,d,rad,dV)
    
    def tilt(self,xcen,ycen,d,rad,dV):
        
        uzx= lambda x: self.model(x,self.get_ys(),xcen,ycen,d,rad,dV)[2]
        uzy= lambda y: self.model(self.get_xs(),y,xcen,ycen,d,rad,dV)[2]
        
        duzx=-scipy.misc.derivative(uzx,self.get_xs(),dx=1e-6)
        duzy=-scipy.misc.derivative(uzy,self.get_ys(),dx=1e-6)
        
        return duzx,duzy

    def model(self, x, y, xcen, ycen, d, rad, dV, nu=0.25, mu=4e9):
       
        """
        3d displacement field from dislocation point source (McTigue, 1987)
        Caution: analysis done in non-dimensional units!
        see also Segall Ch7 p207
        Same as forward, except change in pressure and chamber radius are specified

        Keyword arguments:
        ------------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to point (m)
        rad: chamber radius (m)
        dV: change in volume (m^3)
        dP: change in pressure (Pa)
        nu: poisson's ratio for medium
        mu: shear modulus for medium (Pa)
        order: highest order term to include (up to 2)
        output: 'cart' (cartesian), 'cyl' (cylindrical)

        Set terms=1 to reduce to Mogi Solution
        NOTE: eps**6 term only significant if eps > 0.5
        """
        # center coordinate grid on point source
        x = x - xcen
        y = y - ycen
        
        if rad>d:
            return x*np.Inf,x*np.Inf,x*np.Inf
        
        dP = dV*mu/(np.pi*rad**3)

        # dimensionless scaling term
        scale = dP * d / mu
        eps = rad / d #NOTE eps = 0.5 # mctigue fig3

        # convert to surface cylindrical coordinates
        th, r = util.cart2pol(x,y)
        r = r / d #dimensionless radial distance

        # 1st order mctigue is essentially Mogi solution
        uz = eps**3 * ((1-nu) * (1 / np.hypot(r,1)**3))
        ur = eps**3 * ((1-nu) * (r / np.hypot(r,1)**3))

        # 2nd order term
        A = ((1 - nu) * (1 + nu)) / (2 * (7 - 5*nu))
        B = (15 * (2 - nu) * (1 - nu)) / (4 * (7 - 5*nu))
        uz2 =  -eps**6 * ((A * (1 / np.hypot(r,1)**3)) - (B * (1 / np.hypot(r,1)**5)))
        ur2 =  -eps**6 * ((A * (r / np.hypot(r,1)**3)) - (B * (r / np.hypot(r,1)**5)))
        uz += uz2
        ur += ur2

        # Convert back to dimensional variables
        uz = uz * scale
        ur = ur * scale

        # Convert surface cylindrical to cartesian
        ux, uy = util.pol2cart(th, ur)
        return ux, uy, uz
