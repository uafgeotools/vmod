'''
Penny-shaped Crack Solution from Fialko et al 2001
* Based on dModels scripts Battaglia et. al. 2013
* Check against figure 7.26 in Segall,
which compares Fialko's solution to Davis 1986 point source
* implemented only for single depth right now (no topographic correction)
'''
#from __future__ import division
import numpy as np
import util
import scipy
from scipy import special
from source import Source

class Penny(Source):

    def get_num_params(self):
        return 5

    def get_source_id(self):
        return "Penny"

    def print_model(self, x):
        print("Penny-shaped Crack:")
        print("\tx  = %f (m)" % x[0])
        print("\ty  = %f (m)" % x[1])
        print("\td  = %f (m)" % x[2])
        print("\tP_G= %f (m)" % x[3])
        print("\ta  = %f (m)" % x[4])
    def get_parnames(self):
        return "xcen","ycen","depth","pressure(shear)","radius"
    ##residual function for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0], ycen=x[1], d=x[2], P_G=x[3], a=x[4])
        diff =np.concatenate((ux,uy,uz))-self.get_obs()
        return diff

    # =====================
    # Forward Models
    # =====================
    def forward_mod(self, x):
        return self.forward(xcen=x[0], ycen=x[1], d=x[2], P_G=x[3], a=x[4])

    def forward(self,xcen,ycen,d,P_G,a,nu=0.25):
        """
        Calculates surface deformation based on point pressure source
        References: Fialko

        Args:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)

        Kwargs:
        -----------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to point (m)
        dV: change in volume (m^3)
        nu: poisson's ratio for medium

        Returns:
        -------
        (ux, uy, uz)
        """

        #
        rd=np.copy(a)

        # Center coordinate grid on point source, normalized by radius
        x = (self.get_xs() - xcen) / rd
        y = (self.get_ys() - ycen) / rd
        z = (0 - d)    / rd

        eps=1e-8

        h  = d / rd
        r  = np.sqrt(x ** 2 + y ** 2)

        csi1,w1 = util.gauleg(eps,10,41)
        csi2,w2 = util.gauleg(10,60,41)
        csi     = np.concatenate((csi1,csi2))
        wcsi    = np.concatenate((w1,w2))

        if csi.shape[0] == 1:
            csi=csi.T

        phi1,psi1,t,wt=util.psi_phi(h)

        phi=np.matmul(np.sin(np.outer(csi,t)) , phi1*wt)
        psi=np.matmul(np.divide(np.sin(np.outer(csi,t)), np.outer(csi,t)) - np.cos(np.outer(csi,t)),psi1*wt)
        a=csi * h
        A=np.exp((-1)*a)*(a*psi + (1 + a)*phi)
        B=np.exp((-1)*a)*((1-a)*psi - a*phi)
        Uz=np.zeros(r.shape)
        Ur=np.zeros(r.shape)

        for i in range(r.size):
            J0=special.jv(0,r[i] * csi)
            Uzi=J0*(((1-2*nu)*B - csi*(z+h)*A)*np.sinh(csi*(z+h)) +(2*(1-nu)*A - csi*(z+h)*B)*np.cosh(csi*(z+h)))
            Uz[i]=np.dot(wcsi , Uzi)
            J1=special.jv(1,r[i] * csi)
            Uri=J1*(((1-2*nu)*A + csi*(z+h)*B)*np.sinh(csi*(z+h)) + (2*(1-nu)*B + csi*(z+h)*A)*np.cosh(csi*(z+h)))
            Ur[i]=np.dot(wcsi , Uri)

        ux = rd * P_G * Ur*x / r
        uy = rd * P_G * Ur*y / r
        uz = -rd * P_G * Uz

        return np.array([ux,uy,uz])

    def dP2dV(self, P_G,z0,a,nu=0.25):
        h=z0 / a
        phi,psi,t,wt=util.psi_phi(h)
        dV= -4 * np.pi * (1 - nu) * P_G *a**3 * (t * (wt.T.dot(phi)))
        return dV

    def verify():
        from data import Data
        import matplotlib.pyplot as plt        
        
        x0=0
        y0=0
        z0=1000
        P_G=0.01
        a=1000
        x=np.arange(-5000,5001,100)
        y=np.arange(-5000,5001,100)

        d = Data()
        d.add_locs(x,y)

        penny = Penny(d)

        u,v,w=penny.forward(x0,y0,z0,P_G,a)

        plt.figure()
        plt.plot(x,u,c='red',label='u')

        plt.figure()
        plt.plot(y,v,c='red',label='v')

        plt.figure()
        plt.plot(y,w,c='red',label='w')

        plt.show()

