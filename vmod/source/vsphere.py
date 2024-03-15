import numpy as np
from .. import util
from . import Source

class Vsphere(Source):
    """
    Class that represents a pressurized sphere within a viscoelastic medium Bonafede and Ferrari (2009) model.

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    """
    def get_source_id(self):
        """
        The function defining the name for the model.
          
        Returns:
            str: Name of the model.
        """
        return "Vsphere"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        
        Returns:
            steps (int): Number of steps used in the bayesian inversions.
            burnin (int): discarded number of steps at the begining of the inversion.
            thin (int): number of steps per sample.
        """
        steps=1100000
        burnin=10000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Vsphere")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius = %f" % x[3])
        print("\tdP= %f" % x[4])
    
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","radius","dP","tau")
        
    def model(self, x, y, xcen, ycen, d, rad, dP):
        """
        Initial 3d displacement field on surface for pressurized sphere in a viscoelastic medium (Bonafede and Ferrari, 2009)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            d: depth to point (m)
            rad: chamber radius (m)
            dP: change in pressure (Pa)
            
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        return model_t(x,y,0, xcen, ycen, d, rad, dP)
        
    def model_t(self, x, y, t, xcen, ycen, d, rad, dP, tau, nu=0.25, mu=1):
        """
        3d displacement field on surface for pressurized sphere in a viscoelastic medium (Bonafede and Ferrari, 2009)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            t: input time (s)
            xcen: x-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            d: depth to point (m)
            rad: chamber radius (m)
            dP: change in pressure (in terms of mu if mu=1 if not unit is Pa)
            nu: poisson's ratio for medium (default 0.25)
            mu: shear modulus for medium (Pa) (default 1)
            
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        a=rad
        K=5*mu/3
        
        x = x - xcen
        y = y - ycen
        
        if rad>d:
            return x*np.Inf,x*np.Inf,x*np.Inf
        
        th, r = util.cart2pol(x,y)
        A=1+2*(mu/K)
        B=2*mu**2/(K*(3*K+mu))
        alpha=(3*K+mu)/(3*K)
        taual=alpha*tau
        R=np.sqrt(r**2+d**2)
        ur1=[(r/R**3)*(1+t/tau)+(r/R**3)*(A-B*np.exp(-t/taual)+t/tau)]
        ur2=(dP*a**3)/(4*mu)
        ur=np.array([ur2*urt for urt in ur1])

        uz=[-d/R**3*(1+t/tau)-(d/R**3)*(A-B*np.exp(-t/taual)+t/tau)]
        uz=np.array([ur2*utz for utz in uz])
        
        ux, uy = util.pol2cart(th, ur)
        
        return ux.ravel(),uy.ravel(),uz.ravel()