import numpy as np
from .. import util
from . import Source

class Mogi(Source):
    """
    A class used to represent a point source using the Mogi (1958) model.

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
        return "Mogi"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        
        Returns:
            steps (int): Number of steps used in the bayesian inversions.
            burnin (int): discarded number of steps at the begining of the inversion.
            thin (int): number of steps per sample.
        """
        steps=1100000
        burnin=100000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Mogi")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tdV= %f" % x[3])
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","dV")

    # =====================
    # Forward Models
    # =====================
    
    def model(self,x,y, xcen, ycen, d, dV, nu=0.25):
        """
        3d displacement field on surface from point source (Mogi, 1958)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            d: depth to point (m)
            rad: chamber radius (m)
            dV: change in volume (m^3)
            dP: change in pressure (Pa)
            nu: poisson's ratio for medium (default 0.25)
            mu: shear modulus for medium (Pa) (default 4e9)
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        # Center coordinate grid on point source
        x = x - xcen
        y = y - ycen

        # Convert to surface cylindrical coordinates
        th, rho = util.cart2pol(x,y) # surface angle and radial distance
        R = np.sqrt(d**2+rho**2)     # radial distance from source

        # Mogi displacement calculation
        C = ((1-nu) / np.pi) * dV
        ur = C * rho / R**3    # horizontal displacement, m
        uz = C * d / R**3      # vertical displacement, m

        ux, uy = util.pol2cart(th, ur)
        
        return ux, uy, uz #returns tuple
        #return np.array([ux,uy,uz])
    
    def model_tilt(self, x, y, xcen, ycen, d, dV, nu=0.25):
        """
        Tilt displacement field from point source (Mogi, 1958)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
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

        Returns:
            dx (array) : inclination in the x-axis in radians.
            dy (array) : inclination in the y-axis in radians.
        """
        
        # Center coordinate grid on point source
        x = x - xcen
        y = y - ycen

        # Convert to surface cylindrical coordinates
        th, rho = util.cart2pol(x,y) # surface angle and radial distance
        R = np.sqrt(d**2+rho**2)     # radial distance from source

        # Mogi displacement calculation
        C = ((1-nu) / np.pi) * dV
        
        dx=3*C*d*x/R**5
        dy=3*C*d*y/R**5
        
        #print(dx)
        
        return dx, dy
    
    def calc_genmax(self,t,xcen=0,ycen=0,d=4e3,dP=100e6,a=700,nu=0.25,G=30e9,
                    mu1=0.5,eta=2e16,**kwargs):
        """ Solution for spherical source in a generalized maxwell viscoelastic
        halfspace based on Del Negro et al 2009.

        Required arguments:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)
        t: time (s)

        Keyword arguments:
        -----------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to point (m)
        dV: change in volume (m^3)
        K: bulk modulus (constant b/c incompressible)
        E: Young's moduls
        G: total shear modulus (Gpa)
        mu0: fractional shear modulus (spring part)
        mu1: fractional shear modulus (dashpot part)
        eta: viscosity (Pa s)
        output: 'cart' (cartesian), 'cyl' (cylindrical)

        """
        #WARNING: mu0 != 0
        # center coordinate grid on point source
        x = self.get_xs() - xcen
        y = self.get_ys() - ycen

        # convert to surface cylindrical coordinates
        #th, r = cart2pol(x,y)
        r = np.hypot(x,y) #surface radial distance
        R = np.hypot(d,r) #radial distance from source center

        # Calculate displacements
        #E = 2.0 * G * (1+nu)
        #K = E / (3.0* (1 - 2*nu)) #bulk modulus = (2/3)*E if poisson solid
        K = (2.0*G*(1+nu)) / (3*(1-(2*nu)))
        mu0 = 1.0 - mu1
        alpha = (3.0*K) + G #recurring terms
        beta = (3.0*K) + (G*mu0)

        # Maxwell times
        try:
            tau0 = eta / (G*mu1)
        except:
            tau0 = np.inf
        tau1 = (alpha / beta) * tau0
        tau2 = tau0 / mu0

        #print('relaxation times:\nT0={}\nT1={}\nT2={}'.format(tau0,tau1,tau2))

        term1 = ((3.0*K + 4*G*mu0) / (mu0*beta))
        term2 = ((3.0 * G**2 * np.exp(-t/tau1))*(1-mu0)) / (beta*alpha)
        term3 = ((1.0/mu0) - 1) * np.exp(-t/tau2)

        A = (1.0/(2*G)) * (term1 - term2 - term3)
        C = (dP * a**3) / R**3
        ur = C * A * r
        uz = C * A * d

        return ur, uz