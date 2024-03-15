import numpy as np
from scipy import special
from . import Source
from .. import util
from scipy import integrate

class Wellsite(Source):
    """
    Class that represents a pressurized well following the implementation from Wangen (2018).

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
        return "Wellsite"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        
        Returns:
            steps (int): Number of steps used in the bayesian inversions.
            burnin (int): discarded number of steps at the begining of the inversion.
            thin (int): number of steps per sample.
        """
        steps=3100000
        burnin=10000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Wellsite")
        print("\txcen = %f" % x[0])
        print("\tycen = %f" % x[1])
        print("\twidth = %f" % x[2])
        print("\tpressure = %f" % x[3])
        print("\tdiffusivity= %f" % x[4])
        print("\tYoung Modulus= %f" % x[5])
        
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","width","pressure","diffu")
    
    # =====================
    # Forward Models
    # =====================
    def pressure(self,x,P,diffu,t):
        """
        The function calculates the pressure as a function of time.
        
        Parameters:
           x (float): distance to the injection well (m)
           P (float): initial pressure (Pa)
           diffu (float): diffusivity coefficient (m^2/s)
           t: input time (s)
           
        Results:
            pressure: pressure at given time (Pa)
        """
        return P*special.erfc(x/(2*np.sqrt(diffu*t)))
    
    def dpressure(self,x,an,fn,k):
        return fn*an*np.cos(k*x)
    
    def model_t(self, x, y, t, xcen, ycen, depth, width, P, diffu, mu=1, nu=0.25):
       
        """
        The function 
        
        Parameters:
            xcen: y-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            depth: Reservoir depth (m) 
            width: reservoir width (m)
            P: Initial pressure (in terms of mu if mu=1 if not unit is Pa)
            diffu: Diffusivity (m^2/s)
            nu: poisson's ratio for medium (default 0.25)
            mu: shear modulus for medium (Pa) (default 1)
        
        Returns:
            ux: the model does not compute deformation in the x-axis
            uy: the model does not compute deformation in the y-axis
            uz: deformation in the vertical (m)
        """
        alpha=1

        E=mu*(2*(1+nu))
        
        Lambda = (nu*E)/((1+nu)*(1-2*nu))
        
        h1=width #[10-100]
        h2=depth #[50-300]
        
        # center coordinate grid on point source
        x = x - xcen
        y = y - ycen

        # convert to surface cylindrical coordinates
        th, r = util.cart2pol(x,y)
        
        H = h1+h2
        L =np.nanmax(r)               #m, minimum domain size, >>2*np.pi*H
        N=100

        
        tunique=list(set(t))
        uz=x*np.nan
        for tt in tunique:
            ks=[]
            wlns=[]
            fns=[]
            pus =[]
            ws =[]
            rt=r[t==tt]
            
            rsimp=np.linspace(0,np.max(rt),1000)

            px_simp=self.pressure(rsimp,P,diffu,tt)
            #px_simp=lambda r: self.pressure(r,P,diffu,tt)
            ans=[]
            px_Fs=rt*0

            for n in range(0,N):
                if n == 0:
                    k=0
                    fn = 1/2
                    fns.append(fn)
                    #fun=lambda r: self.pressure(r,P,diffu,tt)*np.cos(n*np.pi*r/L)
                    #an = ((2/L)*integrate.quad_vec(fun,0,L)[0]) #Fourier coeff an
                    an = ((2/L)*np.sum(px_simp*np.cos(n*np.pi*rsimp/L)*np.abs(rsimp[1]-rsimp[0])))
                else:    
                    wln = (2*L)/n        #wavelength n
                    k = (2*np.pi)/wln    #wavenumber n
                    fn = (1/(k*h1))*((np.cosh(k*h2)*np.tanh((k*h1)+(k*h2)))-np.sinh(k*h2)) #dimensionless amplitude dep. on h1 & h2

                    #Fourier coeff an
                    #fun=lambda r: self.pressure(r,P,diffu,tt)*np.cos(n*np.pi*r/L)
                    #an = ((2/L)*integrate.quad_vec(fun,0,L)[0]) #Fourier coeff an
                    #an = ((2/L)*integrate.cumtrapz(px_simp*np.cos(n*np.pi*rt/L)))
                    an = ((2/L)*np.sum(px_simp*np.cos(n*np.pi*rsimp/L)*np.abs(rsimp[1]-rsimp[0])))
                    
                px_Fs+=self.dpressure(rt,an,fn,k)

            uz[t==tt] = (alpha*h1*px_Fs)/(Lambda + (2*mu))
            fns[0]=1

        ux=0*uz
        uy=0*uz
        
        return ux, uy, uz