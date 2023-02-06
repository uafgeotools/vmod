import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import pandas as pd
from . import Source
from .. import util
from scipy import integrate

class Wellsite(Source):

    def get_num_params(self):
        return 7

    def get_source_id(self):
        return "Wellsite"
    
    def bayesian_steps(self):
        steps=3100000
        burnin=10000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        print("Wellsite")
        print("\txcen = %f" % x[0])
        print("\tycen = %f" % x[1])
        print("\twidth = %f" % x[2])
        print("\tpressure = %f" % x[3])
        print("\tdiffusivity= %f" % x[4])
        print("\tYoung Modulus= %f" % x[5])
        
    def set_parnames(self):
        self.parameters=("xcen","ycen","depth","width","pressure","diffu","Young modulus")
    
    # =====================
    # Forward Models
    # =====================
    def pressure(self,x,P,diffu,t):
        return P*special.erfc(x/(2*np.sqrt(diffu*t)))
    
    def dpressure(self,x,an,fn,k):
        return fn*an*np.cos(k*x)
    
    def model_t(self, x, y, t, xcen, ycen, depth, width, P, diffu, E=5, nu=0.2):
       
        """
        Keyword arguments:
        ------------------
        xcen: y-offset of point source epicenter (m): []
        ycen: y-offset of point source epicenter (m): []
        depth: Reservoir depth (m): 
        width: reservoir width (m)
        P: Initial pressure (MPa): [5-50]
        diffu: Diffusivity (m^2/s): [10^-4-10^-1]
        E: Young's modulus (GPa): [5-15]
        nu: Poisson ratio 
        
        """
        alpha=1
        
        mu=E/(2*(1+nu))
        
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