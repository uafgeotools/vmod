import numpy as np
from .. import util
from . import Source

class Vsphere(Source):
    
    def get_source_id(self):
        return "Vsphere"
    
    def time_dependent(self):
        return False
    
    def bayesian_steps(self):
        steps=1100000
        burnin=10000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        print("Vsphere")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius = %f" % x[3])
        print("\tdP= %f" % x[4])
    
    def set_parnames(self):
        self.parameters=("xcen","ycen","depth","radius","dP","tau")
        
    def model(self, x, y, xcen, ycen, d, rad, dP):
        return model_t(x,y,0, xcen, ycen, d, rad, dP)
        
    def model_t(self, x, y, t, xcen, ycen, d, rad, dP, tau, nu=0.25, mu=4e9):
        
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
        
        return ux,uy,uz