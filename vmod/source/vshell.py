import numpy as np
from .. import util
from . import Source

class Vshell(Source):
    
    def get_source_id(self):
        return "Vshell"
    
    def time_dependent(self):
        return False
    
    def bayesian_steps(self):
        steps=1100000
        burnin=10000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        print("Vshell")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius1 = %f" % x[3])
        print("\tradius2 = %f" % x[4])
        print("\tdP= %f" % x[5])
        print("\ttau= %f" % x[6])
    
    def set_parnames(self):
        self.parameters=("xcen","ycen","depth","radius","dP","tau")
        
    def model(self, x, y, xcen, ycen, d, rad, dP):
        return model_t(x,y,0, xcen, ycen, d, rad, dP)
        
    def model_t(self, x, y, t, xcen, ycen, d, rad1, rad2, dP, tau, nu=0.25, mu=4e9):
        
        x = x - xcen
        y = y - ycen
        
        coeff=(1-nu)*dP*rad1**3/(mu*d**2)
        
        th, r = util.cart2pol(x,y)
        
        uz=coeff*(np.exp(-t/tau)+((rad2/rad1)**3)*(1-np.exp(-t/tau)))/(1+r**2)**1.5

        ur=uz*r
        
        ux, uy = util.pol2cart(th, ur)
        
        return ux,uy,uz