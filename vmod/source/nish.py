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
from .. import util
import scipy
from . import Source

class Nish(Source):

    def get_source_id(self):
        return "Nishimura"

    def print_model(self, x):
        print("Nishimura")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius = %f" % x[3])
        print("\theight = %f" % x[4])
        print("\tdP= %f" % x[5])
        
    def set_parnames(self):
        if self.data.ts is None:
            #return "xcen","ycen","depth","radius",'lenght','pressure','tau'
            self.parameters=("xcen","ycen","depth","radius",'lenght','pressure')
        else:
            self.parameters=("xcen","ycen","depth","radius",'lenght','pressure','dtime')

    # =====================
    # Forward Models
    # =====================
    
    def model(self, x, y, xcen, ycen, d, a, h, dP, mu=4e9, rho=2500, nu=0.25):
       
        """
        3d displacement field from pressurize conduit (Nishimura, 2009)

        Keyword arguments:
        ------------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to middle in conduit (m)
        a: radius of the conduit (m)
        h: height of the tube (m)
        dP: change in pressure (Pa)
        nu: poisson's ratio for medium
        mu: shear modulus for medium (Pa)
        rho: density of magma (kg/m3)

        """
        x=x-xcen
        y=y-ycen
        if isinstance(d,float) or isinstance(d,int):
            if h/2>d:
                c1=0
            else:
                c1=d-h/2
        else:
            c1=d-h/2
            c1[c1<0]=0
        c2=d+h/2
        m=dP*a**2/mu
        r=np.sqrt(x**2+y**2)
        R2=np.sqrt(r**2+c2**2)
        R1=np.sqrt(r**2+c1**2)
        angle=np.arctan2(y,x)
        g=9.8
        s=(dP/(c2-c1)-rho*g)*a**2/(4*mu)

        urn=(m*r/(c2-c1))*(0.5*(c2**2/R2**3-c1**2/R1**3)+(c1/(2*r**2))*(c2**3/R2**3-c1**3/R1**3)-nu*(1/R2-1/R1)-((1+nu)*c1/r**2)*(c2/R2-c1/R1))
        urs=s*r*(1/R1-1/R2-(2*nu-1)*(1/(R2+c2)-1/(R1+c1)))
        ur= urn+urs
        uzn=(-1)*m/(2*(c2-c1))*((-1)*(c2**2)*(c2-c1)/R2**3+(2*nu-1)*(c2/R2-c1/R1)-2*nu*c1*(1/R2-1/R1)-(2*nu-1)*np.log((c2+R2)/(c1+R1)))
        uzs=s*(-c2/R2+c1/R1+(2*nu-1)*np.log((R2+c2)/(R1+c2)))
        uz=uzn+uzs
        
        ux=ur*np.cos(angle)
        uy=ur*np.sin(angle)
        
        return ux,uy,uz
    
    def model_tilt(self, x, y, xcen, ycen, d, a, h, dP, mu=4e9, rho=2500, nu=0.25):
       
        """
        Tilt displacement field from pressurize conduit (Nishimura, 2009)

        Keyword arguments:
        ------------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to middle in conduit (m)
        a: radius of the conduit (m)
        h: height of the tube (m)
        dP: change in pressure (Pa)
        nu: poisson's ratio for medium
        mu: shear modulus for medium (Pa)
        rho: density of magma (kg/m3)

        """
        x=x-xcen
        y=y-ycen
        if isinstance(d,float) or isinstance(d,int):
            if h/2>d:
                c1=0
            else:
                c1=d-h/2
        else:
            c1=d-h/2
            c1[c1<0]=0
        c2=d+h/2
        m=dP*a**2/mu
        r=np.sqrt(x**2+y**2)
        R2=np.sqrt(r**2+c2**2)
        R1=np.sqrt(r**2+c1**2)
        angle=np.arctan2(y,x)
        g=9.8
        s=(dP/(c2-c1)-rho*g)*a**2/(4*mu)
        
        tiltrn=(m*r/(2*(c2-c1)))*((3*c2**2/R2**5-2*nu/R2**3)*(c2-c1)+(c2/R2**3-c1/R1**3)-(2*nu-1)*(1/(R2*(R2+c2))-1/(R1*(R1+c1))))
        tiltrs=(s/r)*(2*(1-nu)*(c2/R2-c1/R1)-(c2**3/R2**3-c1**3/R1**3))
        tiltr=tiltrn+tiltrs
        
        tiltx=tiltr*np.cos(angle)
        tilty=tiltr*np.sin(angle)
        
        return tiltx,tilty
    '''
    def model_tilt(self, x, y, xcen, ycen, d, a, h, dP, tau, mu=4e9, rho=2500, nu=0.25):
       
        """
        Tilt displacement field from pressurize conduit (Nishimura, 2009)

        Keyword arguments:
        ------------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to middle in conduit (m)
        a: radius of the conduit (m)
        h: height of the tube (m)
        dP: change in pressure (Pa)
        nu: poisson's ratio for medium
        mu: shear modulus for medium (Pa)
        rho: density of magma (kg/m3)

        """
        x=x-xcen
        y=y-ycen
        if isinstance(d,float) or isinstance(d,int):
            if h/2>d:
                c1=0
            else:
                c1=d-h/2
        else:
            c1=d-h/2
            c1[c1<0]=0
        c2=d+h/2
        m=dP*a**2/mu
        r=np.sqrt(x**2+y**2)
        R2=np.sqrt(r**2+c2**2)
        R1=np.sqrt(r**2+c1**2)
        angle=np.arctan2(y,x)
        g=9.8
        #s=(dP/(c2-c1)-rho*g)*a**2/(4*mu)
        s=tau*a/(2*mu)
        
        tiltrn=(m*r/(2*(c2-c1)))*((3*c2**2/R2**5-2*nu/R2**3)*(c2-c1)+(c2/R2**3-c1/R1**3)-(2*nu-1)*(1/(R2*(R2+c2))-1/(R1*(R1+c1))))
        tiltrs=(s/r)*(2*(1-nu)*(c2/R2-c1/R1)-(c2**3/R2**3-c1**3/R1**3))
        tiltr=tiltrn+tiltrs
        
        tiltx=tiltr*np.cos(angle)
        tilty=tiltr*np.sin(angle)
        
        return tiltx,tilty
    '''
    def timenbg(self,z1,ap,tp,z2):
        return tp*(z1-1-ap*np.log(np.abs((ap-(z2-z1))/(ap-(z2-1)))))

    def depthnbg(self,t,ap,tp,z2):
        diff=lambda z1,tx: (tx-self.timenbg(z1,ap,tp,z2))**2
        if isinstance(t,float):
            res = scipy.optimize.minimize(diff, 1.0, args=(t), method='Nelder-Mead', tol=1e-6)
            z=res.x[0]
        else:
            z=np.zeros(t.shape)
            for i,t_value in enumerate(t):
                res = scipy.optimize.minimize(diff, 1.0, args=(t_value), method='Nelder-Mead', tol=1e-6)
                z[i]=res.x[0]
        return z
    
    
    def depthbg(self,t,c10,td,z2):
        h0=z2-c10
        h=h0*(1-(t/td)**(3/2))
        return z2-h
    
    def depthbr(self,t,c10,tr,z2):
        h0=z2-c10
        h=h0*(1-(t/tr)*5/3)**(-3/5)
        return z2-h
    
    def model_t(self, x, y, t, xcen, ycen, d, a, h, dP, td, nu=0.25, mu=4e9,rho=2500,eta=100,model='nb'):
        g=9.8
        if model=='nb':
            ap=dP/(rho*g*(d-h/2))
            z1=(d-h/2)*self.depthnbg(t,ap,td,(d+h/2)/(d-h/2))
        elif model=='bg':
            z1=self.depthbg(t,d-h/2,td,d+h/2)
        elif model=='br':
            z1=self.depthbr(t,d-h/2,td,d+h/2)
            
        z2=d+(h/2)
        h1=z2-z1
        d1=z2-h1/2
        ux,uy,uz=self.model(x, y, xcen, ycen, d1, a, h1, dP)
        ux0,uy0,uz0=self.model(x, y, xcen, ycen, d, a, h, dP)
        return ux-ux0,uy-uy0,uz-uz0
