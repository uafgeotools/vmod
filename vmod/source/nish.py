import numpy as np
from .. import util
import scipy
from . import Source

class Nish(Source):
    """
    A class used to represent an open conduit with Nishimura (2009) model.

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
        return "Nishimura"

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Nishimura")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius = %f" % x[3])
        print("\theight = %f" % x[4])
        print("\tdP= %f" % x[5])
        
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
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
        3d displacement field on surface from open conduit (Nishimura, 2009)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of open conduit (m)
            ycen: y-offset of open conduit (m)
            d: depth of magma column (m)
            a: conduit radius (m)
            h: height of magma column (m)
            dP: change in pressure (Pa)
            mu: shear modulus for medium (Pa) (default 4e9)
            rho: host rock density (kg/m^3) (default 2500)
            nu: poisson ratio (default 0.25)
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
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
        #m=dP*a**2/mu
        m=1
        #m=1+h
        r=np.sqrt(x**2+y**2)
        R2=np.sqrt(r**2+c2**2)
        R1=np.sqrt(r**2+c1**2)
        angle=np.arctan2(y,x)
        g=9.8
        #s=(dP/(c2-c1)-rho*g)*a**2/(4*mu)
        s=0

        urn=(m*r/(c2-c1))*(0.5*(c2**2/R2**3-c1**2/R1**3)+(c1/(2*r**2))*(c2**3/R2**3-c1**3/R1**3)-nu*(1/R2-1/R1)-((1+nu)*c1/r**2)*(c2/R2-c1/R1))
        urs=s*r*(1/R1-1/R2-(2*nu-1)*(1/(R2+c2)-1/(R1+c1)))
        ur= urn+urs
        uzn=(-1)*m/(2*(c2-c1))*((-1)*(c2**2)*(c2-c1)/R2**3+(2*nu-1)*(c2/R2-c1/R1)-2*nu*c1*(1/R2-1/R1)-(2*nu-1)*np.log((c2+R2)/(c1+R1)))
        uzs=-s*(-c2/R2+c1/R1+(2*nu-1)*np.log((R2+c2)/(R1+c1)))
        uz=uzn+uzs
        
        ux=ur*np.cos(angle)
        uy=ur*np.sin(angle)
        
        return ux,uy,uz
    
    def model_tilt(self, x, y, xcen, ycen, d, a, h, dP, mu=4e9, rho=2500, nu=0.25):
        """
        Tilt displacement field on surface from open conduit (Nishimura, 2009)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of open conduit (m)
            ycen: y-offset of open conduit (m)
            d: depth of magma column (m)
            a: conduit radius (m)
            h: height of magma column (m)
            dP: change in pressure (Pa)
            mu: shear modulus for medium (Pa) (default 4e9)
            rho: host rock density (kg/m^3) (default 2500)
            nu: poisson ratio (default 0.25)
        
        Returns:
            dx (array) : inclination in x-axis in radians.
            dy (array) : inclination in y-axis in radians.
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
        #m=0
        r=np.sqrt(x**2+y**2)
        R2=np.sqrt(r**2+c2**2)
        R1=np.sqrt(r**2+c1**2)
        angle=np.arctan2(y,x)
        g=9.8
        s=(dP/(c2-c1)-rho*g)*a**2/(4*mu)
        #s=1
        #s=tau*a/(2*mu)
        
        tiltrn=(m*r/(2*(c2-c1)))*((3*c2**2/R2**5-2*nu/R2**3)*(c2-c1)+(c2/R2**3-c1/R1**3)-(2*nu-1)*(1/(R2*(R2+c2))-1/(R1*(R1+c1))))
        tiltrs=(s/r)*(2*(1-nu)*(c2/R2-c1/R1)-(c2**3/R2**3-c1**3/R1**3))
        tiltr=tiltrn+tiltrs
        
        tiltx=tiltr*np.cos(angle)
        tilty=tiltr*np.sin(angle)
        
        return tiltx,tilty
    
    def timenbg(self,z1,ap,tp,z2):
        """
        Calculates time that takes for a column to reach certain height with no bubble growth (Nishimura, 2009)

        Parameters:
            z1 (float): final top depth of magma column (m)
            ap (float): normalized pressure
            tp (float): characteristic time (s)
            z2 (float): bottom depth of magma column (m)
        
        Returns:
            t (float) : time to reach the final top depth
        """
        return tp*(z1-1-ap*np.log(np.abs((ap-(z2-z1))/(ap-(z2-1)))))

    def depthnbg(self,t,ap,tp,z2):
        """
        Inverse function of timenbg

        Parameters:
            t (float): input time (s)
            ap (float): normalized pressure
            tp (float): characteristic time (s)
            z2 (float): bottom depth of magma column (m)
        
        Returns:
            z (float) : top depth at given time (m)
        """
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
        """
        Function that gives top depth for magma column for a magma with bubble growth.

        Parameters:
            t (float): input time (s)
            c10 (float): initial top depth for magma column (m)
            td (float): characteristic time (s)
            z2 (float): bottom depth of magma column (m)
        
        Returns:
            z (float) : top depth at given time (m)
        """
        h0=z2-c10
        h=h0*(1+(t/td)**(3/2))
        return z2-h
    
    def depthbr(self,t,c10,tr,z2):
        """
        Function that gives top depth for magma column for a magma with bubble rising.

        Parameters:
            t (float): input time (s)
            c10 (float): initial top depth for magma column (m)
            td (float): characteristic time (s)
            z2 (float): bottom depth of magma column (m)
        
        Returns:
            z (float) : top depth at given time (m)
        """
        h0=z2-c10
        h=h0*(1-(t/tr)*5/3)**(-3/5)
        return z2-h
    
    def model_t(self, x, y, t, xcen, ycen, d, a, h, dP, td, nu=0.25, mu=4e9,rho=2500,model='nb'):
        """
        3d displacement field on surface from open conduit with time dependency(Nishimura, 2009)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of open conduit (m)
            ycen: y-offset of open conduit (m)
            d: depth of magma column (m)
            a: conduit radius (m)
            h: height of magma column (m)
            dP: change in pressure (Pa)
            td: characteristic time (s)
            nu: poisson ratio (default 0.25)
            mu: shear modulus for medium (Pa) (default 4e9)
            rho: host rock density (kg/m^3) (default 2500)
            model: magma rheology model, no bubble growth (nb), bubble growth (bg) and bubble rising (br) (default nb)
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
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
