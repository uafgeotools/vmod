import numpy as np
from .. import util
import scipy
from scipy.integrate import quad,quad_vec
from . import Source

class Mctigue(Source):
    """
    A class used to represent a spherical source using the McTigue (1987) implementation

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
        return "Mctigue"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
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
        print("Mctigue")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tradius = %f" % x[3])
        print("\tdP= %f" % x[4])
    
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","radius","dP")
    
    # =====================
    # Forward Models
    # =====================
    
    def model(self, x, y, xcen, ycen, d, rad, dP, nu=0.25, mu=1):
        """
        3d displacement field on surface from spherical source (McTigue, 1987)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: y-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            d: depth to point (m)
            rad: chamber radius (m)
            dP: change in pressure (Pa)
            nu: poisson's ratio for medium (default 0.25)
            mu: shear modulus for medium (Pa) (default 1)
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        # center coordinate grid on point source
        x = x - xcen
        y = y - ycen
        nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        if np.sum(d<=0)>0 or rad<=0 or np.sum(rad>d)>0:
            return nans

        #dP = dV*mu/(np.pi*rad**3)

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
    
    def model_tilt(self, x, y, xcen, ycen, d, rad, dV, nu=0.25, mu=4e9):
        """
        Tilt displacement field from spherical source (McTigue, 1987)

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
        # center coordinate grid on point source
        x = x - xcen
        y = y - ycen
        nans=np.array([x*0+1e6,x*0+1e6,x*0+1e6])
        if np.sum(d<=0)>0 or rad<=0 or np.sum(rad>d)>0:
            return nans
        
        dP = dV*mu/(np.pi*rad**3)

        # dimensionless scaling term
        scale = dP * d / mu
        eps = rad / d #NOTE eps = 0.5 # mctigue fig3

        # convert to surface cylindrical coordinates
        th, r = util.cart2pol(x,y)
        r = r / d #dimensionless radial distance

        # 1st order mctigue is essentially Mogi solution
        dx = 3 * (x/(d**2)) * eps**3 * ((1-nu) * (1 / np.hypot(r,1)**5))
        dy = 3 * (y/(d**2)) * eps**3 * ((1-nu) * (1 / np.hypot(r,1)**5))
        #print(dx)

        # 2nd order term
        A = ((1 - nu) * (1 + nu)) / (2 * (7 - 5*nu))
        B = (15 * (2 - nu) * (1 - nu)) / (4 * (7 - 5*nu))
        
        dx2 =  -eps**6 * (x/(d**2)) * ((A * (3 / np.hypot(r,1)**5)) - (5 * B * (1 / np.hypot(r,1)**7)))
        dy2 =  -eps**6 * (y/(d**2)) * ((A * (3 / np.hypot(r,1)**5)) - (5 * B * (1 / np.hypot(r,1)**7)))
        
        #print(dx2)
        
        dx += dx2
        dy += dy2
        
        dx = dx*scale
        dy = dy*scale
        
        return dx, dy
    
    def model_depth(self, x, y, z, xcen, ycen, d, rad, dP, nu=0.25, mu=1):
        """
        3d displacement field at depth from dislocation point source (McTigue, 1987)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            z: z-coordinate for displacement (m)
            xcen: y-offset of point source epicenter (m)
            ycen: y-offset of point source epicenter (m)
            d: depth to point (m)
            rad: chamber radius (m)
            dP: change in pressure (Pa)
            nu: poisson's ratio for medium (default 0.25)
            mu: shear modulus for medium (Pa) (default 4e9)
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        # center coordinate grid on point source
        x = x - xcen
        y = y - ycen
        if isinstance(d,float):
            if rad>d:
                return x*np.Inf,x*np.Inf,x*np.Inf
        else:
            if len(d[rad>d])>0:
                return x*np.Inf,x*np.Inf,x*np.Inf
        
        #dP = dV*mu/(np.pi*rad**3)

        # dimensionless scaling term
        scale = dP * d / mu
        eps = rad / d #NOTE eps = 0.5 # mctigue fig3

        # convert to surface cylindrical coordinates
        th, r = util.cart2pol(x,y)
        rho = r / d #dimensionless radial distance
        zeta = z / d
        
        uz0=(eps**3)*0.25*(1-zeta)/(rho**2+(1-zeta)**2)**(1.5)
        ur0=(eps**3)*0.25*rho/(rho**2+(1-zeta)**2)**(1.5)
        
        #Auz1=self.auz1(nu,rho,zeta)
        Auz1=self.uadd(nu,rho,zeta,typ=0,coord='z')
        #Aur1=self.aur1(nu,rho,zeta)
        Aur1=self.uadd(nu,rho,zeta,typ=0,coord='r')
        
        R=np.sqrt(rho**2+(1-zeta)**2)
        sint=rho/R
        cost=(1-zeta)/R
        C3=[eps*(1+nu)/(12*(1-nu)),5*(eps**3)*(2-nu)/(24*(7-5*nu))]
        D3=[-(eps**3)*(1+nu)/12,(eps**5)*(2-nu)/(4*(7-nu))]
        P0=1
        P2=0.5*(3*cost**2-1)
        dP0=0
        dP2=3*cost*sint
        #dP2=3*cost
        ur38=-0.5*P0*D3[0]/R**2+(C3[1]*(5-4*nu)-1.5*D3[1]/R**2)*P2/R**2
        ut39=-(2*C3[0]*(1-nu)-0.5*D3[0]/R**2)*dP0-(C3[1]*(1-2*nu)+0.5*D3[1]/R**2)*dP2/R**2
        
        ut39=ut39*sint
        Auz3=ur38*cost-ut39*sint
        Aur3=ur38*sint+ut39*cost
        
        #Auz6=self.auz6(nu,rho,zeta)
        Auz6=self.uadd(nu,rho,zeta,typ=1,coord='z')
        #Aur6=self.aur6(nu,rho,zeta)
        Aur6=self.uadd(nu,rho,zeta,typ=1,coord='r')
        #print(Auz1,Auz6,Aur1,Aur6)
        uz=uz0+(eps**3)*(Auz1+Auz3)+(eps**6)*Auz6
        #uz=uz0+(eps**3)*(Auz1+Auz3)
        ur=ur0+(eps**3)*(Aur1+Aur3)+(eps**6)*Aur6
        #ur=ur0+(eps**3)*(Aur1+Aur3)
       
        ux=ur*x/r
        uy=ur*y/r
        
        ux=ux*dP*d/mu
        uy=uy*dP*d/mu
        uz=uz*dP*d/mu
        
        return ux, uy, uz

    def uadd(self,nu,r,zeta,typ=0,coord='r'):
        from hankel import HankelTransform
        import hankel
        
        #sigma=lambda tt: (1-typ)*0.5*np.exp(-tt)+typ*(1.5*(1+tt)*np.exp(-tt)/(7-5*nu))#Doesn't include extra tt because Hankel transform ends with tt*dtt
        sigma1=lambda tt: 0.5*np.exp(-tt)
        sigma2=lambda tt: 1.5*(1+tt)*np.exp(-tt)/(7-5*nu)
        tau2= lambda tt: tt*np.exp(-tt)/(7-5*nu)
        #tau=lambda tt: (1-typ)*0.5*np.exp(-tt)+typ*tt*np.exp(-tt)/(7-5*nu)#Doesn't include extra tt because Hankel transform ends with tt*dtt
        f1=lambda tt,ex: 0.5*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)
        f2=lambda tt,ex: 0.5*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)

        if coord=='r':
            if typ==0:
                a8= lambda tt: sigma1(tt)*f2(tt,0)
                a18= lambda tt: sigma1(tt)*f1(tt,0)
            else:
                a8= lambda tt: sigma2(tt)*f2(tt,0)
                a18= lambda tt: tau2(tt)*f1(tt,0)
            dur= lambda tt: a8(tt)+a18(tt)
            #dur= lambda tt: (a8(tt)+a18(tt))*scipy.special.jv(1, tt*r)*tt
            #h=hankel.get_h(dur,nu=1)
            ht = HankelTransform(nu=1, N=1000, h=0.001)    
            result = ht.transform(dur, r)
            
            return result[0]
            #return quad_vec(dur,0,50)[0]
        else:
            if typ==0:
                a7= lambda tt: 0.5*0.5*(2*(1-nu)+tt*zeta)*np.exp(tt*(-zeta-1))
                a17= lambda tt: 0.5*0.5*((1-2*nu)+tt*zeta)*np.exp(tt*(-zeta-1))
            else:
                a7= lambda tt: 0.5*(2*(1-nu)+tt*zeta)*tt*np.exp(tt*(-zeta-1))/(7-5*nu)#(1.5*(1+tt))
                a17= lambda tt: 0.5*((1-2*nu)+tt*zeta)*tt*np.exp(tt*(-zeta-1))/(7-5*nu)
            duz= lambda tt: a7(tt)+a17(tt)
            #h=hankel.get_h(duz,nu=0) 
            R=np.sqrt(r**2+zeta**2)
            #R=r
            #duz= lambda tt: (a7(tt)+a17(tt))*scipy.special.jv(0, tt*R)*tt
            
            #return quad_vec(duz,0,50)[0]

            ht = HankelTransform(nu=0, N=1000, h=0.001)    
            
            result = ht.transform(duz, R)

            return result[0]
    def uadd1(self,nu,r,zeta,typ=0,coord='r'):
        from hankel import HankelTransform
        ht0 = HankelTransform(nu=0, N=1000, h=0.001)
        ht1 = HankelTransform(nu=1, N=1000, h=0.001)
        if typ==0 and coord=='z':
            R=np.sqrt(r**2+zeta**2)
            #R=r
            a7=lambda tt: 0.5*0.5*np.exp(-tt)*(2*(1-nu)-tt*zeta)*np.exp(tt*zeta)
            a17=lambda tt: 0.5*0.5*np.exp(-tt)*((1-2*nu)-tt*zeta)*np.exp(tt*zeta)
            duz=lambda tt: a7(tt)+a17(tt)
            return ht0.transform(duz, R)[0]
            #duz=lambda tt: (a7(tt)+a17(tt))*tt*scipy.special.jv(0, tt*R)
            #return quad_vec(duz,0,50)[0]
        elif typ==1 and coord=='z':
            R=np.sqrt(r**2+zeta**2)
            #R=r
            a7=lambda tt: 0.5*(tt*np.exp(-tt)/(7-5*nu))*(2*(1-nu)-tt*zeta)*np.exp(tt*zeta)
            a17=lambda tt: 0.5*(tt*np.exp(-tt)/(7-5*nu))*((1-2*nu)-tt*zeta)*np.exp(tt*zeta)
            duz=lambda tt: a7(tt)+a17(tt)
            return ht0.transform(duz, R)[0]
            #duz=lambda tt: (a7(tt)+a17(tt))*tt*scipy.special.jv(0, tt*R)
            #return quad_vec(duz,0,50)[0]
        elif typ==0 and coord=='r':
            aur=r*np.nan
            R=r
            #R=np.sqrt(r**2+zeta**2)
            a18=lambda tt: 0.5*0.5*np.exp(-tt)*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)
            a8=lambda tt: 0.5*0.5*np.exp(-tt)*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)
            dur=lambda tt: a8(tt)+a18(tt)
            return ht1.transform(dur, R)[0]
            #dur=lambda tt: (a8(tt)+a18(tt))*tt*scipy.special.jv(1, tt*R)
            #return quad_vec(dur,0,50)[0]
        elif typ==1 and coord=='r':
            R=r
            #R=np.sqrt(r**2+zeta**2)
            a18=lambda tt: 0.5*(tt*np.exp(-tt)/(7-5*nu))*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)
            a8=lambda tt: 0.5*(1.5*(1+tt)*np.exp(-tt)/(7-5*nu))*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)
            dur=lambda tt: a8(tt)+a18(tt)
            return ht1.transform(dur, R)[0]
            #dur=lambda tt: (a8(tt)+a18(tt))*tt*scipy.special.jv(1, tt*R)
            #return quad_vec(dur,0,50)[0]
    
    def auz1(self,nu,r,zeta):
        """
        Auxiliary function to calculate displacements (for detail check McTigue, 1987)
        """
        R=np.sqrt(r**2+zeta**2)
        #R=r
        sigma=lambda tt: 0.5*tt*np.exp(-tt)
        a7=lambda tt: 0.5*sigma(tt)*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(0, tt*R)
        a17=lambda tt: 0.5*sigma(tt)*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(0, tt*R)
        duz=lambda tt: a7(tt)+a17(tt)
        return quad_vec(duz,0,50)[0]
    
    def auz6(self,nu,r,zeta):
        """
        Auxiliary function to calculate displacements (for detail check McTigue, 1987)
        """
        R=np.sqrt(r**2+zeta**2)
        #R=r
        sigma=lambda tt: (tt**2)*np.exp(-tt)/(7-5*nu)
        tau=lambda tt: tt**2*np.exp(-tt)/(7-5*nu)
        a7=lambda tt: 0.5*sigma(tt)*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(0, tt*R)
        a17=lambda tt: 0.5*tau(tt)*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(0, tt*R)
        duz=lambda tt: a7(tt)+a17(tt)
        return quad_vec(duz,0,50)[0]
    
    def aur1(self,nu,r,zeta):
        """
        Auxiliary function to calculate displacements (for detail check McTigue, 1987)
        """
        aur=r*np.nan
        R=r
        #R=np.sqrt(r**2+zeta**2)
        sigma=lambda tt: 0.5*tt*np.exp(-tt)
        a18=lambda tt: 0.5*sigma(tt)*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(1, tt*R)
        a8=lambda tt: 0.5*sigma(tt)*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(1, tt*R)
        dur=lambda tt: a8(tt)+a18(tt)
        return quad_vec(dur,0,50)[0]
    
    def aur6(self,nu,r,zeta):
        """
        Auxiliary function to calculate displacements (for detail check McTigue, 1987)
        """
        aur=r*np.nan
        R=r
        #R=np.sqrt(r**2+zeta**2)
        sigma=lambda tt: 1.5*(tt+tt**2)*np.exp(-tt)/(7-5*nu)
        tau=lambda tt: tt**2*np.exp(-tt)/(7-5*nu)
        a18=lambda tt: 0.5*tau(tt)*(2*(1-nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(1, tt*R)
        a8=lambda tt: 0.5*sigma(tt)*((1-2*nu)-tt*zeta)*np.exp(-tt*zeta)*scipy.special.jv(1, tt*R)
        dur=lambda tt: a8(tt)+a18(tt)
        return quad_vec(dur,0,50)[0]