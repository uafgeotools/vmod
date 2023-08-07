from __future__ import division
import numpy as np
from .. import util
import scipy
from . import Source

class Yang(Source):
    """
    Class that represents a pressurized well following the implementation from Yang et., al. (1988).

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
        return "Yang"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        
        Returns:
            steps (int): Number of steps used in the bayesian inversions.
            burnin (int): discarded number of steps at the begining of the inversion.
            thin (int): number of steps per sample.
        """
        steps=5100000
        burnin=100000
        thin=1000
        return steps,burnin,thin

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Yang")
        print("\txcen = %f" % x[0])
        print("\tycen = %f" % x[1])
        print("\tz0 = %f" % x[2])
        print("\tP= %f" % x[3])
        print("\ta = %f" % x[4])
        print("\tb = %f" % x[5])
        print("\tphi = %f" % x[6])
        print("\ttheta= %f" % x[7])
        print("\tmu = %f" % x[8])
        
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","pressure","a","b","az","dip")
        
    ##residual function for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0],ycen=x[1],z0=x[2],P=x[3],a=x[4],b=x[5],phi=x[6],theta=x[7],mu=x[8])
        diff =np.concatenate((ux,uy,uz))-self.get_obs()
        return diff

    # =====================
    # Forward Models
    # =====================
    
    def model(self,x,y,xcen=0,ycen=0,z0=5e3,P=10,a=2,b=1,phi=0,theta=0,mu=9.6e9,nu=0.25):
        """
        Computes surface deformation due to a prolate spheroid pressurized chamber in elastic half-space
        Yang et al., vol 93, JGR, 4249-4257, 1988)     

        Parameters:
            x: x-coordinate (m)
            y: y-coordinate (m)
            xcen: source easting epicenter (m)
            ycen: source northing epicenter (m)
            z0: source depth (m)
            P: excess pressure (Pa)
            a: major axis [km]
            b: minor axis [km]
            phi: strike (degrees clockwise from north)
            theta: dip (degrees from horizontal)
            mu: normalized shear modulus (Pa)
            nu: poisson ratio (unitless)
        
        Returns:
            ux: deformation in the x-axis (m)
            uy: deformation in the y-axis (m)
            uz: deformation in the z-axis (m)
        """
        
        d_crita = a * np.sin(np.deg2rad(theta))
        d_critb = b * np.cos(np.deg2rad(theta))
        
        nans=np.array([x*np.nan,x*np.nan,x*np.nan])
        if d_crita>=z0 or b>=z0:
            return nans
        if theta==0:
            theta=0.1
        elif theta==90:
            theta=89.9

        tp = 0 #topography vector?
        phi = np.deg2rad(phi) 
        theta = np.deg2rad(theta)  

        # Store some commonly used parameters (material properties)
        coeffs = np.zeros(3)
        coeffs[0] = 1 / (16 * mu * (1 - nu))
        coeffs[1] = 3 - 4 * nu
        coeffs[2] = 4 * (1 - nu) * (1 - 2 * nu)    
        # Elastic constant array
        matrl = np.array([mu,mu,nu])   

        # Geometery
        e_theta = np.zeros(2)
        e_theta[0] = np.sin(theta)
        e_theta[1] = np.cos(theta)
        cosp = np.cos(phi)
        sinp = np.sin(phi)
        c = np.sqrt(a ** 2 - b ** 2)

        xn = x - xcen
        yn = y - ycen

        # Call spheroid function to get geometric paramters
        # NOTE: had to add file name
        sph = Yang.spheroid(a,b,c,matrl,phi,theta,P)

        # Rotate points
        xp = xn * cosp - yn * sinp
        yp = yn * cosp + xn * sinp

        # Run forward model ?why called twice?, for each side of model...
        xi = c
        Up1,Up2,Up3 = Yang.yang(sph,xi,z0,xp,yp,0,matrl,e_theta,coeffs,tp)
        xi = -xi
        Um1,Um2,Um3 = Yang.yang(sph,xi,z0,xp,yp,0,matrl,e_theta,coeffs,tp)

        # Sum
        U1r = -Up1 + Um1
        U2r = -Up2 + Um2

        # Rotate horiz. displacements back to the orig. coordinate system
        U1 = U1r * cosp + U2r * sinp
        U2 = -U1r * sinp + U2r * cosp
        U3 = Up3 - Um3

        return U1,U2,U3
    
    def model_depth(self,x,y,z,xcen=0,ycen=0,z0=5e3,P=10,a=2,b=1,phi=0,theta=0,mu=9.6e9,nu=0.25):
        """
        Computes deformation at depth due to a prolate spheroid pressurized chamber in elastic half-space
        Yang et al., vol 93, JGR, 4249-4257, 1988)     

        Parameters:
            x: x-coordinate (m)
            y: y-coordinate (m)
            z: depth-coordinate (m)
            xcen: source easting epicenter (m)
            ycen: source northing epicenter (m)
            z0: source depth (m)
            P: excess pressure (Pa)
            a: major axis [km]
            b: minor axis [km]
            phi: strike (degrees clockwise from north)
            theta: dip (degrees from horizontal)
            mu: normalized shear modulus (Pa)
            nu: poisson ratio (unitless)
        
        Returns:
            ux: deformation in the x-axis (m)
            uy: deformation in the y-axis (m)
            uz: deformation in the z-axis (m)
        """
        if theta==0:
            theta=0.1
        elif theta==90:
            theta=89.9

        tp = 0 #topography vector?
        phi = np.deg2rad(phi) 
        theta = np.deg2rad(theta)  

        # Store some commonly used parameters (material properties)
        coeffs = np.zeros(3)
        coeffs[0] = 1 / (16 * mu * (1 - nu))
        coeffs[1] = 3 - 4 * nu
        coeffs[2] = 4 * (1 - nu) * (1 - 2 * nu)    
        # Elastic constant array
        matrl = np.array([mu,mu,nu])   

        # Geometery
        e_theta = np.zeros(2)
        e_theta[0] = np.sin(theta)
        e_theta[1] = np.cos(theta)
        cosp = np.cos(phi)
        sinp = np.sin(phi)
        c = np.sqrt(a ** 2 - b ** 2)

        xn = x - xcen
        yn = y - ycen

        # Call spheroid function to get geometric paramters
        # NOTE: had to add file name
        sph = Yang.spheroid(a,b,c,matrl,phi,theta,P)

        # Rotate points
        xp = xn * cosp - yn * sinp
        yp = yn * cosp + xn * sinp

        # Run forward model ?why called twice?, for each side of model...
        xi = c
        Up1,Up2,Up3 = Yang.yang(sph,xi,z0,xp,yp,z,matrl,e_theta,coeffs,tp)
        xi = -xi
        Um1,Um2,Um3 = Yang.yang(sph,xi,z0,xp,yp,z,matrl,e_theta,coeffs,tp)

        # Sum
        U1r = -Up1 + Um1
        U2r = -Up2 + Um2

        # Rotate horiz. displacements back to the orig. coordinate system
        U1 = U1r * cosp + U2r * sinp
        U2 = -U1r * sinp + U2r * cosp
        U3 = Up3 - Um3
        

        return U1,U2,U3

    def spheroid(a,b,c,matrl,phi,theta,P):
        ''' 
        Geometry used in yang pressure source computation 
        '''
        pi = np.pi    
        lamda = matrl[0]
        mu = matrl[1]
        nu = matrl[2]

        ac = (a - c) / (a + c)
        L1 = np.log(ac)
        iia = 2 / a / c ** 2 + L1 / c ** 3
        iiaa = 2 / 3 / a ** 3 / c ** 2 + 2 / a / c ** 4 + L1 / c ** 5
        coef1 = -2 * pi * a * b ** 2
        Ia = coef1 * iia
        Iaa = coef1 * iiaa

        u = 8 * pi * (1 - nu)
        Q = 3 / u
        R = (1 - 2 * nu) / u

        a11 = 2 * R * (Ia - 4 * pi)
        a12 = -2 * R * (Ia + 4 * pi)
        a21 = Q * a ** 2 * Iaa + R * Ia - 1
        a22 = -(Q * a ** 2 * Iaa + Ia * (2 * R - Q))

        coef2 = 3 * lamda + 2 * mu
        w = 1 / (a11 * a22 - a12 * a21)
        e11 = (3 * a22 - a12) * P * w / coef2
        e22 = (a11 - 3 * a21) * P * w / coef2

        Pdila = 2 * mu * (e11 - e22)
        Pstar = lamda * e11 + 2 * (lamda + mu) * e22
        a1 = -2 * b ** 2 * Pdila
        b1 = 3 * b ** 2 * Pdila / c ** 2 + 2 * (1 - 2 * nu) * Pstar

        sph = np.zeros(10)
        sph[0] = a
        sph[1] = b
        sph[2] = c
        sph[3] = phi
        sph[4] = theta
        sph[5] = Pstar
        sph[6] = Pdila
        sph[7] = a1
        sph[8] = b1
        sph[9] = P
        return sph


    def yang(sph,xi,z0,x,y,z,matrl,e_theta,coeffs,tp):
        """
        Auxiliary function for spheroidal model
        """
        pi = np.pi
        
        # Load required spheroid parameters
        a = sph[0]
        b = sph[1]
        c = sph[2]
        #phi=sph[3]
        #theta=sph[4]
        #Pstar=sph[5]
        Pdila = sph[6]
        a1 = sph[7]
        b1 = sph[8]
        #P=sph[9]

        sinth = e_theta[0]
        costh = e_theta[1]

        # Poisson's ratio, Young's modulus, and the Lame coeffiecents mu and lamda
        nu = matrl[2]
        nu4 = coeffs[1]
        #nu2=1 - 2 * nu
        nu1 = 1 - nu
        coeff = a * b ** 2 / c ** 3 * coeffs[0]

        # Introduce new coordinates and parameters (Yang et al., 1988, page 4251):
        xi2 = xi * costh
        xi3 = xi * sinth
        y0 = 0
        z00 = tp + z0
        x1 = x
        x2 = y - y0
        x3 = z - z00
        xbar3 = z + z00
        y1 = x1
        y2 = x2 - xi2
        y3 = x3 - xi3
        ybar3 = xbar3 + xi3
        r2 = x2 * sinth - x3 * costh
        q2 = x2 * sinth + xbar3 * costh
        r3 = x2 * costh + x3 * sinth
        q3 = -x2 * costh + xbar3 * sinth
        rbar3 = r3 - xi
        qbar3 = q3 + xi
        R1 = (y1 ** 2 + y2 ** 2 + y3 ** 2) ** (0.5)
        R2 = (y1 ** 2 + y2 ** 2 + ybar3 ** 2) ** (0.5)

        #C0 = y0 * costh + z00 * sinth # check this?
        C0=z0/sinth

        betatop = (costh * q2 + (1 + sinth) * (R2 + qbar3))
        betabottom = costh * y1
        # Strange replacement for matlab 'find'
        #nz=np.flatnonzero(np.abs(betabottom) != 0) #1D index
        nz  = (np.abs(betabottom) != 0) # 2D index
        atnbeta = pi / 2 * np.sign(betatop)
        # change atan --> arctan, and -1 not needed for 2D boolean index
        #atnbeta[(nz-1)]=np.atan(betatop[(nz-1)] / betabottom[(nz-1)])
        atnbeta[nz] = np.arctan(betatop[nz] / betabottom[nz])

        # Set up other parameters for dipping spheroid (Yang et al., 1988, page 4252):
        # precalculate some repeatedly used natural logs:
        Rr = R1 + rbar3
        Rq = R2 + qbar3
        Ry = R2 + ybar3
        lRr = np.log(Rr)
        lRq = np.log(Rq)
        lRy = np.log(Ry)

        # Note: dot products should in fact be element-wise multiplication
        #A1star=a1 / (R1.dot(Rr)) + b1 * (lRr + (r3 + xi) / Rr)
        #Abar1star=- a1 / (R2.dot(Rq)) - b1 * (lRq + (q3 - xi) / Rq)
        A1star =  a1 / (R1*Rr) + b1*(lRr + (r3 + xi)/Rr)
        Abar1star = -a1 / (R2*Rq) - b1*(lRq + (q3 - xi)/Rq)
        A1 = xi / R1 + lRr
        Abar1 = xi / R2 - lRq
        #A2=R1 - r3.dot(lRr)
        #Abar2=R2 - q3.dot(lRq)
        A2 = R1 - r3*lRr
        Abar2 = R2 - q3*lRq
        A3 = xi * rbar3 / R1 + R1
        Abar3 = xi * qbar3 / R2 - R2

        #B=xi * (xi + C0) / R2 - Abar2 - C0.dot(lRq)
        B = xi * (xi + C0)/R2 - Abar2 - C0*lRq
        Bstar = a1 / R1 + 2 * b1 * A2 + coeffs[1] * (a1 / R2 + 2 * b1 * Abar2)
        F1 = 0
        F1star = 0
        F2 = 0
        F2star = 0


        if z != 0:
            F1 = (-2*sinth*z* (xi*(xi+C0)/R2**3 +
                              (R2+xi+C0)/(R2*(Rq)) +
                              4*(1-nu)*(R2+xi)/(R2*(Rq))
                              )
                 )

            F1star = (2*z*(costh*q2*(a1*(2*Rq)/(R2**3*(Rq)**2) - b1*(R2 + 2*xi)/(R2*(Rq)**2)) +
                           sinth*(a1/R2**3 -2*b1*(R2 + xi)/(R2* (Rq)))
                           )
                     )

            F2 = -2*sinth*z*(xi*(xi+C0)*qbar3/R2**3 + C0/R2 + (5-4*nu)*Abar1)

            F2star = 2*z*(a1*ybar3/R2**3 - 2*b1*(sinth*Abar1 + costh*q2*(R2+xi)/(R2*Rq)))


        # Calculate little f's
        ff1 = (xi*y1/Ry +
               3/(costh)**2*(y1*lRy*sinth -y1*lRq + 2*q2*atnbeta) +
               2*y1*lRq -
               4*xbar3*atnbeta/costh
              )

        ff2 = (xi*y2/Ry +
               3/(costh)**2*(q2*lRq*sinth - q2*lRy + 2*y1*atnbeta*sinth + costh*(R2-ybar3)) -
               2*costh*Abar2 +
               2/costh*(xbar3*lRy - q3*lRq)
              )

        ff3 = ((q2*lRq - q2*lRy*sinth + 2*y1*atnbeta)/costh +
                2*sinth*Abar2 + q3*lRy - xi
              )


        # Assemble into x, y, z displacements (1,2,3):
        u1 = coeff*(A1star + nu4*Abar1star + F1star)*y1

        #u2 = coeff*(sinth*(A1star*r2+(nu4*Abar1star+F1star)*q2) +
        #            costh*(Bstar-F2star) + 2*sinth*costh*z*Abar1star)
        u2 = coeff*(sinth*(A1star*r2+(nu4*Abar1star+F1star)*q2) +
                    costh*(Bstar-F2star))
        #u3 = coeff*(-costh*(Abar1star*r2+(nu4*Abar1star-F1star)*q2) +
        #            sinth*(Bstar+F2star) + 2*(costh)**2*z*Abar1star)
        u3 = coeff*(-costh*(A1star*r2+(nu4*Abar1star-F1star)*q2) +
                    sinth*(Bstar+F2star))


        u1 = u1 + 2*coeff*Pdila*((A1 + nu4*Abar1 + F1)*y1 - coeffs[2]*ff1)

        u2 = u2 + 2*coeff*Pdila*(sinth*(A1*r2+(nu4*Abar1+F1)*q2) -
                                 coeffs[2]*ff2 + 4*nu1*costh*(A2+Abar2) +
                                 costh*(A3 - nu4*Abar3 - F2))

        u3 = u3 + 2*coeff*Pdila*(costh*(-A1*r2 + (nu4*Abar1 + F1)*q2) + coeffs[2]*ff3 +
                                 4*nu1*sinth*(A2+Abar2) +
                                 sinth*(A3 + nu4*Abar3 + F2 - 2*nu4*B))
        
        return u1,u2,u3