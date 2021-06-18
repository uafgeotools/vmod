"""
Functions for forward volcano-geodesy analytic models

Author: Scott Henderson
Date: 8/31/2012



TODO:
-benchmark codes against paper results
-test for georeferenced coordinate grids?
-add function to convert to los
-add sphinx docstrings
"""
import numpy as np
import util
import pandas as pd

class Data:
    def __init__(self):
        self.data = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','ux','uy','uz','sx','sy','sz'])

    def add(self, id, lat, lon, height, x, y, ux, uy, uz, sx, sy, sz):
        self.data.loc[len(self.data.index)] = [id] + list((lat,lon,height,x,y,ux,uy,uz,sx,sy,sz))

    def get_xs(self):
        return self.data['x'].to_numpy()

    def get_ys(self):
        return self.data['y'].to_numpy()

    def get_obs(self):
        ''' returns single vector with [ux1...uxN,uy1...uyN,uz1,...,uzN] as elements'''
        return self.data[['ux','uy','uz']].to_numpy().flatten(order='F')


class Source:
    def __init__(self, data):
        self.data  = data
        self.model = None

    def get_obs(self):
        return self.data.get_obs()

    def get_xs(self):
        return self.data.get_xs()

    def get_ys(self):
        return self.data.get_ys()

    ##inversion methods
    def invert(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = least_squares(self.fun, x0, bounds=bounds)
        return self.model

    def invert_bh(self, x0):
        from scipy.optimize import basinhopping
        return basinhopping(self.fun, x0)

    ##output writers
    def write_forward_gmt(self, prefix):
        if self.model is not None:
            ux,uy,uz = self.forward()

            dat = np.zeros(self.data.data['id'].to_numpy().size, 
                dtype=[ ('lon', float), ('lat', float), ('east', float), ('north', float), 
                        ('esig', float), ('nsig', float), ('corr', float), ('id', 'U6')] )

            dat['lon']   = self.data.data['lon'].to_numpy()
            dat['lat']   = self.data.data['lat'].to_numpy()
            dat['east']  = ux
            dat['north'] = uy
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.data.data['id'].to_numpy()

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )


class Yang(Source):

    ##residual functin for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0], ycen=x[1], z0=x[2], P=x[3], a=x[4], b=x[5], phi=x[6], theta=x[7])
        return np.concatenate((ux,uy,uz))-self.get_obs()


# ================
# Forward models
# ================
    def forward(self, xcen=0,ycen=0,z0=5e3,P=10,a=2,b=1,phi=0,theta=0,mu=1.0,nu=0.25):
    #def forward(params,x,y,matrl,tp):
        '''
        Ellipsoidal pressurized chamber in elastic half-space
        Yang et al., vol 93, JGR, 4249-4257, 1988)     
        
        Inputs:
        -------
        xargs   [eastings, northings, incidence, heading] list of arrays
        xcen    source easting epicenter [km]
        ycen    source northing epicenter [km]
        z0      source depth [km]
        P       excess pressure [mu*10^(-5) Pa] NOTE: weird units
        a       major axis [km]
        b       minor axis [km]
        phi     strike [degrees clockwise from north]
        theta   dip [degrees from horizontal]
        mu      normalized shear modulus [unitless]
        nu      poisson ratio [unitless]
        '''
        # Parameter Checks
        '''
        if a < b:
            print('Error: a must be >= b')
            return
        # Make sure source is inside the grid
        # Shift grid & Make sure source is inside the grid
        minx = np.min(x) #redundant syntax b/c automatically flattens...
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)
        if (xcen < minx) or (xcen > maxx) or (ycen < miny) or (ycen > maxy):
            print('Error: ({0}, {1}) lies outside grid Easting[{2:g}, {3:g}] Northing[{4:g}, {5:g}]'.format(xcen,ycen,minx,maxx,miny,maxy))
            return    
        '''
        
        tp = 0 #topography vector?
        phi = np.deg2rad(phi) 
        theta = np.deg2rad(phi)  
            
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
        
        xn = self.x - xcen
        yn = self.y - ycen
        
        # Call spheroid function to get geometric paramters
        # NOTE: had to add file name
        sph = self.spheroid(a,b,c,matrl,phi,theta,P)
        
        # Rotate points
        xp = xn * cosp + yn * sinp
        yp = yn * cosp - xn * sinp
        
        # Run forward model ?why called twice?, for each side of model...
        xi = c
        Up1,Up2,Up3 = self.yang(sph,xi,z0,xp,yp,0,matrl,e_theta,coeffs,tp)
        xi = -xi
        Um1,Um2,Um3 = self.yang(sph,xi,z0,xp,yp,0,matrl,e_theta,coeffs,tp)
        
        # Sum
        U1r = -Up1 + Um1
        U2r = -Up2 + Um2
        
        # Rotate horiz. displacements back to the orig. coordinate system
        U1 = U1r * cosp - U2r * sinp
        U2 = U1r * sinp + U2r * cosp
        U3 = Up3 - Um3
        
        return U1,U2,U3


    def spheroid(self, a,b,c,matrl,phi,theta,P):
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


    def yang(self, sph,xi,z0,x,y,z,matrl,e_theta,coeffs,tp):
        #epsn=1e-15 #NOTE: yellow underline indicates variable not used
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
        
        C0 = y0 * costh + z00 * sinth # check this?
        
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
        
        u2 = coeff*(sinth*(A1star*r2+(nu4*Abar1star+F1star)*q2) +
                    costh*(Bstar-F2star) + 2*sinth*costh*z*Abar1star)
        
        u3 = coeff*(-costh*(Abar1star*r2+(nu4*Abar1star-F1star)*q2) +
                    sinth*(Bstar+F2star) + 2*(costh)**2*z*Abar1star)
        
        u1 = u1 + 2*coeff*Pdila*((A1 + nu4*Abar1 + F1)*y1 - coeffs[2]*ff1)
        
        u2 = u2 + 2*coeff*Pdila*(sinth*(A1*r2+(nu4*Abar1+F1)*q2) -
                                 coeffs[2]*ff2 + 4*nu1*costh*(A2+Abar2) +
                                 costh*(A3 - nu4*Abar3 - F2))
        
        u3 = u3 + 2*coeff*Pdila*(costh*(-A1*r2 + (nu4*Abar1 + F1)*q2) + coeffs[2]*ff3 +
                                 4*nu1*sinth*(A2+Abar2) +
                                 sinth*(A3 + nu4*Abar3 + F2 - 2*nu4*B))
        
        
        return u1,u2,u3

class Mogi(Source):

    ##residual functin for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0], ycen=x[1], d=x[2], dV=x[3])
        diff =np.concatenate((ux,uy,uz))-self.get_obs()
#        print("Mogi res norm = %f" % np.linalg.norm(diff))
        return diff

    # =====================
    # Forward Models
    # =====================
    def forward(self):
        return self.forward(self.model.x[0], self.model.x[1], self.model.x[2], self.model.x[3])

    def forward(self, xcen=0, ycen=0, d=3e3, dV=1e6, nu=0.25):
        """
        Calculates surface deformation based on point pressure source
        References: Mogi 1958, Segall 2010 p.203

        Args:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)

        Kwargs:
        -----------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to point (m)
        dV: change in volume (m^3)
        nu: poisson's ratio for medium

        Returns:
        -------
        (ux, uy, uz)


        Examples:
        --------

        """
        # Center coordinate grid on point source
        x = self.get_xs() - xcen
        y = self.get_ys() - ycen

        # Convert to surface cylindrical coordinates
        th, rho = util.cart2pol(x,y) # surface angle and radial distance
        R = np.sqrt(d**2+rho**2) # radial distance from source

        # Mogi displacement calculation
        C = ((1-nu) / np.pi) * dV
        ur = C * rho / R**3    # horizontal displacement, m
        uz = C * d / R**3      # vertical displacement, m

        ux, uy = util.pol2cart(th, ur)
        #return ux, uy, uz #returns tuple
        return np.array([ux,uy,uz])

    def forward_dp(self, xcen=0,ycen=0,d=3e3,a=500,dP=100e6,mu=4e9,nu=0.25):
        """
        dP instead of dV, NOTE: dV = pi * dP * a**3 / mu
        981747.7 ~ 1e6
        """
        dV = np.pi * dP * a**3 / mu
        return self.forward(xcen,ycen,d,dV,nu)

    def calc_linmax(self,tn,xcen=0,ycen=0,d=3e3,a=500.0,dP=100e6,mu=4e9,nu=0.25):
        """ Solution for spherical source in a Maxwell Solid viscoelastic halfspace
        based on Bonafede & Ferrari 2009 (Equation 14).

        Simplified equations for z=0 free surface, for instantaneous dP at t=0.

        Note that displacements are in m, but inputs are normalized time.

        Required arguments:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)
        t: normalized time (t/tau)

        Keyword arguments:
        -----------------
        xcen: y-offset of point source epicenter (m)
        ycen: y-offset of point source epicenter (m)
        d: depth to point (m)
        dP: change in pressure at t=0 (Pa)
        K: short-term elastic incompressibility () NOTE=(5/3)mu in poisson approximation
        mu: short-term elastic rigidity (Pa)
        eta: effectice viscosity (Pa*s)
        output: 'cart' (cartesian), 'cyl' (cylindrical)

        """
        # center coordinate grid on point source
        x = self.x - xcen
        y = self.y - ycen

        # convert to surface cylindrical coordinates
        th, r = util.cart2pol(x,y)
        R = np.hypot(d,r)

        # Common variables
        K = (2.0*mu*(1 + nu)) / (3*(1 - 2*nu))
        A = 1 + 2.0*(mu/K)
        B = (2.0*mu**2) / (K*(3*K + mu))
        alpha = (3.0*K + mu) / (3*K)
        #tau = nu / mu #maxwell relaxation time
        #tau_a = alpha * tau
        #Rstar = R #NOTE, this is only true for solution on free surface

        term = 1 + A - (B*np.exp(-tn/alpha)) + (2*tn)

        C = (dP * a**3) / (4*mu)
        ur = C * (r/(R**3)) * term
        uz = C * (d/(R**3)) * term

        print('uz_max = {:.4f}'.format(uz.max()))
        print('ur_max = {:.4f}'.format(ur.max()))

        # Convert surface cylindrical to cartesian
        #if output == 'cart':
        ux, uy = util.pol2cart(th, ur)
        return np.array([ux, uy, uz])
        # returning numpy array allows scaling as:
        #ux,uy,uz = forward(**) * 1e2 + 0.1
        #elif output == 'cyl':
        #    return ur, uz


    def calc_linmax_dPt(self, tn,dVdt,xcen=0,ycen=0,d=3e3,a=500.0,dP=100e6,mu=4e9,
                        nu=0.25):
        """ Instead of constant pressure, have pressure determined by a constant
        supply rate of magma

        From Bonafede 2009 Equation 16

        NOTE: Only Uzmax b/c full solution not given in Segaall
        """
        K = (2.0*mu*(1 + nu)) / (3*(1 - 2*nu))
        tau = nu / mu
        #tauB = ((3*K + 4*mu) / (3*K)) * tau
        tauA = ((3*K + mu) / (3*K)) * tau

        C = (dVdt*tau) / (2*np.pi*d**2)
        term1 = tn/tau
        term2 = (mu / (3*K)) * (1 - np.exp(-tn/tauA))
        uzmax = C * (term1 - term2)
        return uzmax


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
        x = self.x - xcen
        y = self.y - ycen

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



    def calc_mctigue(self, xcen=0,ycen=0,d=3e3,dP=10e6,a=1500.0,nu=0.25,mu=4e9,
                     terms=1):
        """
        3d displacement field from dislocation point source (McTigue, 1987)
        Caution: analysis done in non-dimensional units!
        see also Segall Ch7 p207
        Same as forward, except change in pressure and chamber radius are specified

        Keyword arguments:
        ------------------
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

        Set terms=1 to reduce to Mogi Solution
        NOTE: eps**6 term only significant if eps > 0.5
        """
        # center coordinate grid on point source
        x = self.x - xcen
        y = self.y - ycen

        # dimensionless scaling term
        scale = dP * d / mu
        eps = a / d #NOTE eps = 0.5 # mctigue fig3

        # convert to surface cylindrical coordinates
        th, r = util.cart2pol(x,y)
        r = r / d #dimensionless radial distance

        # 1st order mctigue is essentially Mogi solution
        uz = eps**3 * ((1-nu) * (1 / np.hypot(r,1)**3))
        ur = eps**3 * ((1-nu) * (r / np.hypot(r,1)**3))

        # 2nd order term
        if terms==2:
            print('adding eps**6 term')
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


    def calc_viscoshell(self,t,xcen=0,ycen=0,d=4e3,a=1000.0,b=1200.0,dP=100e6,
                        mu=30e9,nu=0.25,eta=2e16):
        """ Spherical Source surronded by a viscoelastic shell in an elastic halfspace
        Derivation of equations 7.105 in Segall Ch.7 p245
        NOTE: good approximation if duration of intrusion << relation time tR

        Required arguments:
        ------------------
        x: x-coordinate grid (m)
        y: y-coordinate grid (m)
        t: time (s)

        Keyword arguments:
        -----------------
        Same as forward_dp() plus:
        a: inner chamber radius
        b: extent of viscoelastic region
        eta: viscosity
        """
        # characteristic relaxation time
        tR = (3*eta*(1-nu)*b**3) / (mu*(1+nu)*a**3)

        #avoid ZeroDivisionError
        if tR == 0:
            scale = 1
        else:
            scale = (np.exp(-t/tR) + ((b/a)**3)*(1.0 - np.exp(-t/tR)))

        #rho = np.hypot(x,y)
        rho = np.hypot(self.x, self.y) / d #dimensionless!
        uz = (((1-nu)*dP*a**3) / (mu*d**2)) * scale * (1 + rho**2)**(-3/2.0)
        ur = rho * uz

        #print('uz_max = {:.4f}'.format(uz.max()))
        #print('ur_max = {:.4f}'.format(ur.max()))

        return ur, uz


    def calc_viscoshell_dPt(self,t,P0,tS,xcen=0,ycen=0,d=4e3,a=1000.0,b=1200.0,
                                 mu=30e9,nu=0.25,eta=2e16):
        """
        Viscoelastic shell with a exponentially decaying pressure source
        from Segall 2010

        P0 = initial pressure at t0
        tS = relaxation time of pressure source
        NOTE: tS & tR should not equal zero, or else division by zero
        NOTE: eq. 7.113 has an error, when tS=tR, should have t/tR in the formula
        """
        # viscoelastic relaxation time
        tR = (3*eta*(1-nu)*b**3) / (mu*(1+nu)*a**3)
        # pressure source relaxation time = ts

        # Pressure history
        P = P0 * (1 - np.exp(-t/tS))

        # Lambda coefficient
        Lambda = ((tS/tR)*(b/a)**3 - 1) / ((tS/tR) - 1)

        # Scale term
        if tR == tS:
            scale = ( (1-(b/a)**3) * (t/tR) * np.exp(-t/tR) +
                     ((b/a)**3) * (1.0 - np.exp(-t/tR)))
        else:
            scale = (Lambda*np.exp(-t/tR) -
                     Lambda*np.exp(-t/tS) +
                     ((b/a)**3)*(1.0 - np.exp(-t/tR)))

        #rho = np.hypot(x,y)
        rho = np.hypot(self.x,self.y) / d #Dimensionless radius!

        uz = (((1-nu)*P0*a**3) / (mu*d**2)) * (1 + rho**2)**(-3/2.0) * scale
        ur = rho * uz

        print('uz_max = {:.4f}'.format(uz.max()))
        print('ur_max = {:.4f}'.format(ur.max()))

        return ur, uz, P

    def calc_uzmax(self, dP=10e6,a=5e2,d=3e3,mu=30e9,nu=0.25):
        uzmax = ((1-nu)*dP*a**3) / (mu*d**2)
        return uzmax


    def dP2dV(self, dP,a,mu=30e9):
        dV = (np.pi * dP * a**3) / mu
        return dV

    def dV2dP(self,dV,a,mu=30e9):
        dP = (dV * mu) / (np.pi * a**3)
        return dP
