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
import copy

class Data:
    def __init__(self):
        self.data = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','ux','uy','uz','sx','sy','sz'])

    def add(self, id, lat, lon, height, x, y, ux, uy, uz, sx, sy, sz):
        self.data.loc[len(self.data.index)] = [id] + list((lat,lon,height,x,y,ux,uy,uz,sx,sy,sz))

    def get_xs(self):
        return self.data['x'].to_numpy()

    def get_ys(self):
        return self.data['y'].to_numpy()

    def get_zs(self):
        return self.data['y'].to_numpy()*0.0

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

    def get_zs(self):
        return self.data.get_zs()

    def res_norm(self):
        ux,uy,uz = self.forward_mod()
        return np.linalg.norm(self.get_obs()*1000-np.concatenate([ux, uy, uz])*1000)

    ##inversion methods
    def invert(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun, x0, bounds=bounds))
        return self.model

    def invert_dipole(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun_dipole, x0, bounds=bounds))
        return self.model

    def invert_bh(self, x0):
        from scipy.optimize import basinhopping
        return basinhopping(self.fun, x0)

    ##output writers
    def write_forward_gmt(self, prefix):
        if self.model is not None:
            ux,uy,uz = self.forward_mod()

            dat = np.zeros(self.data.data['id'].to_numpy().size, 
                dtype=[ ('lon', float), ('lat', float), ('east', float), ('north', float), 
                        ('esig', float), ('nsig', float), ('corr', float), ('id', 'U6')] )

            dat['lon']   = self.data.data['lon'].to_numpy()
            dat['lat']   = self.data.data['lat'].to_numpy()
            dat['east']  = ux*1000
            dat['north'] = uy*1000
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.data.data['id'].to_numpy()

            print(dat)

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz*1000
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )

#    def make_map(self, region):
#        import pygmt
#
#        fig = pygmt.Figure()
#        df = self.data.data(['lon','lat','ux','uy','sx','sy','id'])
#
#        df = pd.DataFrame(
#            data={
#            "x": self.get_xs()
#            "y": self.get_ys()
#            "east_velocity": [0, 3, 4, 6, -6, 6],
#            "north_velocity": [0, 3, 6, 4, 4, -4],
#            "east_sigma": [4, 0, 4, 6, 6, 6],
#            "north_sigma": [6, 0, 6, 4, 4, 4],
#            "correlation_EN": [0.5, 0.5, 0.5, 0.5, -0.5, -0.5],
#            "SITE": ["0x0", "3x3", "4x6", "6x4", "-6x4", "6x-4"],
#        }
#        )
#        fig.velo(
#            data=df,
#            region=[-10, 8, -10, 6],
#            pen="0.6p,red",
#            uncertaintycolor="lightblue1",
#            line=True,
#            spec="e0.2/0.39/18",
#            frame=["WSne", "2g2f"],
#            projection="x0.8c",
#            vector="0.3c+p1p+e+gred",
#        )#
#
#        fig.show()        

class Mogi(Source):

    ##residual function for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0], ycen=x[1], d=x[2], dV=x[3])
        diff =np.concatenate((ux,uy,uz))-self.get_obs()
#        print("Mogi res norm = %f" % np.linalg.norm(diff))
        return diff

    ##least_squares residaul function for dipole
    def fun_dipole(self, x):
        ux, uy, uz = self.forward_dipole(xcen1=x[0], ycen1=x[1], d1=x[2], dV1=x[3], xcen2=x[4], ycen2=x[5], d2=x[6], dV2=x[7])
        diff = np.concatenate((ux,uy,uz))-self.get_obs()
#        print("Mogi res norm = %f" % np.linalg.norm(diff))
        return diff

    # =====================
    # Forward Models
    # =====================
    def forward_mod(self):
        if len(self.model.x) == 4:
            return self.forward(self.model.x[0], self.model.x[1], self.model.x[2], self.model.x[3])
        elif len(self.model.x) == 8:
            return self.forward_dipole( self.model.x[0], self.model.x[1], self.model.x[2], self.model.x[3], 
                                        self.model.x[4], self.model.x[5], self.model.x[6], self.model.x[7])

    def forward(self, xcen, ycen, d, dV, nu=0.25):
       
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

    def forward_dipole(self, xcen1=0,ycen1=0,d1=3e3,dV1=10e6, xcen2=0,ycen2=0,d2=3e3,dV2=10e6, nu=0.25):
        """
        """
        ux1, uy1, uz1 = self.forward(xcen1,ycen1,d1,dV1,nu)
        ux2, uy2, uz2 = self.forward(xcen2,ycen2,d2,dV2,nu)

        return np.array([ux1+ux2, uy1+uy2, uz1+uz2])

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
        x = self.get_xs() - xcen
        y = self.get_ys() - ycen

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
        x = self.get_xs() - xcen
        y = self.get_ys() - ycen

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
        rho = np.hypot(self.get_xs(), self.get_ys()) / d #dimensionless!
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
        rho = np.hypot(self.get_xs(),self.get_ys()) / d #Dimensionless radius!

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


class Yang(Source):

    ##residual functin for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(x0=x[0], y0=x[1], z0=x[2], a=x[3], 
                           A=x[4], P_G=x[5], theta=x[6], phi=x[7])

        return np.concatenate((ux,uy,uz))-self.get_obs()

    # ================
    # Forward models
    # ================

    def _par_(self,a,b,lamb,mu,nu,P):
        # compute the parameters for the spheroid model
        # formulas from [1] Yang et al (JGR,1988)
        # corrections from [2] Newmann et al (JVGR, 2006), Appendix
        #
        # IN
        # a         semimajor axis [m]
        # b         semiminor axis [m]
        # lamb      Lame's constant [Pa]
        # mu        shear modulus [Pa]
        # nu        Poisson's ratio 
        # P         excess pressure (stress intensity on the surface) [pressure units]
        #
        # OUT
        # a1, b1    pressure (stress) [units of P] from [1]
        # c         prolate ellipsoid focus [m]
        # Pdila     pressure (proportional to double couple forces) [units of P] from [1]
        # Pstar     pressure [units of P]
        #
        # Notes:
        # [-]   : dimensionless
        epsn = 1E-10
        c = np.sqrt(a**2-b**2)                                 # prolate ellipsoid focus [m]

        a2 = a**2; a3 = a**3; b2 = b**2;
        c2 = c**2; c3 = c**3; c4 = c**4; c5 = c**5;
        ac = (a-c)/(a+c);                                   # [-]
        coef1 = 2*np.pi*a*b2;                               # [m^3]
        den1  = 8*np.pi*(1-nu);                             # [-]

        Q   = 3/den1;                                       # [-]       - parameter from [1]
        R   = (1-2*nu)/den1;                                # [-]       - parameter from [1]
        Ia  = -coef1*(2/(a*c2) + np.log(ac)/c3);            # [-]       - parameter from [1]
        Iaa = -coef1*(2/(3*a3*c2) + 2/(a*c4) + np.log(ac)/c5); # [1/m^2] - parameter from [1]

        a11 = 2*R*(Ia-4*np.pi);                             # [-]        - (A-1) from [2]
        a12 = -2*R*(Ia+4*np.pi);                            # [-]        - (A-2) from [2]
        a21 = Q*a2*Iaa + R*Ia - 1;                          # [-]        - (A-3) from [2]
        a22 = -Q*a2*Iaa - Ia*(2*R-Q);                       # [-]        - (A-4) from [2]

        den2 = 3*lamb+2*mu;                                 # [Pa]
        num2 = 3*a22-a12;                                   # [-]
        den3 = a11*a22-a12*a21;                             # [-]
        num3 = a11-3*a21;                                   # [-]

        Pdila= P*(2*mu/den2)*(num2-num3)/den3;              # [units of P]  - (A-5) from [2]
        Pstar= P*(1/den2)*(num2*lamb+2*(lamb+mu)*num3)/den3;# [units of P]  - (A-6) from [2]

        a1 = - 2*b2*Pdila;                                  # [m^2*Pa]  - force from [1]
        b1 = 3*(b2/c2)*Pdila + 2*(1-2*nu)*Pstar;            # [Pa]      - pressure from [1]

        return a1, b1, c, Pdila, Pstar

    def _int_(self,xx, yy, z0,theta,a1,b1,a,b,csi,mu,nu,Pdila):
        # compute the primitive of the displacement for a prolate ellipsoid
        # equation (1)-(8) from Yang et al (JGR, 1988) 
        # corrections to some parameters from Newmann et al (JVGR, 2006)
        # 
        # IN
        # x,y,z     coordinates of the point(s) where the displacement is computed [m], from object
        # y0,z0     coordinates of the center of the prolate spheroid (positive downward) [m]
        # theta     plunge angle [rad]
        # a1,b1     pressure (stress) (output from yangpar.m) [units of P]
        # a         semimajor axis [m]
        # b         semiminor axis [m]
        # c         focus of the prolate spheroid (output from yangpar.m) [m]
        # mu        shear modulus [Pa]
        # nu        Poisson's ratio
        # Pdila     pressure (proportional to double couple forces) [units of P]
        # 
        #
        # OUT
        # U1,U2,U3 : displacement in local coordinates [m] - see Figure 3 of Yang et al (1988)
        #
        # Notes:
        # The location of the center of the prolate spheroid is (x0,y0,z0)
        #     with x0=0 and y0=0;
        # The free surface is z=0;

        # precalculate parameters that are used often
        sint = np.sin(theta); cost = np.cos(theta)                                       # y0 = 0;

        # new coordinates and parameters from Yang et al (JGR, 1988), p. 4251
        # dimensions [m]
        csi2 = csi*cost; csi3 = csi*sint;                                           # see Figure 3 of Yang et al (1988)
        x1 = xx;  x2 = yy; x3 = self.get_zs() - z0; xbar3 = self.get_zs() + z0;
        y1 = x1; y2 = x2 - csi2; y3 = x3 - csi3; ybar3 = xbar3 + csi3;
        r2 = x2*sint - x3*cost; q2 =  x2*sint + xbar3*cost;
        r3 = x2*cost + x3*sint; q3 = -x2*cost + xbar3*sint;
        rbar3 = r3 - csi; qbar3 = q3 + csi;
        R1 = np.sqrt(y1**2 + y2**2 + y3**2); R2 = np.sqrt(y1**2 + y2**2 + ybar3**2);
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #C0 = y0*cost + z0*sint;
        C0= z0/sint;                                                                # correction base on test by FEM by P. Tizzani IREA-CNR Napoli
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        beta = (q2*cost + (1+sint)*(R2+qbar3)) / (cost*y1 + 1E-15);                  # add 1E-15 to avoid a Divide by Zero warning at the origin

        # precalculate parameters that are used often
        drbar3 = R1+rbar3; dqbar3 = R2+qbar3; dybar3 = R2+ybar3;
        lrbar3 = np.log(R1+rbar3); lqbar3 = np.log(R2+qbar3); lybar3 = np.log(R2+ybar3);
        atanb = np.arctan(beta);

        # primitive parameters from Yang et al (1988), p. 4252
        Astar1    =  a1 / (R1*drbar3) + b1*(lrbar3+(r3+csi) / drbar3);
        Astarbar1 = -a1 / (R2*dqbar3) - b1*(lqbar3+(q3-csi) / dqbar3);

        A1 = csi / R1 + lrbar3; Abar1 = csi / R2 - lqbar3;
        A2 = R1 - r3 * lrbar3;  Abar2 = R2 - q3 * lqbar3;
        A3 = csi * rbar3 / R1 + R1; Abar3 = csi * qbar3 / R2 - R2;

        Bstar = (a1/R1+2*b1*A2) + (3-4*nu)*(a1/R2+2*b1*Abar2);
        B = csi*(csi+C0)/R2 - Abar2 - C0*lqbar3;

        # the 4 equations below have been changed to improve the fit to internal deformation
        Fstar1 = 0; 
        Fstar2 = 0; 
        F1     = 0; 
        F2     = 0; 

        f1 = csi*y1/dybar3 + (3/cost**2)*(y1*sint*lybar3 - y1*lqbar3 + \
             2*q2*atanb) + 2*y1*lqbar3 - 4*xbar3*atanb/cost;
        f2 = csi*y2/dybar3 + (3/cost**2)*(q2*sint*lqbar3 - q2*lybar3 + \
             2*y1*sint*atanb + cost*(R2-ybar3)) - 2*cost*Abar2 + \
             (2/cost)*(xbar3*lybar3 - q3*lqbar3);                      # correction after Newmann et al (2006), eq (A-9)
        f3 = (1/cost)*(q2*lqbar3 - q2*sint*lybar3 + 2*y1*atanb) + 2*sint*Abar2 + q3*lybar3 - csi;


        # precalculate coefficients that are used often
        cstar = (a*b**2/csi**3)/(16*mu*(1-nu)); cdila = 2*cstar*Pdila;

        # displacement components (2) to (7): primitive of equation (1) from Yang et al (1988)
        Ustar1 = cstar*(Astar1*y1 + (3-4*nu)*Astarbar1*y1 + Fstar1*y1);          # equation (2) from Yang et al (1988)
        
        # U2star and U3star changed to improve fit to internal deformation
        Ustar2 = cstar*(sint*(Astar1*r2 + (3-4*nu)*Astarbar1*q2 + Fstar1*q2) + \
                 cost*(Bstar-Fstar2));                                           # equation (3) from Yang et al (1988)

        # The formula used in the script by Fialko and Andy is different from
        # equation (4) of Yang et al (1988)
        # I use the same to continue to compare the results 2009 07 23
        # Ustar3 = cstar*(-cost*(Astarbar1.*r2 + (3-4*nu)*Astarbar1.*q2 - Fstar1.*q2) + ...
        #         sint*(Bstar+Fstar2) + 2*cost^2*z.*Astarbar1);           
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # The equation below is correct - follows equation (4) from Yang et al (1988)
        Ustar3 = cstar*(-cost*(Astar1*r2 + (3-4*nu)*Astarbar1*q2 - Fstar1*q2) +\
                 sint*(Bstar+Fstar2));                                              # equation (4) from Yang et al (1988)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Udila1 = cdila*((A1*y1 + (3-4*nu)*Abar1*y1 + F1*y1) - 4*(1-nu)*(1-2*nu)*f1);              # equation (5) from Yang et al (1988)

        Udila2 = cdila*(sint*(A1*r2 + (3-4*nu)*Abar1*q2 + F1*q2) - 4*(1-nu)*(1-2*nu)*f2 + \
                 4*(1-nu)*cost*(A2+Abar2) + cost*(A3-(3-4*nu)*Abar3 - F2));                       # equation (6) from Yang et al (1988)

        Udila3 = cdila*(cost*(-A1*r2 + (3-4*nu)*Abar1*q2 + F1*q2) + 4*(1-nu)*(1-2*nu)*f3 + \
                 4*(1-nu)*sint*(A2+Abar2) + sint*(A3+(3-4*nu)*Abar3 + F2 - 2*(3-4*nu)*B));        # equation (7) from Yang et al (1988)
            
        # displacement: equation (8) from Yang et al (1988) - see Figure 3
        U1 = Ustar1 + Udila1;                                                                       # local x component
        U2 = Ustar2 + Udila2;                                                                       # local y component
        U3 = Ustar3 + Udila3;                                                                       # local z component

        return U1, U2, U3
        
    def _disp_(self,x0,y0,z0,a,b,lamb,mu,nu,P,theta,phi):
        # compute the 3D displacement due to a pressurized ellipsoid  
        #
        # IN
        # a         semimajor axis [m]
        # b         semiminor axis [m]
        # lambda    Lame's constant [Pa]
        # mu        shear modulus [Pa]
        # nu        Poisson's ratio 
        # P         excess pressure (stress intensity on the surface) [pressure units]
        # x0,y0,z0  coordinates of the center of the prolate spheroid (positive downward) [m]
        # theta     plunge angle [rad]
        # phi       trend angle [rad]
        # x,y,x     coordinates of the point(s) where the displacement is computed [m], come from object
        #
        # OUT
        # Ux,Uy,Uz  displacements
        #
        # Note ********************************************************************
        # compute the displacement due to a pressurized ellipsoid 
        # using the finite prolate spheroid model by from Yang et al (JGR,1988)
        # and corrections to the model by Newmann et al (JVGR, 2006).
        # The equations by Yang et al (1988) and Newmann et al (2006) are valid for a
        # vertical prolate spheroid only. There is and additional typo at pg 4251 in 
        # Yang et al (1988), not reported in Newmann et al. (2006), that gives an error 
        # when the spheroid is tilted (plunge different from 90deg):
        #           C0 = y0*cos(theta) + z0*sin(theta)
        # The correct equation is 
        #           C0 = z0/sin(theta)
        # This error has been corrected in this script.
        # *************************************************************************

        # testing parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # clear all; close all; clc;
        # a = 1000; b = 0.99*a;
        # lamb = 1; mu = lamb; nu = 0.25; P = 0.01;
        # theta = pi*89.99/180; phi = 0;
        # x = linspace(0,2E4,7);
        # y = linspace(0,1E4,7);
        # x0 = 0; y0 = 0; z0 = 5E3;
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # compute the parameters for the spheroid model
        [a1,b1,c,Pdila,Pstar] = self._par_(a,b,lamb,mu,nu,P);

        # translate the coordinates of the points where the displacement is computed
        # in the coordinates systen centered in (x0,0)
        xxn = self.get_xs() - x0;
        yyn = self.get_ys() - y0;

        # rotate the coordinate system to be coherent with the model coordinate
        # system of Figure 3 (Yang et al., 1988)
        xxp  = np.cos(phi) * xxn - np.sin(phi) * yyn;
        yyp  = np.sin(phi) * xxn + np.cos(phi) * yyn;

        # compute displacement for a prolate ellipsoid at csi = c
        [U1p,U2p,U3p] = self._int_(xxp,yyp,z0,theta,a1,b1,a,b,c,mu,nu,Pdila);

        # compute displacement for a prolate ellipsoid at csi = -c
        [U1m,U2m,U3m] = self._int_(xxp,yyp,z0,theta,a1,b1,a,b,-c,mu,nu,Pdila);
        Upx = -U1p-U1m;
        Upy = -U2p-U2m;
        Upz =  U3p+U3m;

        # rotate horizontal displacement back (strike)
        Ux =  np.cos(phi) * Upx + np.sin(phi) * Upy;
        Uy = -np.sin(phi) * Upx + np.cos(phi) * Upy;
        Uz = Upz;
        
        return Ux, Uy, Uz

    def forward(self,x0,y0,z0,a,A,P_G,theta,phi,mu=26.6e9,nu=0.25):
        #yang(-500,500,2000,a,b/a,dP/mu,mu,nu,45,90,[-2875.07722612, -2082.40080761,  -526.34373579], [ 606.48515842, -474.04624186,  929.50370699], [0, 0, 0])
        # 3D Green's function for a spheroidal source 
        # all parameters are in SI (MKS) units
        #
        # OUTPUT
        # u         horizontal (East component) deformation
        # v         horizontal (North component) deformation
        # w         vertical (Up component) deformation
        # dwdx      ground tilt (East component)
        # dwdy      ground tilt (North component)
        # eea       areal strain
        # gamma1    shear strain
        # gamma2    shear strain
        #
        # SOURCE PARAMETERS
        # a         semimajor axis
        # A         geometric aspect ratio [dimensionless]
        # P_G       dimennsionless excess pressure (pressure/shear modulus) 
        # x0,y0     surface coordinates of the center of the prolate spheroid
        # z0        depth of the center of the sphere (positive downward and
        #              defined as distance below the reference surface)
        # theta     plunge (dip) angle [deg] [90 = vertical spheroid]
        # phi       trend (strike) angle [deg] [0 = aligned to North]
        #
        # CRUST PARAMETERS
        # mu        shear modulus
        # nu        Poisson's ratio 
        #
        # BENCHMARKS (stored in object)
        # x,y       benchmark location
        # z         depth within the crust (z=0 is the free surface)
        #
        # Reference ***************************************************************
        #
        # Note ********************************************************************
        #
        # 2021-06-18 This is translated from Matlab code that has been around for a while ...
        #
        # BEFORE: compute the displacement due to a pressurized ellipsoid 
        # using the finite prolate spheroid model by from Yang et al (JGR,1988)
        # and corrections to the model by Newmann et al (JVGR, 2006).
        # The equations by Yang et al (1988) and Newmann et al (2006) are valid for a
        # vertical prolate spheroid only. There is and additional typo at pg 4251 in 
        # Yang et al (1988), not reported in Newmann et al. (2006), that gives an error 
        # when the spheroid is tilted (plunge different from 90�):
        #           C0 = y0*cos(theta) + z0*sin(theta)
        # The correct equation is 
        #           C0 = z0/sin(theta)
        # This error has been corrected in this script.
        # *************************************************************************

        # SINGULARITIES ***********************************************************
        if theta >= 89.99:
            theta = 89.99               # solution is singular when theta = 90deg
        if A >= 0.99:
            A = 0.99
        # *************************************************************************

        # DISPLACEMENT ************************************************************
        # define parameters used to compute the displacement
        b     = A*a;                    # semi-minor axis
        lamb  = 2*mu*nu/(1-2*nu)        # first Lame's elatic modulus
        P     = P_G*mu                  # excess pressure
        theta = np.deg2rad(theta)       # dip angle in rad
        phi   = np.deg2rad(phi)         # strike angle in rad

        # compute 3D displacements
        [u, v, w] = self._disp_(x0,y0,z0,a,b,lamb,mu,nu,P,theta,phi)
        # *************************************************************************

        return u, v, w

        # TILT ********************************************************************
#        h = 0.001*abs(max(x)-min(x));                                              % finite difference step
#
#        % East comonent
#        [tmp1, tmp2, wp] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x+h,y,z);
#        [tmp1, tmp2, wm] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x-h,y,z);
#        dwdx = 0.5*(wp - wm)/h;
#
#        % North component
#        [tmp1, tmp2, wp] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x,y+h,z);
#        [tmp1, tmp2, wm] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x,y-h,z);
#        dwdy = 0.5*(wp - wm)/h;
#        % *************************************************************************
#
#
#        % STRAIN ******************************************************************
#        % Displacement gradient tensor
#        [up , tmp1, tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x+h,y,z);
#        [um , tmp1, tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x-h,y,z);
#        dudx = 0.5*(up - um)/h;
#
#        [up , tmp1, tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x,y+h,z);
#        [um , tmp1, tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x,y-h,z);
#        dudy = 0.5*(up - um)/h;
#
#        [tmp1, vp , tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x+h,y,z);
#        [tmp1, vm , tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x-h,y,z);
#        dvdx = 0.5*(vp - vm)/h;
#
#        [tmp1, vp , tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x,y+h,z);
#        [tmp1, vm , tmp2] = yangdisp(x0,y0,z0,a,b,lambda,mu,nu,P,theta,phi,x,y-h,z);
#        dvdy = 0.5*(vp - vm)/h;
#
#        % Strains
#        eea = dudx + dvdy;                                                          % areal strain
#        gamma1 = dudx - dvdy;                                                       % shear strain
#        gamma2 = dudy + dvdx;                                                       % shear strain
#        % *************************************************************************


    def forward_mod(self):
       return self.forward(x0=self.model.x[0], y0=self.model.x[1], z0=self.model.x[2], a=self.model.x[3], 
                           A=self.model.x[4], P_G=self.model.x[5], theta=self.model.x[6], phi=self.model.x[7])


   

