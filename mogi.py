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
from . import util

# =====================
# Inverse Models
# =====================
def invert(xargs,xcen,ycen,depth,dV):
    """
    Wrapper of mogi.forward to project to LOS and adjust arguments to work
    with scipy.omptimize.curvefit. Assumes UTM input for X and Y
    """
    #NOTE: nu fixed to default 0.25 by leaving out
    X,Y,incidence,heading = xargs
    ux, uy, uz = forward(X,Y,xcen,ycen,depth,dV)
    dataVec = np.dstack([ux, uy, uz])
    cart2los = util.get_cart2los(incidence,heading)
    los = -np.sum(dataVec * cart2los, axis=2)

    return los.ravel()


def invert_varres(xargs,xcen,ycen,depth,dV):
    """
    Wrapper of mogi.forward to project to LOS and adjust arguments to work
    with scipy.omptimize.curvefit. Assumes UTM input for X and Y
    """
    #NOTE: nu fixed to default 0.25 by leaving out
    X,Y,EW2los,NS2los,Z2los = xargs
    ux, uy, uz = forward(X,Y,xcen,ycen,depth,dV)
    dataVec = np.dstack([ux, uy, uz])
    cart2los = np.dstack([EW2los, NS2los, Z2los])
    los = -np.sum(dataVec * cart2los, axis=2)

    return los.ravel()


def invert_dipole(xargs,xcen,ycen,depth,dV,xcen1,ycen1,depth1,dV1):
    """
    Wrapper of mogi.forward to project to LOS and adjust arguments to work
    with scipy.omptimize.curvefit. Assumes UTM input for X and Y
    """
    #NOTE: nu fixed to default 0.25 by leaving out
    X,Y,incidence,heading = xargs
    ux, uy, uz = forward(X,Y,xcen,ycen,depth,dV)
    #ux1, uy1, uz1 = forward(X,Y,xcen1,ycen1,depth1,dV1)
    # Fix vertical alignment
    ux1, uy1, uz1 = forward(X,Y,xcen,ycen,depth1,dV1)
    dataVec = np.dstack([ux+ux1, uy+uy1, uz+uz1])
    cart2los = util.get_cart2los(incidence,heading)
    los = -np.sum(dataVec * cart2los, axis=2)

    return los.ravel()

def invert_dipole_nuisance(xargs,xcen,ycen,depth,dV,xcen1,ycen1,depth1,dV1,rampx,rampy,dc):
    """
    Wrapper of mogi.forward to project to LOS and adjust arguments to work
    with scipy.omptimize.curvefit. Assumes UTM input for X and Y

    Add nuisance parameters: ramp in x, ramp in y, dc offset
    My suspicion is this is not a good idea since the ramps can counteract
    differences in viewing geometry, which are probably real!
    """
    #NOTE: nu fixed to default 0.25 by leaving out
    X,Y,incidence,heading = xargs

    ux, uy, uz = forward(X,Y,xcen,ycen,depth,dV)

    #ux1, uy1, uz1 = forward(X,Y,xcen1,ycen1,depth1,dV1)
    # Fix vertical alignment
    ux1, uy1, uz1 = forward(X,Y,xcen,ycen,depth1,dV1)

    dataVec = np.dstack([ux+ux1, uy+uy1, uz+uz1])
    cart2los = util.get_cart2los(incidence,heading)

    los = -np.sum(dataVec * cart2los, axis=2) + rampx*X + rampy*Y + dc

    return los.ravel()

# =====================
# Forward Models
# =====================

def forward(x,y,xcen=0,ycen=0,d=3e3,dV=1e6, nu=0.25):
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
    x = x - xcen
    y = y - ycen

    # Convert to surface cylindrical coordinates
    th, rho = util.cart2pol(x,y) # surface angle and radial distance
    R = np.hypot(d,rho) # radial distance from source

    # Mogi displacement calculation
    C = ((1-nu) / np.pi) * dV
    ur = C * rho / R**3    # horizontal displacement, m
    uz = C * d / R**3   # vertical displacement, m

    ux, uy = util.pol2cart(th, ur)
    #return ux, uy, uz #returns tuple
    return np.array([ux,uy,uz])


def forward_dp(x,y,xcen=0,ycen=0,d=3e3,a=500,dP=100e6,mu=4e9,nu=0.25):
    """
    dP instead of dV, NOTE: dV = pi * dP * a**3 / mu
    981747.7 ~ 1e6
    """
    dV = np.pi * dP * a**3 / mu
    return forward(x,y,xcen,ycen,d,dV,nu)



def calc_linmax(x,y,tn,xcen=0,ycen=0,d=3e3,a=500.0,dP=100e6,mu=4e9,nu=0.25):
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
    x = x - xcen
    y = y - ycen

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


def calc_linmax_dPt(tn,dVdt,xcen=0,ycen=0,d=3e3,a=500.0,dP=100e6,mu=4e9,
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

    C = (dVdt*tau) / (2*pi*d**2)
    term1 = t/tau
    term2 = (mu / (3*K)) * (1 - np.exp(-t/tauA))
    uzmax = C * (term1 - term2)
    return uzmax


def calc_genmax(x,y,t,xcen=0,ycen=0,d=4e3,dP=100e6,a=700,nu=0.25,G=30e9,
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
    x = x - xcen
    y = y - ycen

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



def calc_mctigue(x,y,xcen=0,ycen=0,d=3e3,dP=10e6,a=1500.0,nu=0.25,mu=4e9,
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
    x = x - xcen
    y = y - ycen

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


def calc_viscoshell(x,y,t,xcen=0,ycen=0,d=4e3,a=1000.0,b=1200.0,dP=100e6,
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
    rho = np.hypot(x,y) / d #dimensionless!
    uz = (((1-nu)*dP*a**3) / (mu*d**2)) * scale * (1 + rho**2)**(-3/2.0)
    ur = rho * uz

    #print('uz_max = {:.4f}'.format(uz.max()))
    #print('ur_max = {:.4f}'.format(ur.max()))

    return ur, uz


def calc_viscoshell_dPt(x,y,t,P0,tS,xcen=0,ycen=0,d=4e3,a=1000.0,b=1200.0,
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
    rho = np.hypot(x,y) / d #Dimensionless radius!

    uz = (((1-nu)*P0*a**3) / (mu*d**2)) * (1 + rho**2)**(-3/2.0) * scale
    ur = rho * uz

    print('uz_max = {:.4f}'.format(uz.max()))
    print('ur_max = {:.4f}'.format(ur.max()))

    return ur, uz, P

def calc_uzmax(dP=10e6,a=5e2,d=3e3,mu=30e9,nu=0.25):
    uzmax = ((1-nu)*dP*a**3) / (mu*d**2)
    return uzmax


def dP2dV(dP,a,mu=30e9):
    dV = (np.pi * dP * a**3) / mu
    return dV

def dV2dP(dV,a,mu=30e9):
    dP = (dV * mu) / (np.pi * a**3)
    return dP
