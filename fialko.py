'''
Penny-shaped Crack Solution from Fialko et al 2001

* Based on dModels scripts Battaglia et. al. 2013


* Check against figure 7.26 in Segall,
which compare's Fialko's solution to Davis 1986 point source
'''
#from __future__ import division
import numpy as np
import util
import scipy
from scipy import special
#def forward(x,y,xcen=0,ycen=0,d=3e3,dV=1e6, nu=0.25):
def dP2dV(P_G,z0,a,nu=0.25):
    h=z0 / a
    phi,psi,t,wt=util.psi_phi(h)
    dV= -4 * np.pi * (1 - nu) * P_G *a**3 * (t * (wt.T.dot(phi)))
    return dV

def forward(x0,y0,z0,P_G,a,x,y,z=0,nu=0.25):
    """
    Calculates surface deformation based on point pressure source
    References: Fialko

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
    """
    eps=1e-8
    rd=np.copy(a)
    h=z0 / rd
    x=(x - x0) / rd
    y=(y - y0) / rd
    z=(z - z0) / rd
    r=np.sqrt(x ** 2 + y ** 2)
    csi1,w1=util.gauleg(eps,10,41)
    csi2,w2=util.gauleg(10,60,41)
    csi=np.concatenate((csi1,csi2))
    wcsi=np.concatenate((w1,w2))
    if csi.shape[0] == 1:
        csi=csi.T
    phi1,psi1,t,wt=util.psi_phi(h)
    phi=np.matmul(np.sin(np.outer(csi,t)) , phi1*wt)
    psi=np.matmul(np.divide(np.sin(np.outer(csi,t)), np.outer(csi,t)) - np.cos(np.outer(csi,t)),psi1*wt)
    a=csi * h
    A=np.exp((-1)*a)*(a*psi + (1 + a)*phi)
    B=np.exp((-1)*a)*((1-a)*psi - a*phi)
    Uz=np.zeros(r.shape)
    Ur=np.zeros(r.shape)
    for i in range(r.size):
        J0=special.jv(0,r[i] * csi)
        Uzi=J0*(((1-2*nu)*B - csi*(z+h)*A)*np.sinh(csi*(z+h)) +(2*(1-nu)*A - csi*(z+h)*B)*np.cosh(csi*(z+h)))
        Uz[i]=np.dot(wcsi , Uzi)
        J1=special.jv(1,r[i] * csi)
        Uri=J1*(((1-2*nu)*A + csi*(z+h)*B)*np.sinh(csi*(z+h)) + (2*(1-nu)*B + csi*(z+h)*A)*np.cosh(csi*(z+h)))
        Ur[i]=np.dot(wcsi , Uri)
    u=rd * P_G * Ur*x / r
    v=rd * P_G * Ur*y / r
    w=- rd * P_G * Uz

    return u,v,w
