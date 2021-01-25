'''
Penny-shaped Crack Solution from Fialko et al 2001

* Based on dModels scripts Battaglia et. al. 2013


* Check against figure 7.26 in Segall,
which compare's Fialko's solution to Davis 1986 point source
'''
#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#def forward(x,y,xcen=0,ycen=0,d=3e3,dV=1e6, nu=0.25):
def dV2dP(P, G):
    dV= -4 * np.pi * (1 - nu) * P/G *rd**3 * (t * (wt.T.dot(phi)))
    return dV

def forward(x0,y0,z0,P_G,a,nu,x,y,z):
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
    eps=1e-08
    rd=copy_(a)
    h=z0 / rd
    x=(x - x0) / rd
    y=(y - y0) / rd
    z=(z - z0) / rd
    r=sqrt_(x ** 2 + y ** 2)
    csi1,w1=gauleg_(eps,10,41,nargout=2)
    csi2,w2=gauleg_(10,60,41,nargout=2)
    csi=cat_(2,csi1,csi2)
    wcsi=cat_(2,w1,w2)
    if size_(csi,1) == 1:
        csi=csi.T
    phi,psi,t,wt=psi_phi_(h,nargout=4)
    PHI=sin_(csi * t) * (wt.T.dot(phi))
    PSI=(sin_(csi * t) / (csi * t) - cos_(csi * t)) * (wt.T.dot(psi))
    a=csi * h
    A=exp_(- a).dot((a.dot(PSI) + (1 + a).dot(PHI)))
    B=exp_(- a).dot(((1 - a).dot(PSI) - a.dot(PHI)))
    Uz=zeros_(size_(r))
    Ur=zeros_(size_(r))
    for i in arange_(1,length_(r)).reshape(-1):
        J0=besselj_(0,r[i] * csi)
        Uzi=J0.dot((((1 - 2 * nu) * B - csi * (z + h).dot(A)).dot(sinh_(csi * (z + h))) + (2 * (1 - nu) * A - csi * (z + h).dot(B)).dot(cosh_(csi * (z + h)))))
        Uz[i]=wcsi * Uzi
        J1=besselj_(1,r[i] * csi)
        Uri=J1.dot((((1 - 2 * nu) * A + csi * (z + h).dot(B)).dot(sinh_(csi * (z + h))) + (2 * (1 - nu) * B + csi * (z + h).dot(A)).dot(cosh_(csi * (z + h)))))
        Ur[i]=wcsi * Uri
    u=rd * P_G * Ur.dot(x) / r
    v=rd * P_G * Ur.dot(y) / r
    w=- rd * P_G * Uz

    return u,v,w
