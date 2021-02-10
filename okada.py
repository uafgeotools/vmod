'''
Computes displacements at the surface of an elastic half-space, due to a
dislocation defined by 'slip' and 'rake' for rectangular fault defined by
orientation 'strike' and 'dip', and size 'length' an 'width'. The fault
centroid is located (0,0,-depth). Equations from Okada 1985 & 1992 BSSA.

strike, dip and rake according to the conventions set forth by
Aki and Richards (1980), Quantitative Seismology, Vol. 1.
rake=0 --> left-lateral strike slip.

strike:  Clockwise with respect to north [degrees]
dip: Angle from horizontal, to right side facing strike direction [dip]
rake: Counter-clockwise relative to strike
      examples for positive-valued slip:
      0 --> left-lateral, -90 --> normal, 90-->thrust

depth: postive distance below surface to fault centroid [meters]
width: fault width in dip direction [meters]
length: fault length in strike direction [meters]

slip: positive motion in rake direction [meters]
opening: positive tensile dislocation [meters]

x,y: coordinate grid [meters]
xcen, ycen: epicenter of fault centroid
nu: poisson ratio [unitless]

* tanslated from matlab code by Francois Beauducel
#https://www.mathworks.com/matlabcentral/fileexchange/25982-okada--surface-deformation-due-to-a-finite-rectangular-source

Author: Scott Henderson
'''

from __future__ import division
import numpy as np
import util

eps = 1e-14 #numerical constant
#tests for cos(dip) == 0 are made with "cos(dip) > eps"
#because cos(90*np.pi/180) is not zero but = 6.1232e-17 (!)

def forward(x, y, xcen=0, ycen=0,
            depth=5e3, length=1e3, width=1e3,
            slip=0.0, opening=10.0,
            strike=0.0, dip=0.0, rake=0.0,
            nu=0.25):
    '''
    Calculate surface displacements for Okada85 dislocation model
    '''
    e = x - xcen
    n = y - ycen

    # A few basic parameter checks
    if not (0.0 <= strike <= 360.0) or not (0 <= dip <= 90):
        print('Please use 0<strike<360 clockwise from North')
        print('And 0<dip<90 East of strike convention')
        raise ValueError
    # Don't allow faults that prech the surface
    d_crit = width/2 * np.sin(np.deg2rad(dip))
    assert depth >= d_crit, 'depth must be greater than {}'.format(d_crit)
    assert length >=0, 'fault length must be positive'
    assert width >=0, 'fault length must be positive'
    assert rake <= 180, 'rake should be:  rake <= 180'
    assert -1.0 <= nu <= 0.5, 'Poisson ratio should be: -1 <= nu <= 0.5'

    strike = np.deg2rad(strike) #transformations accounted for below
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)

    L = length
    W = width

    U1 = np.cos(rake) * slip
    U2 = np.sin(rake) * slip
    U3 = opening

    d = depth + np.sin(dip) * W / 2 #fault top edge
    ec = e + np.cos(strike) * np.cos(dip) * W / 2
    nc = n - np.sin(strike) * np.cos(dip) * W / 2
    x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2
    y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W
    p = y * np.cos(dip) + d * np.sin(dip)
    q = y * np.sin(dip) - d * np.cos(dip)

    ux = - U1 / (2 * np.pi) * chinnery(ux_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * np.pi) * chinnery(ux_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * np.pi) * chinnery(ux_tf, x, p, L, W, q, dip, nu)

    uy = - U1 / (2 * np.pi) * chinnery(uy_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * np.pi) * chinnery(uy_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * np.pi) * chinnery(uy_tf, x, p, L, W, q, dip, nu)

    uz = - U1 / (2 * np.pi) * chinnery(uz_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * np.pi) * chinnery(uz_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * np.pi) * chinnery(uz_tf, x, p, L, W, q, dip, nu)

    ue = np.sin(strike) * ux - np.cos(strike) * uy
    un = np.cos(strike) * ux + np.sin(strike) * uy

    return ue,un,uz

def invert(xargs, xoff, yoff,depth,dip,length,width,slip,strike,rake):
    X,Y,incidence,heading = xargs
    nu=0.25
    ux,uy,uz=forward(X,Y,xoff,yoff,depth,length,width,slip,0,strike,dip,rake)
    dataVec = np.dstack([ux, uy, uz])
    cart2los = util.get_cart2los2(incidence,heading)
    los = np.sum(dataVec * cart2los, axis=2)
    return los.ravel()

def chinnery(f, x, p, L, W, q, dip, nu):
    '''Chinnery's notation [equation (24) p. 1143]'''
    u =  (f(x, p, q, dip, nu) -
          f(x, p - W, q, dip, nu) -
          f(x - L, p, q, dip, nu) +
          f(x - L, p - W, q, dip, nu))
    return u


def ux_ss(xi, eta, q, dip, nu):

    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = xi * q / (R * (R + eta)) + \
        I1(xi, eta, q, dip, nu, R) * np.sin(dip)
    k = (q != 0)
    #u[k] = u[k] + np.arctan2( xi[k] * (eta[k]) , (q[k] * (R[k])))
    u[k] = u[k] + np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
    return u


def uy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + \
        q * np.cos(dip) / (R + eta) + \
        I2(eta, q, dip, nu, R) * np.sin(dip)
    return u


def uz_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + eta)) + \
        q * np.sin(dip) / (R + eta) + \
        I4(db, eta, q, dip, nu, R) * np.sin(dip)
    return u


def ux_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = q / R - \
        I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
    return u


def uy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = ( (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) -
           I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip) )
    k = (q != 0)
    u[k] = u[k] + np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def uz_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = ( db * q / (R * (R + xi)) -
          I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip) )
    k = (q != 0)
    #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
    u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def ux_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = q**2 / (R * (R + eta)) - \
        (I3(eta, q, dip, nu, R) * np.sin(dip)**2)
    return u


def uy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = - (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - \
        (np.sin(dip) * xi * q / (R * (R + eta))) - \
        (I1(xi, eta, q, dip, nu, R) * np.sin(dip) ** 2)
    k = (q != 0)
    #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
    u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
    return u


def uz_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + \
         np.cos(dip) * xi * q / (R * (R + eta)) - \
         I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2
    k = (q != 0)
    u[k] = u[k] - np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
    return u


def I1(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * (- xi / (np.cos(dip) * (R + db))) - \
            np.sin(dip) / np.cos(dip) * I5(xi, eta, q, dip, nu, R, db)
    else:
        I = -(1 - 2 * nu)/2 * xi * q / (R + db)**2
    return I


def I2(eta, q, dip, nu, R):
    I = (1 - 2 * nu) * (-np.log(R + eta)) - \
        I3(eta, q, dip, nu, R)
    return I


def I3(eta, q, dip, nu, R):
    yb = eta * np.cos(dip) + q * np.sin(dip)
    db = eta * np.sin(dip) - q * np.cos(dip)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + \
            np.sin(dip) / np.cos(dip) * I4(db, eta, q, dip, nu, R)
    else:
        I = (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))
    return I


def I4(db, eta, q, dip, nu, R):
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * 1.0 / np.cos(dip) * \
            (np.log(R + db) - np.sin(dip) * np.log(R + eta))
    else:
        I = - (1 - 2 * nu) * q / (R + db)
    return I


def I5(xi, eta, q, dip, nu, R, db):
    X = np.sqrt(xi**2 + q**2)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * 2 / np.cos(dip) * \
             np.arctan( (eta * (X + q*np.cos(dip)) + X*(R + X) * np.sin(dip)) /
                        (xi*(R + X) * np.cos(dip)) )
        I[xi == 0] = 0
    else:
        I = -(1 - 2 * nu) * xi * np.sin(dip) / (R + db)
    return I



if __name__ == '__main__':
    print(__doc__)
