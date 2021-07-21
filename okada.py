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
from source import Source


eps = 1e-14 #numerical constant
#tests for cos(dip) == 0 are made with "cos(dip) > eps"
#because cos(90*np.pi/180) is not zero but = 6.1232e-17 (!)

class Okada(Source):

    def get_num_params(self):
        return 4

    def get_source_id(self):
        return "Okada"

    def print_model(self, x):
        print("Okada")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tdV= %f" % x[3])

    ##residual function for least_squares
    def fun(self, x):
        ux, uy, uz = self.forward(xcen=x[0], ycen=x[1], d=x[2], dV=x[3])
        diff =np.concatenate((ux,uy,uz))-self.get_obs()
        return diff

    # =====================
    # Forward Models
    # =====================
    def forward_mod(self, x):
        return self.forward(x[0], x[1], x[2], x[3])

    def forward(self, xcen=0, ycen=0,
                        depth=5e3, length=1e3, width=1e3,
                        slip=0.0, opening=10.0,
                        strike=0.0, dip=0.0, rake=0.0,
                        nu=0.25):
        '''
        Calculate surface displacements for Okada85 dislocation model
        '''
        e = self.get_xs() - xcen
        n = self.get_ys() - ycen

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
        dip    = np.deg2rad(dip)
        rake   = np.deg2rad(rake)

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

        ux = - U1 / (2 * np.pi) * Okada.chinnery(Okada.ux_ss, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(Okada.ux_ds, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(Okada.ux_tf, x, p, L, W, q, dip, nu)

        uy = - U1 / (2 * np.pi) * Okada.chinnery(Okada.uy_ss, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(Okada.uy_ds, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(Okada.uy_tf, x, p, L, W, q, dip, nu)

        uz = - U1 / (2 * np.pi) * Okada.chinnery(Okada.uz_ss, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(Okada.uz_ds, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(Okada.uz_tf, x, p, L, W, q, dip, nu)

        ue = np.sin(strike) * ux - np.cos(strike) * uy
        un = np.cos(strike) * ux + np.sin(strike) * uy

        return np.array([ue,un,uz])

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
            Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip)
        k = (q != 0)
        #u[k] = u[k] + np.arctan2( xi[k] * (eta[k]) , (q[k] * (R[k])))
        u[k] = u[k] + np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u


    def uy_ss(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + \
            q * np.cos(dip) / (R + eta) + \
            Okada.I2(eta, q, dip, nu, R) * np.sin(dip)
        return u


    def uz_ss(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + eta)) + \
            q * np.sin(dip) / (R + eta) + \
            Okada.I4(db, eta, q, dip, nu, R) * np.sin(dip)
        return u


    def ux_ds(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = q / R - \
            Okada.I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        return u


    def uy_ds(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = ( (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) -
               Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip) )
        k = (q != 0)
        u[k] = u[k] + np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        return u


    def uz_ds(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = ( db * q / (R * (R + xi)) -
              Okada.I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip) )
        k = (q != 0)
        #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
        u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        return u


    def ux_tf(xi, eta, q, dip, nu):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = q**2 / (R * (R + eta)) - \
            (Okada.I3(eta, q, dip, nu, R) * np.sin(dip)**2)
        return u


    def uy_tf(xi, eta, q, dip, nu):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = - (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - \
            (np.sin(dip) * xi * q / (R * (R + eta))) - \
            (Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip) ** 2)
        k = (q != 0)
        #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
        u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u


    def uz_tf(xi, eta, q, dip, nu):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + \
             np.cos(dip) * xi * q / (R * (R + eta)) - \
             Okada.I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2
        k = (q != 0)
        u[k] = u[k] - np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u


    def I1(xi, eta, q, dip, nu, R):
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * (- xi / (np.cos(dip) * (R + db))) - \
                np.sin(dip) / np.cos(dip) * Okada.I5(xi, eta, q, dip, nu, R, db)
        else:
            I = -(1 - 2 * nu)/2 * xi * q / (R + db)**2
        return I


    def I2(eta, q, dip, nu, R):
        I = (1 - 2 * nu) * (-np.log(R + eta)) - \
            Okada.I3(eta, q, dip, nu, R)
        return I


    def I3(eta, q, dip, nu, R):
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + \
                np.sin(dip) / np.cos(dip) * Okada.I4(db, eta, q, dip, nu, R)
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

    def verify():
        '''Test for self.forward to replicate Figure 28 of dModels report'''

        from data import Data
        import matplotlib.pyplot as plt        
        
        xi=4000
        xf=25000
        yi=4000
        yf=20000
        zt=1000
        zb=10000
        dip=60

        strike=np.degrees(np.arctan((xf-xi)/(yf-yi)))

        length=np.sqrt(np.power(xf-xi,2)+np.power(yf-yi,2))

        W=(zb-zt)/np.sin(np.radians(dip))
        Wproj=W*np.cos(np.radians(dip))

        xceni=xi+(xf-xi)/2
        yceni=yi+(yf-yi)/2

        xcen=xceni+Wproj*np.cos(np.radians(strike))
        ycen=yceni-Wproj*np.sin(np.radians(strike))

        depth=(zt+zb)/2

        print(np.degrees(strike),xceni,yceni,xcen,ycen,length,W,depth)

        x=np.linspace(-50000,50000,num=100)
        y=x

        mu=1e9
        s=1

        #create data structure
        d = Data()
        d.add_locs(x,y*0)

        #initialize source model
        okada = Okada(d)

        #run forward model
        ux,uy,uz=okada.forward(xcen=xcen, ycen=ycen,
                    depth=depth, length=length, width=W,
                    slip=s, opening=0.0,
                    strike=strike, dip=dip, rake=180.0,
                    nu=0.25)

# CANT DO THE BELOW WITHT HE CURRENT IMPLEMENTATION of DATA (1d only), raises the following error
#   d.add_locs(X,Y)
#  File "/home/roon/REPOSITORIES/vmod/data.py", line 25, in add_locs
#    self.data['x'] = pd.Series(x)
#  File "/home/roon/.conda/envs/mcmc/lib/python3.8/site-packages/pandas/core/series.py", line 364, in __init__
#    data = sanitize_array(data, index, dtype, copy, raise_cast_failure=True)
#  File "/home/roon/.conda/envs/mcmc/lib/python3.8/site-packages/pandas/core/construction.py", line 529, in sanitize_array
#    raise ValueError("Data must be 1-dimensional")
#
#
#        x1=np.linspace(-50000,50000,50)
#        y1=x1
#        X,Y=np.meshgrid(x1,y1)
#
#        #create new data structure
#        d = Data()
#        d.add_locs(X,Y)
#
#        #initialize new source model
#        okada = Okada(d)
        
#        UX,UY,UZ=okada.forward(xcen=xcen, ycen=ycen,
#                    depth=depth, length=length, width=W,
#                    slip=s, opening=0.0,
#                    strike=strike, dip=dip, rake=180.0,
#                    nu=0.25)

#        plt.figure()
#        plt.plot([xi,xf],[yi,yf],c='red',lw=2)
#        plt.quiver(X,Y,UX,UY)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(x/1000,ux,'x',c='green')
        ax1.set_ylabel('U east (meters)')
        ax1.set_xlim([-50,50])
        ax1.set_ylim([-0.08,0.01])

        ax2.plot(y/1000,uy,c='blue')
        ax2.set_ylabel('V north (meters)')
        ax2.set_xlabel('X in km')
        ax2.set_xlim([-50,50])
        ax2.set_ylim([-0.10,0.04])

        ax3.plot(y/1000,uz,c='red')
        ax3.set_ylabel('W up (meters)')
        ax3.set_xlim([-50,50])
        ax3.set_ylim([-0.02,0.16])

        plt.show()


