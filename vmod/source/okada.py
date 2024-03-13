from __future__ import division
import numpy as np
from . import Source
from .. import util
import time

eps = 1e-14 #numerical constant

class Okada(Source):
    """
    A class used to represent a tensile or slip dislocation with Okada (1985) model.

    Attributes
    ----------
    type: str
        describes if dislocation is tensile (open) or slip (slip)
    
    parameters : array
        names for the parameters in the model
    """
    def set_type(self,typ):
        """
        Defines the type of dislocation.
          
        Parameters:
            str: 'slip' to represent faults or 'open' to represent sill/dikes.
        """
        self.type=typ

    def get_source_id(self):
        """
        The function defining the name for the model.
          
        Returns:
            str: Name of the model.
        """
        return "Okada"
    
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
        print("Okada")
        if self.type=='slip':
            print("\txcen = %f" % x[0])
            print("\tycen = %f" % x[1])
            print("\tdepth = %f" % x[2])
            print("\tlength= %f" % x[3])
            print("\twidth = %f" % x[4])
            print("\tslip = %f" % x[5])
            print("\tstrike= %f" % x[6])
            print("\tdip = %f" % x[7])
            print("\trake = %f" % x[8])
        elif self.type=='open':
            print("\txcen = %f" % x[0])
            print("\tycen = %f" % x[1])
            print("\tdepth = %f" % x[2])
            print("\tlength= %f" % x[3])
            print("\twidth = %f" % x[4])
            print("\topening = %f" % x[5])
            print("\tstrike= %f" % x[6])
            print("\tdip = %f" % x[7])
        
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        if self.type=='slip':
            self.parameters=("xcen","ycen","depth","length","width","slip","strike","dip","rake")
        elif self.type=='open':
            self.parameters=("xcen","ycen","depth","length","width","opening","strike","dip")
    
    def get_args(self,args, tilt):
        """
        Function that arranges the parameters for the dislocation model depending on the type.
        
        Parameters:
            args (list) : parameters given by the user
            tilt (boolean) : compute tilt displacements (True) or 3d displacements (False)
        
        Returns:
            rargs (list) : parameters for the Okada model.
        """
        nu=0.25
        if self.type=='slip':
            xcen,ycen,depth,length,width,slip,strike,dip,rake=args
            opening=0.0
        else:
            xcen,ycen,depth,length,width,opening,strike,dip=args
            slip=0.0
            rake=0.0
        rargs=[xcen, ycen,depth, length, width,slip, opening,strike, dip, rake,nu, tilt]
        return rargs
        
    # =====================
    # Forward Models
    # =====================
    def model(self,x,y,*args):
        """
        3d displacement field on surface for dislocation (Okada, 1985)

        Parameters:
            args (list) : parameters given by the user
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        rargs=self.get_args(args,tilt=False)
        return self.model_gen(x,y, *rargs)
    
    def model_tilt(self,x,y,*args):
        """
        Tilt displacement field on surface for dislocation (Okada, 1985)

        Parameters:
            args (list) : parameters given by the user
        
        Returns:
            dx (array) : tilt displacements in the x-axis (radians).
            dy (array) : tilt displacements in the y-axis (radians).
        """
        rargs=self.get_args(args,tilt=True)
        return self.model_gen(x,y, *rargs)
    
    def model_gen(self,x,y, xcen=0, ycen=0,
                        depth=5e3, length=1e3, width=1e3,
                        slip=0.0, opening=10.0,
                        strike=0.0, dip=0.0, rake=0.0,
                        nu=0.25,tilt=False):
        """
        Computes tilt or 3d displacement field on surface for dislocation (Okada, 1985)

        Parameters:
            x: x-coordinate for displacement (m)
            y: y-coordinate for displacement (m)
            xcen: x-offset of dislocation center (m)
            ycen: y-offset of dislocation center (m)
            depth: depth to dislocation center (m)
            length: length of dislocation path (m)
            width: width of dislocation path (m)
            slip: fault movement (m)
            opening: amount of closing or opening of sill/dike (m)
            strike: horizontal clock wise orientation from north of dislocation (degrees)
            dip: dipping angle of dislocation (degrees)
            rake: fault's angle of rupture where 0 represents a strike-slip fault and 90 represents a normal fault (degrees)
            nu: Poisson's ratio
            tilt: boolean to indicate calculation of tilt displacements (True) or 3d displacements (False) (default False)
        
        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        e = x - xcen
        n = y - ycen

        # A few basic parameter checks
        if not (0.0 <= strike <= 360.0) or not (0 <= dip <= 90):
            print('Strike',strike)
            print('Dip',dip)
            print('Please use 0<strike<360 clockwise from North')
            print('And 0<dip<90 East of strike convention')
            raise ValueError

        # Don't allow faults that prech the surface
        d_crit = width/2 * np.sin(np.deg2rad(dip))
        
        if tilt:
            nans=np.array([e*np.nan,e*np.nan])
        else:
            nans=np.array([e*np.nan,e*np.nan,e*np.nan])
        
        if depth<d_crit:
            return nans
        elif length<0:
            return nans
        elif width<0:
            return nans
        elif rake>180:
            return nans
        elif not -1.0 <= nu <= 0.5:
            return nans
        
        #assert depth >= d_crit, 'depth must be greater than {}'.format(d_crit)
        #assert length >=0, 'fault length must be positive'
        #assert width >=0, 'fault length must be positive'
        #assert rake <= 180, 'rake should be:  rake <= 180'
        #assert -1.0 <= nu <= 0.5, 'Poisson ratio should be: -1 <= nu <= 0.5'

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

        if tilt:
            ssx=Okada.dx_ss
            dsx=Okada.dx_ds
            tfx=Okada.dx_tf
            
            ssy=Okada.dy_ss
            dsy=Okada.dy_ds
            tfy=Okada.dy_tf
        else:
            ssx=Okada.ux_ss
            dsx=Okada.ux_ds
            tfx=Okada.ux_tf
            
            ssy=Okada.uy_ss
            dsy=Okada.uy_ds
            tfy=Okada.uy_tf
            
            uz = - U1 / (2 * np.pi) * Okada.chinnery(Okada.uz_ss, x, p, L, W, q, dip, nu) - \
                   U2 / (2 * np.pi) * Okada.chinnery(Okada.uz_ds, x, p, L, W, q, dip, nu) + \
                   U3 / (2 * np.pi) * Okada.chinnery(Okada.uz_tf, x, p, L, W, q, dip, nu)
            
        ux = - U1 / (2 * np.pi) * Okada.chinnery(ssx, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(dsx, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(tfx, x, p, L, W, q, dip, nu)

        uy = - U1 / (2 * np.pi) * Okada.chinnery(ssy, x, p, L, W, q, dip, nu) - \
               U2 / (2 * np.pi) * Okada.chinnery(dsy, x, p, L, W, q, dip, nu) + \
               U3 / (2 * np.pi) * Okada.chinnery(tfy, x, p, L, W, q, dip, nu)


        ue = np.sin(strike) * ux - np.cos(strike) * uy
        un = np.cos(strike) * ux + np.sin(strike) * uy
        
        if tilt:
            return ue,un
        else:
            return ue,un,uz

    def chinnery(f, x, p, L, W, q, dip, nu):
        """
        Chinnery's notation [equation (24) p. 1143]
        """
        u =  (f(x, p, q, dip, nu) -
              f(x, p - W, q, dip, nu) -
              f(x - L, p, q, dip, nu) +
              f(x - L, p - W, q, dip, nu))

        return u


    def ux_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = xi * q / (R * (R + eta)) + \
            Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip)
        k = (q != 0)
        #u[k] = u[k] + np.arctan2( xi[k] * (eta[k]) , (q[k] * (R[k])))
        u[k] = u[k] + np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u


    def uy_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + \
            q * np.cos(dip) / (R + eta) + \
            Okada.I2(eta, q, dip, nu, R) * np.sin(dip)
        return u


    def uz_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = db * q / (R * (R + eta)) + \
            q * np.sin(dip) / (R + eta) + \
            Okada.I4(db, eta, q, dip, nu, R) * np.sin(dip)
        return u
    
    def dx_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = -xi * q**2 * Okada.A(eta,R) * np.cos(dip) + \
            (xi * q / R**3 - Okada.K1(xi, eta, q, dip, nu, R)) * np.sin(dip)
        return u
    
    def dy_ss(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = db * q / R**3 * np.cos(dip) + \
            (xi**2 * q * Okada.A(eta,R) * np.cos(dip) - np.sin(dip) / R + yb * q / R**3 - Okada.K2(xi, eta, q, dip, nu, R)) * np.sin(dip)
        return u


    def ux_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = q / R - \
            Okada.I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        return u


    def uy_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = ( (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) -
               Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip) )
        k = (q != 0)
        u[k] = u[k] + np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        return u


    def uz_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = ( db * q / (R * (R + xi)) -
              Okada.I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip) )
        k = (q != 0)
        #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
        u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        return u
    
    def dx_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        
        u = db * q / R**3 + \
            q * np.sin(dip) / (R * (R + eta)) + \
            Okada.K3(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        
        return u
    
    def dy_ds(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = yb * db * q * Okada.A(xi,R) - \
            (2 * db / (R * (R + xi)) + xi * np.sin(dip) / (R * (R + eta))) * np.sin(dip) + \
            Okada.K1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        
        return u


    def ux_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = q**2 / (R * (R + eta)) - \
            (Okada.I3(eta, q, dip, nu, R) * np.sin(dip)**2)
        return u


    def uy_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = - (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - \
            (np.sin(dip) * xi * q / (R * (R + eta))) - \
            (Okada.I1(xi, eta, q, dip, nu, R) * np.sin(dip) ** 2)
        k = (q != 0)
        #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
        u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u


    def uz_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + \
             np.cos(dip) * xi * q / (R * (R + eta)) - \
             Okada.I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2
        k = (q != 0)
        u[k] = u[k] - np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        return u
    
    def dx_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = q**2 * np.sin(dip) / R**3 - q**3 * Okada.A(eta,R) * np.cos(dip) + Okada.K3(xi, eta, q, dip, nu, R) * (np.sin(dip))**2
        
        return u
    
    def dy_tf(xi, eta, q, dip, nu):
        """
        Auxiliary function for Okada (1985) model.
        """
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = (yb * np.sin(dip) + db * np.cos(dip)) * q**2 * Okada.A(xi,R) + \
            xi * q**2 * Okada.A(eta,R) * np.sin(dip) * np.cos(dip) - \
            (2 * q/(R * (R + xi)) - Okada.K1(xi, eta, q, dip, nu, R)) * (np.sin(dip))**2
        
        return u


    def I1(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * (- xi / (np.cos(dip) * (R + db))) - \
                np.sin(dip) / np.cos(dip) * Okada.I5(xi, eta, q, dip, nu, R, db)
        else:
            I = -(1 - 2 * nu)/2 * xi * q / (R + db)**2
        return I


    def I2(eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        I = (1 - 2 * nu) * (-np.log(R + eta)) - \
            Okada.I3(eta, q, dip, nu, R)
        return I


    def I3(eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + \
                np.sin(dip) / np.cos(dip) * Okada.I4(db, eta, q, dip, nu, R)
        else:
            I = (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))
        return I


    def I4(db, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * 1.0 / np.cos(dip) * \
                (np.log(R + db) - np.sin(dip) * np.log(R + eta))
        else:
            I = - (1 - 2 * nu) * q / (R + db)
        return I


    def I5(xi, eta, q, dip, nu, R, db):
        """
        Auxiliary function for Okada (1985) model.
        """
        X = np.sqrt(xi**2 + q**2)
        if np.cos(dip) > eps:
            I = (1 - 2 * nu) * 2 / np.cos(dip) * \
                 np.arctan( (eta * (X + q*np.cos(dip)) + X*(R + X) * np.sin(dip)) /
                            (xi*(R + X) * np.cos(dip)) )
            I[xi == 0] = 0
        else:
            I = -(1 - 2 * nu) * xi * np.sin(dip) / (R + db)
        return I
    
    def K1(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            K = (1 - 2 * nu) * xi / np.cos(dip) * (1/(R * (R + db)) - np.sin(dip) / (R * (R + eta)))
        else:
            K = (1 - 2 * nu) * xi * q / (R * (R + db)**2)
        return K
    
    def K2(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        K3 = Okada.K3(xi, eta, q, dip, nu, R)
        
        K = (1 - 2 * nu) * (-np.sin(dip) / R + q * np.cos(dip) / (R * (R + eta))) - K3
        
        return K
    
    def K3(xi, eta, q, dip, nu, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            K = (1 - 2 * nu) / np.cos(dip) * ((q / R) * (1 / (R + eta)) - yb / (R * (R + db)))
        else:
            K = (1 - 2 * nu) * np.sin(dip) / (R + db) * (xi**2 / (R * (R + db)) - 1)
        return K
    
    def A(xieta, R):
        """
        Auxiliary function for Okada (1985) model.
        """
        A = (2 * R + xieta) / (R**3 * (R + xieta)**2)
        
        return A

    

