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

Authors: Scott Henderson, Mario Angarita, Ronni Grapenthin
'''

from __future__ import division
import numpy as np
from . import Source
from .. import util
import time

eps = 1e-14 #numerical constant
#tests for cos(dip) == 0 are made with "cos(dip) > eps"
#because cos(90*np.pi/180) is not zero but = 6.1232e-17 (!)

class Okada(Source):

    def set_type(self,typ):
        self.type=typ
        
    def time_dependent(self):
        return False

    def get_source_id(self):
        return "Okada"
    
    def bayesian_steps(self):
        steps=5100000
        burnin=100000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
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
        if self.type=='slip':
            self.parameters=("xcen","ycen","depth","length","width","slip","strike","dip","rake")
        elif self.type=='open':
            self.parameters=("xcen","ycen","depth","length","width","opening","strike","dip")
    
    def rotate_xyz(self,xcen,ycen,depth,length,width,strike,dip):
        cx=xcen
        cy=ycen
        cz=-depth
        wp=width*np.cos(np.radians(dip))
        wr=width*np.sin(np.radians(dip))
        l=length
        phi=strike
        x1 = cx + wp/2 * np.cos(np.radians(phi)) - l/2 * np.sin(np.radians(phi))
        y1 = cy + wp/2 * np.sin(np.radians(phi)) + l/2 * np.cos(np.radians(phi))
        z1 = cz - wr/2
        x2 = cx - wp/2 * np.cos(np.radians(phi)) - l/2 * np.sin(np.radians(phi))
        y2 = cy - wp/2 * np.sin(np.radians(phi)) + l/2 * np.cos(np.radians(phi))
        z2 = cz + wr/2
        x3 = cx - wp/2 * np.cos(np.radians(phi)) + l/2 * np.sin(np.radians(phi))
        y3 = cy - wp/2 * np.sin(np.radians(phi)) - l/2 * np.cos(np.radians(phi))
        z3 = cz + wr/2
        x4 = cx + wp/2 * np.cos(np.radians(phi)) + l/2 * np.sin(np.radians(phi))
        y4 = cy + wp/2 * np.sin(np.radians(phi)) - l/2 * np.cos(np.radians(phi))
        z4 = cz - wr/2
        
        return [x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]
    
    def get_centers(self,xcen,ycen,depth,length,width,strike,dip,ln,wn):
        xc=xcen
        yc=ycen
        zc=-depth
        lslice=length/ln
        wslice=width/wn
        fwc=xcen-width/2+width/(2*wn)
        flc=ycen-length/2+length/(2*ln)
        #print(fwc,flc)
        xcs,ycs,zcs=[],[],[]
        if wn%2==0:
            wi=wn/2
        else:
            wi=(wn-1)/2
            
        if ln%2==0:
            li=ln/2
        else:
            li=(ln-1)/2
            
        for i in range(int(wi)):
            wfake=2*np.abs(fwc-xcen+float(i)*wslice)
            for j in range(int(li)):
                lfake=2*np.abs(flc-ycen+float(j)*lslice)
                xs,ys,zs=self.rotate_xyz(xcen,ycen,depth,lfake,wfake,strike,dip)
                for x in xs:
                    xcs.append(x)
                for y in ys:
                    ycs.append(y)
                for z in zs:
                    zcs.append(z)
        print('Puntos 1',len(xcs),wn%2,ln%2)
        if not ln%2==0:
            for j in range(int(li)):
                wfake=0
                lfake=2*np.abs(flc-ycen+float(j)*lslice)
                xs,ys,zs=self.rotate_xyz(xcen,ycen,depth,lfake,wfake,strike,dip)
                for x in xs[1:3]:
                    xcs.append(x)
                for y in ys[1:3]:
                    ycs.append(y)
                for z in zs[1:3]:
                    zcs.append(z)
        print('Puntos 2',len(xcs))
        if not wn%2==0:
            for i in range(int(wi)):
                wfake=2*np.abs(fwc-xcen+float(i)*wslice)
                lfake=0
                xs,ys,zs=self.rotate_xyz(xcen,ycen,depth,lfake,wfake,strike,dip)
                for x in xs[0:2]:
                    xcs.append(x)
                for y in ys[0:2]:
                    ycs.append(y)
                for z in zs[0:2]:
                    zcs.append(z)
        print('Puntos 3',len(xcs))
        if (not wn%2==0) and (not ln%2==0):
            print('Ninguno')
            xcs.append(xcen)
            ycs.append(ycen)
            zcs.append(-depth)
        print('Puntos 4',len(xcs))
        return xcs,ycs,zcs
    
    def get_greens(self,xcen,ycen,depth,length,width,strike,dip,ln,wn):
        xcs,ycs,zcs=self.get_centers(xcen,ycen,depth,length,width,strike,dip,ln,wn)
        #print(xcs,ycs)
        xo=[xcen,ycen,depth,length,width,1,strike,dip]
        defo=self.get_model(xo)
        slength=length/ln
        swidth=width/wn
        G=np.zeros((len(defo),ln*wn))
        for i in range(len(xcs)):
            #print(xcs[i],ycs[i])
            xp=[xcs[i],ycs[i],-zcs[i],slength,swidth,1,strike,dip]
            defo=self.get_model(xp,wts=1,scale=False)
            G[:,i]=defo
        return G
    
    def get_laplacian(self,xcen,ycen,depth,length,width,strike,dip,ln,wn):
        xcs,ycs,zcs=self.get_centers(xcen,ycen,depth,length,width,strike,dip,ln,wn)
        L=np.zeros((ln*wn,ln*wn))
        for i in range(len(xcs)):
            dist=(np.array(xcs)-xcs[i])**2+(np.array(ycs)-ycs[i])**2+(np.array(zcs)-zcs[i])**2
            pos=np.argsort(dist)
            #print(dist[pos[0:5]])
            if dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]] and dist[pos[1]]==dist[pos[4]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
                L[i,pos[4]]=1
            elif dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
            elif dist[pos[1]]==dist[pos[2]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
        return L
    
    def get_args(self,args, tilt):
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
        rargs=self.get_args(args,tilt=False)
        return self.model_gen(x,y, *rargs)
    
    def model_tilt(self,x,y,*args):
        rargs=self.get_args(args,tilt=True)
        return self.model_gen(x,y, *rargs)
    
    def model_gen(self,x,y, xcen=0, ycen=0,
                        depth=5e3, length=1e3, width=1e3,
                        slip=0.0, opening=10.0,
                        strike=0.0, dip=0.0, rake=0.0,
                        nu=0.25,tilt=False):
        '''
        Calculate surface displacements for Okada85 dislocation model
        '''
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
        u = db * q / (R * (R + eta)) + \
            q * np.sin(dip) / (R + eta) + \
            Okada.I4(db, eta, q, dip, nu, R) * np.sin(dip)
        return u
    
    def dx_ss(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = -xi * q**2 * Okada.A(eta,R) * np.cos(dip) + \
            (xi * q / R**3 - Okada.K1(xi, eta, q, dip, nu, R)) * np.sin(dip)
        return u
    
    def dy_ss(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = db * q / R**3 * np.cos(dip) + \
            (xi**2 * q * Okada.A(eta,R) * np.cos(dip) - np.sin(dip) / R + yb * q / R**3 - Okada.K2(xi, eta, q, dip, nu, R)) * np.sin(dip)
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
    
    def dx_ds(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        
        u = db * q / R**3 + \
            q * np.sin(dip) / (R * (R + eta)) + \
            Okada.K3(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        
        return u
    
    def dy_ds(xi, eta, q, dip, nu):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = yb * db * q * Okada.A(xi,R) - \
            (2 * db / (R * (R + xi)) + xi * np.sin(dip) / (R * (R + eta))) * np.sin(dip) + \
            Okada.K1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
        
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
    
    def dx_tf(xi, eta, q, dip, nu):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = q**2 * np.sin(dip) / R**3 - q**3 * Okada.A(eta,R) * np.cos(dip) + Okada.K3(xi, eta, q, dip, nu, R) * (np.sin(dip))**2
        
        return u
    
    def dy_tf(xi, eta, q, dip, nu):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        
        u = (yb * np.sin(dip) + db * np.cos(dip)) * q**2 * Okada.A(xi,R) + \
            xi * q**2 * Okada.A(eta,R) * np.sin(dip) * np.cos(dip) - \
            (2 * q/(R * (R + xi)) - Okada.K1(xi, eta, q, dip, nu, R)) * (np.sin(dip))**2
        
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
    
    def K1(xi, eta, q, dip, nu, R):
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            K = (1 - 2 * nu) * xi / np.cos(dip) * (1/(R * (R + db)) - np.sin(dip) / (R * (R + eta)))
        else:
            K = (1 - 2 * nu) * xi * q / (R * (R + db)**2)
        return K
    
    def K2(xi, eta, q, dip, nu, R):
        K3 = Okada.K3(xi, eta, q, dip, nu, R)
        
        K = (1 - 2 * nu) * (-np.sin(dip) / R + q * np.cos(dip) / (R * (R + eta))) - K3
        
        return K
    
    def K3(xi, eta, q, dip, nu, R):
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > eps:
            K = (1 - 2 * nu) / np.cos(dip) * ((q / R) * (1 / (R + eta)) - yb / (R * (R + db)))
        else:
            K = (1 - 2 * nu) * np.sin(dip) / (R + dp) * (xi**2 / (R * (R + dp)) - 1)
        return K
    
    def A(xieta, R):
        A = (2 * R + xieta) / (R**3 * (R + xieta)**2)
        
        return A

    

