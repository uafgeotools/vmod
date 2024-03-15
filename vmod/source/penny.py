import numpy as np
from .. import util
import scipy
from scipy import special
from . import Source

class Penny(Source):
    """
    Class used to represent a penny-shaped crack using the Fialko (2001) model.

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
        return "Penny"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        """
        steps=110000
        burnin=10000
        thin=100
        
        return steps,burnin,thin

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Penny-shaped Crack:")
        print("\tx  = %f (m)" % x[0])
        print("\ty  = %f (m)" % x[1])
        print("\td  = %f (m)" % x[2])
        print("\tP_G= %f (m)" % x[3])
        print("\ta  = %f (m)" % x[4])
        
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","pressure","radius")
        
    # =====================
    # Forward Models
    # =====================
    
    def model(self,x,y,xcen,ycen,d,dP,a,nu=0.25):
        """
        Calculates surface deformation based on pressurized penny-shaped crack
        References: Fialko

        Parameters:
            x: x-coordinate grid (m)
            y: y-coordinate grid (m)
            xcen: x-offset of penny-shaped crack (m)
            ycen: y-offset of penny-shaped crack (m)
            d: depth to penny-shaped crack (m)
            dP: change in pressure (in terms of mu if mu=1 if not unit is Pa)
            a: radius for penny shaped crack (m)
            mu: shear modulus for medium (Pa) (default 1)
            nu: poisson's ratio for medium

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """

        #
        rd=np.copy(a)

        # Center coordinate grid on point source, normalized by radius
        x = (x - xcen) / rd
        y = (y - ycen) / rd
        z = (0 - d)    / rd

        eps=1e-8

        h  = d / rd
        r  = np.sqrt(x ** 2 + y ** 2)

        csi1,w1 = self.gauleg(eps,10,41)
        csi2,w2 = self.gauleg(10,60,41)
        csi     = np.concatenate((csi1,csi2))
        wcsi    = np.concatenate((w1,w2))

        if csi.shape[0] == 1:
            csi=csi.T

        phi1,psi1,t,wt=self.psi_phi(h)

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

        ux = rd * P_G * Ur*x / r
        uy = rd * P_G * Ur*y / r
        uz = -rd * P_G * Uz

        return np.array([ux,uy,uz])
    
    def model_depth(self,x,y,z,xcen,ycen,d,P,a,mu=1e10,nu=0.25):
        """
        Calculates deformation at depth based on pressurized penny-shaped crack
        References: Fialko

        Parameters:
            x: x-coordinate grid (m)
            y: y-coordinate grid (m)
            z: z-coordinate grid (m)
            xcen: x-offset of penny-shaped crack (m)
            ycen: y-offset of penny-shaped crack (m)
            d: depth to penny-shaped crack (m)
            P: pressure (Pa)
            a: radius penny-shaped crack (Pa)
            mu: shear modulus (Pa)
            nu: poisson's ratio for medium

        Returns:
            ux (array) : displacements in east in meters.
            uy (array) : displacements in north in meters.
            uz (array) : displacements in vertical in meters.
        """
        P_G=P/mu

        #
        rd=np.copy(a)

        # Center coordinate grid on point source, normalized by radius
        x = (x - xcen) / rd
        y = (y - ycen) / rd
        z = (0 - (d-z)) / rd

        eps=1e-8

        h  = d / rd
        r  = np.sqrt(x ** 2 + y ** 2)

        csi1,w1 = self.gauleg(eps,10,41)
        csi2,w2 = self.gauleg(10,60,41)
        csi     = np.concatenate((csi1,csi2))
        wcsi    = np.concatenate((w1,w2))

        if csi.shape[0] == 1:
            csi=csi.T

        phi1,psi1,t,wt=self.psi_phi(h)

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

        ux = rd * P_G * Ur*x / r
        uy = rd * P_G * Ur*y / r
        uz = -rd * P_G * Uz

        return np.array([ux,uy,uz])

    def psi_phi(self,h):
        """
        Auxiliary function for the Fialko (2001) model
        """
        t,w=self.gauleg(0,1,41)
        t=np.array(t)
        g=-2.0*t/np.pi
        d=np.concatenate((g,np.zeros(g.size)))
        T1,T2,T3,T4=self.giveT(h,t,t)
        T1p=np.zeros(T1.shape)
        T2p=np.zeros(T1.shape)
        T3p=np.zeros(T1.shape)
        T4p=np.zeros(T1.shape)
        N=t.size
        for j in range(N):
            T1p[:,j]=w[j]*T1[:,j]
            T2p[:,j]=w[j]*T2[:,j]
            T3p[:,j]=w[j]*T3[:,j]
            T4p[:,j]=w[j]*T4[:,j]
        M1=np.concatenate((T1p,T3p),axis=1)
        M2=np.concatenate((T4p,T2p),axis=1)
        Kp=np.concatenate((M1,M2),axis=0)
        y=np.matmul(np.linalg.inv(np.eye(2*N,2*N)-(2/np.pi)*Kp),d)
        phi=y[0:N]
        psi=y[N:2*N]
        return phi,psi,t,w

    def giveP(self,h,x):
        """
        Auxiliary function for the Fialko (2001) model
        """
        P=np.zeros((4,x.size))
        P[0]=(12*np.power(h,2)-np.power(x,2))/np.power((4*np.power(h,2)+np.power(x,2)),3)
        P[1] = np.log(4*np.power(h,2)+np.power(x,2)) + (8*np.power(h,4)+2*np.power(x,2)*np.power(h,2)-np.power(x,4))/np.power(4*np.power(h,2)+np.power(x,2),2)
        P[2] = 2*(8*np.power(h,4)-2*np.power(x,2)*np.power(h,2)+np.power(x,4))/np.power(4*np.power(h,2)+np.power(x,2),3)
        P[3] = (4*np.power(h,2)-np.power(x,2))/np.power((4*np.power(h,2)+np.power(x,2)),2)
        return P

    def giveT(self,h,t,r):
        """
        Auxiliary function for the Fialko (2001) model
        """
        M = t.size
        N = r.size
        T1 = np.zeros((M,N)) 
        T2 = np.zeros((M,N)) 
        T3 = np.zeros((M,N))
        for i in range(M):
            Pm=self.giveP(h,t[i]-r)
            Pp=self.giveP(h,t[i]+r)
            T1[i] = 4*np.power(h,3)*(Pm[0,:]-Pp[0,:])
            T2[i] = (h/(t[i]*r))*(Pm[1,:]-Pp[1,:]) +h*(Pm[2,:]+Pp[2,:])
            T3[i] = (np.power(h,2)/r)*(Pm[3,:]-Pp[3,:]-2*r*((t[i]-r)*Pm[0,:]+(t[i]+r)*Pp[0,:]))
        T4=np.copy(T3.T)
        return T1,T2,T3,T4
    
    def gauleg(self,a,b,n):
        """
        Auxiliary function for the Fialko (2001) model
        """
        xs, cs = self.gauleg_params1(n)
        coeffp = 0.5*(b+a) 
        coeffm = 0.5*(b-a)
        ts = coeffp - coeffm*xs
        ws=cs*coeffm
        #contribs = cs*f(ts)
        #return coeffm*np.sum(contribs)
        return ts[::-1],ws
    
    def gauleg_params1(self,n):
        """
        Auxiliary function for the Fialko (2001) model
        """
        xs,cs=np.polynomial.legendre.leggauss(n)
        return xs,cs