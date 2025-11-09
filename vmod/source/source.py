import numpy as np
import random
import scipy
from ..data import Data
from vmod.util import derivative


class Source:
    """
    Base class for analytical magmatic source models. Implements
    common functions required from all child classes.

    Attributes
        data (Data): data object
        x0 (array): initial guess for parameter values
        offsets (boolean): compute offsets as parameters for each component in the data object
        low_bounds (array): lower limit for the parameter values
        high_bounds (array): upper limit for the parameter values
    """
    def __init__(self, data):
        self.data        = data
        self.x0          = None
        self.offsets     = False
        self.parameters  = None
        self.reg         = False
        self.low_bounds  = []
        self.high_bounds = []

    def add_offsets(self):
        """
        Add offsets as parameters for each component in the dara object
        """
        if self.data.ts is None:
            self.offsets=True
            self.get_parnames()
        else:
            raise Exception('The data has a time dependency')

    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        
        Returns:
            steps (int): Number of steps used in the bayesian inversions.
            burnin (int): discarded number of steps at the begining of the inversion.
            thin (int): number of steps per sample.
        """
        steps=1100000
        burnin=100000
        thin=1000
        
        return steps,burnin,thin

    def get_parnames(self):
        """
        Function that add offsets to the list of parameters.
        """
        self.set_parnames()
        if self.offsets:
            for i,c in enumerate(self.data.comps):
                self.parameters=(*self.parameters,'offset'+str(i))
        return self.parameters

    def get_num_params(self):
        """
        Function that give the number of parameters.
        
        Returns:
            size (int): length of the parameters.
        """
        if self.parameters is None:
            self.get_parnames()
        return len(self.parameters)

    def set_x0(self, x0):
        """
        Function that sets the initial guess for the model.
        
        Parameters:
            x0 (list): list of values.
        """
        self.get_parnames()
        self.x0 = x0

    def set_bounds(self, low_bounds, high_bounds):
        """
        Function that sets the low and upper bounds for the parameters.
        
        Parameters:
            low_bounds (list): lower bounds for the parameters.
            high_bounds (list): upper bounds for the parameters.
        """
        self.get_parnames()
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds

    def get_xs(self):
        """
        Function that gives the data points positions in east.
        
        Returns:
            xs (list): positions in east.
        """
        return self.data.xs

    def get_ys(self):
        """
        Function that gives the data points positions in north.
        
        Returns:
            ys (list): positions in north.
        """
        return self.data.ys

    def get_ts(self):
        """
        Function that gives the times for the observations.
        
        Returns:
            ts (list): times for the observations.
        """
        return self.data.ts

    def draw_x0(self):
        """
        Function to draw a combination of parameters that are valid
        
        Returns:
            x0 (list): List of parameters.
        """
        novalid = True
        while(novalid):
            x0 = []
            for i in range(self.get_num_params()):
                low = float(self.low_bounds[i])
                high = float(self.high_bounds[i])
                x0.append(random.uniform(low,high))
            ux,uy,uz = self.model(np.array([1]),np.array([1]),*x0)
            if np.sum(ux)<1e4:
                novalid = False
        return x0

    def get_zs(self):
        """
        Function that gives the data points positions in vertical.
        
        Returns:
            zs (list): positions in vertical.
        """
        return self.data.zs

    def get_orders(self):
        """
        Function that gives the orders for the parameters value.
        
        Returns:
            orders (list): orders for the parameters.
        """
        orders=[]
        for i in range(len(self.low_bounds)):
            order=int(np.log10(np.max([np.abs(self.low_bounds[i]),np.abs(self.high_bounds[i])])))-1
            orders.append(10**order)
        orders=np.array(orders)
        return orders

    def strain(self,x,y,args):
        """
        Function that computes stresses in the horizontal plane.
        Parameters:
            x: x-coordinate (m)
            y: y-coordinate (m)
            args: parameters for the model
        Returns:
            sxx (list): normal strain in the x direction.
            syy (list): normal strain in the y direction.
            sxy (list): horizontal shear strain.
        """
        hx=0.001*np.abs(np.max(x)-np.min(x))
        hy=0.001*np.abs(np.max(y)-np.min(y))
        if hx==0:
            h=hy
        elif hy==0:
            h=hx
        elif hx<hy:
            h=hx
        elif hy<hx:
            h=hy
            
        u,v,w=self.model(x, y, *args)
        
        upx,vpx,wpx=self.model(x+h, y, *args)
        umx,vmx,wmx=self.model(x-h, y, *args)
        dudx=0.5*(upx-umx)/h
        dvdx=0.5*(vpx-vmx)/h
        dwdx=0.5*(wpx-wmx)/h
        
        upy,vpy,wpy=self.model(x, y+h, *args)
        umy,vmy,wmy=self.model(x, y-h, *args)
        dudy=0.5*(upy-umy)/h
        dvdy=0.5*(vpy-vmy)/h
        dwdy=0.5*(wpy-wmy)/h
        
        sxx=2*dudx
        syy=2*dvdy
        sxy=(dudy+dvdx)
        
        return sxx,syy,sxy

    def stress(self, x, y, z, args):
        """
        Function that computes stresses in the horizontal plane.

        Parameters:
            x: x-coordinate (m)
            y: y-coordinate (m)
            args: parameters for the model

        Returns:
            sxx (list): normal stress in the x direction (Pa).
            syy (list): normal stress in the y direction (Pa).
            szz (list): normal stress in the z direction (Pa).
            sxy (list): shear stress in the xy direction (Pa).
            sxz (list): shear stress in the xz direction (Pa).
            syz (list): shear stress in the yz direction (Pa).
        """
        ux=lambda h: self.model_depth(h, y, z, *args)[0]
        vx=lambda h: self.model_depth(h, y, z, *args)[1]
        wx=lambda h: self.model_depth(h, y, z, *args)[2]

        uy=lambda h: self.model_depth(x, h, z, *args)[0]
        vy=lambda h: self.model_depth(x, h, z, *args)[1]
        wy=lambda h: self.model_depth(x, h, z, *args)[2]

        uz=lambda h: self.model_depth(x, y, h, *args)[0]
        vz=lambda h: self.model_depth(x, y, h, *args)[1]
        wz=lambda h: self.model_depth(x, y, h, *args)[2]

        dudx=derivative(ux,x,delta=1e-8)
        dvdx=derivative(vx,x,delta=1e-8)
        dwdx=derivative(wx,x,delta=1e-8)

        dudy=derivative(uy,y,delta=1e-8)
        dvdy=derivative(vy,y,delta=1e-8)
        dwdy=derivative(wy,y,delta=1e-8)

        dudz=-derivative(uz,z,delta=1e-8)
        dvdz=-derivative(vz,z,delta=1e-8)
        dwdz=-derivative(wz,z,delta=1e-8)

        nu=args[-2]
        mu=args[-1]

        sxx=2*(1+nu)*dudx*mu
        syy=2*(1+nu)*dvdy*mu
        szz=2*(1+nu)*dwdz*mu
        sxy=(1+nu)*(dudy+dvdx)*mu
        sxz=(1+nu)*(dudz+dwdx)*mu
        syz=(1+nu)*(dvdz+dwdy)*mu

        return sxx,syy,szz,sxy,sxz,syz

    def fault_vectors(self,strike, dip, rake):
        """
        Function that calculates the normal and slip vectors for a fault geometry.
        
        Parameters:
            strike: strike angle in degrees for the receiver fault
            dip: dip angle in degrees for the receiver fault
            rake: rake angle in degrees for the receiver fault
        
        Returns:
            n: normal vector
            s: slip vector
        """
        strike=np.radians(strike)
        dip=np.radians(dip)
        rake=np.radians(rake)
        n = np.array([
             np.sin(dip) * np.cos(strike),
            -np.sin(dip) * np.sin(strike),
             np.cos(dip)
        ])
    
        s = np.array([
            -np.cos(strike) * np.cos(dip) * np.sin(rake) + np.sin(strike) * np.cos(rake),
             np.sin(strike) * np.cos(dip) * np.sin(rake) + np.cos(strike) * np.cos(rake),
             np.sin(dip) * np.sin(rake)
        ])
    
        return n / np.linalg.norm(n), s / np.linalg.norm(s)

    def principal_stresses_2d(self,args):
        """
        Function that calculates the value and orientation for the first two principal stresses.

        Parameters:
            args: parameters for the model

        Returns:
            s1s: value for the first principal stresses
            s2s: value for the second principal stresses
            d1s: orientation for the first principal stresses
            d2s: orientation for the second principal stresses
        """
        xs=np.copy(self.data.xs)
        ys=np.copy(self.data.ys)
        sxx,syy,sxy=self.strain(xs,ys,args)
        sxx=(1+args[-2])*args[-1]*sxx
        syy=(1+args[-2])*args[-1]*syy
        sxy=(1+args[-2])*args[-1]*sxy
        s1s=sxx*np.nan
        s2s=sxx*np.nan
        d1s=np.ones((len(sxx),2))*np.nan
        d2s=np.ones((len(sxx),2))*np.nan
        for i in range(len(sxx)):
            st=np.ones((2,2))*np.nan
            st[0,0]=sxx[i]
            st[1,1]=syy[i]
            st[0,1]=sxy[i]
            st[1,0]=sxy[i]
            eigenvalues, eigenvectors = np.linalg.eig(st)
            s1s[i]=np.max(eigenvalues)
            d1s[i,:]=eigenvectors[:,np.argmax(eigenvalues)]/np.linalg.norm(eigenvectors[:,np.argmax(eigenvalues)])
            s2s[i]=np.min(eigenvalues)
            d2s[i,:]=eigenvectors[:,np.argmin(eigenvalues)]/np.linalg.norm(eigenvectors[:,np.argmin(eigenvalues)])
        return s1s,s2s,d1s,d2s

    def principal_stresses(self,z,args):
        """
        Function that calculates the value and orientation of principal stresses.

        Parameters:
            z: slide to calculate for Coulomb stress change (depth is positive)
            args: parameters for the model

        Returns:
            s1s: value for the first principal stresses
            s2s: value for the second principal stresses
            s3s: value for the third principal stresses
            d1s: orientation for the first principal stresses
            d2s: orientation for the second principal stresses
            d3s: orientation for the third principal stresses
        """
        if not 'model_depth' in dir(self):
            raise Exception('The current model cannot compute internal displacements please define the function \'model_depth\' or use function \'principal_stresses_2d\'')
        xs=np.copy(self.data.xs)
        ys=np.copy(self.data.ys)
        sxx,syy,szz,sxy,sxz,syz=self.stress(xs,ys,z,args)
        s1s=sxx*np.nan
        s2s=sxx*np.nan
        s3s=sxx*np.nan
        d1s=np.ones((len(sxx),3))*np.nan
        d2s=np.ones((len(sxx),3))*np.nan
        d3s=np.ones((len(sxx),3))*np.nan
        for i in range(len(sxx)):
            st=np.ones((3,3))*np.nan
            st[0,0]=sxx[i]
            st[1,1]=syy[i]
            st[2,2]=szz[i]
            st[0,1]=sxy[i]
            st[1,0]=sxy[i]
            st[0,2]=sxz[i]
            st[2,0]=sxz[i]
            st[1,2]=syz[i]
            st[2,1]=syz[i]
            st[2,2]=szz[i]
            eigenvalues, eigenvectors = np.linalg.eig(st)
            s1s[i]=np.max(eigenvalues)
            d1s[i,:]=eigenvectors[:,np.argmax(eigenvalues)]/np.linalg.norm(eigenvectors[:,np.argmax(eigenvalues)])
            s3s[i]=np.min(eigenvalues)
            d3s[i,:]=eigenvectors[:,np.argmin(eigenvalues)]/np.linalg.norm(eigenvectors[:,np.argmin(eigenvalues)])
            for j,s in enumerate(eigenvalues):
                if not s==s1s[i] and not s==s3s[i]:
                    s2s[i]=s
                    d2s[i,:]=eigenvectors[j]
        return s1s,s2s,s3s,d1s,d2s,d3s

    def coulomb_change(self,strike,dip,rake,z,friction,args):
        """
        Function that computes the Coulomb stress change on a receiver fault geometry.

        Parameters:
            strike: strike angle in degrees for the receiver fault
            dip: dip angle in degrees for the receiver fault
            rake: rake angle in degrees for the receiver fault
            z: slide to calculate for Coulomb stress change (depth is positive)
            friction: friction coefficient
            args: parameters for the model

        Returns:
            sshear: shear stress in the receiver fault
            snormal: normal stress in the receiver fault
            coulomb: Coulomb stress change for the receiver fault
        """
        xs=np.copy(self.data.xs)
        ys=np.copy(self.data.ys)

        if not 'model_depth' in dir(self) and (not dip==90 or not (rake==0 or rake==180) or not z==0):
            raise Exception('The current model cannot compute internal displacements please define the function \'model_depth\' or change rake to 0 or 180, dip to 90 and z to 0')
        elif not 'model_depth' in dir(self) and (dip==90 and (rake==0 or rake==180) and z==0):
            print('Calculating only horizontal stresses on the free surface')
            sxx,syy,sxy=self.strain(xs,ys,args)
            sxx=(1+args[-2])*args[-1]*sxx
            syy=(1+args[-2])*args[-1]*syy
            sxy=(1+args[-2])*args[-1]*sxy
            sxz,syz,szz=sxx*0,sxx*0,sxx*0
        else:
            sxx,syy,szz,sxy,sxz,syz=self.stress(xs,ys,z,args)

        n,s=self.fault_vectors(strike,dip,rake)
        nx,ny,nz=n
        sx,sy,sz=s

        tx=sxx*nx+sxy*ny+sxz*nz
        ty=sxy*nx+syy*ny+syz*nz
        tz=sxz*nx+syz*ny+szz*nz

        snormal=tx*nx+ty*ny+tz*nz
        sshear=(tx*sx+ty*sy+tz*sz)

        return sshear,snormal,sshear+friction*snormal

    def forward(self,args,unravel=True):
        """
        Function that computes the forward model.

        Parameters:
            args: parameters for the model

        Returns:
            output (list): output in certain datatype according to the data object.
        """
        self.get_parnames()
        if self.offsets:
            offsets=args[-len(self.data.comps)::]
            args=args[0:len(args)-len(self.data.comps)]
        else:
            offsets=None

        if not self.data.zs is None and 'depth' in self.parameters:
            pos=np.argwhere(np.array(self.parameters)=='depth')[0][0]
            if isinstance(args, np.ndarray):
                args=args.tolist()
            args[pos]=self.data.zs+args[pos]

        if self.data.ts is None:
            if self.data.__class__.__name__=='Tilt' and 'model_tilt' in dir(self):
                func_tilt=lambda x,y: self.model_tilt(x,y,*args)
                func_tilt.__name__ = 'func_tilt'
                return self.data.from_model(func_tilt,offsets,unravel)
            elif 'model' in dir(self):
                func=lambda x,y: self.model(x,y,*args)
                return self.data.from_model(func,offsets,unravel)
            else:
                raise Exception('The source does not have a time-independent model defined')
        else:
            if self.data.__class__.__name__=='Tilt' and 'model_tilt_t' in dir(self):
                func_tilt_time=lambda x,y,t: self.model_tilt_t(x,y,t,*args)
                func_tilt_time.__name__ = 'func_tilt_time'
                return self.data.from_model(func_tilt_time,offsets,unravel)
            elif 'model_t' in dir(self):
                func_time=lambda x,y,t: self.model_t(x,y,t,*args)
                func_time.__name__ = 'func_time'
                return self.data.from_model(func_time,offsets,unravel)
            else:
                raise Exception('The source does not have a time-dependent model defined')

