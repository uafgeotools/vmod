import numpy as np
import scipy
from ..data import Data

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
            
        u,v,w=self.model_depth(x, y, z, *args)
        
        upx,vpx,wpx=self.model_depth(x+h, y, z, *args)
        umx,vmx,wmx=self.model_depth(x-h, y, z, *args)
        dudx=0.5*(upx-umx)/h
        dvdx=0.5*(vpx-vmx)/h
        dwdx=0.5*(wpx-wmx)/h
        
        upy,vpy,wpy=self.model_depth(x, y+h, z, *args)
        umy,vmy,wmy=self.model_depth(x, y-h, z, *args)
        dudy=0.5*(upy-umy)/h
        dvdy=0.5*(vpy-vmy)/h
        dwdy=0.5*(wpy-wmy)/h
        
        double=False
        if isinstance(z,float):
            if z==0:
                double=True
            else:
                double==False
        elif len(z[z==0])>0:
            double=True
            
        if double:
            upz,vpz,wpz=self.model_depth(x, y, z+2*h, *args)
            umz,vmz,wmz=self.model_depth(x, y, z, *args)
        else:
            upz,vpz,wpz=self.model_depth(x, y, z+h, *args)
            umz,vmz,wmz=self.model_depth(x, y, z-h, *args)
        dudz=0.5*(upz-umz)/h
        dvdz=0.5*(vpz-vmz)/h
        dwdz=0.5*(wpz-wmz)/h
        
        
        nu=args[-1]
        mu=args[-2]
        
        sxx=2*(1+nu)*dudx*mu
        syy=2*(1+nu)*dvdy*mu
        szz=2*(1+nu)*dwdz*mu
        sxy=(1+nu)*(dudy+dvdx)*mu
        sxz=(1+nu)*(dudz+dwdx)*mu
        syz=(1+nu)*(dvdz+dwdy)*mu
        
        return sxx,syy,szz,sxy,sxz,syz
    
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
            print(args,pos)
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
                print('Name',func_tilt_time.__name__)
                return self.data.from_model(func_tilt_time,offsets,unravel)
            elif 'model_t' in dir(self):
                func_time=lambda x,y,t: self.model_t(x,y,t,*args)
                func_time.__name__ = 'func_time'
                return self.data.from_model(func_time,offsets,unravel)
            else:
                raise Exception('The source does not have a time-dependent model defined')
    