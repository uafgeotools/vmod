"""
Base class for analytical magmatic source models. Implements
common functions required from all child classes.

Author: Ronni Grapenthin, UAF
Date: 6/23/2021


TODO:
-add sphinx docstrings
"""
import numpy as np
import util
import pandas as pd
import copy
import scipy
from data import Data

class Source:
    def __init__(self, data, typ='elastic'):
        self.data        = data
        self.type        = typ
        self.x0          = None
        self.low_bounds  = []
        self.high_bounds = []

    def set_x0(self, x0):
        self.x0 = x0

    def set_bounds(self, low_bounds, high_bounds):
        self.low_bounds  = low_bounds
        self.high_bounds = high_bounds
        
    def forward(self, xpos, ypos, x):
        if self.get_source_id()=='Mogi':
            return self.model(xpos,ypos,x[0], x[1], x[2], x[3])
        elif self.get_source_id()=='Mctigue':
            return self.model(xpos,ypos,x[0], x[1], x[2], x[3], x[4])
        elif self.get_source_id()=='Okada':
            if self.type=='slip':
                return self.model(xpos,ypos,x[0], x[1], x[2], x[3], x[4], x[5], 0.0, x[6], x[7], x[8])
            elif self.type=='open':
                return self.model(xpos,ypos,x[0], x[1], x[2], x[3], x[4], 0.0, x[5], x[6], x[7], 0.0)
        elif self.get_source_id()=='Penny':
            return self.model(xpos,ypos,x[0], x[1], x[2], x[3], x[4])
        elif self.get_source_id()=='Nishimura':
            ist=np.sum(~np.isnan(self.data.data['t'].to_numpy()))
            if ist>0:
                return self.model_t(xpos,ypos,x[0],x[1],x[2],x[3],x[4],x[5],x[6])
            else:
                return self.model(xpos,ypos,x[0],x[1],x[2],x[3],x[4],x[5])
        elif self.get_source_id()=='Yang':
            return self.model(xpos,ypos,x[0], x[1], x[2], x[3],x[4], x[5], x[6], x[7])
    
    def forward_insar(self,x):
        xpos=self.get_xs(dataset='insar')
        ypos=self.get_ys(dataset='insar')
        
        ux,uy,uz=self.forward(xpos,ypos,x)
        
        data=self.data.get_data('insar')
        
        los=ux*np.sin(data['lk'].to_numpy())*np.cos(data['az'].to_numpy())-uy*np.sin(data['lk'].to_numpy())*np.sin(data['az'].to_numpy())- uz*np.cos(data['lk'].to_numpy())
        los=-los
        
        return los
    
    def forward_gps(self,x):
        xpos=self.get_xs(dataset='gps')
        ypos=self.get_ys(dataset='gps')
        
        return self.forward(xpos,ypos,x)
    
    def forward_tilt(self, x):
        uzx= lambda xpos: self.forward(xpos,self.get_ys(dataset='tilt'),x)[2]
        uzy= lambda ypos: self.forward(self.get_xs(dataset='tilt'),ypos,x)[2]
        
        duzx=-scipy.misc.derivative(uzx,self.get_xs(dataset='tilt'),dx=1e-6)
        duzy=-scipy.misc.derivative(uzy,self.get_ys(dataset='tilt'),dx=1e-6)
        
        return duzx,duzy

    def get_obs(self):
        return self.data.get_obs()

    def get_xs(self,dataset=None):
        return self.data.get_xs(dataset)

    def get_ys(self,dataset=None):
        return self.data.get_ys(dataset)

    def get_ts(self,dataset=None):
        return self.data.get_ts(dataset)
    
    def get_zs(self,dataset=None):
        return self.data.get_zs(dataset)

    def get_site_ids(self,dataset=None):
        return self.data.get_site_ids(dataset)

    def get_lats(self,dataset=None):
        return self.data.get_lats(dataset)

    def get_lons(self,dataset=None):
        return self.data.get_lons(dataset)
    
    def get_orders(self):
        orders=[]
        for i in range(len(self.low_bounds)):
            order=int(np.log10(np.max([np.abs(self.low_bounds[i]),np.abs(self.high_bounds[i])])))-1
            orders.append(10**order)
        orders=np.array(orders)
        return orders
    
    def get_model_los(self,x0):
        return self.forward_insar(x0)
        
    def get_model_3d(self,x0):
        ux,uy,uz = self.forward_gps(x0)
        if not self.data.refidx==None:
            ux-=ux[self.data.refidx]
            uy-=uy[self.data.refidx]
            uz-=uz[self.data.refidx]
        return np.concatenate((ux,uy,uz)).ravel()
    
    def get_model_tilt(self,x0):
        dux,duy = self.forward_tilt(x0)
        return np.concatenate((dux,duy)).ravel()
    
    def get_model(self,x0,wts=1):
        pars=np.copy(x0)
        parnames=self.get_parnames()
        for i in range(len(pars)):
            order=int(np.log10(np.max([np.abs(self.low_bounds[i]),np.abs(self.high_bounds[i])])))-1
            if parnames[i]=='pressure':
                pars[i]=10**pars[i]
            else:
                pars[i]=pars[i]*10**order
        if self.data.type=='gps' or self.data.type=='joint':
            gps=self.get_model_3d(pars)
            if self.data.type=='gps':
                return gps
        if self.data.type=='insar' or self.data.type=='joint':
            insar=self.get_model_los(pars)
            if self.data.type=='insar':
                return insar
        if self.data.type=='tilt' or self.data.type=='joint':
            tilt=self.get_model_tilt(pars)*1e6
            if self.data.type=='tilt':
                return tilt
        if self.data.type=='joint':
            data=np.concatenate((gps,insar,tilt))
            return data*wts
    def get_weights(self,wts):
        if self.data.type=='joint':
            if len(wts)<=3:
                if len(self.get_xs('gps'))==0:
                    winsar=self.get_xs('insar')*0+wts[0]
                    wtilt=self.get_xs('tilt')*0+wts[1]
                    return np.concatenate((winsar,wtilt,wtilt))
                elif len(self.get_xs('insar'))==0:
                    wgps=self.get_xs('gps')*0+wts[0]
                    wtilt=self.get_xs('tilt')*0+wts[1]
                    return np.concatenate((wgps,wgps,wgps,wtilt,wtilt))
                elif len(self.get_xs('tilt'))==0:
                    wgps=self.get_xs('gps')*0+wts[0]
                    winsar=self.get_xs('insar')*0+wts[1]
                    return np.concatenate((wgps,wgps,wgps,winsar))
                else:
                    wgps=self.get_xs('gps')*0+wts[0]
                    winsar=self.get_xs('insar')*0+wts[1]
                    wtilt=self.get_xs('tilt')*0+wts[2]
                    return np.concatenate((wgps,wgps,wgps,winsar,wtilt,wtilt))
            else:
                return wts
            
    def get_residual(self,x0):
        ux,uy,uz = self.forward_mod(x0)
        return self.data-np.concatenate((ux,uy,uz)).ravel()
    
    def res_norm(self):
        ux,uy,uz = self.forward_mod()
        return np.linalg.norm(self.get_obs()*1000-np.concatenate([ux, uy, uz])*1000)

    ##inversion methods
    def invert(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun, x0, bounds=bounds))
        return self.model

    def invert_dipole(self, x0, bounds=None):
        from scipy.optimize import least_squares
        self.model = copy.deepcopy(least_squares(self.fun_dipole, x0, bounds=bounds))
        return self.model

    def invert_bh(self, x0):
        from scipy.optimize import basinhopping
        return basinhopping(self.fun, x0)

    ##output writers
    def write_forward_gmt(self, prefix):
        if self.model is not None:
            ux,uy,uz = self.forward_mod()

            dat = np.zeros(self.data.data['id'].to_numpy().size, 
                dtype=[ ('lon', float), ('lat', float), ('east', float), ('north', float), 
                        ('esig', float), ('nsig', float), ('corr', float), ('id', 'U6')] )

            dat['lon']   = self.get_lons()
            dat['lat']   = self.get_lats()
            dat['east']  = ux*1000
            dat['north'] = uy*1000
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.get_site_ids()

            print(dat)

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz*1000
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )

    def make_map(self, west, east, south, north, proj):
        import pygmt

        fig = pygmt.Figure()

        if self.model is not None:
            ux,uy,uz = self.forward_mod()
            df_mod_hori = pd.DataFrame(
                            data={
                                "x": self.get_xs(),
                                "y": self.get_ys(),
                                "east_velocity":  ux,
                                "north_velocity": uy,
                                "east_sigma": ux*0,
                                "north_sigma": uy*0,
                                "correlation_EN": ux*0,
                                "SITE": self.get_site_ids(),
                                }
                    )
        fig.velo(
            data=df_mod_hori,
            region=[west, east, south, north],
            pen="0.6p,red",
            uncertaintycolor="lightblue1",
            line=True,
            spec="e0.2/0.39/18",
            frame=["WSne", "2g2f"],
            projection=proj, 
            vector="0.3c+p1p+e+gred",
        )

        fig.show()        
   



