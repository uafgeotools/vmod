"""
Class to implement interface to geodetic data types used in source inversions

Author: Ronni Grapenthin
Date: 6/23/2021


TODO:
- integrate (In)SAR LOS
- integrate tilt
"""

import pandas as pd
import numpy as np
#from pyproj import CRS
#from pyproj import Transformer

class Data:
    def __init__(self,typ):
        if typ not in ['gps','insar','tilt']:
            raise Exception('The data type is not supported')
        else:
            self.type = typ
        if typ=='gps':
            self.refidx=None
            self.data = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','ux','uy','uz','sx','sy','sz','t'])
        elif typ=='insar':
            self.refidx=None
            self.data = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','los','az','lk','t'])
        elif typ=='tilt':
            self.refidx=None
            self.data = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','dux','duy','azx','t'])

    #this is useful if you've got a few GNSS stations with offsets
    def add(self, id, lat, lon, height, x, y, ux, uy, uz, sx, sy, sz):
        self.data.loc[len(self.data.index)] = [id] + list((lat,lon,height,x,y,ux,uy,uz,sx,sy,sz))

    def add_angles(self,az,lk):
        if isinstance(az,float):
            az=self.data['x'].to_numpy()*0+az
            lk=self.data['x'].to_numpy()*0+lk
        self.data['az'] = pd.Series(az)
        self.data['lk'] = pd.Series(lk)
        
    def add_los(self,los):
        self.data['los']=pd.Series(los)
    
    def add_tilt(self,tiltx,tilty):
        self.data['dux']=pd.Series(tiltx)
        self.data['duy']=pd.Series(tilty)
    
    def add_disp_time(self,x,y,ux,uy,uz,t):
        self.data.loc[len(self.data.index)] =list((np.nan,np.nan,np.nan,np.nan,x,y,ux,uy,uz,np.nan,np.nan,np.nan,t))
    
    def add_los_time(self,x,y,ux,az,lk,t):
        self.data.loc[len(self.data.index)] =list((np.nan,np.nan,np.nan,np.nan,x,y,los,az,lk,t))
    
    def add_tilt_time(self,x,y,ux,az,lk,t):
        self.data.loc[len(self.data.index)] =list((np.nan,np.nan,np.nan,np.nan,x,y,dux,duy,azx,t))
    
    def add_azimuthx(self,az):
        if isinstance(az,float):
            az=self.data['x'].to_numpy()*0+az
        self.data['azx'] = pd.Series(az)
        print(self.data['azx'])
        
    #useful if all that's to be done is run a forward model
    def add_locs(self, x, y, unit='m'):
        if unit=='m':
            self.data['x'] = pd.Series(x)
            self.data['y'] = pd.Series(y)
        elif unit=='deg':
            self.data['lon']=pd.Series(x)
            self.data['lat']=pd.Series(y)
            mlon=np.mean(x)
            mlat=np.mean(y)
            #self.data['x']=(x-mlon)*np.cos(np.radians(self.data['lat'].to_numpy()))*111e3
            self.data['x']=(x-mlon)*111e3
            self.data['y']=(y-mlat)*111e3
    
    def add_disp(self,ux,uy,uz):
        if self.type=='gps':
            self.data['ux'] = pd.Series(ux)
            self.data['uy'] = pd.Series(uy)
            self.data['uz'] = pd.Series(uz)
        elif self.type=='insar':
            self.data['los']=pd.Series(ux)
            self.data['az']=pd.Series(uy)
            self.data['lk']=pd.Series(uz)
        elif self.type=='tilt':
            self.data['dux']=pd.Series(ux)
            self.data['duy']=pd.Series(uy)
            self.data['azx']=pd.Series(uz)
    
    def set_refidx(self,idx):
        self.refidx=idx
        
    def get_refidx(self):
        return self.refidx
    
    def get_reduced_obs(self):
        rux=np.copy(self.data['ux'])-self.data['ux'][self.refidx]
        ruy=np.copy(self.data['uy'])-self.data['uy'][self.refidx]
        ruz=np.copy(self.data['uz'])-self.data['uz'][self.refidx]
      
        return rux,ruy,ruz
        
    def get_xs(self):
        return self.data['x'].to_numpy()

    def get_ys(self):
        return self.data['y'].to_numpy()
    
    def get_ts(self):
        return self.data['t'].to_numpy()

    def get_zs(self):
        return self.data['y'].to_numpy()*0.0

    def get_site_ids(self):
        return self.data['id'].to_numpy()

    def get_lats(self):
        return self.data['lat'].to_numpy()

    def get_lons(self):
        return self.data['lon'].to_numpy()

    def get_obs(self):
        ''' returns single vector with [ux1...uxN,uy1...uyN,uz1,...,uzN] as elements'''
        if self.type=='gps':
            if not self.refidx==None:
                print('Reduced')
                rux=np.copy(self.data['ux'])-self.data['ux'][self.refidx]
                ruy=np.copy(self.data['uy'])-self.data['uy'][self.refidx]
                ruz=np.copy(self.data['uz'])-self.data['uz'][self.refidx]
                return np.concatenate((rux,ruy,ruz)).ravel()
            else:
                print('Not reduced')
                return self.data[['ux','uy','uz']].to_numpy().flatten(order='F')
        elif self.type=='insar':
            return self.data[['los']].to_numpy().flatten(order='F')
        elif self.type=='tilt':
            rot=self.data['azx'].to_numpy()-np.pi/2
            dux=self.data['dux'].to_numpy()*np.cos(rot)-self.data['duy'].to_numpy()*np.sin(rot)
            duy=self.data['dux'].to_numpy()*np.sin(rot)+self.data['duy'].to_numpy()*np.cos(rot)
            return np.concatenate((dux,duy)).ravel()*1e6
'''
    def set_projection(self):
        lons=self.get_lons()
        lats=self.get_lats()
        xs=[]
        ys=[]
        crs = CRS.from_epsg(4326)
        crs1 = CRS.from_proj4("+proj=lcca +lat_0="+str(int(np.mean(lats)))+" +lon_0="+str(int(np.mean(lons))))
        transformer = Transformer.from_crs(crs, crs1, always_xy=True)
        for i in range(len(lons)):
            x,y=transformer.transform(lons[i],lats[i])
            xs.append(x)
            ys.append(y)
        self.add_locs(xs,ys)
'''
#class GNSS(Data):

#class LOS(Data):

#class Tilt(Data):
    

