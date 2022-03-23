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
        if typ not in ['gps','insar','tilt','joint']:
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
        elif typ=='joint':
            self.refidx=None
            #self.data=dict()
            data_gps = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','ux','uy','uz','sx','sy','sz','t'])
            data_insar = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','los','az','lk','t'])
            data_tilt = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','dux','duy','azx','t'])
            #self.data['gps']=data_gps
            #self.data['insar']=data_insar
            #self.data['tilt']=data_tilt
            self.gps=data_gps
            self.insar=data_insar
            self.tilt=data_tilt

    #this is useful if you've got a few GNSS stations with offsets
    def add(self, id, lat, lon, height, x, y, ux, uy, uz, sx, sy, sz):
        data=self.get_data('gps')
        data.loc[len(self.data.index)] = [id] + list((lat,lon,height,x,y,ux,uy,uz,sx,sy,sz))

    def get_data(self,typ):
        if not self.type=='joint':
            data=self.data
        elif typ=='gps':
            data=self.gps
        elif typ=='insar':
            data=self.insar
        elif typ=='tilt':
            data=self.tilt
        return data
        
    def add_angles(self,az,lk):
        data=self.get_data('insar')
        if isinstance(az,float):
            az=data['x'].to_numpy()*0+az
            lk=data['x'].to_numpy()*0+lk
        data['az'] = pd.Series(az)
        data['lk'] = pd.Series(lk)
        
    def add_los(self,los):
        data=self.get_data('insar')
        data['los']=pd.Series(los)
    
    def add_tilt(self,tiltx,tilty):
        data=self.get_data('tilt')
        data['dux']=pd.Series(tiltx)
        data['duy']=pd.Series(tilty)
    
    def add_disp_time(self,x,y,ux,uy,uz,t):
        data=self.get_data('gps')
        data.loc[len(self.data.index)] =list((np.nan,np.nan,np.nan,np.nan,x,y,ux,uy,uz,np.nan,np.nan,np.nan,t))
    
    def add_los_time(self,x,y,ux,az,lk,t):
        data=self.get_data('insar')
        data.loc[len(self.data.index)] =list((np.nan,np.nan,np.nan,np.nan,x,y,los,az,lk,t))
    
    def add_tilt_time(self,x,y,ux,az,lk,t):
        data=self.get_data('tilt')
        data.loc[len(self.data.index)] =list((np.nan,np.nan,np.nan,np.nan,x,y,dux,duy,azx,t))
    
    def add_azimuthx(self,az):
        data=self.get_data('tilt')
        if isinstance(az,float):
            az=self.data['x'].to_numpy()*0+az
        self.data['azx'] = pd.Series(az)
        print(self.data['azx'])
        
    #useful if all that's to be done is run a forward model
    def add_locs(self, x, y, unit='m',dataset=None):
        data=self.get_data(dataset)
        if unit=='m':
            data['x'] = pd.Series(x)
            data['y'] = pd.Series(y)
        elif unit=='deg':
            data['lon']=pd.Series(x)
            data['lat']=pd.Series(y)
            mlon=np.mean(x)
            mlat=np.mean(y)
            #self.data['x']=(x-mlon)*np.cos(np.radians(self.data['lat'].to_numpy()))*111e3
            data['x']=(x-mlon)*111e3
            data['y']=(y-mlat)*111e3
    
    def add_disp(self,ux,uy,uz,dataset=None):
        data=self.get_data(dataset)
        if self.type=='gps' or dataset=='gps':
            data['ux'] = pd.Series(ux)
            data['uy'] = pd.Series(uy)
            data['uz'] = pd.Series(uz)
        elif self.type=='insar' or dataset=='insar':
            data['los']=pd.Series(ux)
            data['az']=pd.Series(uy)
            data['lk']=pd.Series(uz)
        elif self.type=='tilt' or dataset=='tilt':
            data['dux']=pd.Series(ux)
            data['duy']=pd.Series(uy)
            data['azx']=pd.Series(uz)
    
    def set_refidx(self,idx):
        self.refidx=idx
        
    def get_refidx(self):
        return self.refidx
    
    def get_reduced_obs(self):
        data=self.get_data('gps')
        rux=np.copy(data['ux'])-data['ux'][self.refidx]
        ruy=np.copy(data['uy'])-data['uy'][self.refidx]
        ruz=np.copy(data['uz'])-data['uz'][self.refidx]
      
        return rux,ruy,ruz
        
    def get_xs(self,dataset=None):
        data=self.get_data(dataset)
        return data['x'].to_numpy()

    def get_ys(self,dataset=None):
        data=self.get_data(dataset)
        return data['y'].to_numpy()
    
    def get_ts(self,dataset=None):
        data=self.get_data(dataset)
        return data['t'].to_numpy()

    def get_zs(self,dataset=None):
        data=self.get_data(dataset)
        return data['y'].to_numpy()*0.0

    def get_site_ids(self,dataset=None):
        data=self.get_data(dataset)
        return data['id'].to_numpy()

    def get_lats(self,dataset=None):
        data=self.get_data(dataset)
        return data['lat'].to_numpy()

    def get_lons(self,dataset=None):
        data=self.get_data(dataset)
        return data['lon'].to_numpy()

    def get_obs(self):
        ''' returns single vector with [ux1...uxN,uy1...uyN,uz1,...,uzN] as elements'''
        if self.type=='gps' or self.type=='joint':
            data=self.get_data('gps')
            if not self.refidx==None:
                print('Reduced')
                rux=np.copy(data['ux'])-data['ux'][self.refidx]
                ruy=np.copy(data['uy'])-data['uy'][self.refidx]
                ruz=np.copy(data['uz'])-data['uz'][self.refidx]
                gps=np.concatenate((rux,ruy,ruz)).ravel()
            else:
                print('Not reduced')
                gps=data[['ux','uy','uz']].to_numpy().flatten(order='F')
            if self.type=='gps':
                return gps
        if self.type=='insar' or self.type=='joint':
            data=self.get_data('insar')
            insar=data[['los']].to_numpy().flatten(order='F')
            if self.type=='insar':
                return insar
        if self.type=='tilt' or self.type=='joint':
            data=self.get_data('tilt')
            rot=data['azx'].to_numpy()-np.pi/2
            dux=data['dux'].to_numpy()*np.cos(rot)-data['duy'].to_numpy()*np.sin(rot)
            duy=data['dux'].to_numpy()*np.sin(rot)+data['duy'].to_numpy()*np.cos(rot)
            tilt=np.concatenate((dux,duy)).ravel()*1e6
            if self.type=='tilt':
                return tilt
        if self.type=='joint':
            data=np.concatenate((gps,insar,tilt))
            return data
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
    

