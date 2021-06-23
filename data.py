"""
Functions for forward volcano-geodesy analytic models

Author: Ronni Grapenthin
Date: 6/23/2021


TODO:
- think about how to integrate (In)SAR LOS
- integrate tilt
"""

import pandas as pd

class Data:
    def __init__(self):
        self.data = pd.DataFrame(columns=['id', 'lat', 'lon', 'height', 'x','y','ux','uy','uz','sx','sy','sz'])

    def add(self, id, lat, lon, height, x, y, ux, uy, uz, sx, sy, sz):
        self.data.loc[len(self.data.index)] = [id] + list((lat,lon,height,x,y,ux,uy,uz,sx,sy,sz))

    def get_xs(self):
        return self.data['x'].to_numpy()

    def get_ys(self):
        return self.data['y'].to_numpy()

    def get_zs(self):
        return self.data['y'].to_numpy()*0.0

    def get_obs(self):
        ''' returns single vector with [ux1...uxN,uy1...uyN,uz1,...,uzN] as elements'''
        return self.data[['ux','uy','uz']].to_numpy().flatten(order='F')

#class GNSS(Data):

#class LOS(Data):

#class Tilt(Data):
    

