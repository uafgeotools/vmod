#!/usr/bin/env python

"""
Mogi inversion for Mogi model using Markov Chain Monte-Carlo
Author: Mario Angarita
Date: 03/25/2021
"""

import numpy as np
import utm

from source import Source
from data import Data
from inverse import Inverse
from mogi import Mogi
from yang import Yang


#timeseries analysis directory
ts_dir = "/gps/standard-solutions/erebus/2004_2011"
#ts_dir = "/gps/standard-solutions/erebus/erebus_mogi_test"
#ts_dir = "/gps/standard-solutions/erebus/2020_2021"
#ts_dir = "/gps/standard-solutions/erebus/erebus_yang_test"

#erebus summit 77.53°S, 167.17°E https://volcano.si.edu/volcano.cfm?vn=390020
erebus = utm.from_latlon(-77.53, 167.17)

print(erebus[0], erebus[1])

#stations of interest
sites = ("ABBZ", "HOOZ", "CONG", "E1G2", "NAUS", "MACG")
#sites = ("CONG", "E1G2", "NAUS", "MACG")
#sites = ("PHIG", "CON2", "NAU2", "HOG2")

data = Data()

#create data arrays from CATS estimates on disk
for s in sites:
    d_hori = np.genfromtxt( ts_dir+"/"+s.upper()+"_igs14.series.ANTA.cats_out.hori.gmtvec", dtype=None, encoding=None, names=['lon', 'lat', 'east_vel', 'north_vel', 'east_sig', 'north_sig', 'east_inter', 'north_inter'])
    d_vert = np.genfromtxt( ts_dir+"/"+s.upper()+"_igs14.series.ANTA.cats_out.up.gmtvec", dtype=None, encoding=None, names=['lon', 'lat', 'east_vel', 'north_vel', 'east_sig', 'north_sig', 'east_inter', 'north_inter'])
    s_utm  = utm.from_latlon(d_hori['lat'], d_hori['lon'])
    #find station x and y locations, append to vector
    x = s_utm[0]-erebus[0]
    y = s_utm[1]-erebus[1]
    #displacements / velocities convert to meteres
    ux=d_hori['east_vel']/1000.
    uy=d_hori['north_vel']/1000.
    uz=d_vert['north_vel']/1000.
    #data uncertainties, convert to meters
    sx=d_hori['east_sig']/1000.
    sy=d_hori['north_sig']/1000.
    sz=d_vert['north_sig']/1000.
    data.add(s.lower(), d_hori['lat'], d_hori['lon'], 0, x, y, ux, uy, uz, sx, sy, sz)

print(data.data)

inv = Inverse(data)

mogi = Mogi(data)
mogi.set_x0(np.array([-10000, -10000, 40000, 1e9]))
mogi.set_bounds(low_bounds = [-10000, -10000, 200, -1e9], high_bounds = [10000, 10000, 40000, 1e9])

mogi2 = Mogi(data)
mogi2.set_x0(np.array([-10000, -10000, 40000, 1e9]))
mogi2.set_bounds(low_bounds = [-10000, -10000, 200, -1e9], high_bounds = [10000, 10000, 40000, 1e9])

inv.register_source(mogi)
#inv.register_source(mogi2)
inv.nlsq()

inv.write_forward_gmt(ts_dir+'/erebus_mogi')

yang = Yang(data)

a = 100
b = 10
V = 4/3 * np.pi * a * b**2
mu=26.6E9
nu=0.25
delta_V = 1e6
dP = ( (delta_V/V) * mu ) / ( ((b/a)**2)/3 - 0.7*(b/a) + 1.37 )

#yang(-500,500,2000,a,b/a,dP/mu,mu,nu,45,90,[-2875.07722612, -2082.40080761,  -526.34373579], [ 606.48515842, -474.04624186,  929.50370699], [0, 0, 0])

yang.set_x0(np.array([0, 0, 1000, a, b/a, dP/mu, 1, 1]))
yang.set_bounds(low_bounds  = [-10000, -10000, 0,     0,     0, -1e9,  0, 0], 
                high_bounds = [ 10000,  10000, 40000, 20000, 1,  1e9, 90, 360])

inv.register_source(yang)
inv.nlsq()
#inv.write_forward_gmt(ts_dir+'/erebus_mogi_yang')

inv.print_model()
#print("residual norm: %f" %(yang.res_norm()))
#yang.write_forward_gmt(ts_dir+'/erebus_yang')




