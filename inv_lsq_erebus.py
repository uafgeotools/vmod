#!/usr/bin/env python

"""
Mogi inversion for Mogi model using Markov Chain Monte-Carlo
Author: Mario Angarita
Date: 03/25/2021
"""

import numpy as np
import utm

from source import Data, Source, Mogi, Yang

#timeseries analysis directory
ts_dir = "/gps/standard-solutions/erebus/erebus"
#ts_dir = "/gps/standard-solutions/erebus/erebus_mogi_test"
ts_dir = "/gps/standard-solutions/erebus/erebus_2021"
#ts_dir = "/gps/standard-solutions/erebus/erebus_yang_test"

#erebus summit 77.53°S, 167.17°E https://volcano.si.edu/volcano.cfm?vn=390020
erebus = utm.from_latlon(-77.53, 167.17)

print(erebus[0], erebus[1])

#stations of interest
#sites = ("ABBZ", "HOOZ", "CONG", "E1G2", "NAUS", "MACG")
sites = ("PHIG", "CON2", "NAU2")

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

#mogi = Mogi(data)
#x0   = np.array([-10000, -10000, 40000, 1e9])
#mod  = mogi.invert(x0, ([-10000, -10000, 200, -1e9], [10000, 10000, 40000, 1e9]))
#print("monopole:", mod.x)
#print("residual norm: %f" %(mogi.res_norm()))
#mogi.write_forward_gmt('erebus_mogi')

#x0   = np.array([-10000, -10000, 1000, 1e9, -10000, -10000, 40000, 1e9])
#mod  = mogi.invert_dipole(x0, ([-10000, -10000, 0, -1e9, -10000, -10000, 0, -1e9], [10000, 10000, 5000, 1e9, 10000, 10000, 40000, 1e9]))
#print("dipole:", mod.x)
#print("residual norm: %f" %(mogi.res_norm()))
#mogi.write_forward_gmt('erebus_2mogi')

yang = Yang(data)

a = 100
b = 10
V = 4/3 * np.pi * a * b**2
mu=26.6E9
nu=0.25
delta_V = 1e6
dP = ( (delta_V/V) * mu ) / ( ((b/a)**2)/3 - 0.7*(b/a) + 1.37 )
ux,uy,uz = yang.forward(-500,500,2000,a,b/a,dP/mu,45,90,mu,nu)
print(ux,uy,uz)

#yang(-500,500,2000,a,b/a,dP/mu,mu,nu,45,90,[-2875.07722612, -2082.40080761,  -526.34373579], [ 606.48515842, -474.04624186,  929.50370699], [0, 0, 0])

x0   = np.array([0, 0, 1000, a, b/a, dP/mu, 1, 1])
print([-10000, -10000, 0, 0, 0, -1e9, 0, 0])
print(x0)
print([10000, 10000, 40000, 20000, 1, 1e9, 90, 360])
mod  = yang.invert(x0, ([-10000, -10000, 0, 0, 0, -1e9, 0, 0], [10000, 10000, 40000, 20000, 1, 1e9, 90, 360]))
print("Yang:",mod.x)
print("residual norm: %f" %(yang.res_norm()))
yang.write_forward_gmt('erebus_yang')




