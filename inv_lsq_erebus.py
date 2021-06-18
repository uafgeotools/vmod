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
ts_dir = "/gps/standard-solutions/erebus"
#ts_dir = "/gps/standard-solutions/erebus_mogi_test"
#ts_dir = "/gps/standard-solutions/erebus_2021"

#erebus summit 77.53°S, 167.17°E https://volcano.si.edu/volcano.cfm?vn=390020
erebus = utm.from_latlon(-77.53, 167.17)

print(erebus[0], erebus[1])

#stations of interest
sites = ("ABBZ", "HOOZ", "CONG", "E1G2", "NAUS", "MACG")
#sites = ("PHIG", "CONG", "NAUS")

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

mogi = Mogi(data)
x0   = np.array([-10000, -10000, 40000, 1e9])
mod  = mogi.invert(x0, ([-10000, -10000, 0, -1e9], [10000, 10000, 40000, 1e9]))
print(mod)

mogi.write_forward_gmt('erebus_mogi')

#yang = Yang(x,y,x*0,obs,sigmas)
#x0   = np.array([0, 0, 100, 10, 2, 1, 0, 0])
#mod  = yang.invert(x0)
#print(mod)




