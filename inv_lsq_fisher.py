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
from visco_shell import ViscoShell
from penny import Penny

#timeseries analysis directory
ts_dir = "/gps/standard-solutions/timeseries"

#54.65°N 164.43°W
volcano = utm.from_latlon(54.65, -164.43)

print(volcano)

print(volcano[0], volcano[1])

#stations of interest
sites = ("FC01", "FC02", "FC03", "FC04", "FC05")

data = Data()

#create data arrays from CATS estimates on disk
for s in sites:
    d_hori = np.genfromtxt( ts_dir+"/"+s.upper()+"_igs14.series.NOAM.cats_out.hori.gmtvec", dtype=None, encoding=None, names=['lon', 'lat', 'east_vel', 'north_vel', 'east_sig', 'north_sig', 'east_inter', 'north_inter'])
    d_vert = np.genfromtxt( ts_dir+"/"+s.upper()+"_igs14.series.NOAM.cats_out.up.gmtvec", dtype=None, encoding=None, names=['lon', 'lat', 'east_vel', 'north_vel', 'east_sig', 'north_sig', 'east_inter', 'north_inter'])
    s_utm  = utm.from_latlon(d_hori['lat'], d_hori['lon'])
    #find station x and y locations, append to vector
    x = s_utm[0]-volcano[0]
    y = s_utm[1]-volcano[1]
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

yang = Yang(data)

a = 100
b = 10
V = 4/3 * np.pi * a * b**2
mu=26.6E9
nu=0.25
delta_V = 1e6
dP = ( (delta_V/V) * mu ) / ( ((b/a)**2)/3 - 0.7*(b/a) + 1.37 )

yang.set_x0(np.array([0, 0, 1000, a, b/a, dP/mu, 1, 1]))
yang.set_bounds(low_bounds  = [-10000, -10000, 0,     0,     0, -1e9,  0, 0], 
                high_bounds = [ 10000,  10000, 40000, 20000, 1,  1e9, 90, 360])


vshell = ViscoShell(data)
vshell.set_x0(np.array([2, 0, 0, 500, 200, 400, -100e5, 30e9, 2e16]))
vshell.set_bounds(low_bounds = [0, -1000, -1000, 0, 0, 0, -100e9, 10e8, 2e15], high_bounds = [60*60*24*365*10, 1000, 1000, 40000, 5000, 10000, 100e9, 30e10, 2e18])

penny = Penny(data)
penny.set_x0(np.array([0, 0, 500, 0.01, 1000]))
penny.set_bounds(low_bounds = [-10000, -10000, 0, -10, 0], high_bounds = [10000, 10000, 40000, 100, 10000])

#inv.register_source(yang)
#inv.register_source(yang)
inv.register_source(penny)
#inv.register_source(mogi)
inv.nlsq()

inv.write_forward_gmt(ts_dir+'/fisher_mogi', volcano)
#inv.write_forward_gmt(ts_dir+'/erebus_penny')
inv.print_model()
#print("residual norm: %f" %(yang.res_norm()))
#yang.write_forward_gmt(ts_dir+'/erebus_yang')




