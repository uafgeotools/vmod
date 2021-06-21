#!/usr/bin/env python

"""
Mogi inversion for Mogi model using Markov Chain Monte-Carlo
Author: Mario Angarita
Date: 03/25/2021
"""

import numpy as np
import pymc3 as pm
import mogi
import corner
import utm

#timeseries analysis directory
#ts_dir = "/gps/standard-solutions/erebus"
ts_dir = "/gps/standard-solutions/erebus_mogi_test"
#ts_dir = "/gps/standard-solutions/erebus_2021"

#erebus summit 77.53°S, 167.17°E https://volcano.si.edu/volcano.cfm?vn=390020
erebus = utm.from_latlon(-77.53, 167.17)

print(erebus[0], erebus[1])

#stations of interest
sites = ("ABBZ", "HOOZ", "CONG", "E1G2", "NAUS", "MACG")
#sites = ("PHIG", "CONG", "NAUS")

data  = {}

x = np.array([])
y = np.array([])

ux = np.array([])
uy = np.array([])
uz = np.array([])

sx = np.array([])
sy = np.array([])
sz = np.array([])

#create data arrays from CATS estimates on disk
for s in sites:
    d_hori = np.genfromtxt( ts_dir+"/"+s.upper()+"_igs14.series.ANTA.cats_out.hori.gmtvec", dtype=None, encoding=None, names=['lon', 'lat', 'east_vel', 'north_vel', 'east_sig', 'north_sig', 'east_inter', 'north_inter'])
    d_vert = np.genfromtxt( ts_dir+"/"+s.upper()+"_igs14.series.ANTA.cats_out.up.gmtvec", dtype=None, encoding=None, names=['lon', 'lat', 'east_vel', 'north_vel', 'east_sig', 'north_sig', 'east_inter', 'north_inter'])
    s_utm  = utm.from_latlon(d_hori['lat'], d_hori['lon'])
    #find station x and y locations, append to vector
    x = np.append(x, s_utm[0]-erebus[0])
    y = np.append(y, s_utm[1]-erebus[1])
    #displacements / velocities convert to meteres
    ux=np.append(ux, d_hori['east_vel']/1000.)
    uy=np.append(uy, d_hori['north_vel']/1000.)
    uz=np.append(uz, d_vert['north_vel']/1000.)
    #data uncertainties, convert to meters
    sx=np.append(sx, d_hori['east_sig']/1000.)
    sy=np.append(sy, d_hori['north_sig']/1000.)
    sz=np.append(sz, d_vert['north_sig']/1000.)

#single observation, uncertainty vectors
obs=np.copy(np.concatenate((ux,uy,uz)))
sigmas=np.copy(np.concatenate((sx,sy,sz)))

print(x)
print(y)
print(obs)
print(sigmas)

######Forward Model function##################
def forward(model_pars):
    
    #Scaling parameters
    xoff1,yoff1,depth,dV=model_pars
    xoff1=xoff1
    yoff1=yoff1
    depth1=depth
    dV=dV
    
    ux,uy,uz=mogi.forward(x,y,xcen=xoff1,ycen=yoff1,d=depth1,dV=dV, nu=0.25)
    
    rpta=np.concatenate((np.ravel(ux),np.ravel(uy),np.ravel(uz)))
    
    return rpta


######MCMC inversion#########################
with pm.Model() as model:
    xcen=pm.Uniform('xcen',lower=-10000,upper=10000)
    ycen=pm.Uniform('ycen',lower=-10000,upper=10000)
    depth=pm.Uniform('depth',lower=0, upper=10000)
    dV=pm.Uniform('dV',lower=-4e7,upper=4e7)
    model_pars=[xcen,ycen,depth,dV]
    likelihood = pm.Normal('likelihood', mu=mogi.forward(x,y,xcen=xcen,ycen=ycen,d=depth,dV=dV, nu=0.25), sigma=1.0, value=obs, observed=True)
#    likelihood = pm.Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)
    start = pm.find_MAP()
    step = pm.AdaptiveMetropolis()
    trace = pm.sample(200000, step, start=start, progressbar=True)

Listing the parameters output
xc_values=MDL.trace('xcen')[:]
yc_values=MDL.trace('ycen')[:]
d_values=MDL.trace('depth')[:]
dv_values=MDL.trace('dV')[:]

#calculate forward model for mean values
x_mogi = np.mean(xc_values)
y_mogi = np.mean(yc_values)
d_mogi = np.mean(d_values)
dv_mogi = np.mean(dv_values)

ux_mod,uy_mod,uz_mod=mogi.forward(x,y,xcen=x_mogi,ycen=y_mogi,d=d_mogi,dV=dv_mogi, nu=0.25)

print("UX:")
print(ux)
print(ux_mod)
print(ux-ux_mod)
print("-----------")
print("UY:")
print(uy)
print(uy_mod)
print(uy-uy_mod)
print("-----------")
print("UZ:")
print(uz)
print(uz_mod)
print(uz-uz_mod)


#Plot histograms
samples=np.vstack([xc_values,yc_values,d_values,dv_values])
pymc.Matplot.plot(MDL)
#corner.corner(samples.T,labels=['xcen','ycen','d','dV'],quantiles=[0.26, 0.5, 0.74],show_titles=True,verbose=True)
figure = corner.corner(samples.T,labels=['xcen','ycen','d','dV'],show_titles=True,verbose=True)
figure.savefig('corner.png')



