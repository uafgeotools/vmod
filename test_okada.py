# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:57:29 2021

Author: Mario Angarita
"""

import okada
import numpy as np
import matplotlib.pyplot as plt

#Test for okada.py replicating Figure 28 of dModels report

xi=4000
xf=25000
yi=4000
yf=20000
zt=1000
zb=10000
dip=60
strike=np.degrees(np.arctan((xf-xi)/(yf-yi)))

length=np.sqrt(np.power(xf-xi,2)+np.power(yf-yi,2))

W=(zb-zt)/np.sin(np.radians(dip))
Wproj=W*np.cos(np.radians(dip))

xceni=xi+(xf-xi)/2
yceni=yi+(yf-yi)/2

xcen=xceni+Wproj*np.cos(np.radians(strike))
ycen=yceni-Wproj*np.sin(np.radians(strike))

depth=(zt+zb)/2

print(np.degrees(strike),xceni,yceni,xcen,ycen,length,W,depth)

x=np.linspace(-50000,50000,10000)
y=x



mu=1e9
s=1

ux,uy,uz=okada.forward(x, y*0, xcen=xcen, ycen=ycen,
            depth=depth, length=length, width=W,
            slip=s, opening=0.0,
            strike=strike, dip=dip, rake=180.0,
            nu=0.25)

x1=np.linspace(-50000,50000,50)
y1=x1
X,Y=np.meshgrid(x1,y1)

UX,UY,UZ=okada.forward(X, Y, xcen=xcen, ycen=ycen,
            depth=depth, length=length, width=W,
            slip=s, opening=0.0,
            strike=strike, dip=dip, rake=180.0,
            nu=0.25)



plt.figure()
plt.plot([xi,xf],[yi,yf],c='red',lw=2)
plt.quiver(X,Y,UX,UY)

plt.figure()
plt.plot(x,ux,c='green',label='ux')
plt.figure()
plt.plot(y,uy,c='blue',label='uy')
plt.figure()
plt.plot(y,uz,c='red',label='uz')



plt.show()
