# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:22:54 2021

@author: elfer
"""

import mogi
import numpy as np
import matplotlib.pyplot as plt

#Test for mogi.py replicating Figure 7.6b of Earthquake and volcano deformation (Segall, 2010)

x=np.linspace(0,3000/np.sqrt(2),10000)
y=x

mu=9e9
dP=0.001273*mu
d=1000


############# MOGI SOLUTION TEST ################################################
ux,uy,uz=mogi.calc_mctigue(x,y,xcen=0,ycen=0,d=1000,dP=dP,a=500,nu=0.25,mu=mu,terms=1)

uz=uz*mu/(d*dP)

ur=np.sqrt(np.power(ux,2)+np.power(uy,2))

ur=ur*mu/(d*dP)


x=x/d
y=x

r=np.sqrt(np.power(x,2)+np.power(y,2))


plt.figure()
plt.plot(r,ur,c='b',label='Mogi')
plt.plot(r,uz,c='b')
##################################################################################

x=np.linspace(0,3000/np.sqrt(2),10000)
y=x


############# MCTIGUE SOLUTION TEST ################################################
ux,uy,uz=mogi.calc_mctigue(x,y,xcen=0,ycen=0,d=1000,dP=dP,a=500,nu=0.25,mu=mu,terms=2)


x=x/d
y=x

r=np.sqrt(np.power(x,2)+np.power(y,2))

uz=uz*mu/(d*dP)

ur=np.sqrt(np.power(ux,2)+np.power(uy,2))

ur=ur*mu/(d*dP)

plt.plot(r,ur,c='r',label='McTigue')
plt.plot(r,uz,c='r')

plt.ylabel('Displacement')
plt.xlabel('Distance (r/d)')

plt.legend()
###################################################################################

plt.show()
