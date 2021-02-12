import yang
import numpy as np
import matplotlib.pyplot as plt


# Trying to reporduce fig 7.16 from Earthquake and volcano deformation (Segall, 2010)

# def forward(x,y,xcen=0,ycen=0,z0=5e3,P=10,a=2,b=1,phi=0,theta=0,mu=1.0,nu=0.25):

xcen = 0
ycen = 0
a = 1000
theta = 90
mu = 9e9
dP = 0.001273*mu

b = a / 2
z0 = a / 0.3
d = z0


x = np.linspace(0, 5*(z0), 100000)
y = x
r = np.sqrt(np.power(x,2), np.power(y,2))

U12, U22, U32 = yang.forward(x, y, a=a, b=b, z0=z0, theta=theta, mu=mu, P=dP)
uz2 = U32
zmax2 = np.max(uz2)

ur2 = np.sqrt(np.power(U12, 2), np.power(U22, 2))

yticks = np.arange(0,1.2,0.2)
xticks = np.arange(0, 5.5, .5)
b = a / 5

U15, U25, U35 = yang.forward(x, y, a=a, b=b, z0=z0, theta=theta, mu=mu, P=dP)
uz5 = U35
zmax5 = np.max(uz5)
ur5 = np.sqrt(np.power(U15, 2), np.power(U25, 2))




plt.figure()
ax = plt.subplot(2,1,1)
plt.plot(r/d, uz2/zmax2, label='Aspect Ratio = 2')
plt.plot(r/d, uz5/zmax5, label='Aspect Ratio = 5')
ax.set_xlim(0,5)
ax.set_ylim(0,1.2)
ax.grid(True)
ax.set_yticks(yticks)
ax.set_xticks(xticks)
plt.legend()

yticks = np.arange(0,1.2,0.2)
ax = plt.subplot(2,1,2)
plt.plot(r/d, ur2/zmax2)
plt.plot(r/d, ur5/zmax5)
ax.set_xlim(0,5)
ax.set_ylim(0,1.0)
ax.grid(True)
ax.set_yticks(yticks)
ax.set_xticks(xticks)

# plt.show()
plt.savefig("test_yang.png")