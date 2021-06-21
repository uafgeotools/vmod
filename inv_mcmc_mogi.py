
"""
Mogi inversion for Mogi model using Markov Chain Monte-Carlo
Author: Mario Angarita
Date: 03/25/2021
"""

import numpy as np
import pymc
import mogi
import corner

#####Location data############################
x=np.array([-10e3,-5e3,-5e3,0,0,0,5e3,5e3,10e3]) #meters
y=np.array([0,-5e3,5e3,-10e3,0,10e3,-5e3,5e3,0]) #meters

#####Synthetic data###########################
ux,uy,uz=mogi.forward(x,y,xcen=1e3,ycen=1e3,d=1e3,dV=2e6, nu=0.25) #meters

ux=np.ravel(ux)+np.random.uniform(-0.1,0.1,ux.size)*np.mean(ux)
uy=np.ravel(uy)+np.random.uniform(-0.1,0.1,uy.size)*np.mean(uy)
uz=np.ravel(uz)+np.random.uniform(-0.1,0.1,uz.size)*np.mean(uz)
obs=np.copy(np.concatenate((ux,uy,uz)))

######Uncertainties in data###################
sigmas=np.abs(np.ones(obs.shape)*np.mean(obs)*0.1)


######Forward Model function##################
def forward(model_pars):
    
    #Scaling parameters
    xoff1,yoff1,depth,dV=model_pars
    xoff1=xoff1*1e3
    yoff1=yoff1*1e3
    depth1=depth*1e3
    dV=dV*1e6
    
    ux,uy,uz=mogi.forward(x,y,xcen=xoff1,ycen=yoff1,d=depth1,dV=dV, nu=0.25)
    
    rpta=np.concatenate((np.ravel(ux),np.ravel(uy),np.ravel(uz)))
    
    return rpta

######MCMC inversion#########################
def model(data):
    #Distribution for every parameter
    
    xcen=pymc.Uniform('xcen',0.1,10,value=7.01)

    ycen=pymc.Uniform('ycen',0.1,10,value=7.01)

    depth=pymc.Uniform('depth',0.1,10,value=7.01)

    dV=pymc.Uniform('dV',0,10,value=7.0)
    
    model_pars=[xcen,ycen,depth,dV]

    #Deformation is the deterministic variable
    @pymc.deterministic(plot=False)
    def defo(model_pars=model_pars):
        return forward(model_pars=model_pars)

    #Probability distribution   	
    z = pymc.Normal('z', mu=defo, tau=1.0/sigmas, value=data, observed=True)
    return locals()

#Making MCMC model
MDL = pymc.MCMC(model(obs))

#Choosing Metropolis-Hastings for step method
MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.model_pars,scales={MDL.model_pars[0]:1,MDL.model_pars[1]:1,MDL.model_pars[2]:1,MDL.model_pars[3]:1},delay=30001)

#Number of runs
MDL.sample(200000) 

#Writing the output in a csv file
MDL.write_csv("mapcmc_rpta.csv", variables=["xcen","ycen","depth","dV"])

#Listing the parameters output
xc_values=MDL.trace('xcen')[:]
yc_values=MDL.trace('ycen')[:]
d_values=MDL.trace('depth')[:]
dv_values=MDL.trace('dV')[:]

#Plot histograms
samples=np.vstack([xc_values,yc_values,d_values,dv_values])
pymc.Matplot.plot(MDL)
corner.corner(samples.T,labels=['xcen','ycen','d','dV'],quantiles=[0.26, 0.5, 0.74],show_titles=True,verbose=True)
