# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:20:56 2021

@author: elfer
"""

import pymc
import numpy as np
import matplotlib.pyplot as plt
import sys
def inversion(source,data,wts=1.0):
    parnames=source.get_parnames()
    #print(data)
    def model(data):
        #Distribution for every parameter
        theta=[]
        orders=[]
        for i in range(source.get_num_params()):
            order=int(np.log10(np.max([np.abs(source.low_bounds[i]),np.abs(source.high_bounds[i])])))-1
            if not parnames[i]=='pressure':
                low=source.low_bounds[i]/(10**order)
                high=source.high_bounds[i]/(10**order)
                ini=source.x0[i]/(10**order)
            else:
                low=np.log10(source.low_bounds[i])
                high=np.log10(source.high_bounds[i])
                ini=np.log10(source.x0[i])
            thetat=pymc.Uniform(parnames[i],low,high,value=ini)
            theta.append(thetat)
        model_pars=theta
        sigma=pymc.Uniform('sigma',0,1,value=0.5)
        
        #Deformation is the deterministic variable
        @pymc.deterministic(plot=False)
        def defo(model_pars=model_pars):
            return source.get_model(model_pars)

        #Probability distribution
        z = pymc.Normal('z', mu=defo, tau=1.0*wts/sigma, value=data, observed=True)
        return locals()

    #Making MCMC model
    MDL = pymc.MCMC(model(data))

    #Choosing Adaptive Metropolis-Hastings for step method
    MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.model_pars)
    
    #Steps, burnin and thining for each model
    if source.get_source_id()=='Mogi':
        steps=1100000
        burnin=10000
        thin=1000
    elif source.get_source_id()=='Mctigue':
        steps=1100000
        burnin=10000
        thin=1000
    elif source.get_source_id()=='Nishimura':
        steps=11000
        burnin=1000
        thin=10
    elif source.get_source_id()=='Okada':
        steps=1100000
        burnin=10000
        thin=1000
    elif source.get_source_id()=='Penny':
        steps=110000
        burnin=10000
        thin=100
    elif source.get_source_id()=='Yang':
        steps=1100000
        burnin=100000
        thin=1000
        
    #Number of runs
    MDL.sample(steps,burnin,thin) 
    
    #Getting the traces
    traces=[]
    for i in range(source.get_num_params()):
        traces.append(MDL.trace(parnames[i]))
    traces=np.array(traces)
    return traces,MDL

