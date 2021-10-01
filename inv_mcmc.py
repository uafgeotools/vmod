# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:20:56 2021

@author: elfer
"""

import pymc
import numpy as np
import matplotlib.pyplot as plt
import sys
def inversion(source,data):
    parnames=source.get_parnames()
    def model(data):
        #Distribution for every parameter
        theta=[]
        orders=[]
        for i in range(source.get_num_params()):
            order=int(np.log10(np.max([np.abs(source.low_bounds[i]),np.abs(source.high_bounds[i])])))-1
            low=source.low_bounds[i]/(10**order)
            high=source.high_bounds[i]/(10**order)
            ini=source.x0[i]/(10**order)
            #print(low,high,ini,order)
            thetat=pymc.Uniform(parnames[i],low,high,value=ini)
            theta.append(thetat)
        model_pars=theta
        sigma=pymc.Uniform('sigma',0,1,value=0.5)
        #sigma=100
        #Deformation is the deterministic variable
        @pymc.deterministic(plot=False)
        def defo(model_pars=model_pars):
            return source.get_model_reduced(model_pars)

        #Probability distribution
        z = pymc.Normal('z', mu=defo, tau=1.0/sigma, value=data, observed=True)
        return locals()

    #Making MCMC model
    MDL = pymc.MCMC(model(data))

    #Choosing Metropolis-Hastings for step method
    #MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.model_pars,scales={MDL.model_pars[0]:1000,MDL.model_pars[1]:1000})
    MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.model_pars)
    #MDL.use_step_method(pymc.Metropolis,MDL.model_pars)
    if source.get_source_id()=='Mogi':
        steps=110000
        burnin=10000
        thin=100
    elif source.get_source_id()=='Okada':
        steps=1100000
        burnin=10000
        thin=1000
    elif source.get_source_id()=='Penny':
        steps=41000
        burnin=1000
        thin=100
    elif source.get_source_id()=='Yang':
        steps=110000
        burnin=10000
        thin=1000
    #Number of runs
    MDL.sample(steps,burnin,thin) 
    traces=[]
    for i in range(source.get_num_params()):
        traces.append(MDL.trace(parnames[i]))
    #traces.append(MDL.trace('sigma'))
    traces=np.array(traces)
    return traces,MDL

