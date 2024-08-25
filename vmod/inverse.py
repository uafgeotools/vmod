import copy
import subprocess
import numpy as np
import sys
import pickle
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
import random
import string
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"

class Inverse:
    """
    Class to implement inverse methods operating on one or more analytical volcanic source models
    
    Attributes:
        sources (array): sources that will be part of a model
        obs (Data): data object containing one or more datasets
        minresidual (float): minimum residual achieved with any inversion algorithm
        minparms (array): parameters corresponding to the minimum residual
    """
    def __init__(self, obs):
        self.sources = []       #simple list that will contain instances of class Source
        self.obs     = obs      #instance of Data
        #self.model   = None
        #self.iter=0
        self.steps=None
        self.burnin=None
        self.thin=None
        self.minresidual=None
        self.minparms=None
    
    #add a new source to the geometry
    def register_source(self, source):
        """
        Adds a source to the model
        
        Parameters:
            source (Source): source object to be added to the model
        """
        self.sources.append(source)

    #interface to scipy bounded nonlinear least squares implementation
    def nlsq(self):
        """
        Non-linear least squares approach using the trust reflective algorithm
        
        Returns:
            params (array): parameters for the model with the minimum residual
        """
        self.minresidual=1e6
        if len(self.sources)==0:
            raise Exception('You need to include at least one source')

        params = copy.deepcopy(least_squares(self.residual, self.get_x0(), bounds=self.get_bounds(), jac="3-point"))

        print(self.minresidual)
        return params
    
    def bh(self):
        """
        Optimization using basin-hopping algorithm from scipy
        
        Returns:
            params (array): parameters for the model with the minimum residual
        """
        self.minresidual=1e6
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=self.get_bounds_de())
        if len(self.sources)==0:
            raise Exception('You need to include at least one source')
        
        params = copy.deepcopy(basinhopping(self.residual_bh, self.get_x0(),niter=100,minimizer_kwargs=minimizer_kwargs))
        print(self.minresidual)
        return params
    
    def de(self):
        """
        Optimization using differential evolution algorithm from scipy
        
        Returns:
            params (array): parameters for the model with the minimum residual
        """
        self.minresidual=1e6
        if len(self.sources)==0:
            raise Exception('You need to include at least one source')
        
        params = copy.deepcopy(differential_evolution(self.residual_bh, bounds=self.get_bounds_de()))
        print(self.minresidual)
        return params
    
    def shg(self):
        """
        Optimization using simplicial homology global algorithm from scipy
        
        Returns:
            params (array): parameters for the model with the minimum residual
        """
        self.minresidual=1e6
        if len(self.sources)==0:
            raise Exception('You need to include at least one source')
        start=time.time()
        params = copy.deepcopy(shgo(self.residual_bh, bounds=self.get_bounds_de()))
        end=time.time()
        print('Time:',end-start)
        print(self.minresidual)
        return params
    
    def log_prior(self,theta):
        """
        Logarithmic uniform prior probabilities for the model parameters
        
        Parameters:
            theta (array): parameter values
            
        Returns:
            log_prior (float): logarithm of the prior probability 
        """
        j=0
        if not -10<theta[-1]<10:
            return -np.inf
        
        for k,source in enumerate(self.sources):
            parnames=source.get_parnames()
            for i in range(source.get_num_params()):
                low,high,ini=self.par2log(source,i)
                if not low<theta[j]<high:
                    return -np.inf
                j+=1
        
        return 0.0

    def log_likelihood(self, theta):
        """
        Calculate the log likelihood with a normal distribution.

        Parameters:
            theta (array): parameter values

        Returns:
            likeli (float): logarithm of the likelihood 
        """
        model = self.get_model(theta[0:-1])
        data = self.obs.get_data()
        diff = data - model
        std_devs = self.obs.get_errors()/(10**theta[-1])
        log_std_devs = np.log(std_devs)

        likeli = -0.5 * (len(model) * np.log(2 * np.pi) + np.sum(2 * log_std_devs) + np.sum(((diff/std_devs) ** 2)))
        
        if np.isnan(likeli):
            return -np.inf
        
        return likeli
    
    def log_probability(self,theta):
        """
        Logarithm of the probability for a set of parameters
        
        Parameters:
            theta (array): parameter values
            
        Returns:
            log_prob (float): logarithm of the probability for the set of parameters 
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        likeli=self.log_likelihood(theta)
        #print('lp',lp)
        #print('likeli',likeli)
        #print('likeli',np.isnan(likeli))
        return lp + likeli
    
    def mcmc_em(self,name=None,move=None):
        """
        Bayesian inversion approach with multiple algorithms using the emcee library
        
        Parameters:
            name (str): filename for the h5 file that will contain the traces
            move (str): update algorithm for the steps (metropolis, stretch, kde, de, desnooker, redblue)
            
        Returns:
            traces (array): traces that will give the posterior distribution for the parameters
        """
        import emcee
        
        inis=[]
        for k,source in enumerate(self.sources):
            parnames=source.get_parnames()
            for i in range(source.get_num_params()):
                low,high,ini=self.par2log(source,i)
                inis.append(ini)
        inis.append(0.0)
        
        print(inis)
        
        steps,burnin,thin=self.get_numsteps()
        
        pos = np.array(inis) + 1e-4 * np.random.randn(2*len(inis), len(inis))
        nwalkers, ndim = pos.shape

        if move=='metropolis':
            moves=emcee.moves.GaussianMove(1.0)
        elif move=='stretch':
            moves=emcee.moves.StretchMove()
        elif move=='kde':
            moves=emcee.moves.KDEMove()
        elif move=='de':
            moves=emcee.moves.DEMove()
        elif move=='desnoooker':
            moves=emcee.moves.DESnookerMove()
        elif move=='redblue':
            moves=emcee.moves.RedBlueMove()

        if name is None:
            name='mcmc'
        
        backend = emcee.backends.HDFBackend(name+'.h5')
        backend.reset(nwalkers, ndim)
        
        with Pool() as pool:
            if move is None:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool=pool, backend=backend)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool=pool,moves=moves, backend=backend)
            try:
                sampler.run_mcmc(pos, int(steps/nwalkers),skip_initial_state_check=True, progress=True)
                traces = sampler.get_chain(discard=int(burnin/nwalkers), thin=int(thin/nwalkers), flat=True)
            except KeyboardInterrupt:
                print('Inversion interrupted')
                reader = emcee.backends.HDFBackend(name+'.h5')
                traces = reader.get_chain(discard=int(burnin/nwalkers), thin=int(thin/nwalkers), flat=True)

        traces=traces.T.tolist()[0:-1]
        
        traces,labels=self.traces2lin(traces)
        
        if name is None:
            name=''.join(random.choices(string.ascii_lowercase, k=5))
        
        solution=dict()
        for i,trace in enumerate(traces):
            solution[labels[i]]=trace
        
        with open(name+'.pkl', 'wb') as f:
            pickle.dump(solution, f)
        subprocess.call('rm -rf '+name+'.h5',shell=True)
        
        return traces
    
    def mcmc(self,name=None):
        """
        Bayesian inversion approach with multiple algorithms using the pymc library
        
        Parameters:
            name (str): filename for the h5 file that will contain the traces
            
        Returns:
            traces (array): traces that will give the posterior distribution for the parameters
            MDL (pymc.MCMC): inversion object
        """
        try:
            import pymc
        except ModuleNotFoundError:
            raise ImportError("PyMC is not installed. Please use mcmc_em instead.")

        if(pymc.__version__ != "2.3.8"):
            raise ImportError("PyMC version 2.3.8 is required. Installed PyMC version is %s" % (pymc.__version__))

        self.minresidual=1e6
        data=self.obs.get_data()
        errors=self.obs.get_errors()
        
        #############THIS PART IS FOR REGULARIZATION#################
        reg=False
        for source in self.sources:
            if source.reg:
                reg=True
                break
        
        if reg:
            data=data.tolist()
            errors=errors.tolist()
            for source in self.sources:
                if source.reg:
                    data+=[0.0 for i in range(source.get_num_params())]
                    errors+=[1.0 for i in range(source.get_num_params())]
            data=np.array(data)
            errors=np.array(errors)
        #############THIS PART IS FOR REGULARIZATION#################
        
        if self.obs.err is None:
            wts=1.0
        else:
            wts=1.0/errors
        def model(data):
            #Distribution for every parameter
            theta=[]
            orders=[]
            for k,source in enumerate(self.sources):
                parnames=source.get_parnames()
                for i in range(source.get_num_params()):
                    low,high,ini=self.par2log(source,i)
                    thetat=pymc.Uniform(parnames[i]+str(k),low,high,value=ini)
                    theta.append(thetat)
            model_pars=theta
            #sigma=pymc.Uniform('sigma',0,100,value=0.5)
            sigma=pymc.Uniform('sigma',-10,10,value=0)
            #sigma=-7

            #Deformation is the deterministic variable
            @pymc.deterministic(plot=False)
            def defo(model_pars=model_pars):
                return self.get_model(model_pars)

            #Probability distribution
            z = pymc.Normal('z', mu=defo*wts, tau=1.0/10**sigma, value=data*wts, observed=True)
            return locals()
        
        #Making MCMC model
        MDL = pymc.MCMC(model(data))

        #Choosing Adaptive Metropolis-Hastings for step method
        MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.model_pars)

        #Steps, burnin and thining for each model
        steps,burnin,thin=self.get_numsteps()

        #Number of runs
        MDL.sample(steps,burnin,thin) 

        #Getting the traces
        traces=[]
        for k,source in enumerate(self.sources):
            parnames=source.get_parnames()
            for i in range(source.get_num_params()):
                traces.append(MDL.trace(parnames[i]+str(k)))
        traces=np.array(traces)
        traces,labels=self.traces2lin(traces)
        
        if name is None:
            name=''.join(random.choices(string.ascii_lowercase, k=5))
        
        solution=dict()
        for i,trace in enumerate(traces):
            solution[labels[i]]=trace
        #solution['MDL']=MDL
        
        with open(name+'.pkl', 'wb') as f:
            pickle.dump(solution, f)
        
        return traces,MDL
    
    def doublelog(self,x):
        """
        Double logarithmic function to sweep the parameter space more efficiently
        
        Parameters:
            x (float): parameter value 
            
        Returns:
            logx (float): logarithm in base 10 of the parameter value 
            if value > 1e-2 or value < -1e-2 if not it will give a linear function
        """
        if x>=-1e-2 and x<=1e-2:
            m=2.0/0.02
            return m*x
        elif x<-1e-2:
            return -np.log10(-x)-3
        else:
            return np.log10(x)+3
    
    def invdoublelog(self,y):
        """
        Inverse for double logarithmic function to get the parameter value
        
        Parameters:
            logx (float): value given by the doublelog function
            
        Returns:
            x (float): parameter value
        """
        if y>=-1 and y<=1:
            m=0.02/2.0
            return m*y
        elif y<-1:
            return -10**(-(y+3.0))
        else:
            return 10**(y-3.0)
    
    def par2log(self,source,i):
        """
        Normalizes the bounds and initial guess for the parameters  
        
        Parameters:
            source (Source): source where the parameters will be normalized
            i (int): index for the parameter
            
        Returns:
            low (float): normalized lower bound for the parameter
            high (float): normalized upper bound for the parameter
            ini (float): normalized initial guess for the parameter
        """
        #orders,parnames=self.get_parnames_orders()
        parnames=source.get_parnames()
        order=int(np.log10(np.max([np.abs(source.low_bounds[i]),np.abs(source.high_bounds[i])])))-1
        if not parnames[i]=='pressure':
            low=source.low_bounds[i]/(10**order)
            high=source.high_bounds[i]/(10**order)
            ini=source.x0[i]/(10**order)
        else:
            low=self.doublelog(source.low_bounds[i])
            high=self.doublelog(source.high_bounds[i])
            ini=self.doublelog(source.x0[i])
            
        return low,high,ini
    
    def par2lin(self,pars):
        """
        Multiplies normalized parameters with the correspondant magnitude  
        
        Parameters:
            pars (array): normalized parameters
            
        Returns:
            linpars (array): normalized parameters multiply by their correspondant magnitude
        """
        parnames,orders=self.get_parnames_orders()
        
        linpars=[]
        for i in range(len(pars)):
            par=pars[i]
            if parnames[i]=='pressure':
                linpars.append(self.invdoublelog(par))
            else:
                linpars.append(par*orders[i])
        return linpars
    
    def traces2lin(self,traces):
        """
        Multiplies normalized parameters with the correspondant magnitude
        for all the values in the traces
        
        Parameters:
            traces (array): traces with normalized values 
            
        Returns:
            data (array): normalized traces multiply by their correspondant magnitude
            labels (array): names for the parameters
        """
        data=[]
        labels=[]
        parnames,orders=self.get_parnames_orders()
        for i,trace in enumerate(traces):
            temp=np.array(trace[:])
            if parnames[i]=='pressure':
                vdata=np.array([self.invdoublelog(temp[j]) for j in range(len(temp))])
                data.append(vdata)
            else:
                data.append(temp*orders[i])
            labels.append(parnames[i])
        data=np.vstack(data)
        return data,labels
    
    def get_parnames_orders(self):
        """
        Gives the magnitudes for the parameters
        
        Returns:
            parnames (array): names for the parameters
            orders (array): magnitudes for the parameters
        """
        if len(self.sources)==1:
            source=self.sources[0]
            return source.get_parnames(),source.get_orders()
        else:
            parnamest=[]
            orderst=[]
            for k,source in enumerate(self.sources):
                parnames=source.get_parnames()
                orders=source.get_orders()
                for j,name in enumerate(parnames):
                    parnamest.append(name+str(k))
                    orderst.append(orders[j])
            
            return parnamest,orderst
    
    def set_numsteps(self,steps,burnin,thin):
        """
        Set the number of steps, discarded steps and steps 
        per sample for a Bayesian inversion, if the model has 
        more than one source the default is
        steps=6600000, burnin=6000000, thin=1000, if not it will
        take the values defined in the source object
        
        Returns:
            steps (int): number of steps 
            burnin (int): number of initial discarded steps
            thin (int): number of steps per sample
        """
        self.steps=steps
        self.burnin=burnin
        self.thin=thin
    
    def get_numsteps(self):
        """
        Gives the number of steps, discarded steps and steps 
        per sample for a Bayesian inversion, if the model has 
        more than one source the default is
        steps=6600000, burnin=6000000, thin=1000, if not it will
        take the values defined in the source object
        
        Returns:
            steps (int): number of steps 
            burnin (int): number of initial discarded steps
            thin (int): number of steps per sample
        """
        if not (self.steps is None or self.burnin is None or self.thin is None):
            steps=self.steps
            burnin=self.burnin
            thin=self.thin
        elif len(self.sources)>1:
            steps=6600000
            burnin=600000
            thin=1000
        else:
            source=self.sources[0]
            steps,burnin,thin=source.bayesian_steps()
        return steps,burnin,thin
    
    #initial guess of source model characteristics, defined when creating the source model
    def get_x0(self):
        """
        Concatenates the initial values for the parameters for all sources
        in the model
        
        Returns:
            x0 (array): intial values for the parameters
        """
        x0 = []
        
        for s in self.sources:
            x0 = np.concatenate((x0, s.x0))

        return x0

    #high and low bounds for the parameters
    def get_bounds(self):
        """
        Concatenates and returns the lower and upper bounds for the parameters for all sources
        in the model
        
        Returns:
            low_b (array): lower bounds for the parameters
            high_b (array): upper bounds for the parameters
        """
        low_b  = []
        high_b = []

        for s in self.sources:
            low_b  = np.concatenate((low_b,  s.low_bounds))
            high_b = np.concatenate((high_b, s.high_bounds))

        return (low_b, high_b)
    
    def get_bounds_de(self):
        """
        Arranges lower and upper bounds in one array for the differential evolution algorithm
        
        Returns:
            bounds (array): bounds for the parameters
        """
        bounds  = []

        for s in self.sources:
            for i in range(len(s.high_bounds)):
                if len(bounds)==0:
                    bounds=[(s.low_bounds[i],s.high_bounds[i])]
                else:
                    bounds = np.concatenate((bounds,[(s.low_bounds[i],s.high_bounds[i])]))

        return bounds
    
    def forward(self, x, unravel=True):
        """
        Computes the forward model for single or multisource model
        
        Parameters:
            unravel (boolean): if True it will return just one array,
            if False it will separate the deformation in components
        
        Returns:
            data (array): modeled deformation
        """
        param_cnt = 0
        data=None
        for s in self.sources:
            datat = s.forward(x[param_cnt:param_cnt+s.get_num_params()],unravel)
            param_cnt += s.get_num_params()
            
            if data is None:
                data=datat
            else:
                if unravel:
                    data += datat
                else:
                    for i,comp in enumerate(data):
                        comp+=datat[i]

        return data
    
    def residual(self,x):
        """
        Computes the residuals for the non-linear least squares algorithm
        
        Parameters:
            x (array): parameters in an iteration
            
        Returns:
            res (float): residual for the iteration
        """
        if self.obs.get_errors() is None:
            res=self.obs.get_data()-self.forward(x)
        else:
            res=(self.obs.get_data()-self.forward(x))/self.obs.get_errors()
        rest=np.sqrt(np.sum(res**2))
        if self.minresidual>rest:
            self.minresidual=rest
            self.minparms=x
        
        return res
    
    def residual_bh(self,x):
        """
        Computes the residuals for the optimization algorithms
        
        Parameters:
            x (array): parameters in an iteration
            
        Returns:
            rest (float): residual for the iteration
        """
        if self.obs.get_errors() is None:
            res=self.obs.get_data()-self.forward(x)
        else:
            res=(self.obs.get_data()-self.forward(x))/self.obs.get_errors()
        rest=np.sqrt(np.sum(res**2))
        if self.minresidual>rest:
            self.minresidual=rest
            self.minparms=x
        
        return rest
    
    def inv_opening(self,xcen,ycen,depth,length,width,strike,dip,reg=False,lamb=1):
        """
        Inversion for opening values in a discretized sill
        
        Parameters:
            xcen (float): x-coordinate location for center of the sill (m)
            ycen (float): y-coordinate location for center of the sill (m)
            depth (float): depth for center of the sill (m)
            length (float): length of the discretized sill (m)
            width (float): width of the discretized sill (m)
            strike (float): azimuth angle for the sill clockwise from north (degrees)
            dip (float): dipping angle for the sill (degrees)
            reg (boolean): if True apply regularization, if False regularization
            won't be applied
            lamb (float): regularization constant 
            
        Returns:
            ops (array): openings for the sill patches
        """
        s=self.sources[0]
        G=s.get_greens(xcen,ycen,depth,length,width,strike,dip)
        if reg==True:
            d=np.array(self.obs.get_data().tolist()+np.zeros((s.ln*s.wn,)).tolist())
            L=s.get_laplacian(xcen,ycen,depth,length,width,strike,dip)
            #print(L)
            newG=np.concatenate((G,lamb*L),axis=0)
            ops=np.linalg.lstsq(newG, d, rcond=None)[0]
        else:
            ops=np.linalg.lstsq(G, self.obs.get_obs(), rcond=None)[0]
        return ops
    
    def get_params_openings(self,xcen,ycen,depth,length,width,strike,dip,ln,wn,ops):
        """
        Gives the parameters for each sill patch
        
        Parameters:
            xcen (float): x-coordinate location for center of the sill (m)
            ycen (float): y-coordinate location for center of the sill (m)
            depth (float): depth for center of the sill (m)
            length (float): length of the discretized sill (m)
            width (float): width of the discretized sill (m)
            strike (float): azimuth angle for the sill clockwise from north (degrees)
            dip (float): dipping angle for the sill (degrees)
            ln (int): divisions in length
            wn (int): divisions in width
            ops (array): openings for sill patches 
            
        Returns:
            params (array): parameters for sill patches
        """
        s=self.sources[0]
        xcs,ycs,zcs=s.get_centers(xcen,ycen,depth,length,width,strike,dip,ln,wn)
        params=[]
        slength=length/ln
        swidth=width/wn
        for i in range(len(xcs)):
            params+=[xcs[i],ycs[i],zcs[i],slength,swidth,ops[i],strike,dip]
        return params

    def get_model(self,x):
        """
        Computes the deformation values for a given set of parameters
        
        Parameters:
            x (array): normalized values for the parameters 
            
        Returns:
            defo (array): deformation values for the given parameters
        """
        defo=None
        rmod=None
        param_cnt = 0
        xlin=[]
        linparst=self.par2lin(x)
        
        for s in self.sources:
            linpars=linparst[param_cnt:param_cnt+s.get_num_params()]
            #############THIS PART IS FOR REGULARIZATION#################
            if s.reg:
                if rmod is None:
                    rmod=(s.get_laplacian()@np.array(linpars)).tolist()
                else:
                    rmod+=(s.get_laplacian()@np.array(linpars)).tolist()
            #############THIS PART IS FOR REGULARIZATION#################
            
            if defo is None:
                defo=s.forward(linpars)
            else:
                defo+=s.forward(linpars)
            param_cnt += s.get_num_params()
        
        self.residual(linparst)
        
        #############THIS PART IS FOR REGULARIZATION#################
        if not rmod is None:
            defo=np.array(defo.tolist()+rmod)
        #############THIS PART IS FOR REGULARIZATION#################
        
        return defo
    
    ##output writers
    def print_model(self):
        """
        Prints the parameter names and their values
        """
        param_cnt = 0

        for s in self.sources:
            s.print_model(self.model.x[param_cnt:param_cnt+s.get_num_params()])
            param_cnt += s.get_num_params()

    ##writes gmt files for horizontal and vertical deformation, each, to use with gmt velo.
    def write_forward_gmt(self, prefix, volcano=None):
        if self.model is not None:

            ux,uy,uz = self.forward(self.model.x)

            dat = np.zeros(self.obs.data['id'].to_numpy().size, 
                dtype=[ ('lon', float), ('lat', float), ('east', float), ('north', float), 
                        ('esig', float), ('nsig', float), ('corr', float), ('id', 'U6')] )

            dat['lon']   = self.obs.data['lon'].to_numpy()
            dat['lat']   = self.obs.data['lat'].to_numpy()
            dat['east']  = ux*1000
            dat['north'] = uy*1000
            dat['esig']  = ux*0
            dat['nsig']  = ux*0
            dat['corr']  = ux*0
            dat['id']    = self.obs.data['id'].to_numpy()

            print(dat)

            #horizontal predictions    
            np.savetxt(prefix+"_hori.gmt", dat, fmt='%s' )

            #vertical predictions    
            dat['east']  = ux*0
            dat['north'] = uz*1000
            np.savetxt(prefix+"_vert.gmt", dat, fmt='%s' )
            
            if volcano is not None:
                import utm
                dat = np.zeros(len(self.sources), dtype=[ ('lon', float), ('lat', float), ('id', 'U6')] )

                print(len(self.sources))                    

                param_cnt = 0
                source_cnt = 0

                for s in self.sources:
                    e_loc = volcano[0] + self.model.x[param_cnt]
                    n_loc = volcano[1] + self.model.x[param_cnt+1]

                    param_cnt += s.get_num_params()

                    loc_ll = utm.to_latlon(e_loc, n_loc, volcano[2], volcano[3])

                    dat["lat"][source_cnt] = loc_ll[0]
                    dat["lon"][source_cnt] = loc_ll[1]
                    dat["id"][source_cnt]  = s.get_source_id()

                    source_cnt += 1

                print(dat)                    

                np.savetxt(prefix+"_source_loc.gmt", dat, fmt='%s' )
        else:
            print("No model, nothing to write")
