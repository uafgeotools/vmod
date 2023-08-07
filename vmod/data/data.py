import numpy as np
import os
from vmod import util


class Data:
    """
    Class to implement interface to geodetic data types used in source inversions

    Parameters:
        names: names for the datapoints (stations)
        xs: positions of datapoints in the x-axis (m)
        ys: positions of datapoints in the y-axis (m)
        lons: longitudes of datapoints (m)
        lats: latitudes of datapoints (m)
        zs: elevations of datapoints (m)
        ts: times for datapoints (s)
        comps: names for the components in the dataset (e.g., for gps, comps=['ux','uy','uz'])
        data: deformation dataset, they array is ordered according to comps (m)
        err: uncertainties in the dataset (m)
        refs: names for the datapoints that are the references
        utmz: utm zone for the projection
        
    """
    def __init__(self):
        self.names=None
        self.xs=None
        self.ys=None
        self.lons=None
        self.lats=None
        self.zs=None
        self.ts=None
        self.comps=None
        self.data=None
        self.err=None
        self.refs=None
        self.utmz=None
        #self.data_size_per_point=0
    
    def get_size_per_point(self):
        """
        Gives number of components per datapoint (time-space)
        
        Returns:
            numcomps: number of components
        """
        if self.comps==None:
            return 0
        else:
            return len(self.comps)
        
    def add_names(self,names):
        """
        Assign names to the datapoints
        
        Parameters:
            names: names for the stations or datapoints
        """
        self.assert_size(names,'names')
        self.names=names
    
    def add_lls(self,lons,lats,ori=None):
        """
        Assign spatial positions using longitudes and latitudes
        
        Parameters:
            lons: longitudes for the stations or datapoints
            lats: latitudes for the stations or datapoints
            ori: origin for the projection of longitudes and latitudes if None the mean position is the origin
        """
        xs,ys,z1s,z2s=util.ll2utm(lons,lats)
        if ori is None:
            self.utmz=[np.mean(xs),np.mean(ys),z1s,z2s]
            self.add_xs(xs-np.mean(xs))
            self.add_ys(ys-np.mean(ys))
        else:
            orix,oriy,z1sor,z2sor=util.ll2utm([ori[0]],[ori[1]],z1=z1s,z2=z2s)
            self.utmz=[orix[0],oriy[0],z1sor,z2sor]
            self.add_xs(xs-orix[0])
            self.add_ys(ys-oriy[0])
        self.lons=lons
        self.lats=lats
        
    def add_ys(self,ys):
        """
        Assign y-axis positions for the stations or datapoints
        
        Parameters:
            ys: y-coordinates for stations or datapoints
        """
        self.assert_size(ys,'ys')
        self.ys=ys
    
    def add_xs(self,xs):
        """
        Assign x-axis positions for the stations or datapoints
        
        Parameters:
            xs: x-coordinates for stations or datapoints
        """
        self.assert_size(xs,'xs')
        self.xs=xs
    
    def add_zs(self,zs):
        """
        Assign z-axis positions for the stations or datapoints
        
        Parameters:
            zs: z-coordinates for stations or datapoints
        """
        self.assert_size(zs,'zs')
        self.zs=zs
    
    def add_ts(self,ts):
        """
        Assign times for the stations or datapoints
        
        Parameters:
            ts: times for stations or datapoints
        """
        self.assert_size(ts,'ts')
        self.ts=ts
        
    def add_comp(self,comp,name):
        """
        Add component to the dataset
        
        Parameters:
            comp: data points for the component
            name: name for the component
        """
        self.assert_size(comp,name)
        i=self.get_index(name)
        if self.comps==None:
            new_data=comp
            new_err=comp*0+1
            self.comps=[name]
        else:
            replace=False
            k=self.get_order(name)
            if name in self.comps:
                print('This component was already included; it will be replaced')
                replace=True
            if replace:
                new_data=np.concatenate((self.data[0:k*len(comp)],comp,self.data[(k+1)*len(comp)::]))
                new_err=np.concatenate((self.err[0:k*len(comp)],comp*0+1.0,self.err[(k+1)*len(comp)::]))
            else:
                new_data=np.concatenate((self.data[0:k*len(comp)],comp,self.data[k*len(comp)::]))
                new_err=np.concatenate((self.err[0:k*len(comp)],comp*0+1.0,self.err[k*len(comp)::]))
                self.comps=self.comps[0:k]+[name]+self.comps[k::]
        self.data=new_data
        self.err=new_err
        
    def get_order(self,name):
        """
        Gives the position of the component in the dataset
        
        Parameters:
            name: name for the component
        """
        i=self.get_index(name)
        for k,compname in enumerate(self.comps):
            j=self.get_index(compname)
            if j>=i:
                break
            elif k==len(self.comps)-1:
                k+=1
        return k
        
    def add_errcomp(self,err,name):
        """
        Adds the uncertainties for certain component
        
        Parameters:
            err: uncertainties for the component
            name: component's name to add the uncertainties
        """
        assert name in self.comps,'The component '+name+' has not been included'
        self.assert_size(err,name)
        k=self.get_order(name)
        self.err[k*len(err):(k+1)*len(err)]=err
    
    def add_locs(self, xs, ys, ts=None, zs=None):
        """
        Adds the spatial-temporal positions for the datapoints
        
        Parameters:
            xs: x-coordinates for the datapoints
            ys: y-coordinates for the datapoints
            ts: times for the datapoints
            zs: z-coordinates for the datapoints
        """
        self.add_xs(xs)
        self.add_ys(ys)
        if isinstance(zs,(list,np.ndarray)):
            self.add_zs(zs)
        if isinstance(ts,(list,np.ndarray)):
            self.add_ts(ts)
    
    def get_index(self,name):
        """
        Gives the position for the component
        
        Parameters:
            name: component's name
        """
        return 0
    
    def add_data(self,data):
        """
        Add the deformation values to the dataset
        
        Parameters:
            data: deformation data
        """
        self.assert_size(data,'data')
        self.data=data
        
    def assert_size(self,arr,name):
        """
        Asserts the size of the array. The correspondance should be 1-1 with deformation data and uncertainties
        the correspondance of spatial datapoints with deformation data is 1-number of components
        
        Parameters:
            data: deformation data
        """
        assert isinstance(arr,(list,np.ndarray)),name+' must be an array'
        if (name=='data' or name=='errors') and not self.get_size_per_point()==0:
            assert len(arr)/self.get_size_per_point()==self.get_num_points() or self.get_num_points()==None,"The number of "+name+" points does not have the proper size"
        else:
            assert len(arr)==self.get_num_points() or self.get_num_points()==None,"The number of "+name+" points does not have the proper size"
        
    def add_err(self,err):
        """
        Adds the uncertainties for all the dataset
        
        Parameters:
            err: uncertainties for the dataset
        """
        self.assert_size(err,'errors')
        self.err=err
    
    def add(self, xs, ys, data, ts=None, zs=None):
        """
        Adds spatial-temporal locations and deformation data
        
        Parameters:
            xs: x-coordinates for the datapoints
            ys: y-coordinates for the datapoints
            data: deformation data
            zs: z-coordinates for the datapoints
        """
        self.add_locs(xs,ys,ts,zs)
        self.add_data(data)
        
    def get_num_points(self):
        """
        Gives the number of stations or datapoints in the dataset
        
        Returns:
            num: number of datapoints, None if there are not datapoints.
        """
        params=[self.xs,self.ys,self.zs,self.ts,self.data]
        for i,l in enumerate(params):
            if isinstance(l,(list,np.ndarray)):
                if i==len(params)-1:
                    return len(l)/self.get_size_per_point()
                else:
                    return len(l)
        return None
    
    def check_ref(self,ref):
        """
        Checks if a station has the same timesteps as the rest of the stations
        so itcan be taken as reference
        
        Returns:
            ref: name of reference station or datapoint
        """
        for sta in self.names:
            tref=self.ts[self.names==ref].tolist()
            newtref=self.ts[self.names==sta].tolist()
            if not newtref in tref:
                return False
        return True
            
    
    def add_ref(self,sta):
        """
        Adds station or datapoint as a reference for the rest of the datapoints
        
        Parameters:
            sta: name of the possible reference station
        """
        if self.ref_possible():
            if self.names is None:
                raise Exception('Names have not been defined')
            else:
                if not sta in self.names:
                    raise Exception('The station does not exist')
        else:
            raise Exception('This type of data cannot be referenced')
            
        if self.refs==None and self.ts==None:
            self.refs=[sta]
        elif not self.refs==None:
            if sta in self.refs:
                raise Exception('The station has already been included as a reference')
        elif self.ts==None:
            self.refs.append(sta)
        else:
            if not self.check_ref(sta):
                raise Exception('The new reference do not have the same time steps as the rest of the stations')
            else:
                if self.refs==None:
                    self.refs=[sta]
                else:
                    self.refs.append(sta)
    
    def remove_ref(self,sta):
        """
        Removes station or datapoint as a reference for the rest of the datapoints
        
        Parameters:
            sta: name of reference station
        """
        if not self.refs==None:
            if not sta in self.refs:
                raise Exception('This station has not been set as a reference') 
            elif len(self.refs)==1:
                self.refs=None
            else:
                self.refs.remove(sta)
        else:
            print('References have not been set')
    
    def ref_possible(self):
        """
        Defines if references can be taken for certain datatype
        
        Returns:
            possible: references can be taken for the datatype (default True but can be override)
        """
        return True
    
    def get_locs(self):
        """
        Gives the locations for the datapoints
        
        Returns:
            xs: x-coordinates
            ys: y-coordinates
            ts: times for datapoints
            zs: z-coordinates
        """
        return self.xs,self.ys,self.ts,self.zs
    
    def get_data(self,unravel=True):
        """
        Gives the referenced deformation datapoints
        
        Parameters:
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
        
        Returns:
            data: referenced deformation
        """
        return self.reference_dataset(self.data,unravel)
        
    def reference_dataset(self,data,unravel=True):
        """
        References the deformation according to the reference stations
        
        Parameters:
            data: data to be referenced
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
        
        Returns:
            data: referenced deformation
        """
        if self.refs is None:
            ref_data=data
        elif self.ts is None:
            ref_data=np.copy(data)
            for i in range(len(self.comps)):
                ref=np.mean([ref_data[i*len(self.xs):(i+1)*len(self.xs)][self.names==r] for r in self.refs])
                ref_data[i*len(self.xs):(i+1)*len(self.xs)]-=ref
        else:
            tref=self.ts[self.names==self.refs[0]]
            ref_data=np.copy(data)
            for i in range(len(self.comps)):
                for t in tref:
                    ref=0
                    for r in self.refs:
                        pos=np.argwhere(np.logical_and(self.ts==t,self.names==r))[0]
                        ref+=data[i*len(self.xs)+pos]/len(self.refs)
                    for sta in self.names:
                        stapos=np.argwhere(np.logical_and(self.ts==t,self.names==sta))[0]
                        ref_data[i*len(self.xs)+stapos]-=ref
        
        if not unravel:
            ref_comps=()
            for i in range(len(self.comps)):
                ref_comps+=(ref_data[i*len(self.xs):(i+1)*len(self.xs)],)
            ref_data=ref_comps

        return ref_data
    
    def reference_errors(self,err,unravel=True):
        """
        References the uncertainties using error propagation
        
        Parameters:
            err: uncertainties to be referenced
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
        
        Returns:
            err: referenced uncertainties
        """
        if self.refs is None:
            ref_err=err
        elif self.ts is None:
            ref_err=np.copy(err)
            for i in range(len(self.comps)):
                ref=np.sum([ref_err[i*len(self.xs):(i+1)*len(self.xs)][self.names==r]**2 for r in self.refs])
                ref_err[i*len(self.xs):(i+1)*len(self.xs)]=np.sqrt((ref_err[i*len(self.xs):(i+1)*len(self.xs)]**2+ref)/(len(self.refs)+1))
        else:
            tref=self.ts[self.names==self.refs[0]]
            ref_err=np.copy(err)
            for i in range(len(self.comps)):
                for t in tref:
                    ref=0
                    for r in self.refs:
                        pos=np.argwhere(np.logical_and(self.ts==t,self.names==r))[0]
                        ref+=err[i*len(self.xs)+pos]**2
                    for sta in self.names:
                        stapos=np.argwhere(np.logical_and(self.ts==t,self.names==sta))[0]
                        ref_err[i*len(self.xs)+stapos]=np.sqrt((ref_err[i*len(self.xs)+stapos]**2+ref)/(len(self.refs)+1))
        
        if not unravel:
            ref_comps=()
            for i in range(len(self.comps)):
                ref_comps+=(ref_err[i*len(self.xs):(i+1)*len(self.xs)],)
            ref_err=ref_comps

        return ref_err
    
    def from_model(self,func,offsets=None,unravel=True):
        """
        Uses the function from the forward model and add the offsets if they are computed
        
        Parameters:
            func: forward model function
            offsets: None or array that contains the offset for each component
            unravel (boolean): If True will give a single list with all the deformation in the datapoints
            if False, it will several array depending on the number of components in the dataset.
            
        Returns:
            ux: modeled deformation in the x-axis
            uy: modeled deformation in the y-axis
            uz: modeled deformation in the vertical
        """
        if 'time' in func.__name__:
            funct=lambda x,y: func(x,y,self.ts)
            model=self.from_model3d(funct)
        else:
            model=self.from_model3d(func)
        new_model=np.array([])
        if not offsets is None:
            if not len(model)==len(offsets):
                raise Exception('Not enough number of offsets')
            for i,comp in enumerate(model):
                comp+=offsets[i]
                
        for comp in model:
            new_model=np.concatenate((new_model,comp))
            
        ref_model=self.reference_dataset(new_model,unravel)
        
        return ref_model
    
    def get_errors(self):
        """
        Gives the referenced uncertainties for the datapoints
        
        Returns:
            errors: referenced uncertainties
        """
        return self.reference_errors(self.err)