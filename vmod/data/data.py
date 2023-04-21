"""
Class to implement interface to geodetic data types used in source inversions

Author: Mario Angarita & Ronni Grapenthin
Date: 9/22/2022
"""

import numpy as np
import os
from vmod import util


class Data:
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
        self.data_size_per_point=0
    
    def get_size_per_point(self):
        if self.comps==None:
            return 0
        else:
            return len(self.comps)
        
    def add_names(self,names):
        self.assert_size(names,'names')
        self.names=names
    
    def add_lls(self,lons,lats,ori=None):
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
        self.assert_size(ys,'ys')
        self.ys=ys
    
    def add_xs(self,xs):
        self.assert_size(xs,'xs')
        self.xs=xs
    
    def add_zs(self,zs):
        self.assert_size(zs,'zs')
        self.zs=zs
    
    def add_ts(self,ts):
        self.assert_size(ts,'ts')
        self.ts=ts
        
    def add_comp(self,comp,name):
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
        i=self.get_index(name)
        for k,compname in enumerate(self.comps):
            j=self.get_index(compname)
            if j>=i:
                break
            elif k==len(self.comps)-1:
                k+=1
        return k
        
    def add_errcomp(self,err,name):
        assert name in self.comps,'The component '+name+' has not been included'
        self.assert_size(err,name)
        k=self.get_order(name)
        self.err[k*len(err):(k+1)*len(err)]=err
    
    def add_locs(self, xs, ys, ts=None, zs=None):
        self.add_xs(xs)
        self.add_ys(ys)
        if isinstance(zs,(list,np.ndarray)):
            self.add_zs(zs)
        if isinstance(ts,(list,np.ndarray)):
            self.add_ts(ts)
    
    def get_index(self,name):
        return 0
    
    def add_data(self,data):
        self.assert_size(data,'data')
        self.data=data
        
    def assert_size(self,arr,name):
        assert isinstance(arr,(list,np.ndarray)),name+' must be an array'
        if (name=='data' or name=='errors') and not self.get_size_per_point()==0:
            assert len(arr)/self.get_size_per_point()==self.get_num_points() or self.get_num_points()==None,"The number of "+name+" points does not have the proper size"
        else:
            assert len(arr)==self.get_num_points() or self.get_num_points()==None,"The number of "+name+" points does not have the proper size"
        
    def add_err(self,err):
        self.assert_size(err,'errors')
        self.err=err
    
    def add(self, xs, ys, data, ts=None, zs=None):
        self.add_locs(xs,ys,ts,zs)
        self.add_data(data)
        
    def get_num_points(self):
        params=[self.xs,self.ys,self.zs,self.ts,self.data]
        for i,l in enumerate(params):
            if isinstance(l,(list,np.ndarray)):
                if i==len(params)-1:
                    return len(l)/self.get_size_per_point()
                else:
                    return len(l)
        return None
    
    def check_ref(self,sta):
        for name in self.names:
            tref=self.ts[self.names==ref].tolist()
            newtref=self.ts[self.names==sta].tolist()
            if not newtref in tref:
                return False
        return True
            
    
    def add_ref(self,sta):
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
        return True
    
    def get_locs(self):
        return self.xs,self.ys,self.ts,self.zs
    
    def get_data(self,unravel=True):
        return self.reference_dataset(self.data,unravel)
        
    def reference_dataset(self,data,unravel=True):
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
        return self.reference_errors(self.err)