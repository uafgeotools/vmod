from . import Data
import numpy as np
import copy

class Joint(Data):
    
    def __init__(self):
        self.datasets=None
        super().__init__()
    
    def add_dataset(self,data,wt=1.0):
        dataset=copy.deepcopy(data)
        if isinstance(dataset,Data):
            if dataset.err is not None:
                dataset.err=dataset.err/(np.min(dataset.err)*wt)
            if self.datasets is None:
                if dataset.data is None:
                    raise Exception('There are not data points in this dataset')
                else:
                    self.datasets=[dataset]
                    self.comps=copy.deepcopy(dataset.comps)
            else:
                self.datasets.append(dataset)
                self.comps+=dataset.comps
                
                
    def ref_possible(self):
        return False
    
    def get_data(self):
        data=[]
        for dataset in self.datasets:
            data+=dataset.get_data().tolist()
        return np.array(data)
    
    def get_errors(self):
        errors=[]
        for dataset in self.datasets:
            errors+=dataset.get_errors().tolist()
        return np.array(errors)
    
    def from_model(self,func,offsets=None,unravel=True):
        if self.datasets is None:
            raise Exception('No datasets included')
        if offsets is None:
            offsets=np.array([0 for i in self.comps])
        data=[]
        ini=0
        for dataset in self.datasets:
            ncomps=ini+len(dataset.comps)
            datat=dataset.from_model(func,offsets[ini:ncomps],unravel)
            data+=datat.tolist()
            ini=len(dataset.comps)
        return np.array(data)
            