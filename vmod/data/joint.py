from . import Data
import numpy as np
import copy

class Joint(Data):
    """
    Class that represents a joint dataset composed by one or many different datasets

    Attributes:
        datasets (array): array with multiple data objects
        comps (array): array containing all the components in the data objects
    """
    def __init__(self):
        self.datasets=None
        super().__init__()
    
    def add_dataset(self,data,wt=1.0):
        """
        Adds a data object to the joint dataset
        
        Parameters:
            data (Data): data object that represents a dataset
            wt (float): relative weight for the dataset
        """
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
        """
        Overrides the function to specify that the joint dataset cannot take references
        """
        return False
    
    def get_data(self):
        """
        Concatenates and gives the data in the datasets
        
        Returns:
            data (array): deformation data
        """
        data=[]
        for dataset in self.datasets:
            data+=dataset.get_data().tolist()
        return np.array(data)
    
    def get_errors(self):
        """
        Concatenates and gives the uncertainties in the datasets
        
        Returns:
            errors (array): uncertainties deformation data
        """
        errors=[]
        for dataset in self.datasets:
            errors+=dataset.get_errors().tolist()
        return np.array(errors)
    
    def from_model(self,func,offsets=None,unravel=True):
        """
        Uses the function from_model on each dataset to compute the forward model
        
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
            