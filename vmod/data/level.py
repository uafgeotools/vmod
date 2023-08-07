from .gnss import Gnss

class Level(Gnss):
    """
    Class that represents a a levelling dataset. It uses the Gnss class
    """
    def add_data(self,deltas):
        """
        Adds the vertical components
        
        Parameters:
            deltas (array): changes in the vertical (m)
        """
        self.add_uz(deltas)
    
    def add_err(self,err):
        """
        Adds the uncertainties in the vertical components
        
        Parameters:
            err (array): uncertainties in the changes in the vertical (m)
        """
        self.add_errz(err)