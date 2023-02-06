from .gnss import Gnss

class Level(Gnss):
    def add_data(self,deltas):
        self.add_uz(deltas)
    
    def add_err(self,err):
        self.add_errz(err)