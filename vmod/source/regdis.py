import numpy as np
from .. import util
from . import Source
from .okada import Okada
from ..data import Gnss

class Regdis(Source):
    """
    Class to represent a regularized dislocation using dislocation patches from Okada (1985) model.

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    typ : str
        type of dislocation, open for tensile and slip for fault
    ln : number of segments in the lenght
    wn : number of segments in the width
    xcen : x-coordinate for the center of the sill/fault
    ycen : y-coordinate for the center of the sill/fault
    depth : depth of the center of the sill/fault
    length : length of the sill/fault
    width : width of the sill/fault
    strike : orientation of the sill/fault clockwise from north
    dip : dip angle of the fault 0 horizontal, 90 vertical
    lamb : regularization constant
    """
    def __init__(self, data, typ=None, ln=None,wn=None, xcen=None,ycen=None,depth=None,length=None,width=None,strike=None,dip=None,lamb=None,rake=None):
        self.typ = 'open' if typ == 'open' or typ is None else 'slip'
        self.ln     = 1 if ln     is None else ln
        self.wn     = 1 if wn     is None else wn
        self.xcen   = 0 if xcen   is None else xcen
        self.ycen   = 0 if ycen   is None else ycen
        self.depth  = 1 if depth  is None else depth
        self.length = 0 if length is None else length
        self.width  = 0 if width  is None else width
        self.strike = 0 if strike is None else strike
        self.dip    = 0 if dip    is None else dip
        self.rake   = 0 if rake   is None else rake
        self.lamb   = 0 if lamb   is None else lamb
        
        super().__init__(data)
        
        self.reg    = True
        
    def time_dependent(self):
        return False
    
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=()
        for i in range(self.ln*self.wn):
            if self.typ=='open':
                self.parameters+=('open'+str(i),)
            else:
                self.parameters+=('slip'+str(i),)
    
    def set_x0(self,x0):
        """
        Initial guess for the openings/slips
        """
        self.get_parnames()
        self.x0=[x0]*(self.ln*self.wn)
    
    def set_bounds(self,low_bound,high_bound):
        """
        Bounds for the openings/slips
        """
        self.get_parnames()
        self.low_bounds=[low_bound]*(self.ln*self.wn)
        self.high_bounds=[high_bound]*(self.ln*self.wn)
        
    def rotate_xyz(self,length,width):
        """
        Rotation for the patches in the sill/fault
        """
        cx=self.xcen
        cy=self.ycen
        cz=-self.depth
        wp=width*np.cos(np.radians(self.dip))
        wr=width*np.sin(np.radians(self.dip))
        l=length
        phi=self.strike
        x1 = cx + wp/2 * np.cos(np.radians(phi)) - l/2 * np.sin(np.radians(phi))
        y1 = cy + wp/2 * np.sin(np.radians(phi)) + l/2 * np.cos(np.radians(phi))
        z1 = cz - wr/2
        x2 = cx - wp/2 * np.cos(np.radians(phi)) - l/2 * np.sin(np.radians(phi))
        y2 = cy - wp/2 * np.sin(np.radians(phi)) + l/2 * np.cos(np.radians(phi))
        z2 = cz + wr/2
        x3 = cx - wp/2 * np.cos(np.radians(phi)) + l/2 * np.sin(np.radians(phi))
        y3 = cy - wp/2 * np.sin(np.radians(phi)) - l/2 * np.cos(np.radians(phi))
        z3 = cz + wr/2
        x4 = cx + wp/2 * np.cos(np.radians(phi)) + l/2 * np.sin(np.radians(phi))
        y4 = cy + wp/2 * np.sin(np.radians(phi)) - l/2 * np.cos(np.radians(phi))
        z4 = cz - wr/2
        
        return [x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]
        
    def get_centers(self):
        """
        Returns the centers for sill/fault patches
        """
        xc=self.xcen
        yc=self.ycen
        zc=-self.depth
        ln=self.ln
        wn=self.wn
        lslice=self.length/self.ln
        wslice=self.width/self.wn
        fwc=self.xcen-self.width/2+self.width/(2*self.wn)
        flc=self.ycen-self.length/2+self.length/(2*self.ln)
        #print(fwc,flc)
        xcs,ycs,zcs=[],[],[]
        if wn%2==0:
            wi=self.wn/2
        else:
            wi=(self.wn-1)/2
            
        if ln%2==0:
            li=self.ln/2
        else:
            li=(self.ln-1)/2
            
        for i in range(int(wi)):
            wfake=2*np.abs(fwc-self.xcen+float(i)*wslice)
            for j in range(int(li)):
                lfake=2*np.abs(flc-self.ycen+float(j)*lslice)
                xs,ys,zs=self.rotate_xyz(lfake,wfake)
                for x in xs:
                    xcs.append(x)
                for y in ys:
                    ycs.append(y)
                for z in zs:
                    zcs.append(z)
        #print('Puntos 1',len(xcs),wn%2,ln%2)
        if not self.wn%2==0:
            for j in range(int(li)):
                wfake=0
                lfake=2*np.abs(flc-self.ycen+float(j)*lslice)
                xs,ys,zs=self.rotate_xyz(lfake,wfake)
                for x in xs[1:3]:
                    xcs.append(x)
                for y in ys[1:3]:
                    ycs.append(y)
                for z in zs[1:3]:
                    zcs.append(z)
        #print('Puntos 2',len(xcs))
        if not self.ln%2==0:
            for i in range(int(wi)):
                wfake=2*np.abs(fwc-self.xcen+float(i)*wslice)
                lfake=0
                xs,ys,zs=self.rotate_xyz(lfake,wfake)
                for x in xs[0:2]:
                    xcs.append(x)
                for y in ys[0:2]:
                    ycs.append(y)
                for z in zs[0:2]:
                    zcs.append(z)
        #print('Puntos 3',len(xcs))
        if (not self.wn%2==0) and (not self.ln%2==0):
            xcs.append(self.xcen)
            ycs.append(self.ycen)
            zcs.append(-self.depth)
        #print('Puntos 4',len(xcs))
        return xcs,ycs,zcs
    
    def get_greens(self):
        """
        Greens functions for the patches
        """
        ln=self.ln
        wn=self.wn
        x=self.data.xs
        y=self.data.ys
        
        dat1 = Gnss()
        dat1.add_xs(x)
        dat1.add_ys(y)
        
        dat1.add_data(x*0,x*0,x*0)
        
        oki=Okada(dat1)
        #print('Tipo',self.typ)
        oki.set_type(self.typ)
        
        xcs,ycs,zcs=self.get_centers()
        xo=[self.xcen,self.ycen,self.depth,self.length,self.width,1,self.strike,self.dip]
        
        defo=oki.forward(xo)
        
        slength=self.length/ln
        swidth=self.width/wn
        
        if self.typ=='open':
            op=1
            sl=0
        else:
            op=0
            sl=1
        G=np.zeros((len(defo),ln*wn))
        for i in range(len(xcs)):
            if self.typ=='open':
                xp=[xcs[i],ycs[i],-zcs[i],slength,swidth,op,self.strike,self.dip]
                #print(xp)
            else:
                xp=[xcs[i],ycs[i],-zcs[i],slength,swidth,sl,self.strike,self.dip,self.rake]
            defo=oki.forward(xp)
            G[:,i]=defo
        return G
    
    def get_laplacian(self):
        """
        Second order laplacian to regularized the patches in the sill/fault
        """
        ln=self.ln
        wn=self.wn
        xcs,ycs,zcs=self.get_centers()
        L=np.zeros((ln*wn,ln*wn))
        for i in range(len(xcs)):
            dist=(np.array(xcs)-xcs[i])**2+(np.array(ycs)-ycs[i])**2+(np.array(zcs)-zcs[i])**2
            pos=np.argsort(dist)
            if dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]] and dist[pos[1]]==dist[pos[4]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
                L[i,pos[4]]=1
            elif dist[pos[1]]==dist[pos[2]] and dist[pos[1]]==dist[pos[3]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
                L[i,pos[3]]=1
            elif dist[pos[1]]==dist[pos[2]]:
                L[i,pos[0]]=-2
                L[i,pos[1]]=1
                L[i,pos[2]]=1
        return self.lamb*L
    
    def model(self,x,y,*ops):
        """
        3d displacement produce by the discretized sill
          
        Parameters:
            ops: array that contains the opening/slip values for the patches
        """
        G=self.get_greens()
        model=np.array(ops)
        data=G@model
        
        ux=data[0:len(x)]
        uy=data[len(x):2*len(x)]
        uz=data[2*len(x):3*len(x)]
        
        return ux,uy,uz
        
    def plot_patches(self,ops,colormap='viridis'):
        """
        Auxiliary function to plot the opening/slip values
        
        Parameters:
            ops: array that contains the opening/slip values for the patches
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        
        ln=self.ln
        wn=self.wn
        
        reg_proj = Regdis(self.data,typ=self.typ,ln=self.ln,wn=self.wn,length=self.length,width=self.width)
        
        xs,ys,zs=reg_proj.get_centers()
        print('xs',xs)
        print('ys',ys)
        patches=[]
        fig, ax = plt.subplots()
        for i in range(len(xs)):
            rect = matplotlib.patches.Rectangle((xs[i]-self.width/(2*wn),ys[i]-self.length/(2*ln)), self.width/wn, self.length/ln)
            patches.append(rect)

        # values as numpy array
        values = ops

        # define the norm 
        norm = plt.Normalize(values.min(), values.max())
        coll = matplotlib.collections.PatchCollection(patches, cmap=colormap,
                                                      norm=norm, match_original = True)

        coll.set_array(values)
        polys = ax.add_collection(coll)
        fig.colorbar(polys, label='Opening(m)')

        plt.xlim(-self.width/2,self.width/2)
        plt.ylim(-self.length/2,self.length/2)

        plt.show()

    def get_reg_sill(self,*ops):
        xs,ys,zs=self.get_centers()
        oks=[]
        params=[]
        ops = ops[0]
        for i in range(len(xs)):
            oki = Okada(self.data)
            oki.set_type('open')
        #Initial parameters [xcen,ycen,depth,length,width,opening,strike,dip]
            oki.set_bounds(low_bounds = [0, 0, 1e3, 1e3, 1e3,10.0,1.0,1.0], high_bounds = [0, 0, 1e3, 1e3, 1e3,10.0,1.0,1.0])
            oks.append(oki)
            params+=[xs[i],ys[i],-zs[i],self.length/self.ln,self.width/self.wn,ops[i],self.strike,self.dip]
        return oks,params

    def transform_order_natural2regdis(self,natural_array):
        ln = self.ln
        wn = self.wn
        natural_matrix = natural_array.reshape((ln, wn))
        regdis_array = np.zeros_like(natural_array)
    
        counter = 0
        for base_col in range(wn // 2):
            for base_row in range(ln // 2):
                base_row_adjusted = ln - 1 - base_row
                regdis_array[counter] =  natural_matrix[base_row, wn - 1 - base_col]
                regdis_array[counter + 1] = natural_matrix[base_row, base_col]
                regdis_array[counter + 2] = natural_matrix[base_row_adjusted, base_col]
                regdis_array[counter + 3] = natural_matrix[base_row_adjusted, wn - 1 - base_col]
                counter += 4
        if wn % 2 != 0:
            center_col = wn // 2
            for base_row in range(ln // 2):
                base_row_adjusted = ln - 1 - base_row
                regdis_array[counter] = natural_matrix[base_row, center_col]
                regdis_array[counter + 1] = natural_matrix[base_row_adjusted, center_col]
                counter += 2
        if ln % 2 != 0:
            center_row = ln // 2
            for base_col in range(wn // 2):
                regdis_array[counter] = natural_matrix[center_row, base_col]
                regdis_array[counter + 1] = natural_matrix[center_row, wn - 1 - base_col]
                counter += 2
        if wn % 2 != 0 and ln % 2 != 0:
            regdis_array[counter] = natural_matrix[center_row, center_col]
        return regdis_array

    def transform_order_regdis2natural(self,regdis_array):
        ln = self.ln
        wn = self.wn
        natural_matrix = np.zeros_like(regdis_array.reshape((ln, wn)))

        counter = 0
        for base_col in range(wn // 2):
            for base_row in range(ln // 2):
                base_row = wn-1 - base_row
                natural_matrix[ln-1-base_row,wn-1-base_col] = regdis_array[counter]
                natural_matrix[ln-1-base_row,base_col] = regdis_array[counter+1]
                natural_matrix[base_row,base_col] = regdis_array[counter+2]
                natural_matrix[base_row,wn-1-base_col] = regdis_array[counter+3]
                counter += 4
        if wn % 2 != 0:
            center_col = wn // 2
            for base_row in range(ln // 2):
                base_row_adjusted = ln - 1 - base_row
                natural_matrix[base_row_adjusted, center_col] = regdis_array[counter+1]
                natural_matrix[base_row, center_col] = regdis_array[counter]
                counter += 2
        if ln % 2 != 0:
            center_row = ln // 2
            for base_col in range(wn // 2):
                natural_matrix[center_row, base_col] = regdis_array[counter+1]
                natural_matrix[center_row, wn - 1 - base_col] = regdis_array[counter]
                counter += 2
        if wn % 2 != 0 and ln % 2 != 0:
            natural_matrix[center_row, center_col] = regdis_array[counter]
        return natural_matrix.flatten()
        