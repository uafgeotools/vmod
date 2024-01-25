import numpy as np
from .. import util
from . import Source

class Cdm(Source):
    """
    A class used to represent a point source using the Mogi (1958) model.

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    """
    def get_source_id(self):
        """
        The function defining the name for the model.
          
        Returns:
            str: Name of the model.
        """
        return "Compound Dislocation Model"
    
    def bayesian_steps(self):
        """
        Function that defines the number of steps for a bayesian inversion.
        
        Returns:
            steps (int): Number of steps used in the bayesian inversions.
            burnin (int): discarded number of steps at the begining of the inversion.
            thin (int): number of steps per sample.
        """
        steps=1100000
        burnin=100000
        thin=1000
        
        return steps,burnin,thin

    def print_model(self, x):
        """
        The function prints the parameters for the model.
        
        Parameters:
           x (list): Parameters for the model.
        """
        print("Mogi")
        print("\tx = %f" % x[0])
        print("\ty = %f" % x[1])
        print("\td = %f" % x[2])
        print("\tw1= %f" % x[3])
        print("\tw2= %f" % x[3])
        print("\tw3= %f" % x[3])
        print("\tax= %f" % x[3])
        print("\tay= %f" % x[3])
        print("\taz= %f" % x[3])
        print("\topening= %f" % x[3])
        
    def set_parnames(self):
        """
        Function defining the names for the parameters in the model.
        """
        self.parameters=("xcen","ycen","depth","w1","w2","w3","ax","ay","az","opening")

    # =====================
    # Forward Models
    # =====================    
    def model(self,x,y,xcen,ycen,depth,wx,wy,wz,ax,ay,az,opening,nu=0.25):
        ax=2*ax
        ay=2*ay
        az=2*az
        
        wx=np.radians(wx)
        wy=np.radians(wy)
        wz=np.radians(wz)
        
        rx=np.vstack(([1,0,0],[0,np.cos(wx),np.sin(wx)],[0,-np.sin(wx),np.cos(wx)]))
        ry=np.vstack(([np.cos(wy),0,-np.sin(wy)],[0,1,0],[np.sin(wy),0,np.cos(wy)]))
        rz=np.vstack(([np.cos(wz),np.sin(wz),0],[-np.sin(wz),np.cos(wz),0],[0,0,1]))
        
        r=rx@ry@rz
        
        #print(r)
        
        p0=np.array([xcen,ycen,-depth])
        p1=p0+ay*r[:,1]/2+az*r[:,2]/2
        p2=p1-ay*r[:,1]
        p3=p2-az*r[:,2]
        p4=p1-az*r[:,2]
        
        q1=p0-ax*r[:,0]/2+az*r[:,2]/2
        q2=q1+ax*r[:,0]
        q3=q2-az*r[:,2]
        q4=q1-az*r[:,2]
        
        r1=p0+ax*r[:,0]/2+ay*r[:,1]/2
        r2=r1-ax*r[:,0]
        r3=r2-ay*r[:,1]
        r4=r1-ay*r[:,1]
        
        #print('p0',p0)
        #print('p1',p1)
        #print('p2',p2)
        #print('p3',p3)
        #print('p4',p4)
            
        #print(dsadsa)
        
        vert_vec=np.array([p1[2],p2[2],p3[2],p4[2],q1[2],q2[2],q3[2],q4[2],r1[2],r2[2], r3[2],r4[2]])
        
        if np.sum(vert_vec>0)>0:
            raise Exception('Half-space solution: The CDM must be under the free surface!')
            
        if ax==0 and ay==0 and az==0:
            ue=np.zeros(x.shape)
            un=np.zeros(x.shape)
            uv=np.zeros(x.shape)
        elif ax==0 and not ay==0 and not az==0:
            ue,un,uv=self.RD_disp_surf(x,y,p1,p2,p3,p4,opening,nu)
        elif not ax==0 and ay==0 and not az==0:
            ue,un,uv=self.RD_disp_surf(x,y,q1,q2,q3,q4,opening,nu)
        elif not ax==0 and not ay==0 and az==0:
            ue,un,uv=self.RD_disp_surf(x,y,r1,r2,r3,r4,opening,nu)
        else:
            ue1,un1,uv1=self.RD_disp_surf(x,y,p1,p2,p3,p4,opening,nu)
            ue2,un2,uv2=self.RD_disp_surf(x,y,q1,q2,q3,q4,opening,nu)
            ue3,un3,uv3=self.RD_disp_surf(x,y,r1,r2,r3,r4,opening,nu)
            ue=ue1+ue2+ue3
            un=un1+un2+un3
            uv=uv1+uv2+uv3
            
        return ue,un,uv
            
    def RD_disp_surf(self,x,y,p1,p2,p3,p4,opening,nu):
        bx=opening
        
        vnorm=np.cross(p2-p1,p4-p1)
        vnorm=vnorm/np.linalg.norm(vnorm)
        
        bxm=bx*vnorm[0]
        bym=bx*vnorm[1]
        bzm=bx*vnorm[2]
        
        u1,v1,w1 = self.ang_setup_fsc(x,y,bxm,bym,bzm,p1,p2,nu); # Side P1P2
        u2,v2,w2 = self.ang_setup_fsc(x,y,bxm,bym,bzm,p2,p3,nu); # Side P2P3
        u3,v3,w3 = self.ang_setup_fsc(x,y,bxm,bym,bzm,p3,p4,nu); # Side P3P4
        u4,v4,w4 = self.ang_setup_fsc(x,y,bxm,bym,bzm,p4,p1,nu); # Side P4P1
        
        ue=u1+u2+u3+u4
        un=v1+v2+v3+v4
        uv=w1+w2+w3+w4
        
        return ue,un,uv
    
    def coord_trans(self,x1,x2,x3,A):
        
        B=np.vstack((x1,x2,x3))
        
        r=A@B
        
        x1m=r[0,:]
        x2m=r[1,:]
        x3m=r[2,:]
        
        return x1m,x2m,x3m
    
    def ang_setup_fsc(self,x,y,bx,by,bz,pa,pb,nu):
        side_vec=pb-pa
        
        eZ = np.array([0,0,1])
        beta = np.arccos(-np.sum(side_vec*eZ)/np.linalg.norm(side_vec))
        
        eps=np.finfo(float).eps
        
        if abs(beta)<eps or abs(np.pi-beta)<eps:
            ue = np.zeros(x.shape)
            un = np.zeros(x.shape)
            uv = np.zeros(x.shape)
        else:
            ey1 = np.array([side_vec[0],side_vec[1],0])
            ey1 = ey1/np.linalg.norm(ey1)
            ey3 = -eZ
            ey2 = np.cross(ey3,ey1);
            A = np.hstack((ey1.reshape((3,1)),ey2.reshape((3,1)),ey3.reshape((3,1)))) # Transformation matrix
            
            #print('ey1',ey1)
            #print('ey2',ey2)
            #print('ey3',ey3)
            #print('A',A)
            
            #print(dsadsa)
            
            y1a,y2a,nada=self.coord_trans(x-pa[0],y-pa[1],np.zeros(x.shape)-pa[2],A)
            y1ab,y2ab,nada=self.coord_trans(side_vec[0],side_vec[1],side_vec[2],A)
            
            y1b=y1a-y1ab
            y2b=y2a-y2ab
            
            
            
            b1,b2,b3=self.coord_trans(bx,by,bz,A)
            
            cond=((beta*y1a)>=0)
            
            v1a=np.ones(cond.shape)*0
            v2a=np.ones(cond.shape)*0
            v3a=np.ones(cond.shape)*0
            
            v1a[cond],v2a[cond],v3a[cond]=self.ang_dis_disp_surf(y1a[cond],y2a[cond],-np.pi+beta,b1,b2,b3,nu,-pa[2])
            
            v1b=np.ones(cond.shape)*0
            v2b=np.ones(cond.shape)*0
            v3b=np.ones(cond.shape)*0
            
            v1b[cond],v2b[cond],v3b[cond]=self.ang_dis_disp_surf(y1b[cond],y2b[cond],-np.pi+beta,b1,b2,b3,nu,-pb[2])
            
            v1a[~cond],v2a[~cond],v3a[~cond]=self.ang_dis_disp_surf(y1a[~cond],y2a[~cond],beta,b1,b2,b3,nu,-pa[2])
            
            v1b[~cond],v2b[~cond],v3b[~cond]=self.ang_dis_disp_surf(y1b[~cond],y2b[~cond],beta,b1,b2,b3,nu,-pb[2])
            
            v1=v1b-v1a
            v2=v2b-v2a
            v3=v3b-v3a
            
            
            ue,un,uv=self.coord_trans(v1,v2,v3,A.T)
        
        return ue,un,uv
    
    def ang_dis_disp_surf(self,y1,y2,beta,b1,b2,b3,nu,a):
        sinB=np.sin(beta)
        cosB=np.cos(beta)
        cotB=1.0/np.tan(beta)
        
        z1 = y1*cosB+a*sinB
        z3 = y1*sinB-a*cosB
        r2 = y1**2+y2**2+a**2
        r = np.sqrt(r2)
        
        Fi = 2*np.arctan2(y2,(r+a)*(1/np.tan(beta/2))-y1) # The Burgers function
        
        v1b1 = b1/2/np.pi*((1-(1-2*nu)*cotB**2)*Fi+y2/(r+a)*((1-2*nu)*(cotB+y1/2/(r+a))-y1/r)-y2*(r*sinB-y1)*cosB/r/(r-z3))
        v2b1 = b1/2/np.pi*((1-2*nu)*((0.5+cotB**2)*np.log(r+a)-cotB/sinB*np.log(r-z3))-1.0/(r+a)*((1-2*nu)*(y1*cotB-a/2-y2**2/2/(r+a))+(y2**2)/r)+(y2**2)*cosB/r/(r-z3))
        v3b1 = b1/2/np.pi*((1-2*nu)*Fi*cotB+y2/(r+a)*(2*nu+a/r)-y2*cosB/(r-z3)*(cosB+a/r))
        
        v1b2 = b2/2/np.pi*(-(1-2*nu)*((0.5-cotB**2)*np.log(r+a)+cotB**2*cosB*np.log(r-z3))-1/(r+a)*((1-2*nu)*(y1*cotB+0.5*a+(y1**2)/2/(r+a))-y1**2/r)+z1*(r*sinB-y1)/r/(r-z3))
        v2b2 = b2/2/np.pi*((1+(1-2*nu)*cotB**2)*Fi-y2/(r+a)*((1-2*nu)*(cotB+y1/2/(r+a))-y1/r)-y2*z1/r/(r-z3))
        v3b2 = b2/2/np.pi*(-(1-2*nu)*cotB*(np.log(r+a)-cosB*np.log(r-z3))-y1/(r+a)*(2*nu+a/r)+z1/(r-z3)*(cosB+a/r))
        
        v1b3 = b3/2/np.pi*(y2*(r*sinB-y1)*sinB/r/(r-z3))
        v2b3 = b3/2/np.pi*(-y2**2*sinB/r/(r-z3))
        v3b3 = b3/2/np.pi*(Fi+y2*(r*cosB+a)*sinB/r/(r-z3))
        
        v1 = v1b1+v1b2+v1b3
        v2 = v2b1+v2b2+v2b3
        v3 = v3b1+v3b2+v3b3
        
        return v1,v2,v3