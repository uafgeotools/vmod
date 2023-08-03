# -*- coding: utf-8 -*-
"""
General utility functions to accompany vmodels (mogi, yang, okada...)
Created on Tue Jun 21 14:59:15 2016

@author: scott
"""
import numpy as np
import numpy.ma as ma
import time
import math
import os
import subprocess
import utm
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import h5py
from IPython.display import Markdown
from IPython.display import display
import matplotlib
from scipy import stats
from scipy.interpolate import interp1d
from global_land_mask import globe
from skimage.restoration import denoise_nl_means, estimate_sigma

def psi_phi(h):
    t,w=gauleg(0,1,41)
    t=np.array(t)
    g=-2.0*t/np.pi
    d=np.concatenate((g,np.zeros(g.size)))
    T1,T2,T3,T4=giveT(h,t,t)
    T1p=np.zeros(T1.shape)
    T2p=np.zeros(T1.shape)
    T3p=np.zeros(T1.shape)
    T4p=np.zeros(T1.shape)
    N=t.size
    for j in range(N):
        T1p[:,j]=w[j]*T1[:,j]
        T2p[:,j]=w[j]*T2[:,j]
        T3p[:,j]=w[j]*T3[:,j]
        T4p[:,j]=w[j]*T4[:,j]
    M1=np.concatenate((T1p,T3p),axis=1)
    M2=np.concatenate((T4p,T2p),axis=1)
    Kp=np.concatenate((M1,M2),axis=0)
    y=np.matmul(np.linalg.inv(np.eye(2*N,2*N)-(2/np.pi)*Kp),d)
    phi=y[0:N]
    psi=y[N:2*N]
    return phi,psi,t,w

def giveP(h,x):
    P=np.zeros((4,x.size))
    P[0]=(12*np.power(h,2)-np.power(x,2))/np.power((4*np.power(h,2)+np.power(x,2)),3)
    P[1] = np.log(4*np.power(h,2)+np.power(x,2)) + (8*np.power(h,4)+2*np.power(x,2)*np.power(h,2)-np.power(x,4))/np.power(4*np.power(h,2)+np.power(x,2),2)
    P[2] = 2*(8*np.power(h,4)-2*np.power(x,2)*np.power(h,2)+np.power(x,4))/np.power(4*np.power(h,2)+np.power(x,2),3)
    P[3] = (4*np.power(h,2)-np.power(x,2))/np.power((4*np.power(h,2)+np.power(x,2)),2)
    return P

def giveT(h,t,r):
    M = t.size
    N = r.size
    T1 = np.zeros((M,N)) 
    T2 = np.zeros((M,N)) 
    T3 = np.zeros((M,N))
    for i in range(M):
        Pm=giveP(h,t[i]-r)
        Pp=giveP(h,t[i]+r)
        T1[i] = 4*np.power(h,3)*(Pm[0,:]-Pp[0,:])
        T2[i] = (h/(t[i]*r))*(Pm[1,:]-Pp[1,:]) +h*(Pm[2,:]+Pp[2,:])
        T3[i] = (np.power(h,2)/r)*(Pm[3,:]-Pp[3,:]-2*r*((t[i]-r)*Pm[0,:]+(t[i]+r)*Pp[0,:]))
    T4=np.copy(T3.T)
    return T1,T2,T3,T4

def legpol(x,N):
    dim=x.size
    if not dim==1:
        dP=np.zeros((N,dim))
        P=np.zeros((N+1,dim))
        P[0]=np.ones(dim)
    else:
        dP=np.zeros(N)
        P=np.zeros(N+1)
        P[0]=1.0
    P[1]=x
    for j in range(1,N):
        P[j+1] = ((2*j+1)*x*P[j] - j*P[j-1])/(j+1)
        dP[j] = j*(x*P[j] - P[j-1])/(np.power(x,2)-1)
    return P[N-1],dP[N-1]

def legendre(n,x):
    if n==0:
        val2 = 1.
        dval2 = 0.
    elif n==1:
        val2 = x
        dval2 = 1.
    else:
        val0 = 1.; val1 = x
        for j in range(1,n):
            val2 = ((2*j+1)*x*val1 - j*val0)/(j+1)
            val0, val1 = val1, val2
        dval2 = n*(val0-x*val1)/(1.-x**2)
    return val2, dval2

def legnewton(n,xold,kmax=200,tol=1.e-8):
    for k in range(1,kmax):
        val, dval = legendre(n,xold)
        xnew = xold - val/dval

        xdiff = xnew - xold
        if abs(xdiff/xnew) < tol:
            break

        xold = xnew
    else:
        xnew = None
    return xnew

def legroots(n):
    roots = np.zeros(n)
    npos = n//2
    for i in range(npos):
        xold = np.cos(np.pi*(4*i+3)/(4*n+2))
        root = legnewton(n,xold) 
        roots[i] = -root
        roots[-1-i] = root
    return roots

def gauleg_params(n):
    xs = legroots(n)
    cs = 2/((1-xs**2)*legendre(n,xs)[1]**2)
    return xs, cs

def gauleg(a,b,n):
    xs, cs = gauleg_params1(n)
    coeffp = 0.5*(b+a) 
    coeffm = 0.5*(b-a)
    ts = coeffp - coeffm*xs
    ws=cs*coeffm
    #contribs = cs*f(ts)
    #return coeffm*np.sum(contribs)
    return ts[::-1],ws

'''
def gauleg(x1,x2,N):
    eps=1e-8
    z=np.zeros(N)
    xm = 0.5*(x2+x1)
    xl = 0.5*(x2-x1)
    for n in range(N):
        z[n] = np.cos(np.pi*((n+1)-0.25)/(N+0.5))
        z1 = 100*z[n]
        while np.abs(z1-z[n])>eps:
            pN,dpN=legpol(z[n],N+1)
            z1=z[n]
            z[n]=z1-pN/dpN
    pN,dpN=legpol(z,N+1)
    x=xm-xl*z
    w = 2*xl/((1-np.power(z,2))*np.power(dpN,2))
    return x,w
'''

def gauleg_params1(n):
    xs,cs=np.polynomial.legendre.leggauss(n)
    return xs,cs

def f(x):
    return 1/np.sqrt(x**2 + 1)

def world2rc(x,y,affine, inverse=False):
    '''
    World coordinates (lon,lat) to image (row,col) center pixel coordinates
    '''
    import rasterio
    #NOTE: src.xy() does this I think...
    #T0 = src.meta['affine']
    T0 = affine
    T1 = T0 * rasterio.Affine.translation(0.5, 0.5)
    rc2xy = lambda r, c: (c, r) * T1
    # can probable simpligy,,, also int() acts like floor()
    xy2rc = lambda x, y: [int(i) for i in [x, y] * ~T1][::-1]

    if inverse:
        return rc2xy(y,x)
    else:
        return xy2rc(x,y)


def save_rasterio(path, data, profile):
    '''
    save single band raster file
    intended to use with load_rasterio() to open georeferenced data manipulate
    with numpy and then resave modified data
    '''
    import rasterio
    with rasterio.drivers():
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1) #single band


def load_rasterio(path):
    '''
    load single bad georeference data as 'f4', convert NoDATA to np.nan
    not sure this works with 'scale' in vrt
    '''
    import rasterio
    with rasterio.drivers():
        with rasterio.open(path, 'r') as src:
            data = src.read()
            meta = src.profile
            extent = src.bounds[::2] + src.bounds[1::2]

    return data, extent, meta


def load_cor_mask(path='phsig.cor.8alks_8rlks.geo.vrt', corthresh=0.1):
    '''
    load geocoded correlation file to use as mask
    '''
    import rasterio
    cordata, extent, meta = load_rasterio(path)
    cor = cordata[0]
    # Geocoding seems to create outliers beyond realistic range (maybe interpolation)
    ind_outliers = (np.abs(cor) > 1)
    cor[ind_outliers] = 0.0
    # Can go further and remove pixels with low coherence (or just set to 0.0)
    mask = (cor < corthresh)
    #data[mask] = np.nan

    return mask


def calc_ramp(array, ramp='quadratic', custom_mask=None):
    '''
    Remove a quadratic surface from the interferogram Subtracting
    the best-fit quadratic surface forces the background mean surface
    displacement to be zero

    Note: exclude known signal, unwrapping errors, etc. with custom_mask

    ramp = 'dc','linear','quadratic'

    returns ramp
    '''
    X,Y = np.indices(array.shape)
    x = X.reshape((-1,1))
    y = Y.reshape((-1,1))

    # Work with numpy mask array
    phs = np.ma.masked_invalid(array)

    if custom_mask != None:
        phs[custom_mask] = ma.masked

    d = phs.reshape((-1,1))
    g = ~d.mask
    dgood = d[g].reshape((-1,1))

    if ramp == 'quadratic':
        print('fit quadtratic surface')
        G = np.concatenate([x, y, x*y, x**2, y**2, np.ones_like(x)], axis=1) #all pixels
        Ggood = np.vstack([x[g], y[g], x[g]*y[g], x[g]**2, y[g]**2, np.ones_like(x[g])]).T
        try:
            m,resid,rank,s = np.linalg.lstsq(Ggood,dgood)
        except ValueError as ex:
            print('{}: Unable to fit ramp with np.linalg.lstsq'.format(ex))


    elif ramp == 'linear':
        print('fit linear surface')
        G = np.concatenate([x, y, x*y, np.ones_like(x)], axis=1)
        Ggood = np.vstack([x[g], y[g], x[g]*y[g], np.ones_like(x[g])]).T
        try:
            m,resid,rank,s = np.linalg.lstsq(Ggood,dgood)
        except ValueError as ex:
            print('{}: Unable to fit ramp with np.linalg.lstsq'.format(ex))

    elif ramp == 'dc':
        G = np.ones_like(array)
        m = np.mean(phs)
        print('fit dc offset')

    ramp = np.dot(G,m)
    ramp = ramp.reshape(phs.shape)

    return ramp

def get_enu2los(enuFile='enu.rdr.geo'):
    '''
    Get conversion of cartesian ground displacements to radar LOS

    To avoid incidence and heading convention issues, assume 3 band GDAL file with
    conversion factors. Convension used is RECENT MASTER - OLDER SLAVE such that
    positive phase in interferogram is uplift.

    Example:
    model = np.array([ux,uy,uz])
    enu2los = get_enu2los('enu.rdr.geo')
    dlos = model*enu2los

    To generate this file with ISCE:
    imageMath.py --eval='sin(rad(a_0))*cos(rad(a_1+90)); sin(rad(a_0)) * sin(rad(a_1+90)); cos(rad(a_0))' --a=los.rdr.geo -t FLOAT -s BIL -o enu.rdr.geo
    imageMath.py --eval='a_0*b_0;a_1*b_1;a_2*b_2' --a=enu.rdr.geo --b=model.geo -t FLOAT -o model_LOS.geo
    '''
    import rasterio
    data,junk,junk = load_rasterio(enuFile)
    data[data==0] = np.nan
    e2los,n2los,u2los = data
    # NOTE: some sort of bug in conversion code why is there z=1 in z2los
    u2los[u2los==1] = np.nan

    cart2los = np.dstack([e2los, n2los, u2los])
    return cart2los


def get_cart2los(incidence,heading):
    '''
    coefficients for projecting cartesian displacements into LOS vector
    assuming convention of Hannsen text figure 5.1 where angles are clockwise +
    relative to north!
    is Azimuth look direction (ALD=heading-270)
    vm.util.get_cart2los(23,190)

    ISCE descending los file has heading=-100
    '''
    incidence = np.deg2rad(incidence)
    ALD = np.deg2rad(heading-270)

    EW2los = -np.sin(ALD) * np.sin(incidence)
    NS2los = -np.cos(ALD) * np.sin(incidence)
    Z2los = np.cos(incidence)

    cart2los = np.dstack([EW2los, NS2los, Z2los])

    return cart2los



def cart2pol(x1,x2):
    #theta = np.arctan(x2/x1)
    theta = np.arctan2(x2,x1) #sign matters -SH
    r = np.sqrt(x1**2 + x2**2)
    return theta, r

def pol2cart(theta,r):
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1,x2

def shift_utm(X,Y,xcen,ycen):
    '''
    Avoid large numbers in UTM grid by creating local (0,0) origin
    '''
    x0 = X.min()
    y0 = Y.min()

    X = X - x0
    Y = Y - y0
    xcen = xcen - x0
    ycen = ycen - y0

    return X,Y,xcen,ycen


def get_cart2los_bak(inc, ald, x):
    '''
    NOTE: possible sign convention issues with this function...
    '''
    # x is data array
    # converted to LOS
    # los = data.dot(cart2los) * 1e2 # maybe use numpy.tensordot? not sure...
    # For now fake it:
    look = np.deg2rad(inc) * np.ones_like(x)  # incidence
    # heading (degreees clockwise from north)
    head = np.deg2rad(ald) * np.ones_like(x)
    # NOTE: matlab code default is -167 'clockwise from north' (same as hannsen text fig 5.1)
    # This is for descending envisat beam 2, asizmuth look direction (ALD) is
    # perpendicular to heading (-77)

    # however, make_los.pl generates unw file with [Incidence, ALD], ALD for ascending data is 77
    # make_los.pl defines "(alpha) azimuth pointing of s/c"
    EW2los = np.sin(head) * np.sin(look)
    NS2los = np.cos(head) * np.sin(look)
    Z2los = -np.cos(look)
    # NOTE: negative here implies uplift=positive in LOS
    cart2los = -np.dstack([EW2los, NS2los, Z2los])

    return cart2los

def write_rsc(los,az,lk,wl,extent,units):
    stepx=(extent[1]-extent[0])/los.shape[1]
    stepy=(extent[2]-extent[3])/los.shape[0]
    if units=='m':
        unit='meters'
    elif units=='deg':
        unit='degrees'
    files=['vmod/varres/insar.unw.rsc','vmod/varres/incidence.unw.rsc']
    for archivo in files:
        rsc=open(archivo,'w')
        rsc.write('FILE_LENGTH '+str(los.shape[0])+'\n')
        rsc.write('WIDTH '+str(los.shape[1])+'\n')
        rsc.write('X_FIRST '+str(extent[0])+'\n')
        rsc.write('X_STEP '+str(stepx)+'\n')
        rsc.write('X_UNIT '+unit+'\n')
        rsc.write('Y_FIRST '+str(extent[3])+'\n')
        rsc.write('Y_STEP '+str(stepy)+'\n')
        rsc.write('Y_UNIT '+unit+'\n')
        rsc.write('WAVELENGTH '+str(wl))
        rsc.close()

def write_unw(los,az,lk):
    nxx=los.reshape(los.shape).shape[1]
    nyy=los.reshape(los.shape).shape[0]
    amp=np.ones((nyy,nxx),dtype=np.float32)
    inp=np.zeros((nyy,2*nxx),dtype=np.float32)
    inp[:,0:nxx]=amp
    inp[:,nxx:]=los
    archivo=open('vmod/varres/insar.unw','wb')
    inp.tofile(archivo)
    archivo.close()
    if isinstance(az,float):
        inp[:,0:nxx]=amp*(az)
    else:
        inp[:,0:nxx]=az
    if isinstance(lk,float):
        inp[:,0:nxx]=amp*(lk)
    else:
        inp[:,0:nxx]=lk
    inp=inp.astype(np.float32)
    archivo=open('vmod/varres/incidence.unw','wb')
    inp.tofile(archivo)
    archivo.close()
    
def ll2utm(lons,lats,z1=None,z2=None):
    xs,ys=[],[]
    for i in range(len(lats)):
        if z1 is None:
            x,y,z1,z2=utm.from_latlon(lats[i], lons[i])
        else:
            x,y,z1,z2=utm.from_latlon(lats[i], lons[i],force_zone_number=z1, force_zone_letter=z2)
        xs.append(x)
        ys.append(y)
    xs=np.array(xs)
    ys=np.array(ys)
    return xs,ys,z1,z2

def utm2ll(xs,ys,z1,z2):
    lons,lats=[],[]
    for i in range(len(xs)):
        lat,lon=utm.to_latlon(xs[i], ys[i], z1, z2)
        lons.append(lon)
        lats.append(lat)
    lons=np.array(lons)
    lats=np.array(lats)
    return lons,lats

def get_quadtree(ref,az,lk,name='quadtree.txt',th=None):
    im=ref.dataset
    if th is None:
        th=np.nanvar(im)/50
    
    quadtree_var(im,az,lk,ref.extent,th,name)

'''
def get_quadtree(ref,az,lk,name='quadtree.txt',per=100):
    get_quadtree2(ref,az,lk,per)
    
    
    
    archivo=open('./vmod/varres/newtest.txt','r')
    lineas=archivo.readlines()
    archivo.close()

    lns,lts,cats=[],[],[]
    for i,linea in enumerate(lineas[2::]):
        lns.append(float(linea.split()[3]))
        lts.append(float(linea.split()[4]))
        cats.append(i)
    lns=np.array(lns)
    lts=np.array(lts)
    cats=np.array(cats)
    
    lons=np.linspace(ref.extent[0],ref.extent[1],ref.dataset.shape[1])
    lats=np.linspace(ref.extent[2],ref.extent[3],ref.dataset.shape[0])[::-1]
    LATS,LONS=np.meshgrid(lons,lats)
    
    quadobs=np.empty(ref.dataset.shape)
    quadobs[:,:]=np.nan
    for i in range(len(lons)):
        for j in range(len(lats)):
            minn=np.argmin((lns-lons[i])**2+(lts-lats[j])**2)
            quadobs[j,i]=cats[minn]
    quadobs[np.isnan(ref.dataset)]=np.nan
            
    result=open(name,'w')
    result.write('%Reference (Lon,Lat): '+str(ref.xref)+','+str(ref.yref)+'\n')
    for i in range(len(cats)):
        cond=quadobs==cats[i]
        nrows=5
        ncols=5
        inrow=int((nrows-1)/2)
        incol=int((ncols-1)/2)
        if len(ref.dataset[cond])<ncols*nrows:
            std=0
            while(std==0):
                rows,cols=np.nonzero(cond)
                row,col=rows[0],cols[0]
                if row-nrows<0:
                    row=int((nrows-1)/2)
                elif row+nrows>=quadobs.shape[0]:
                    row=(quadobs.shape[0]-1)-int((nrows-1)/2)
                if col-ncols<0:
                    col=int((ncols-1)/2)
                elif col+ncols>=quadobs.shape[1]:
                    col=(quadobs.shape[1]-1)-int((ncols-1)/2)

                value=np.nanmedian(ref.dataset[row-inrow:row+inrow,col-incol:col+incol])
                std=np.nanstd(ref.dataset[row-inrow:row+inrow,col-incol:col+incol])
                nrows=2*nrows+1
                ncols=2*ncols+1
                inrow=int((nrows-1)/2)
                incol=int((ncols-1)/2)
        else:
            value=np.nanmedian(ref.dataset[cond])
            std=np.nanstd(ref.dataset[cond])
        lon=lns[i]
        lat=lts[i]
        azm=np.mean(az[cond])
        inc=np.mean(lk[cond])
        wgt=np.sum(quadobs==cats[i])
        if std==0:
            print('std=0',i,wgt)
        elif std/float(wgt)<1e-9:
            print('stdw',i,std,wgt)
        
        line="%3.6f %3.6f %1.6f %1.6f %1.6f %1.9f\n"\
            % (lon,lat,azm,inc,value,std/float(wgt))
        
        result.write(line)
    result.close()

def rewrite_csv(xs,ys,azs,lks,los,elos,ref,name='output.txt'):
    result=open(name,'w')
    result.write('%Reference (Lon,Lat): '+str(ref[0])+','+str(ref[1])+'\n')
    for i in range(len(xs)):
        line="%3.6f %3.6f %1.6f %1.6f %1.6f %1.9f\n"\
            % (xs[i],ys[i],azs[i],lks[i],los[i],elos[i])
        result.write(line)
    result.close()
    rewrite_csv(self.los,[self.xref,self.yref],ori=csvfile,name=csvfile.split('.')[0]+'_ref.'+csvfile.split('.')[1])
'''
def rewrite_csv(los,ref,old,name='output.txt'):
    archivo=open(old,'r')
    lines=archivo.readlines()
    archivo.close()
    
    result=open(name,'w')
    result.write(lines[0].split(':')[0]+': '+str(ref[0])+','+str(ref[1])+', Dimensions:'+lines[0].split('Dimensions:')[1])
    for i in range(len(lines)-1):
        i+=1
        linef=lines[i].split()
        line="%6.3f %6.3f %1.6f %1.6f %1.6f %1.9f %5.0f %5.0f %5.0f %5.0f\n"\
                % (float(linef[0]),float(linef[1]),float(linef[2]),float(linef[3]),los[i-1],float(linef[5]),float(linef[6]),float(linef[7]),float(linef[8]),float(linef[9]))
        
        result.write(line)
    result.close()

def get_quadtree2(ref,az,lk,per=100,th=1e-10,unit='deg',name='newtest'):
    los=np.copy(ref.dataset)
    los[np.isnan(los)]=0
    extent=ref.extent
    wl=ref.wl
    
    quadtree(los,az,lk,extent,per,unit,th,wl,name)

def quadtree(los,az,lk,extent,per=100,unit='deg',th=1e-10,wl=1.0,name='newtest'):
    eths=[np.log10(th)]
    if not per==100:
        eths=np.linspace(-8,0,9)[::-1]
    
    write_unw(los,az,lk)
    write_rsc(los,az,lk,wl,extent,unit)
    
    ini=run_decompose(th)
    
    for i,eth in enumerate(eths):
        th=10**eth
        tam=run_decompose(th)
        
        if i==0:
            an=tam
            opt=ini*per/100
            low=tam
        if not i==0: 
            if not an==tam:
                anp=tam
                if per/100<tam/ini:
                    anp=run_decompose(10**eths[i-1])
                    ths=np.sort(10**np.linspace(eths[i],eths[i-1],7))[::-1]
                else:
                    ths=np.sort(10**np.linspace(eths[i+1],eths[i],7))[::-1]
                pers=[]
                thts=[]
                for j,th in enumerate(ths):
                    anc=run_decompose(th)
                    if anc not in [low,ini]:
                        pers.append(anc/ini)
                        thts.append(th)
                
                f2 = interp1d(pers, thts, kind='slinear')
                optth=f2(per/100)
                
                if optth>0:
                    anc=run_decompose(optth)
                else:
                    anc=np.inf
                '''    
                if not ((anc/ini-0.05)<=per/100 and (anc/ini+0.05)>=per/100):
                    pos=np.argmin(np.abs(per/100-np.array(pers)))
                    if pers[pos]>(per/100):
                        if pos==0:
                            nths=np.linspace(thts[pos]/10,thts[pos],5)
                        else:
                            nths=np.linspace(thts[pos-1],thts[pos],5)
                    else:
                        if pos==len(pers)-1:
                            nths=np.linspace(thts[pos],thts[pos]*10,5)
                        else:
                            nths=np.linspace(thts[pos],thts[pos+1],5)
                    for k,th in enumerate(nths):
                        anc=run_decompose(th)
                        if anc not in [low,ini]:
                            pers.append(anc/ini)
                            thts.append(th)
                    print(pers,thts)
                    f2 = interp1d(pers, thts, kind='cubic')
                    optth=f2(per/100)
                    anc=run_decompose(optth)
                '''
                return


def quadtree_var(im,az,inc,extent,th,name='quadtree.txt',ref=None,denoise=True):
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=15,  # 13x13 search area
                    multichannel=True)

    imcp=np.copy(im)
    imcp[np.isnan(im)]=0
    
    
    if denoise:
        imfil=denoise_nl_means(imcp, h=0.6 * 0.5, sigma=0.5, fast_mode=True, **patch_kw)
    else:
        imfil=np.copy(imcp)

    imfil[imcp==0]=np.nan
    
    
    
    fverts=[]
    pointsx=[]
    pointsy=[]
    
    
    def quadtree_im(im,inverts,th):
        #if im.shape[0]*im.shape[1]<=10 or np.sum(np.isnan(im))>=0.9*im.shape[0]*im.shape[1] or np.nanvar(im)<=th:
        #if np.sum(np.isnan(im))>=0.9*im.shape[0]*im.shape[1] or np.nanvar(im)<=th:
        #if np.sum(np.isnan(im))<=10 or np.sum(np.isnan(im))>=0.9*im.shape[0]*im.shape[1] or np.nanvar(im)<=th:
        #if im.shape[0]*im.shape[1]<=10 or np.sum(np.isnan(im))>=0.9*im.shape[0]*im.shape[1] or np.nanvar(im)<=th:
        if im.shape[0]*im.shape[1]<=10 or np.sum(np.logical_not(np.isnan(im)))<2 or np.nanvar(im)<=th:
            y=np.arange(im.shape[0]).astype(float)+inverts[0]
            x=np.arange(im.shape[1]).astype(float)+inverts[1]
            xx,yy=np.meshgrid(x,y)
            xx[np.isnan(im)]=np.nan
            yy[np.isnan(im)]=np.nan
            if not np.isnan(np.nanmean(xx)):
                pointsx.append(int(np.nanmean(xx)))
                pointsy.append(int(np.nanmean(yy)))
                fverts.append((inverts[0],inverts[0]+im.shape[0],inverts[1],inverts[1]+im.shape[1]))

        else:
            halfr=int(im.shape[0]/2)
            halfc=int(im.shape[1]/2)
            quadtree_im(im[0:halfr,0:halfc],inverts,th)
            quadtree_im(im[halfr::,0:halfc],[inverts[0]+halfr,inverts[1]],th)
            quadtree_im(im[0:halfr,halfc::],[inverts[0],inverts[1]+halfc],th)
            quadtree_im(im[halfr::,halfc::],[inverts[0]+halfr,inverts[1]+halfc],th)
            
    quadtree_im(imfil,[0,0],th)
    
    print('Final samples: ',len(fverts))
    ar=open(name,'w')
    if ref is None:
        ar.write('%Reference (Lon,Lat): None, Dimensions: '+str(im.shape[0])+','+str(im.shape[1])+', Extent: '+','.join(str(v) for v in extent)+'\n')
    else:
        ar.write('%Reference (Lon,Lat): '+str(ref[0])+','+str(ref[1])+', Dimensions: '+str(im.shape[0])+','+str(im.shape[1])+', Extent: '+','.join(extent)+'\n')
    xcoords=np.linspace(extent[0],extent[1],im.shape[1])
    ycoords=np.linspace(extent[2],extent[3],im.shape[0])[::-1]
    for i,fvert in enumerate(fverts):
        mean=np.nanmean(im[fvert[0]:fvert[1],fvert[2]:fvert[3]])
        azmean=np.nanmean(az[fvert[0]:fvert[1],fvert[2]:fvert[3]])
        incmean=np.nanmean(inc[fvert[0]:fvert[1],fvert[2]:fvert[3]])
        std=calc_std(im,fvert)
        line="%6.3f %6.3f %1.6f %1.6f %1.6f %1.9f %5.0f %5.0f %5.0f %5.0f\n"\
                % (xcoords[pointsx[i]],ycoords[pointsy[i]],azmean,incmean,mean,std,fvert[0],fvert[1],fvert[2],fvert[3])
        ar.write(line)
    ar.close()

def calc_std(mat,verts):
    std = np.nanstd(mat[verts[0]:verts[1],verts[2]:verts[3]]) / np.sum(np.logical_not(np.isnan(mat[verts[0]:verts[1],verts[2]:verts[3]])))
    
    if not std==0:
        return std
    
    if verts[0]==0 and not verts[2]==0:
        newverts=[verts[0],verts[1]+2,verts[2]-1,verts[3]+1]
    elif verts[0]==0 and verts[2]==0:
        newverts=[verts[0],verts[1]+2,verts[2],verts[3]+2]
    elif verts[2]==0 and not verts[0]==0:
        newverts=[verts[0]-1,verts[1]+1,verts[2],verts[3]+2]
    elif verts[1]==mat.shape[0]-1 and not verts[3]==mat.shape[1]-1:
        newverts=[verts[0]-2,verts[1],verts[2]-1,verts[3]+1]
    elif verts[1]==mat.shape[0]-1 and verts[3]==mat.shape[1]-1:
        newverts=[verts[0]-2,verts[1],verts[2]-2,verts[3]]
    elif verts[3]==mat.shape[1]-1 and not verts[1]==mat.shape[0]-1:
        newverts=[verts[0]-1,verts[1]+1,verts[2]-2,verts[3]]
    else:
        newverts=[verts[0]-1,verts[1]+1,verts[2]-1,verts[3]+1]
        
    return calc_std(mat,newverts)
    
    
def mat2quad(los,az,lk,extent,name,per=100,wl=1.0,unit='deg',ref=None):
    quadtree(los,az,lk,extent,per,unit,th=1e-10,wl=1.0,name='newtest')
    archivo=open('./vmod/varres/newtest.txt','r')
    lineas=archivo.readlines()
    archivo.close()

    lns,lts,cats=[],[],[]
    for i,linea in enumerate(lineas[2::]):
        lns.append(float(linea.split()[3]))
        lts.append(float(linea.split()[4]))
        cats.append(i)
    lns=np.array(lns)
    lts=np.array(lts)
    cats=np.array(cats)
    
    lons=np.linspace(extent[0],extent[1],los.shape[1])
    lats=np.linspace(extent[2],extent[3],los.shape[0])[::-1]
    LATS,LONS=np.meshgrid(lons,lats)
    
    quadobs=np.empty(los.shape)
    quadobs[:,:]=np.nan
    for i in range(len(lons)):
        for j in range(len(lats)):
            minn=np.argmin((lns-lons[i])**2+(lts-lats[j])**2)
            quadobs[j,i]=cats[minn]
    quadobs[np.isnan(los)]=np.nan
            
    result=open(name,'w')
    if unit=='deg' and ref is not None:
        result.write('%Reference (Lon,Lat): '+str(ref[0])+','+str(ref[1])+'\n')
    elif unit=='m' and ref is not None:
        result.write('%Reference (x,y): '+str(ref[0])+','+str(ref[1])+'\n')
    for i in range(len(cats)):
        cond=quadobs==cats[i]
        nrows=5
        ncols=5
        inrow=int((nrows-1)/2)
        incol=int((ncols-1)/2)
        if len(los[cond])<ncols*nrows:
            std=0
            while(std==0):
                rows,cols=np.nonzero(cond)
                row,col=rows[0],cols[0]
                if row-nrows<0:
                    row=int((nrows-1)/2)
                elif row+nrows>=quadobs.shape[0]:
                    row=(quadobs.shape[0]-1)-int((nrows-1)/2)
                if col-ncols<0:
                    col=int((ncols-1)/2)
                elif col+ncols>=quadobs.shape[1]:
                    col=(quadobs.shape[1]-1)-int((ncols-1)/2)

                value=np.nanmedian(los[row-inrow:row+inrow,col-incol:col+incol])
                std=np.nanstd(los[row-inrow:row+inrow,col-incol:col+incol])
                nrows=2*nrows+1
                ncols=2*ncols+1
                inrow=int((nrows-1)/2)
                incol=int((ncols-1)/2)
        else:
            value=np.nanmedian(los[cond])
            std=np.nanstd(los[cond])
        lon=lns[i]
        lat=lts[i]
        azm=np.mean(az[cond])
        inc=np.mean(lk[cond])
        wgt=np.sum(quadobs==cats[i])
        if std==0:
            print('std=0',i,wgt)
        elif std/float(wgt)<1e-9:
            print('stdw',i,std,wgt)
        if unit=='deg':
            line="%3.6f %3.6f %1.6f %1.6f %1.6f %1.9f\n"\
                % (lon,lat,azm,inc,value,std/float(wgt))
        else:
            line="%6.3f %6.3f %1.6f %1.6f %1.6f %1.9f\n"\
                % (lon,lat,azm,inc,value,std/float(wgt))
        result.write(line)
    result.close()

def points2map(xs,ys,data):
    uxs=np.array(list(set(xs)))
    uys=np.array(list(set(ys)))
    data=np.array(data)
    xint=min_distance(uxs)
    yint=min_distance(uys)
    xnum=int((np.max(xs)-np.min(xs))/xint)
    ynum=int((np.max(ys)-np.min(ys))/yint)
    
    extent=[np.min(xs),np.max(xs),np.min(ys),np.max(ys)]
    
    qmap=np.zeros((ynum,xnum))
    
    lons=np.linspace(np.min(xs),np.max(xs),xnum)
    lats=np.linspace(np.min(ys),np.max(ys),ynum)[::-1]
    
    for i,lon in enumerate(lons):
        for j,lat in enumerate(lats):
            #minn=np.argmin((xs-lon)**2+(ys-lat)**2)
            qmap[j,i]=data[i*len(lons)+j]
    return qmap,extent

def get_defmap(quadfile='quadtree.txt',mask=None, trans=False,cref=True):
    quad=open(quadfile)
    linesor=quad.readlines()
    quad.close()
    
    dim=(int(linesor[0].split(':')[2].split(',')[0]),int(linesor[0].split(':')[2].split(',')[1]))
    ext=(float(linesor[0].split(':')[3].split(',')[0]), float(linesor[0].split(':')[3].split(',')[1]), float(linesor[0].split(':')[3].split(',')[2]), float(linesor[0].split(':')[3].split(',')[3]))
    rcoords=None
    
    if not 'None' in linesor[0]:
        rcoords=[float(linesor[0].split(':')[1].split(',')[0]),float(linesor[0].split(':')[1].split(',')[1])]
        intc=(ext[1]-ext[0])/dim[1]
        intr=(ext[3]-ext[2])/dim[0]
        col=int((rcoords[0]-ext[0])/intc)
        row=int((ext[3]-rcoords[1])/intr)
    
    lines=[line for line in linesor if not line[0]=='%']
    
    quad=np.zeros(dim)
    quad[:,:]=np.nan
    
    xs,ys,qlos=[],[],[]
    for i,line in enumerate(lines):
        xs.append(float(line.split()[0]))
        ys.append(float(line.split()[1]))
        qlos.append(float(line.split()[4]))
        vert=[int(line.split()[6]),int(line.split()[7]),int(line.split()[8]),int(line.split()[9])]
        quad[vert[0]:vert[1],vert[2]:vert[3]]=float(line.split()[4])
        if rcoords is not None:
            if vert[0]<=row<=vert[1] and vert[2]<=col<=vert[3]:
                posmin=i
        
    xs=np.array(xs)
    ys=np.array(ys)
    uxs=np.array(list(set(xs)))
    uys=np.array(list(set(ys)))
    qlos=np.array(qlos)
    
    if 'None' in linesor[0]:
        posmin=np.argmin(np.abs(qlos))
        rcoords=[xs[posmin],ys[posmin]]
    
    if cref:
        quad[:,:]-=qlos[posmin]
    if mask is not None:
        quad[mask]=np.nan
    
    if trans:
        utmxs,utmys,z1s,z2s=ll2utm([ext[0],ext[1]],[ext[2],ext[3]])
        refxs,refys,z1s,z2s=ll2utm([rcoords[0]],[rcoords[1]])
        ext=[utmxs[0],utmxs[1],utmys[0],utmys[1]]
        rcoords=[refxs[0],refys[0]]
        
    return quad,ext,rcoords
'''    
def get_defmap(quadfile='quadtree.txt',mask=None,unit='deg'):
    quad=open(quadfile)
    linesor=quad.readlines()
    quad.close()
    
    lines=[line for line in linesor if not len(line.split())<5]
    
    xs,ys,qlos=[],[],[]
    for line in lines:
        xs.append(float(line.split()[0]))
        ys.append(float(line.split()[1]))
        qlos.append(float(line.split()[4]))
    xs=np.array(xs)
    ys=np.array(ys)
    uxs=np.array(list(set(xs)))
    uys=np.array(list(set(ys)))
    qlos=np.array(qlos)
    
    if linesor[0][0]=='%' and not 'None' in linesor[0]:
        rcoords=[float(linesor[0].split(':')[1].split(',')[0]),float(linesor[0].split(':')[1].split(',')[1])]
        posmin=np.argmin((xs-rcoords[0])**2+(ys-rcoords[1])**2)
    else:
        posmin=np.argmin(qlos)
        rcoords=[xs[posmin],ys[posmin]]
    qlos-=qlos[posmin]
    
    if mask is None:
        xint=min_distance(uxs)
        yint=min_distance(uys)
        xnum=int((np.max(xs)-np.min(xs))/xint)
        ynum=int((np.max(ys)-np.min(ys))/yint)
    else:
        xnum=mask.shape[1]
        ynum=mask.shape[0]
    
    lons=np.linspace(np.min(xs),np.max(xs),xnum)
    lats=np.linspace(np.min(ys),np.max(ys),ynum)[::-1]
    
    if not unit=='m':
        extent=[np.min(xs),np.max(xs),np.min(ys),np.max(ys)]
    else:
        utmxs,utmys,z1s,z2s=ll2utm([np.min(xs),np.max(xs)],[np.min(ys),np.max(ys)])
        refxs,refys,z1s,z2s=ll2utm([rcoords[0]],[rcoords[1]])
        extent=[utmxs[0],utmxs[1],utmys[0],utmys[1]]
        rcoords=[refxs[0],refys[0]]
    
    qmap=np.zeros((ynum,xnum))
    for i,lon in enumerate(lons):
        for j,lat in enumerate(lats):
            minn=np.argmin((xs-lon)**2+(ys-lat)**2)
            if mask is None:
                cond=globe.is_land(lat,lon)
            else:
                cond=~mask[j,i]
            if cond:
                qmap[j,i]=qlos[minn]
            else:
                qmap[j,i]=np.nan
    return qmap,extent,rcoords
'''

def min_distance(points):
    if len(points)<=3:
        mini=np.inf
        for i in range(len(points)):
            for j in range(i+1,len(points)):
                if np.abs(points[i]-points[j])<mini:
                    mini=np.abs(points[i]-points[j])
        return mini
    
    spoints=np.array(sorted(np.copy(points).tolist()))
    mid=len(points)//2
    midpoint=spoints[mid]
    left=spoints[:mid]
    right=spoints[mid::]
    
    minleft=min_distance(left)
    minright=min_distance(right)
    
    if minleft<minright:
        return minleft
    else:
        return minright

def run_decompose(th):
    cwd=os.getcwd()
    os.chdir('./vmod/varres')
    cmd='python decompose.py -i insar.unw -t '+str(th)+' -o newtest'
    result = subprocess.check_output(cmd, shell=True).decode()
    res=open('newtest.txt')
    lines=res.readlines()
    tam=len(lines)
    res.close()
    os.chdir(cwd)
    
    return tam
    
def ll2rc(lon,lat,extent,dims):
    lonr1, lonr2, latr1, latr2=extent
    lons=np.linspace(lonr1,lonr2,dims[1])
    lats=np.linspace(latr1,latr2,dims[0])[::-1]

    row=np.argmin(np.abs(lats-lat))

    col=np.argmin(np.abs(lons-lon))
    
    return row,col

def get_closest_point(row,col,dataset):
    if not np.isnan(dataset[row,col]):
        return row,col
    else:
        x=np.linspace(0,dataset.shape[1],dataset.shape[1])
        y=np.linspace(0,dataset.shape[0],dataset.shape[0])
        XX,YY=np.meshgrid(x,y)
        
        XX[np.isnan(dataset)]=np.nan
        YY[np.isnan(dataset)]=np.nan
        
        dif=np.sqrt((XX-col)**2+(YY-row)**2)
        
        row,col=np.unravel_index(np.nanargmin(dif), dif.shape)
        return row,col
    
def read_dataset_h5(h5file,key,plot=True,aoi=None):
    h5f=h5py.File(h5file)
    dataset=h5f[key][:]
    
    lons=[float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2'])]
    lats=[float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])]
    
    lonr1,lonr2,latr1,latr2=np.min(lons),np.max(lons),np.min(lats),np.max(lats)
    #lonr1, lonr2, latr1, latr2 = float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']), float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])
    
    extent=[lonr1,lonr2,latr1,latr2]
    h5f.close()
    
    if aoi is not None:
        row1,col1=ll2rc(aoi.x1,aoi.y2,extent,dataset.shape)
        row2,col2=ll2rc(aoi.x2,aoi.y1,extent,dataset.shape)
        extent=[aoi.x1,aoi.x2,aoi.y1,aoi.y2]
        dataset=dataset[row1:row2,col1:col2]
    
    vmin = np.nanpercentile(dataset, 1)
    vmax = np.nanpercentile(dataset, 99)
    
    fig, ax = plt.subplots()
        
    fig.suptitle(key, fontsize=16)
    
    if 'coherence' in key.lower():
        cmap = matplotlib.cm.gist_gray.copy()
        cmap.set_bad('black')
        im=ax.imshow(dataset, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    else:
        cmap=plt.cm.jet
        im=ax.imshow(dataset,cmap=cmap,extent=extent,vmin=vmin, vmax=vmax)
    ax.set_ylabel('Latitude (°)')
    ax.set_xlabel('Longitude (°)')
    plt.colorbar(im,orientation='horizontal')
    print(extent)
    return dataset

def read_gnss_csv(csvfile,trans=False):
    archivo=open(csvfile,'r')
    linesor=archivo.readlines()
    archivo.close()
    
    lines=[line for line in linesor if not '%' in line]
    
    names,lons,lats,uxs,uys,uzs,euxs,euys,euzs=[],[],[],[],[],[],[],[],[]
    for line in lines:
        names.append(line.split()[0])
        lons.append(float(line.split()[1]))
        lats.append(float(line.split()[2]))
        uxs.append(float(line.split()[3]))
        uys.append(float(line.split()[4]))
        uzs.append(float(line.split()[5]))
        euxs.append(float(line.split()[6]))
        euys.append(float(line.split()[7]))
        euzs.append(float(line.split()[8]))
        
    lons=np.array(lons)
    lots=np.array(lats)
    uxs=np.array(uxs)
    uys=np.array(uys)
    uzs=np.array(uzs)
    euxs=np.array(euxs)
    euys=np.array(euys)
    euzs=np.array(euzs)
    
    if trans:
        xs,ys,z1s,z2s=ll2utm(lons,lats)
        meanx=np.mean(xs)
        meany=np.mean(ys)
        xs-=meanx
        ys-=meany
        ref=[meanx,meany,z1s,z2s]
        return names,xs,ys,uxs,uys,uzs,euxs,euys,euzs,ref
    else:
        return names,np.array(lons),np.array(lats),uxs,uys,uzs,euxs,euys,euzs

def read_insar_csv(csvfile,trans=False,unit='m',ori=None,cref=True):
    #if ref and os.path.exists('./'+csvfile.split('.')[0]+'_ref.'+csvfile.split('.')[1]):
    #    csvfile=csvfile.split('.')[0]+'_ref.'+csvfile.split('.')[1]
    archivo=open(csvfile,'r')
    linesor=archivo.readlines()
    archivo.close()
    
    lines=[line for line in linesor if not '%' in line]
    
    lons,lats,azs,lks,los,elos=[],[],[],[],[],[]
    for line in lines:
        lons.append(float(line.split()[0]))
        lats.append(float(line.split()[1]))
        azs.append(float(line.split()[2]))
        lks.append(float(line.split()[3]))
        los.append(float(line.split()[4]))
        elos.append(float(line.split()[5]))
    lons=np.array(lons)
    lats=np.array(lats)
    azs=np.array(azs)
    lks=np.array(lks)
    los=np.array(los)
    elos=np.array(elos)
    
    if linesor[0][0]=='%' and not 'None' in linesor[0]:
        ref=[float(linesor[0].split(':')[1].split(',')[0]),float(linesor[0].split(':')[1].split(',')[1])]
        posmin=np.argmin((lons-ref[0])**2+(lats-ref[1])**2)
    elif linesor[0][0]=='%' and cref:
        posmin=np.argmin(np.abs(los))
        ref=[lons[posmin],lats[posmin]]
        los-=los[posmin]
    else:
        ref=None
    if trans and unit=='m':
        xs,ys,z1s,z2s=ll2utm(lons,lats)
        xref,yref,z1s,z2s=ll2utm([ref[0]],[ref[1]])
        meanx=np.mean(xs)
        meany=np.mean(ys)
        xs-=meanx
        ys-=meany
        ref=[xref-meanx,yref-meany,meanx,meany,str(z1s)+str(z2s)]
        return xs,ys,azs,lks,los,elos,ref
    else:
        return lons,lats,azs,lks,los,elos,ref

def plot_gnss(xs,ys,uxs,uys,uzs,title=None,names=None,euxs=None,euys=None,euzs=None,scl=None,unit='m',figsize=None):
    #Plotting GPS deformation
    if unit=='m':
        norm=1e3
    else:
        norm=1.0
    ratio=0.2
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(5,5))
    
    if title:
        plt.title(title)
    else:
        plt.title('GPS data')
    plt.scatter(xs/norm,ys/norm,s=10)
    hmax=np.max(np.sqrt(uxs**2+uys**2))
    extentx=(np.max(xs)-np.min(xs))/norm
    extenty=(np.max(ys)-np.min(ys))/norm
    if unit=='m':
        limsx=[-extentx/2-ratio*extentx,extentx/2+ratio*extentx]
        limsy=[-extenty/2-ratio*extenty,extenty/2+ratio*extenty]
    else:
        limsx=[np.min(xs)-2.25*ratio*extentx,np.max(xs)+2.25*ratio*extentx]
        limsy=[np.min(ys)-2.25*ratio*extenty,np.max(ys)+2.25*ratio*extenty]
    extent=np.max([limsx[1]-limsx[0],limsy[1]-limsy[0]])

    if unit=='m':
        sposx=limsx[0]+ratio/4*(limsx[1]-limsx[0])
        sposy=limsy[1]-ratio/4*(limsy[1]-limsy[0])
        sposy2=limsy[0]+ratio/4*(limsx[1]-limsx[0])
    else:
        sposx=limsx[0]+ratio/10*(limsx[1]-limsx[0])
        sposy=limsy[1]-ratio/10*(limsy[1]-limsy[0])
        sposy2=limsy[0]+ratio/10*(limsx[1]-limsx[0])-0.05*(limsy[1]-limsy[0])
    sposy1=sposy-0.05*(limsy[1]-limsy[0])
    
    
    if scl is None:
        scale=ratio*extent/hmax
        sc=hmax/2
        scl=round(sc*100,2)
    else:
        sc=scl/100
        hmax=2*sc
        scale=ratio*extent/hmax
    logsc=scale/10**int(np.log10(scale))
    ax = plt.gca()
    for i in range(len(xs)):
        plt.annotate("", xy=(xs[i]/norm+uxs[i]*scale, ys[i]/norm+uys[i]*scale), xytext=(xs[i]/norm, ys[i]/norm),arrowprops=dict(arrowstyle="->",color="red"))
        if not (euxs is None and euys is None):
            ax.add_patch(Ellipse(xy=(xs[i]/norm+uxs[i]*scale, ys[i]/norm+uys[i]*scale), width=euxs[i]*scale*2, height=euys[i]*scale*2, color="grey", fill=False, lw=2))
        plt.annotate("", xy=(xs[i]/norm, ys[i]/norm+uzs[i]*scale), xytext=(xs[i]/norm, ys[i]/norm),arrowprops=dict(arrowstyle="-",color="black"))
        if names is not None:
            plt.annotate(names[i], xy=(xs[i]/norm, ys[i]/norm-0.05*(limsy[1]-limsy[0])), xytext=(xs[i]/norm, ys[i]/norm-0.05*(limsy[1]-limsy[0])),color='blue')
    if euzs is not None:
        plt.errorbar(xs/norm,ys/norm+uzs*scale,euzs*scale,fmt='bo',ms=1)
    plt.annotate("", xy=(sposx+sc*scale, sposy), xytext=(sposx, sposy),arrowprops=dict(arrowstyle="->",color="red"))
    plt.annotate(str(scl)+r"cm/yr", xy=(sposx, sposy1), xytext=(sposx, sposy1),color='red')
    
    plt.annotate("", xy=(sposx, sposy2+sc*scale), xytext=(sposx, sposy2),arrowprops=dict(arrowstyle="-",color="black"))
    plt.annotate(str(scl)+r"cm/yr", xy=(sposx, sposy2), xytext=(sposx, sposy2),color='black')
    if unit=='m':
        plt.ylabel('Y(km)')
        plt.xlabel('X(km)')
    else:
        plt.ylabel('Lat(°)')
        plt.xlabel('Lon(°)')
    
    plt.xlim(limsx)
    plt.ylim(limsy)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    #plt.axis('square')
    #plt.axis('equal')
    #plt.axis('scaled')
    plt.show()

def los2npy(los,quadfile,maskfile=None,output=None,cref=False):
    archivo=open(quadfile,'r')
    lines=archivo.readlines()
    archivo.close()

    dim=[int(lines[0].split('Dimensions:')[1].split(',')[i]) for i in range(2)]
    
    result=open('temp.txt','w')
    result.write(lines[0])
    for i in range(len(lines)-1):
        i+=1
        linef=lines[i].split()
        line="%6.3f %6.3f %1.6f %1.6f %1.6f %1.9f %5.0f %5.0f %5.0f %5.0f\n"\
                    % (float(linef[0]),float(linef[1]),float(linef[2]),float(linef[3]),los[i-1],float(linef[5]),float(linef[6]),float(linef[7]),float(linef[8]),float(linef[9]))

        result.write(line)
    result.close()
    
    if maskfile:
        mask_des=np.load(maskfile)
    else:
        mask_des=np.zeros((dim[0],dim[1]))
        mask_des=mask_des>0
        
    qmap,extent,rcoords=get_defmap('temp.txt',mask=mask_des,trans=False,cref=cref)

    subprocess.call('rm -rf temp.txt',shell=True)
    
    if output:
        np.save(output,qmap)
    
    return qmap,extent
    
class AOI_Selector:
    def __init__(self,
                 h5file,
                 key,
                 coh=None,cohth=0,
                 fig_xsize=None, fig_ysize=None,
                 cmap=plt.cm.gist_gray,
                 vmin=None, vmax=None,
                 drawtype='box'
                ):
        
        h5f=h5py.File(h5file)
        keys=[ke for ke in h5f.keys()]
        if 'timeseries' in keys:
            try:
                dates=np.array([h5f['date'][:][i].decode('utf-8') for i in range(len(h5f['date'][:]))])
                print('The possible dates are:',dates)
                timeseries=h5f['timeseries'][:]
                if key in dates:
                    velocity=timeseries[dates==key][0]
                else:
                    print('The date was not found in this dataset')
                    velocity=timeseries[-1]
            except:
                raise Exception('The dataset does not exist in this file')
        else:
            if key in keys:
                velocity=h5f[key][:]
            else:
                try:
                    velocity=h5f['velocity'][:]
                except:
                    raise Exception('This dataset does not have LOS deformation')
                    
        lons=[float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2'])]
        lats=[float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])]
        lonr1,lonr2,latr1,latr2=np.min(lons),np.max(lons),np.min(lats),np.max(lats)
        #lonr1, lonr2, latr1, latr2 = float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']), float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])
        self.wl=float(h5f.attrs['WAVELENGTH'])
        
        h5f.close()

        velocity[velocity==0]=np.nan
        
        if coh is not None:
            self.coh=coh
            self.cohth=cohth
            velocity[coh<cohth]=np.nan
        
        print('Please select the area of interest')
        
        self.image = velocity
        self.extent=[lonr1,lonr2,latr1,latr2]
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        if not vmin:
            self.vmin = -np.nanmax(np.abs(velocity))
        else:
            self.vmin = vmin
        if not vmax:
            self.vmax = np.nanmax(np.abs(velocity))
        else:
            self.vmax = vmax
        if fig_xsize and fig_ysize:
            self.fig, self.current_ax = plt.subplots(figsize=(fig_xsize, fig_ysize))
        else:
            self.fig, self.current_ax = plt.subplots()
        self.fig.suptitle('Area-Of-Interest Selector', fontsize=16)
        
        if 'coherence' in key.lower():
            self.cmap = matplotlib.cm.gist_gray.copy()
            cmap.set_bad('black')
            im=self.current_ax.imshow(self.image, cmap=self.cmap, extent=self.extent, vmin=self.vmin, vmax=self.vmax)
        else:
            self.cmap=plt.cm.jet
            im=self.current_ax.imshow(self.image, cmap=plt.cm.jet, extent=self.extent, vmin=self.vmin, vmax=self.vmax)
        
        self.current_ax.set_ylabel('Latitude (°)')
        self.current_ax.set_xlabel('Longitude (°)')
        plt.colorbar(im,orientation='horizontal')

        def toggle_selector(self, event):
            print(' Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

        toggle_selector.RS = RectangleSelector(self.current_ax, self.line_select_callback,
                                               drawtype=drawtype, useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=0, minspany=0,
                                               spancoords='pixels',
                                               rectprops = dict(facecolor='red', edgecolor = 'yellow',
                                                                alpha=0.3, fill=True),
                                               interactive=True)
        plt.connect('key_press_event', toggle_selector)

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (self.x1, self.y1, self.x2, self.y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    
class Ref_Insar_Selector_Pre:
    def __init__(self,aoi):
        self.xref=None
        self.yref=None
        
        velocity=aoi.image
        extent=aoi.extent
        lonr1,lonr2,latr1,latr2=extent
        self.wl=aoi.wl

        velocity[velocity==0]=np.nan
        if aoi.coh is not None:
            velocity[aoi.coh<aoi.cohth]=np.nan

        lons=np.linspace(lonr1,lonr2,velocity.shape[1])
        lats=np.linspace(latr1,latr2,velocity.shape[0])[::-1]
        
        if aoi.x1 is None:
            row1=0
            row2=velocity.shape[0]
            col1=0
            col2=velocity.shape[1]
            self.extent=extent
        else:
            row1,col1=ll2rc(aoi.x1,aoi.y2,extent,velocity.shape)
            row2,col2=ll2rc(aoi.x2,aoi.y1,extent,velocity.shape)
            self.extent=[aoi.x1,aoi.x2,aoi.y1,aoi.y2]
        
        self.dataset=np.copy(velocity[row1:row2,col1:col2])
        fig, ax = plt.subplots()
        
        fig.suptitle('Reference Selector', fontsize=16)
        
        im=ax.imshow(self.dataset,cmap=aoi.cmap,extent=self.extent,vmin=aoi.vmin, vmax=aoi.vmax)
        line,=ax.plot([], [],'ko')
        
        ax.set_ylabel('Latitude (°)')
        ax.set_xlabel('Longitude (°)')

        plt.colorbar(im,orientation='horizontal')

        def on_click(event):
            row,col=ll2rc(event.xdata,event.ydata,self.extent,self.dataset.shape)
            
            row,col=get_closest_point(row,col,self.dataset)
            line.set_xdata(event.xdata)
            line.set_ydata(event.ydata)
            
            self.xref=event.xdata
            self.yref=event.ydata
            
            self.dataset-=self.dataset[row,col]
            
            im.set_data(self.dataset)

        plt.connect('button_press_event', on_click)

        plt.show()
        
class Ref_Insar_Selector:
    def __init__(self,csvfile,mask=None,vmin=None,vmax=None):
        
        velocity,extent,refcoords=get_defmap(csvfile,mask)
        
        lons,lats,azs,lks,los,elos,ref=read_insar_csv(csvfile,unit='deg')
        
        self.xref=ref[0]
        self.yref=ref[1]
        self.extent=extent
        self.auxdata=(lons,lats,azs,lks,elos)
        self.los=np.copy(los)
        self.filename=csvfile
        
        lonr1,lonr2,latr1,latr2=extent

        self.dataset=np.copy(velocity)
        fig, ax = plt.subplots()
        
        fig.suptitle('Reference Selector', fontsize=16)
        
        if vmin is None:
            im=ax.imshow(self.dataset,cmap='jet',extent=self.extent,vmin=-np.nanmax(np.abs(velocity)), vmax=np.nanmax(np.abs(velocity)))
        else:
            im=ax.imshow(self.dataset,cmap='jet',extent=self.extent,vmin=vmin, vmax=vmax)
        #line,=ax.plot([refcoords[0]], [refcoords[1]],'ko',label='Reference')
        line,=ax.plot([ref[0]], [ref[1]],'ko',label='Reference')
        
        ax.set_ylabel('Latitude (°)')
        ax.set_xlabel('Longitude (°)')

        plt.colorbar(im,orientation='horizontal')

        def on_click(event):
            row,col=ll2rc(event.xdata,event.ydata,self.extent,self.dataset.shape)
            
            row,col=get_closest_point(row,col,self.dataset)
            line.set_xdata(event.xdata)
            line.set_ydata(event.ydata)
            
            self.xref=event.xdata
            self.yref=event.ydata
            
            self.dataset-=self.dataset[row,col]
            
            im.set_data(self.dataset)
            
            self.los-=self.dataset[row,col]
            
            rewrite_csv(self.los,[self.xref,self.yref],old=csvfile,name=csvfile.split('.')[0]+'_ref.'+csvfile.split('.')[1])
            
        plt.legend()

        plt.connect('button_press_event', on_click)

        plt.show()
    