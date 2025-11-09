# -*- coding: utf-8 -*-
"""
General utility functions to accompany vmod
"""
import time
import math
import os
import subprocess

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import rasterio
import utm

from IPython.display import Markdown
from IPython.display import display
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.interpolate import interp1d
from skimage.restoration import denoise_nl_means, estimate_sigma


def derivative(f, x, delta):
    """
    Calculates the derivative for a function f at the point x using finite difference with the factor delta

    Parameters:
        f (callable): function with one input variable
        x (float or array): points where the derivative will be calculated
        delta: factor to be used for the finite differences

    Returns:
        dfdx (float or array): result of the derivative at x
    """
    fpx=f(x+delta)
    fnx=f(x-delta)

    return (fpx-fnx)/(2*delta)

def cart2pol(x1,x2):
    """
    Converts cartesian coordinates to polar coordinates
    
    Parameters:
        x1 (float): x-coordinate (m)
        x2 (float): y-coordinate (m)
        
    Returns:
        theta (float): polar angle (radians)
        r (float): polar radius (m)
    """
    #theta = np.arctan(x2/x1)
    theta = np.arctan2(x2,x1) #sign matters -SH
    r = np.sqrt(x1**2 + x2**2)
    return theta, r

def pol2cart(theta,r):
    """
    Converts polar coordinates to cartesian coordinates
    
    Parameters:
        theta (float): polar angle (radians)
        r (float): polar radius (m)
        
    Returns:
        x1 (float): x-coordinate (m)
        x2 (float): y-coordinate (m)
    """
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1,x2

def ll2utm(lons,lats,z1=None,z2=None):
    """
    Projects lon/lat coordinates using certain utm zone
    
    Parameters:
        lons (array): longitudes to be projected
        lats (array): latitudes to be projected
        z1 (str): utm zone number (optional)
        z2 (str): utm zone letter (optional)
    
    Returns:
        xs (array): projected x-coordinates
        ys (array): projected y-coordinates
        z1 (str): utm zone number
        z2 (str): utm zone letter
    """
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

def utm2ll(xs,ys,z1,z2,northern=None):
    """
    Converts utm coordinates into lon/lat
    
    Parameters:
        xs (array): projected x-coordinates to be converted
        ys (array): projected y-coordinates to be converted
        z1 (str): utm zone number
        z2 (str): utm zone letter
    
    Returns:
        lons (array): converted longitudes
        lats (array): converted latitudes
    """
    lons,lats=[],[]
    for i in range(len(xs)):
        if northern is not None and z2 is None:
            lat,lon=utm.to_latlon(xs[i], ys[i], z1, northern=northern)
        elif northern is None and z2 is not None:
            lat,lon=utm.to_latlon(xs[i], ys[i], z1, z2)
        lons.append(lon)
        lats.append(lat)
    lons=np.array(lons)
    lats=np.array(lats)
    return lons,lats

def get_quadtree(ref,az,lk,name='quadtree.txt',th=None):
    """
    Downsample an image that is inside a Ref_Selector object
    
    Parameters:
        ref (Ref_Selector): Ref_Selector object that contains the image to be downsampled
        az (array or float): interferogram azimuth angle clockwise from north (degrees)
        lk (array or float): interferogram incidence angle (degrees)
        name (str): csv filename for the output
        th (float): variance threshold
    """
    im=ref.dataset
    if th is None:
        th=np.nanvar(im)/50

    quadtree_var(im,az,lk,ref.extent,th,name)

def rewrite_csv(los,ref,old,name='output.txt'):
    """
    Rewrites a csv file that contains the downsampled
    interferogram and has changed the reference
    
    Parameters:
        los (array): new line of sight data
        ref (array): longitude and latitude for the reference pixel (degrees)
        old (str): filename of the csvfile
        name (str): new csv filename for the output
    """
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

def quadtree_var(im,az,inc,extent,th,name='quadtree.txt',ref=None,denoise=True):
    """
    Downsample an image (im) acoording to a variance threshold (th)

    Parameters:
        im (array): image (represented by a matrix) to be downsampled
        az (array or float): interferogram azimuth angle clockwise from north (degrees)
        inc (array or float): interferogram incidence angle (degrees)
        extent (array): array with the extent of the image (min_lon,max_lon,min_lat,max_lat)
        th (float): variance threshold
        name (str): filename for the output
        ref (array): reference pixel (lon,lat)
        denoise (boolean): if True a non-local means filter will be applied to the image
    """
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=15,  # 13x13 search area
                    )

    imcp=np.copy(im)
    imcp[np.isnan(im)]=0

    if isinstance(az, np.ndarray) and isinstance(inc, np.ndarray):
        pass
    elif (isinstance(az, float) or isinstance(az, int)) and (isinstance(inc, float) or isinstance(inc, int)):
        pass
    else:
        raise Exception('Azimuth or Incidence not in the right format')

    if denoise:
        imfil=denoise_nl_means(imcp, h=0.6 * 0.5, sigma=0.5, fast_mode=True, **patch_kw)
    else:
        imfil=np.copy(imcp)

    imfil[imcp==0]=np.nan

    fverts=[]
    pointsx=[]
    pointsy=[]

    def quadtree_im(im,inverts,th):
        """
        Recurrent split function, if the image has a variance lower than th
        the recurrence will stop
        
        Parameters:
            im (array): image to be downsampled
            inverts (array): upper left coordinates (row,col)
            th (float): variance threshold
        """
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
        if isinstance(az, np.ndarray) and isinstance(inc, np.ndarray):
            azmean=np.nanmean(az[fvert[0]:fvert[1],fvert[2]:fvert[3]])
            incmean=np.nanmean(inc[fvert[0]:fvert[1],fvert[2]:fvert[3]])
        else:
            azmean=az
            incmean=inc
        std=calc_std(im,fvert)
        line="%6.3f %6.3f %1.6f %1.6f %1.6f %1.9f %5.0f %5.0f %5.0f %5.0f\n"\
                % (xcoords[pointsx[i]],ycoords[pointsy[i]],azmean,incmean,mean,std,fvert[0],fvert[1],fvert[2],fvert[3])
        ar.write(line)
    ar.close()

def calc_std(mat,verts):
    """
    Calculates the standard deviation within a sub-matrix. If the std is 0, 
    it makes the sub-matrix bigger
    
    Parameters:
        mat (array): input matrix
        verts (array): indexes to define the sub-matrix (min_row, max_row, min_col, max_col)
        
    Returns:
        std (float): standard deviation of the submatrix
    """
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
    
def points2map(xs,ys,data):
    """
    Converts a list of points into a matrix
    
    Parameters:
        xs (array): x-coordinates or longitudes 
        ys (array): y-coordinates or latitudes
        data (array): value for each coordinate
    
    Returns:
        qmap (array): matrix with the data
        extent (array): extent of the matrix or image (min_x,max_x,min_y,max_y)
    """
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
    """
    Converts a downsampled image (represented by a text file) into a matrix
    
    Parameters:
        quadfile (str): path for the downsampled image 
        mask (array): boolean matrix for nan values
        trans (array): if True transforms lon/lat into planar coordinates
        cref (boolean): if True reference the matrix with respect to the reference pixel,
        if None reference the matrix with respect to closest value to 0
    
    Returns:
        qmap (array): matrix with the data
        ext (array): extent of the matrix or image (min_x,max_x,min_y,max_y)
        rcoords (array): coordinates for the reference (x-coord,y-coord)
    """
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

def min_distance(points):
    """
    Gives the minimum distance in a list of coordinates
    
    Parameters:
        points (array): list of points
        
    Returns:
        mini: minimum distance in the list of points
    """
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

def ll2rc(lon,lat,extent,dims):
    """
    Gives the row and column that correspond to a lon/lat coordinate
    
    Parameters:
        lon (float): longitude coordinate
        lat (float): latitude coordinate
        extent (array): extent of the image (min_lon,max_lon,min_lat,max_lat)
        dims (array): dimensions of the image (rows,columns)
        
    Returns:
        row (int): row that corresponds to the coordinate
        col (int): column that corresponds to the coordinate
    """
    lonr1, lonr2, latr1, latr2=extent
    lons=np.linspace(lonr1,lonr2,dims[1])
    lats=np.linspace(latr1,latr2,dims[0])[::-1]

    row=np.argmin(np.abs(lats-lat))

    col=np.argmin(np.abs(lons-lon))

    return row,col

def get_closest_point(row,col,dataset):
    """
    Gets the closest point for a pixel that is not nan
    
    Parameters:
        row (int): row for the pixel
        col (int): column for the pixel
        dataset (array): matrix that represents an image
    
    Returns:
        row (int): row for the closest pixel that is not nan
        col (int): column for the closest pixel that is not nan
    """
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

def read_dataset_h5(h5file,key,index=None,plot=True,aoi=None):
    """
    Reads and plots the output from mintpy 

    Parameters:
        h5file (str): path to h5 file
        key (str): key for certain dataset (e.g., coherence)
        plot (boolean): if True plots the dataset
        aoi (AOI_Selector): AOI_Selector to get the area of interest in the dataset
    
    Returns:
        dataset (array): matrix that represents the dataset
    """
    h5f=h5py.File(h5file)
    if index is None:
        dataset=h5f[key][:]
    else:
        dataset=h5f[key][:][index,:,:]

    #lons=[float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']),float(h5f.attrs['LON_REF3']), float(h5f.attrs['LON_REF4'])]
    lons=[float(h5f.attrs['X_FIRST']),float(h5f.attrs['X_FIRST'])+dataset.shape[1]*float(h5f.attrs['X_STEP'])]
    #lats=[float(h5f.attrs['LAT_REF1']), float(h5f.attrs['LAT_REF2']),float(h5f.attrs['LAT_REF3']), float(h5f.attrs['LAT_REF4'])]
    lats=[float(h5f.attrs['Y_FIRST'])+dataset.shape[0]*float(h5f.attrs['Y_STEP']),float(h5f.attrs['Y_FIRST'])]

    lonr1,lonr2,latr1,latr2=np.min(lons),np.max(lons),np.min(lats),np.max(lats)
    #lonr1, lonr2, latr1, latr2 = float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']), float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])

    extent=[lonr1,lonr2,latr1,latr2]
    if np.mean(np.abs(extent))>180:
        utm=h5f.attrs['UTM_ZONE']
        z2=utm[-1]
        northern=False
        if z2=='N':
            northern=True
        z1=int(utm.replace(z2,''))
        lons,lats=utm2ll(extent[0:2],extent[2::],z1,None,northern=northern)
        extent=[np.min(lons),np.max(lons),np.min(lats),np.max(lats)]
    h5f.close()

    if aoi is not None:
        row1,col1=ll2rc(aoi.x1,aoi.y2,extent,dataset.shape)
        row2,col2=ll2rc(aoi.x2,aoi.y1,extent,dataset.shape)
        extent=[aoi.x1,aoi.x2,aoi.y1,aoi.y2]
        dataset=dataset[row1:row2,col1:col2]

    vmin = np.nanpercentile(dataset, 1)
    vmax = np.nanpercentile(dataset, 99)

    ratio=(extent[1]-extent[0])/(extent[3]-extent[2])

    fig, ax = plt.subplots()

    if ratio<0.3 or ratio>4:
        print('bad ratio')
        ax.set_aspect(1/ratio)

    fig.suptitle(key, fontsize=16)

    if 'coherence' in key.lower():
        cmap = matplotlib.cm.gist_gray.copy()
        cmap.set_bad('black')
        im=ax.imshow(dataset, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    else:
        cmap=plt.cm.jet
        im=ax.imshow(dataset,cmap=cmap,extent=extent,vmin=vmin, vmax=vmax)

    if np.mean(np.abs(extent))<=180:
        ax.set_ylabel('Latitude (°)')
        ax.set_xlabel('Longitude (°)')
    else:
        ax.set_ylabel('North (m)')
        ax.set_xlabel('East (m)')

    plt.colorbar(im,orientation='horizontal')

    return dataset

def read_dataset_tif(tiffile,plot=True,aoi=None):
    """
    Reads and plots the output from mintpy 
    
    Parameters:
        tiffile (str): path to tif file
        plot (boolean): if True plots the dataset
        aoi (AOI_Selector): AOI_Selector to get the area of interest in the dataset
    
    Returns:
        dataset (array): matrix that represents the dataset
    """
    with rasterio.open(tiffile) as src:
        # Read the raster data
        dataset = src.read()[0,:,:]
        bounds = src.bounds

    #lonr1, lonr2, latr1, latr2 = float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']), float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])

    extent=[bounds.left,bounds.right,bounds.bottom,bounds.top]

    if aoi is not None:
        row1,col1=ll2rc(aoi.x1,aoi.y2,extent,dataset.shape)
        row2,col2=ll2rc(aoi.x2,aoi.y1,extent,dataset.shape)
        extent=[aoi.x1,aoi.x2,aoi.y1,aoi.y2]
        dataset=dataset[row1:row2,col1:col2]

    vmin = np.nanpercentile(dataset, 1)
    vmax = np.nanpercentile(dataset, 99)

    fig, ax = plt.subplots()

    if 'coh' in tiffile:
        fig.suptitle('Coherence', fontsize=16)
        cmap = matplotlib.cm.gist_gray.copy()
        cmap.set_bad('black')
        im=ax.imshow(dataset, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    else:
        fig.suptitle('Velocity', fontsize=16)
        cmap=plt.cm.jet
        im=ax.imshow(dataset,cmap=cmap,extent=extent,vmin=vmin, vmax=vmax)

    if np.mean(np.abs(extent))<=180:
        ax.set_ylabel('Latitude (°)')
        ax.set_xlabel('Longitude (°)')
    else:
        ax.set_ylabel('North (m)')
        ax.set_xlabel('East (m)')
    plt.colorbar(im,orientation='horizontal')

    return dataset

def read_gnss_csv(csvfile,trans=False,ignore=[]):
    """
    Reads csv file with a gnss dataset 
    
    Parameters:
        csvfile (str): path to csv file
        trans (boolean): projects the lon/lat coordinates into plane coordinates
        
    Returns:
        names (array): names of the stations
        xs/lons (array): x-coordinates or longitudes for the stations
        ys/lats (array): y-coordinates or latitudes for the stations
        uxs (array): deformation in the east component
        uys (array): deformation in the north component
        uzs (array): deformation in the vertical component
        euxs (array): uncertainties in the deformation in the east component
        euys (array): uncertainties in the deformation in the north component
        euzs (array): uncertainties in the deformation in the vertical component
        ref (array): origin coordinate if a projection was made
    """
    archivo=open(csvfile,'r')
    linesor=archivo.readlines()
    archivo.close()

    lines=[line for line in linesor if not '%' in line]

    names,lons,lats,uxs,uys,uzs,euxs,euys,euzs=[],[],[],[],[],[],[],[],[]
    for line in lines:
        if not line.split()[0] in ignore:
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
    """
    Reads csv file with a downsampled InSAR dataset 
    
    Parameters:
        csvfile (str): path to csv file
        trans (boolean): if True projects the lon/lat coordinates into plane coordinates (default False)
        unit (str): units for the coordinates (m for meters or deg for degrees) (default m)
        ori (array): coordinates (lon,lat) that will be the origin for the projection
        cref
        
    Returns:
        names (array): names of the stations
        xs/lons (array): x-coordinates or longitudes for the stations
        ys/lats (array): y-coordinates or latitudes for the stations
        uxs (array): deformation in the east component
        uys (array): deformation in the north component
        uzs (array): deformation in the vertical component
        euxs (array): uncertainties in the deformation in the east component
        euys (array): uncertainties in the deformation in the north component
        euzs (array): uncertainties in the deformation in the vertical component
        ref (array): origin coordinate if a projection was made
    """
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
    """
    Plots gnss dataset, horizontal velocities are represented by red arrows
    vertical velocities are represented by black lines

    Parameters:
        xs/lons (array): x-coordinates or longitudes for the stations (m/deg)
        ys/lats (array): y-coordinates or latitudes for the stations (m/deg)
        uxs (array): deformation in the east component
        uys (array): deformation in the north component
        uzs (array): deformation in the vertical component
        title (str): title for the plot
        names (array): names for the stations
        euxs (array): uncertainties in the deformation in the east component
        euys (array): uncertainties in the deformation in the north component
        euzs (array): uncertainties in the deformation in the vertical component
        scl (float): scale for the arrows
        unit (str): unit for the coordinates (m for meters or deg for degrees)
        figsize (tuple): size for the matplotlib figure
    """
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
        vmax=4*sc
        scale=ratio*extent/hmax
        vscale=ratio*extent/vmax
    logsc=scale/10**int(np.log10(scale))
    ax = plt.gca()
    for i in range(len(xs)):
        plt.annotate("", xy=(xs[i]/norm+uxs[i]*scale, ys[i]/norm+uys[i]*scale), xytext=(xs[i]/norm, ys[i]/norm),arrowprops=dict(arrowstyle="->",color="red"))
        if not (euxs is None and euys is None):
            ax.add_patch(Ellipse(xy=(xs[i]/norm+uxs[i]*scale, ys[i]/norm+uys[i]*scale), width=euxs[i]*scale*2, height=euys[i]*scale*2, color="grey", fill=False, lw=2))
        plt.annotate("", xy=(xs[i]/norm, ys[i]/norm+uzs[i]*vscale), xytext=(xs[i]/norm, ys[i]/norm),arrowprops=dict(arrowstyle="-",color="black"))
        if names is not None:
            plt.annotate(names[i], xy=(xs[i]/norm, ys[i]/norm-0.05*(limsy[1]-limsy[0])), xytext=(xs[i]/norm, ys[i]/norm-0.05*(limsy[1]-limsy[0])),color='blue')
    if euzs is not None:
        plt.errorbar(xs/norm,ys/norm+uzs*vscale,euzs*vscale,fmt='bo',ms=1)
    plt.annotate("", xy=(sposx+sc*scale, sposy), xytext=(sposx, sposy),arrowprops=dict(arrowstyle="->",color="red"))
    plt.annotate(str(scl)+r"cm/yr", xy=(sposx, sposy1), xytext=(sposx, sposy1),color='red')
    
    plt.annotate("", xy=(sposx, sposy2+sc*vscale), xytext=(sposx, sposy2),arrowprops=dict(arrowstyle="-",color="black"))
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
    plt.show()

def plot_gnss_pygmt(csvfile, uxs=None, uys=None, uzs=None, scalebar=10, output='figure_pygmt.png', title=None, points=None, epoints=None, lpoints=None, errx=None, erry=None, errz=None, arrowscale=0.01, ignore=[]):
    """
    Plots GNSS dataset with pygmt, horizontal velocities are represented by blue arrows
    vertical velocities are represented by red arrows

    Parameters:
        csvfile (str): filename for the csv that contains the GNSS velocities
        uxs (array): deformation in the east component, if None it will plot data from the csv file
        uys (array): deformation in the north component
        uzs (array): deformation in the vertical component
        scalebar (int): scalebar for the map in kilometers
        output (str): filename for the output figure
        title (str): title on the figure
        points (array): coordinate points, if None no points will be plotted
        epoints (array): error bars in degrees for 'points'. It needs to have the same size of 'points', if None no error bars will be plotted
        lpoints (array): labels for 'points', if None no labels will be plotted
        errx (array): uncertainties in the deformation in the north component
        erry (array): uncertainties in the deformation in the north component
        errz (array): uncertainties in the deformation in the vertical component
        arrowscale (float): scale for the velocities in meters per year, default 1cm/yr
        ignore (array): names of the stations that will not be plotted
    """
    import pygmt
    import xarray as xr
    import pandas as pd

    names,lons,lats,uxsf,uysf,uzsf,euxs,euys,euzs=read_gnss_csv(csvfile,ignore=ignore)

    if uxs is None:
        uxs=uxsf
        uys=uysf
        uzs=uzsf
    else:
        if len(ignore)>0:
            namest,lonst,latst,uxsft,uysft,uzsft,euxst,euyst,euzst=read_gnss_csv(csvfile)
            namesf=list(set(namest)-set(names))
            uxs=np.array([uxs[i] for i in range(len(uxs)) if namest[i] not in namesf])
            uys=np.array([uys[i] for i in range(len(uys)) if namest[i] not in namesf])
            uzs=np.array([uzs[i] for i in range(len(uzs)) if namest[i] not in namesf])

    if erry is None:
        sxs=euxs
        sys=euys
        szs=euzs
    else:
        sxs=errx
        sys=erry
        szs=errz

    interlon=np.round(np.abs(np.max(lons)-np.min(lons))*0.5,1)
    interlat=np.round(np.abs(np.max(lats)-np.min(lats))*0.5,1)
    inter=np.max([interlon,interlat])
    region=[np.min(lons)-interlon,np.max(lons)+interlon,np.min(lats)-interlat,np.max(lats)+interlat]

    stns=[[lons[i],lats[i]] for i in range(len(lons))]

    lons=lons.tolist()+[region[0]+interlon/2]
    lats=lats.tolist()+[region[-1]-interlat]
    if arrowscale*1e2>=1:
        names=names+[str(int(arrowscale*1e2))+"cm/yr"]
    else:
        names=names+[str(float(arrowscale*1e2))+"cm/yr"]

    uxs=np.array(uxs.tolist()+[arrowscale])
    uys=np.array(uys.tolist()+[0.000])
    uzs=np.array(uzs.tolist()+[arrowscale])

    sxs=np.array(sxs.tolist()+[0.000])
    sys=np.array(sys.tolist()+[0.000])
    szs=np.array(szs.tolist()+[0.000])

    df = pd.DataFrame(
        data={
            "x": lons,
            "y": lats,
            "east_velocity": uxs*1e3,
            "north_velocity": uys*1e3,
            "east_sigma": sxs*0,
            "north_sigma": sys*0,
        }
    )

    df1 = pd.DataFrame(
        data={
            "x": lons,
            "y": lats,
            "east_velocity": uxs*0*3e6,
            "north_velocity": uzs*1e3,
            "east_sigma": sxs*0,
            "north_sigma": szs*0,
        }
    )

    grid_data = '@earth_relief_03s' 
    grid = pygmt.grdcut(grid_data,
                                 region=region,
                                )

    dgrid = pygmt.grdgradient(grid=grid,azimuth=270)
    grid.data[np.logical_and(grid>177,grid<180)]=np.nan
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP='D')  # Use decimal degrees
    pygmt.config(MAP_FRAME_TYPE='plain')  # Use decimal degrees
    pygmt.config(FONT_ANNOT_PRIMARY='30p,Helvetica,black')
    pygmt.config(FONT_LABEL='30p,Helvetica,black')
    pygmt.config(FONT_TITLE='40p,Helvetica,black')
    pygmt.config(COLOR_FOREGROUND='lightgray')
    pygmt.makecpt(cmap="gray", series=[-np.nanmax(grid.data), np.nanmax(grid.data)])

    if title is not None:
        frame=['a'+str(inter),'+t'+str(title)]
    else:
        frame=['a'+str(inter)]

    fig.grdimage(
        grid=grid,
        region=region,
        projection='M8i',
        frame=frame,
        shading=True,
    )

    lonll=np.round(region[0]+interlon/2,1)
    latll=np.round(region[2]+interlat/2,2)

    fig.coast(shorelines="0.5p,black",lakes='+l',map_scale=str(lonll)+'/'+str(latll)+'/'+str(latll)+'/'+str(scalebar),water="white")

    fig.text(x=lons,y=np.array(lats)-float(inter/20),text=names,fill='white',font="30p,Helvetica,black")


    fig.velo(
        data=df,
        region=region,
        pen="5p,blue",
        line="5p,blue",
        projection='M8i',
        spec="e"+str(0.25/(arrowscale*1e2))+"/0.39/10",
        vector="0.7c+p0.5p+e+gblue",
    )

    fig.velo(
        data=df1,
        region=region,
        pen="5p,red",
        line="5p,red",
        projection='M8i',
        spec="e"+str(0.25/(arrowscale*1e2))+"/0.39/10",
        vector="0.7c+p0.5p+e+gred",
    )

    if points is not None:
        for i in range(len(points)):
            fig.plot(x=points[i][0], y=points[i][1], style="a0.3", pen="3p,black")
            if epoints is not None:
                if not len(epoints)==len(points):
                    raise Exception('The uncertainties and points do not have the same size')
                fig.plot(x=[points[i][0]-epoints[i][0],points[i][0]+epoints[i][0]], y=[points[i][1],points[i][1]], pen="3p,gray30")
                fig.plot(x=[points[i][0],points[i][0]], y=[points[i][1]-epoints[i][1],points[i][1]+epoints[i][1]], pen="3p,gray30")
            if lpoints is not None:
                if not len(lpoints)==len(points):
                    raise Exception('The labels and points do not have the same size')
                fig.text(x=points[i][0],y=points[i][1]-float(inter/20),text=lpoints[i],font="20p,Helvetica,black")



    fig.savefig(output)
    fig.show()

def plot_insar_pygmt(csvfile, data=None, maskfile=None, scalebar=10, output='figure_pygmt.png', title=None, points=None, epoints=None, lpoints=None):
    """
    Plots InSAR dataset with pygmt

    Parameters:
        csvfile (str): filename for the csv that contains the downsampled InSAR dataset
        data (array): LOS deformation data, if None it will plot the data from the csv file
        maskfile (array): filename for the mask file in npy format, if None no mask will be applied
        scalebar (int): scalebar for the map in kilometers
        output (str): filename for the output figure
        title (str): title on the figure
        points (array): coordinate points, if None no points will be plotted
        epoints (array): error bars in degrees for 'points'. It needs to have the same size of 'points', if None no error bars will be plotted
        lpoints (array): labels for 'points', if None no labels will be plotted
    """
    import pygmt
    import xarray as xr

    if data is None:
        archivo=open(quadfile,'r')
        lines=archivo.readlines()
        archivo.close()
    
        dim=[int(lines[0].split('Dimensions:')[1].split(',')[i]) for i in range(2)]
        if maskfile:
            mask_des=np.load(maskfile)
        else:
            mask_des=np.zeros((dim[0],dim[1]))
            mask_des=mask_des>0
        dataset,extent,rcoords=get_defmap(csvfile,mask=maskfile,trans=False,cref=False)
    else:
        dataset,extent=los2npy(data,csvfile,maskfile=maskfile)
    velocitycp=dataset

    quad=open(csvfile,'r')
    lines=quad.readlines()
    quad.close()

    region=list(extent)

    ref=[float(lines[0].split(':')[1].split(',')[i]) for i in range(2)] 
    line=lines[0].split('Extent:')[1]

    coords=[float(line.split(',')[i]) for i in range(len(line.split(',')))]
    for coord in coords:
        if np.abs(coord)>180:
            raise Exception('The dataset does not have lon/lat coordinates cannot use pygmt')

    lons=np.linspace(coords[0],coords[1],velocitycp.shape[1])
    lats=np.linspace(coords[2],coords[3],velocitycp.shape[0])[::-1]

    quadobsa1=np.copy(velocitycp)*1e2

    box_x=[lons[0],lons[-1],lons[-1],lons[0],lons[0]]
    box_y=[lats[-1],lats[-1],lats[0],lats[0],lats[-1]]

    data=xr.DataArray(data=quadobsa1,dims=['lat','lon'],coords={'lon':lons,'lat':lats})

    grid_data = '@earth_relief_03s' 

    grid=pygmt.datasets.load_earth_relief(resolution="03s", region=[lons[0],lons[-1],lats[-1],lats[0]])
    orgrid=np.copy(grid)
    dgrid = pygmt.grdgradient(grid=grid,azimuth=270)
    grid.data[np.logical_and(grid.data>-1000,grid.data<0)]=np.nan
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP='D')  # Use decimal degrees
    pygmt.config(MAP_FRAME_TYPE='plain')  # Use decimal degrees
    pygmt.config(FONT_ANNOT_PRIMARY='30p,Helvetica,black')
    pygmt.config(FONT_LABEL='30p,Helvetica,black')
    pygmt.config(FONT_TITLE='40p,Helvetica,black')
    pygmt.config(COLOR_FOREGROUND='lightgray')

    pygmt.makecpt(cmap="gray", series=[-np.nanmax(grid.data), np.nanmax(grid.data)])

    interlon=np.round(np.abs(np.max(lons)-np.min(lons))/5,1)
    interlat=np.round(np.abs(np.max(lats)-np.min(lats))/5,1)
    inter=np.max([interlon,interlat])

    if title is not None:
        frame=['a'+str(inter),'+t'+str(title)]
    else:
        frame=['a'+str(inter)]

    fig.grdimage(
        grid=grid,
        region=region,
        projection='M8i',
        frame=frame,
        shading=True,
    )

    max=np.nanmax(np.abs(dataset)*1e2)*0.8

    pygmt.makecpt(cmap="jet", series=[-max, max])

    fig.grdimage(
        grid=data,
        region=region,
        projection='M8i',
        frame=frame,
        transparency=30,
        nan_transparent=True,
    )
    fig.colorbar(frame="af+lLOS deformation (cm)")

    pygmt.makecpt(cmap="jet", series=[-max, max])

    lonll=np.percentile(lons,20)
    latll=np.percentile(lats,20)

    fig.coast(shorelines="0.5p,black",lakes='+l',map_scale=str(lonll)+'/'+str(latll)+'/'+str(latll)+'/'+str(scalebar),water="white")

    if points is not None:
        for i in range(len(points)):
            fig.plot(x=points[i][0], y=points[i][1], style="a0.3", pen="3p,black")
            if epoints is not None:
                if not len(epoints)==len(points):
                    raise Exception('The uncertainties and points do not have the same size')
                fig.plot(x=[points[i][0]-epoints[i][0],points[i][0]+epoints[i][0]], y=[points[i][1],points[i][1]], pen="3p,gray30")
                fig.plot(x=[points[i][0],points[i][0]], y=[points[i][1]-epoints[i][1],points[i][1]+epoints[i][1]], pen="3p,gray30")
            if lpoints is not None:
                if not len(lpoints)==len(points):
                    raise Exception('The labels and points do not have the same size')
                fig.text(x=points[i][0],y=points[i][1]-float(inter/20),text=lpoints[i],font="20p,Helvetica,black")

    fig.savefig(output)
    fig.show()


def los2npy(los,quadfile,maskfile=None,output=None,cref=False):
    """
    Creates matrix representing downsampled InSAR dataset replacing los deformation
    and save it in a npy file
    
    Parameters:
        los (array): new line of sight deformation dataset
        quadfile (array): downsampled InSAR dataset
        maskfile (str): npy file with boolean matrix that indicates nan values
        output (str): npy filename for the output
        cref (boolean): if True reference the matrix with respect to the reference pixel,
        if None reference the matrix with respect to closest value to 0
        
    Returns:
        qmap (array): saved matrix that represents downsampled InSAR dataset
        extent (array): coordinates with the extent of the image
    """
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
    """
    Class to create an interactive tool to select the area of interest from
    a mintpy output h5 file
    
    Attributes:
        wl (float): wavelength of the mission in meters
        coh (array): coherence dataset in h5 file
        cohth (float): the pixels that have a coherence value below this won't be plot
        image (array): dataset to be plot
        extent (array): coordinates for the extent of the image (min_lon,max_lon,min_lat,max_lat)
        x1 (float): upper left x-coordinate or longitude of area of interest
        y1 (float): upper left y-coordinate or latitude of area of interest
        x2 (float): lower right x-coordinate or longitude of area of interest
        y2 (float): lower right y-coordinate or latitude of area of interest
        vmin (float): minimum value for colorbar
        vmax (float): maximum value for colorbar
        fig (matplotlib Figure): figure that plots the dataset
        current_ax (matplotlib subplot): subplot that has the plot for the dataset
    """
    def __init__(self,
                 filename,
                 key=None,
                 index=None,
                 wvl=None,
                 los=True,
                 coh=None,cohth=0,ref_date=None,
                 fig_xsize=None, fig_ysize=None,
                 cmap=plt.cm.gist_gray,
                 vmin=None, vmax=None
                ):

        if '.h5' in filename:
            h5file=filename
            h5f=h5py.File(h5file)
            keys=[ke for ke in h5f.keys()]
            if key is None:
                raise Exception('For h5 files you need to specify a key')
            elif 'timeseries' in keys:
                try:
                    dates=np.array([h5f['date'][:][i].decode('utf-8') for i in range(len(h5f['date'][:]))])
                    print('The possible dates are:',dates)
                    timeseries=h5f['timeseries'][:]
                    if key in dates:
                        if ref_date:
                            if ref_date in dates:
                                velocity=timeseries[dates==key][0]-timeseries[dates==ref_date][0]
                            else:
                                print('The reference date was not found!!')
                        else:
                            velocity=timeseries[dates==key][0]
                    else:
                        print('The date was not found in this dataset')
                        velocity=timeseries[-1]
                except:
                    raise Exception('The dataset does not exist in this file')
            else:
                if key in keys:
                    if index is None:
                        velocity=h5f[key][:]
                    else:
                        velocity=h5f[key][:][index,:,:]
                else:
                    try:
                        velocity=h5f['velocity'][:]
                    except:
                        raise Exception('This dataset does not have LOS deformation')

            #lons=[float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']),float(h5f.attrs['LON_REF3']), float(h5f.attrs['LON_REF4'])]
            lons=[float(h5f.attrs['X_FIRST']),float(h5f.attrs['X_FIRST'])+velocity.shape[1]*float(h5f.attrs['X_STEP'])]
            #lats=[float(h5f.attrs['LAT_REF1']),float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3']), float(h5f.attrs['LAT_REF4'])]
            lats=[float(h5f.attrs['Y_FIRST'])+velocity.shape[0]*float(h5f.attrs['Y_STEP']),float(h5f.attrs['Y_FIRST'])]
            lonr1,lonr2,latr1,latr2=np.min(lons),np.max(lons),np.min(lats),np.max(lats)
            extent=[lonr1,lonr2,latr1,latr2]
            if np.mean(np.abs(extent))>180:
                utm=h5f.attrs['UTM_ZONE']
                z2=utm[-1]
                northern=False
                if z2=='N':
                    northern=True
                z1=int(utm.replace(z2,''))
                lons,lats=utm2ll(extent[0:2],extent[2::],z1,None,northern=northern)
                lonr1,lonr2,latr1,latr2=np.min(lons),np.max(lons),np.min(lats),np.max(lats)
            #lonr1, lonr2, latr1, latr2 = float(h5f.attrs['LON_REF1']), float(h5f.attrs['LON_REF2']), float(h5f.attrs['LAT_REF2']), float(h5f.attrs['LAT_REF3'])
            #self.wl=float(h5f.attrs['WAVELENGTH'])

            h5f.close()

        elif '.tif' in filename:
            with rasterio.open(filename) as src:
                # Read the raster data
                velocity = src.read()[0,:,:]
                bounds = src.bounds
                lonr1,lonr2,latr1,latr2=bounds.left,bounds.right,bounds.bottom,bounds.top
        else:
            raise Exception('Unknown format. VMOD can only read h5 or tif files.')

        if not los:
            if not wvl is None:
                velocity=velocity*wvl/(-4*np.pi)
            else:
                raise Exception('You need to provide the wavelength of the SAR mission')

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

        self.cmap=plt.cm.jet
        im=self.current_ax.imshow(self.image, cmap=plt.cm.jet, extent=self.extent, vmin=self.vmin, vmax=self.vmax)

        if np.mean(np.abs(self.extent))<=180:
            self.current_ax.set_ylabel('Latitude (°)')
            self.current_ax.set_xlabel('Longitude (°)')
        else:
            self.current_ax.set_ylabel('North (m)')
            self.current_ax.set_xlabel('East (m)')
        plt.colorbar(im,orientation='horizontal')

        def toggle_selector(self, event):
            """
            Activates the selector tool in matplotlib
            """
            print(' Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

        toggle_selector.RS = RectangleSelector(self.current_ax, self.line_select_callback, useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=0, minspany=0,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('key_press_event', toggle_selector)

    def line_select_callback(self, eclick, erelease):
        """
        Catch the coordinates from the selector tool
        """
        'eclick and erelease are the press and release events'
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (self.x1, self.y1, self.x2, self.y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

class Ref_Insar_Selector_Pre:
    """
    Class to create an interactive tool to select a reference pixel from 
    an AOI_Selector object
    
    Attributes:
        wl (float): wavelength of the mission
        xref (float): x-coordinate for the reference
        yref (float): y-coordinate for the reference
        coh (array): coherence dataset in h5 file
        dataset (array): dataset to be plot
        extent (array): coordinates for the extent of the image (min_lon,max_lon,min_lat,max_lat)
    """
    def __init__(self,aoi):
        self.xref=None
        self.yref=None

        velocity=aoi.image
        extent=aoi.extent
        lonr1,lonr2,latr1,latr2=extent
        #self.wl=aoi.wl

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

        import ipywidgets as widgets
        out = widgets.Output()

        self.dataset=np.copy(velocity[row1:row2,col1:col2])
        fig, ax = plt.subplots()
        fig.suptitle('Reference Selector', fontsize=16)

        im=ax.imshow(self.dataset,cmap=aoi.cmap,extent=self.extent,vmin=aoi.vmin, vmax=aoi.vmax)
        line,=ax.plot([], [],'ko')

        if np.mean(np.abs(self.extent))<=180:
            ax.set_ylabel('Latitude (°)')
            ax.set_xlabel('Longitude (°)')
        else:
            ax.set_ylabel('North (m)')
            ax.set_xlabel('East (m)')

        plt.colorbar(im,orientation='horizontal')

        def on_click(event):
            """
            Captures the coordinates when a user click on the plot
            """
            row,col=ll2rc(event.xdata,event.ydata,self.extent,self.dataset.shape)

            row,col=get_closest_point(row,col,self.dataset)

            #line.set_xdata(event.xdata)
            #line.set_ydata(event.ydata)

            self.xref=event.xdata
            self.yref=event.ydata

            self.dataset-=self.dataset[row,col]
            ax.clear()
            ax.imshow(self.dataset,cmap=aoi.cmap,extent=self.extent,vmin=aoi.vmin, vmax=aoi.vmax)
            ax.plot([self.xref],[self.yref],'ko')
            if np.mean(np.abs(self.extent))<=180:
                ax.set_ylabel('Latitude (°)')
                ax.set_xlabel('Longitude (°)')
            else:
                ax.set_ylabel('North (m)')
                ax.set_xlabel('East (m)')
            #im.set_data(self.dataset)

        plt.connect('button_press_event', on_click)
        #cid=fig.canvas.mpl_connect('button_press_event', on_click)

        plt.show()

class Ref_Insar_Selector:
    """
    Class to create an interactive tool to select a reference pixel from 
    a csv file that represents a downsampled InSAR dataset
    
    Attributes:
        wl (float): wavelength of the mission
        xref (float): x-coordinate for the reference
        yref (float): y-coordinate for the reference
        auxdata (tuple): auxiliary information (longitudes, latitudes, azimuth angles, incidence angles, uncertainties)
        los (array): list with line of sight deformation
        filename (str): path to the csv file with the downsampled InSAR dataset
        dataset (array): matrix with the line of sight deformation
    """
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
            self.vmin=-np.nanmax(np.abs(velocity))
            self.vmax=np.nanmax(np.abs(velocity))
        else:
            self.vmin=vmin
            self.vmax=vmax
        im=ax.imshow(self.dataset,cmap='jet',extent=self.extent,vmin=self.vmin, vmax=self.vmax)
        #line,=ax.plot([refcoords[0]], [refcoords[1]],'ko',label='Reference')
        line,=ax.plot([ref[0]], [ref[1]],'ko',label='Reference')
        
        if np.mean(np.abs(self.extent))<=180:
            ax.set_ylabel('Latitude (°)')
            ax.set_xlabel('Longitude (°)')
        else:
            ax.set_ylabel('North (m)')
            ax.set_xlabel('East (m)')

        plt.colorbar(im,orientation='horizontal')

        def on_click(event):
            row,col=ll2rc(event.xdata,event.ydata,self.extent,self.dataset.shape)
            
            row,col=get_closest_point(row,col,self.dataset)
            #line.set_xdata(event.xdata)
            #line.set_ydata(event.ydata)
            
            self.xref=event.xdata
            self.yref=event.ydata
            
            self.dataset-=self.dataset[row,col]
            
            #im.set_data(self.dataset)
            ax.clear()
            ax.imshow(self.dataset,cmap='jet',extent=self.extent,vmin=self.vmin, vmax=self.vmax)
            ax.plot([self.xref],[self.yref],'ko')
            if np.mean(np.abs(self.extent))<=180:
                ax.set_ylabel('Latitude (°)')
                ax.set_xlabel('Longitude (°)')
            else:
                ax.set_ylabel('North (m)')
                ax.set_xlabel('East (m)')
            
            self.los-=self.dataset[row,col]
            
            rewrite_csv(self.los,[self.xref,self.yref],old=csvfile,name=csvfile.split('.')[0]+'_ref.'+csvfile.split('.')[1])
            
        plt.legend()

        plt.connect('button_press_event', on_click)

        plt.show()
    
