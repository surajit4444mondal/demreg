import numpy as np
from matplotlib import pyplot as plt
from demmap_pos import demmap_pos
from dn2dem_pos import dn2dem_pos
import astropy.units as u
import os,pickle,sys
from sunpy.map import Map
from scipy.signal import convolve2d
import h5py
import scipy.io as io
from astropy.coordinates import SkyCoord
import threadpoolctl
threadpoolctl.threadpool_limits(5)

#nt=14
fits_dir=sys.argv[1]
outfile=sys.argv[2]
fov=sys.argv[3]

temp=fov.split(',')
x1=int(temp[0][1:])
x2=int(temp[1])
y1=int(temp[2])
y2=int(temp[3][:-1])

low_temp=5.2   ## in log10
high_temp=7.2  ## in log10
num_temp=40

files=sys.argv[4][1:-1].split(',')
fits_files=[fits_dir+filename for filename in files]

print (fits_files)

avg_length=int(sys.argv[5])  ### a kernel of avg_length x avg_length will be used to smooth the data



trin=io.readsav('/home/surajit/demreg/python/aia_tresp_en.dat')

#print (trin)

#correction_table = get_correction_table()
wavenum=['94','131','171','193','211','335']
channels = []
for i in np.arange(len(wavenum)):
    channels.append(float(wavenum[i])*u.angstrom)

#tresp_logt=tresp[:,0]
tresp_logt=np.array(trin['logt'])
tresp_calibration=np.array(trin['tr'])
trmatrix=tresp_calibration
nt=len(tresp_logt)
nf=len(trin['tr'][:])
trmatrix=np.zeros((nt,nf))
for i in range(0,nf):
    trmatrix[:,i]=trin['tr'][i]

temperatures=10**np.linspace(low_temp,high_temp,num=num_temp+1)
logtemps=np.linspace(low_temp,high_temp,num=num_temp+1)
#tresp = data=np.genfromtxt('tresp.csv',skip_header=1,delimiter=',')####read_csv('tresp.csv').to_numpy()
#start datetime

#we only want optically thin coronal wavelengths
wavenum=['94','131','171','193','211','335']

#load the fits with sunpy
aia = Map(fits_files)


#read dimensions from the header
nf=len(wavenum)

#normalise to dn/s
aia = [Map(m.data/m.exposure_time.value, m.meta) for m in aia]

#convert from our list to an array of data
for j in range(nf):
    bottom_left=SkyCoord(x1*u.arcsec,y1*u.arcsec,frame=aia[j].coordinate_frame)
    top_right=SkyCoord(x2*u.arcsec,y2*u.arcsec,frame=aia[j].coordinate_frame)
    if avg_length!=1:
        submap=aia[j].submap(bottom_left=bottom_left,top_right=top_right)
        if j==0:
            nx=int(submap.dimensions.x.value)
            ny=int(submap.dimensions.y.value)
            new_dimensions=(int(nx/avg_length),int(ny/avg_length))*u.pixel
        aia_resampled = submap.resample(new_dimensions)
        if j==0:
            nx=int(aia_resampled.dimensions.x.value)
            ny=int(aia_resampled.dimensions.y.value)
            #create data array
            data=np.zeros([ny,nx,nf])
        data[:,:,j]=aia_resampled.data
        del aia_resampled,submap,bottom_left,top_right
    else:
        if j==0:
            #create data array
            nx=int(aia[j].dimensions.x.value)
            ny=int(aia[j].dimensions.y.value)
            data=np.zeros([ny,nx,nf])
        data[:,:,j]=aia[j].data
data[data < 0]=0

shape=data.shape




serr_per=10.0
#errors in dn/px/s
npix=avg_length*avg_length
edata=np.zeros_like(data)
gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
dn2ph=gains*[94,131,171,193,211,335]/3397.0
rdnse=np.array([1.14,1.18,1.15,1.20,1.20,1.18])*np.sqrt(npix)/npix  ## previously it was 1.15
drknse=0.17
qntnse=0.288819*np.sqrt(npix)/npix
for j in np.arange(nf):
    etemp=np.sqrt(rdnse[j]**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[:,:,j]))/(npix*dn2ph[j]**2))
    esys=serr_per*data[:,:,j]/100.
    edata[:,:,j]=np.sqrt(etemp**2. + esys**2.)

dem,edem,elogt,chisq,dn_reg=dn2dem_pos(data,edata,trmatrix,tresp_logt,temperatures,max_iter=50)


logt_bin=np.zeros(num_temp)
for i in np.arange(num_temp):
    logt_bin[i]=(logtemps[i]+logtemps[i+1])/2

output={'logt_bin':logt_bin,'elogt':elogt,'dem':dem,'edem':edem,'chisq':chisq,'dn_reg':dn_reg}

hf=h5py.File(outfile,"w")

hf.attrs['files']=sys.argv[5][1:-1]
hf.attrs['x1']=x1
hf.attrs['y1']=y1
hf.attrs['x2']=x2
hf.attrs['y2']=y2
hf.attrs['coord unit']='arcsec'
hf.attrs['avg_length']=avg_length
hf.create_dataset('logt_bin',data=logt_bin)
hf.create_dataset('elogt',data=elogt)
hf.create_dataset('dem',data=dem)
hf.create_dataset('chisq',data=chisq)
hf.create_dataset('dn_reg',data=dn_reg)
hf.create_dataset('edem',data=edem)
hf.close()
#pickle.dump(output,open(outfile,"wb"))

