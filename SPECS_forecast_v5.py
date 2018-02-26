# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import unicode_literals
import os, sys, glob, re, pickle, time
import numpy as np
import numpy.ma as ma
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from netCDF4 import Dataset
from scipy import stats
from scipy.stats import linregress,pearsonr
from sklearn import  linear_model
from sklearn.preprocessing import Imputer
import urllib2
import zipfile
from SPECS_forecast_v5_tools import *
from cdo import *
cdo = Cdo()
from pyresample import geometry,image, kd_tree
import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xarray as xr
import pandas as pd

# TODO
# Check climatology period for different datasets.. jonathan had 1981-2010
# Check if there are higher resolution datafiles when choosing higher resolution..

dt = datetime.date.today()
date_list = [dt.year, dt.month, dt.day]
start0 = time.time()

predictands = ["GCEcom","20CRslp","GPCCcom"]
#predictands = ["20CRslp","GPCCcom"]
predictands = ["GPCCcom"]
#predictands = ['GCEcom']

bd = '/nobackup/users/krikken/SPESv2/'
bdid = bd+'inputdata/'
bdp = bd+'testplots/'
bdnc = bd+'ncfiles/test/'

# Load these predictors, this does not mean that these are neceserally used.. see predictorz for those
predictors_1d = ['CO2EQ','NINO34','PDO','AMO','IOD']
predictors_3d = ['PERS','CPREC']
predictors = ['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS']

# Select method how to run

# NAMELIST

## Resolution, currently only 25 or 50 is supported..
resolution = 25             # 10, 25 or 50

## Redo full hindcast period and remove original nc output file?
overwrite = True

## Redo a specific month / year?
overwrite_m = False         # Overwrite only the month specified with overw_m and overw_y
overw_m = 5                 # Jan = 1, Feb = 2.. etc
overw_y = 2017



UPDATE = True

## Save a figure with the correlation between predictors and predictand
PLOT_PREDCOR =  True

##
VALIDATION =    True         # Validates and makes figures of predicted values 

DYN_MONAVG = False          # Include the dynamical monthly averaging in the predictors
MLR_PRED = True           # Include the trend of the last 3 month as predictor

FORECAST = True           # Do forecast for given date?
HINDCAST = True       # Validate forecast using leave n-out cross validation?
CROSVAL = True
CAUSAL = False
cv_years = 3                       # Leave n out cross validation

## Validation period is 1961 - current

ens_size =      51
styear  =       1901    # Use data from this year until current
stvalyear =     1961    # Start validation from this year until previous year
endyear =       dt.year
endmonth =      dt.month-1  # -1 as numpy arrays start with 0
tot_time = (dt.year - styear) * 12 + endmonth

# Defining some arrays used for writing labels and loading data
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthzz = 'JFMAMJJASONDJFMAMJJASOND'

print 'Data from Jan '+str(styear)+' up to '+str(months[dt.month-2])+' '+str(dt.year)

print 'Predictands = ',predictands
print 'Predictors = ',predictors
print 'Horizontal resolution is ',str(resolution/10.),' degrees'

## Predefine arrays and make lats,lons according to specified resolution

## All url for data downloads

times = pd.date_range('1901-01',str(dt.year)+'-'+str(dt.month),freq='M')

if resolution == 10:
    targetgrid = 'griddes10.txt'        # grid description file used for regridding
    latx = 180; lonx = 360
    latz = np.arange(89.5,-90.,-1.)
    lonz = np.arange(-179.5,180.,1.)
    #predadata = np.zeros((len(predictands),tot_time,180,360))
elif resolution == 25:
    targetgrid = 'griddes25.txt'        # grid description file used for regridding
    #predadata = np.zeros((len(predictands),tot_time,72,144))
    latx = 72; lonx = 144
    latz = np.arange(88.75,-90.,-2.5)
    lonz = np.arange(-178.75,180.,2.5)
    #predadata = xr.DataArray(
elif resolution == 50:
    targetgrid = 'griddes50.txt'        # grid description file used for regridding
    #predadata = np.zeros((len(predictands),tot_time,36,82))
    latx = 36; lonx = 82
    latz = np.arange(87.5,-90.,-5)
    lonz = np.arange(-177.5,180.,5)

#sys.exit()   

# ************************************************************************
# Read in predictand data for fitting 
# ************************************************************************
start1 = time.time()
print '-- Read in predictand data for fitting --'
predadata = []
predodata = []
predictorz = []     # Predefine empty array, fill with specified predictors for predictand
#predictorz_1d = []
#predictorz_3d = []

UPDATE_INDICES = check_updates2()
if UPDATE_INDICES and UPDATE:
    
    import subprocess
    print "start updating monthly observations"
    subprocess.check_call(["./update_indices.sh",str(resolution)])
    print "done updating monthly opbserations"


for p,predictand in enumerate(predictands):
    
        
    if predictand == 'GISTEMP':
        print 'not done yet..'
        
    
    elif predictand == 'GCEcom':
        ## These predictors are selelected for GCEcom in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS'])
        gcecom = xr.open_dataset('inputdata/gcecom_r'+str(resolution)+'.nc').squeeze()
        gcecom = gcecom.drop('lev')
        gcecom = gcecom.rename({'sst':'tas'})
        gcecom.time.values = rewrite_time(gcecom)
        
        mask = np.sum((gcecom.tas.round(3) == 271.350),axis=0)>100.
        mask[61:,:] = True
        
        gcecom.tas.values[:,mask]= np.nan
        gcecom.tas.values = anom_df(gcecom.tas,1980,2010,1948)
        
        hadcrucw = xr.open_dataset('inputdata/had4_krig_v2_0_0_r'+str(resolution)+'.nc')
        hadcrucw = hadcrucw.rename({'temperature_anomaly':'tas'})
        hadcrucw = hadcrucw.drop('year')
        hadcrucw = hadcrucw.drop('month')
        hadcrucw.time.values = rewrite_time(hadcrucw)
        
        hadcrucw.tas.values = anom_df(hadcrucw.tas,1980,2010,styear)
         
        
        # Hadcrucw until 1957, and gcecom from 1958
        com = xr.concat([hadcrucw.tas[:684,:],gcecom.tas[120:,:]],dim='time')
        
        #com = com.rename('GCEcom')
        if p == 0: predadata = xr.Dataset({'GCEcom':com.astype('float32')}) 
        else: predadata = xr.merge([predadata,com.astype('float32')])
        

    elif predictand == 'HadCRU4CW':
        # to do
        print 'to do'

    elif predictand == 'GPCCcom':
        # These predictors are selelected for GPCCcom in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','AMO','IOD','PERS'])
        gpcccom = xr.open_dataset('inputdata/gpcc_10_combined_r'+str(resolution)+'.nc')
        gpcccom.prcp.values = anom_df(gpcccom.prcp,1980,2010,styear)
        gpcccom.time.values = rewrite_time(gpcccom)

        #gpccprec[gpccprec<=-1000.] = np.nan
        #clim_gpcc.values = clim(gpcccom.prcp,1980,2010,styear)
        #clim_gpcc2 = gpcccom.groupby('time.month').mean('time')
        gpcccom = gpcccom.prcp.rename('GPCCcom')
        if p == 0: predadata = xr.Dataset({'GPCCcom':gpcccom.astype('float32')})
        else: predadata = xr.merge([predadata,gpcccom.astype('float32')])
        #predadata[p,:] = anom(gpccprec,1980,2010,styear)

    elif predictand == '20CRslp':
        # These predictors are selelected for 20CRslp in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS'])
        slp = xr.open_dataset('inputdata/slp_mon_mean_1901-current_r25.nc')
        slp.prmsl.values = anom_df(slp.prmsl,1980,2010,styear)
        slp = slp.prmsl.rename('20CRslp')
        slp.time.values = rewrite_time(slp)
        if p == 0: predadata = xr.Dataset({'20CRslp':slp.astype('float32')})
        else: predadata = xr.merge([predadata,slp.astype('float32')])

    else:
        print 'predictand not yet known.. exiting!'
        sys.exit()

 
print '-- Done reading in predictand data for fitting, time = ',str(np.int(time.time()-start1)),' seconds --' 

# ************************************************************************
# Read in predictor data for fitting 
# ************************************************************************
start1 = time.time()
print '-- Read in predictor data for fitting --'
#predodata_1d = np.zeros((len(predictors_1d),predadata.shape[1]))
#predodata_3d = np.zeros((len(predictors_3d),predadata.shape[1],latx,lonx))
#predodata = []
#predodata = np.zeros((len(predictors),predadata.shape[1],latx,lonx))

indices_locs = {'CO2EQ':'inputdata/RCP45_CO2EQ_mo.dat',
                'NINO34':'inputdata/ersst_nino3.4a.dat',
                'QBO':'inputdata/qbo?.dat',
                'IOD':'inputdata/dmi_ersst.dat',
                'PDO':'inputdata/pdo_ersst.dat',
                'AMO':'inputdata/amo_ersst_ts.dat',
                'PERS':'not needed',
                'PERS_TREND':'not needed'}

# Load predictors (time,lat,lon)
for i,pred in enumerate(predictors):
    print i,pred
    if pred in ['PERS','PERS_TREND']:
        continue

    elif pred == 'CPREC':    # Cum precip [time,lat,lon] - 1901 -current
        gpcccom = xr.open_dataset('inputdata/gpcc_10_combined_r'+str(resolution)+'.nc')
        gpcccom.prcp.values = anom_df(gpcccom.prcp,1980,2010,styear)
        gpcccom = gpcccom.prcp.rename('CPREC')
        gpcccom.time.values = rewrite_time(gpcccom)
        if i == 0: predodata = df
        else: predodata = xr.merge([predodata,gpcccom.astype('float32')])

    else:
        df = load_clim_index(indices_locs[pred],styear,endyear,endmonth,pred)
        if i==0: predodata = df
        else: predodata = xr.merge([predodata,df.astype('float32')])

        
print '-- Done reading in predictor data for fitting, time = ',str(np.int(time.time()-start1)),' seconds --' 



# *************************************************************************   
# Now start the predictor selection and the forecasting / hindcasting loop
# *************************************************************************

for p,predictand in enumerate(predictands):
    # Fill persistence predictor with predictand data
    if 'PERS' in predictorz[p]:
        if 'PERS' in predodata:  predodata = predodata.drop('PERS')
        predodata = xr.merge([predodata,predadata[predictand].rename('PERS')])
    
    print 'Predictand: ',predictand
    print 'Predictors: ',predictorz[p]

    if overwrite:
        print 'overwrite is True, so remove all data and redo complete forecast cycle'
        mon_range = range(12)
        filez = glob.glob('ncfiles/test/*'+predictand+'.nc')
        for fil in filez: os.remove(fil)
    elif overwrite_m:
        mon_range = [overw_m]
    else:
        try:    
            datanc = xr.open_dataset(bdnc+'pred_v2_'+predictand+'.nc')
            # Create list of dates which are needed to update the data
            monrange=pd.date_range(pd.to_datetime(datanc['time'].values[-1]),pd.to_datetime(predadata['time'].values[-1])+pd.DateOffset(months=1),freq='MS')[1:]
            if len(monrange) == 0:
                print 'Data already up to date... continue to next in loop'
                continue
            mon_range = range(monrange[0].month-1,monrange[-1].month)
            #print mon_range
        except IOError:  # If file does not exist do full hindcast
            mon_range = range(12)
            print 'no previous output, do full hindcast!'

    #if FORECAST and not HINDCAST: # Only redo forecast loop
    #    mon_range = range(12)
        

    
    # Rolling 3-month mean, first and last timestep become nans, so remove last timestep (slice(None,-1))
    predodata_3m  = predodata[predictorz[p]].rolling(time=3,center=True).mean().isel(time=slice(None,-1))
    predadata_3m = predadata[predictand].rolling(time=3,center=True).mean().isel(time=slice(None,-1))
    
    # Change time values of predictor and predictand data for simplicity. Add 2 months for predictor data and subtract 2 months for predicand data.
    predodata_3m['time'].values = pd.DatetimeIndex(predodata_3m['time'].values) + pd.DateOffset(months=2)
    predadata_3m['time'].values = pd.DatetimeIndex(predadata_3m['time'].values) - pd.DateOffset(months=2)
    
    if MLR_PRED: 
        print 'MLR_PRED is True, calculating trend over previous 3 months for all predictors'
        predodata_3m_trend = pred_trend(predodata[predictorz[p]]).isel(time=slice(None,-1))
        predodata_3m_trend['time'].values = pd.DatetimeIndex(predodata_3m_trend['time'].values) + pd.DateOffset(months=2)
                
    for m in mon_range:
        
        print 'prediction month = ',months[m]
        print 'predictor season = ', '-'.join([months[m-3],months[m-2],months[m-1]])
        try: print 'predictor season = ', '-'.join([months[m+1],months[m+2],months[m+3]])
        except IndexError: print 'predictor season = ', '-'.join([months[m-11],months[m-10],months[m-9]])
        
        if HINDCAST and CAUSAL:
            print 'Hindcasting mode, causal ',str(stvalyear),'-current'
            timez = pd.DatetimeIndex(predodata['time'].sel(time=predodata['time.month']==m+1).values)[1:]
            if timez[-1].month + 2 >= times[-1].month: timez = timez[:-1]
            #times_p, times_f = timez - pd.DateOffset(months=2), timez + pd.DateOffset(months=2)
            
            for y in range(np.where(timez.year == stvalyear)[0],len(timez)):
                train = timez[:y]
                test = timez[y]

                if predictand == 'GPCCcom':
                    train = train[50:]
                
                print 'test years: ',test

                regr_loop(predodata_3m,predodata_3m_trend,predadata_3m,timez,train,test,ens_size,bdnc,25,predictand,predictorz[p], MLR_PRED,FC=False,sig_val = 0.1) # hier wil ik naartoe

            
        elif HINDCAST and CROSVAL:
            print 'Hindcasting mode, leave ',str(cv_years),' out cross-validation'
            timez = pd.DatetimeIndex(predodata['time'].sel(time=predodata['time.month']==m+1).values)[1:-1]

            for y in range(np.where(timez.year == stvalyear)[0],len(timez),cv_years):
                test = timez[y:y+3]
                train = timez.drop(test)
                
                if predictand == 'GPCCcom':
                    train = train[50:]
                
                print 'test years: ',test

                regr_loop(predodata_3m,predodata_3m_trend,predadata_3m,timez,train,test,ens_size,bdnc,25,predictand,predictorz[p], MLR_PRED,FC=False,sig_val = 0.1) # hier wil ik naartoe
                

        elif HINDCAST:
            print 'either CROSVAL or CAUSAL should be set to true'
            sys.exit()

        if FORECAST:
            print 'Forecasting mode'
            timez = pd.DatetimeIndex(predodata_3m['time'].sel(time=predodata_3m['time.month']==m+1).values)
            if timez[0].year == 1901: timez = timez[1:]

            # In forecast mode, the last timestep is the test data, the rest is training data.
            train = timez[:-1]
            test = timez[-1]
            if predictand == 'GPCCcom':
                train = train[50:]

            print 'test years: ',test

            regr_loop(predodata_3m,predodata_3m_trend,predadata_3m,timez,train,test,ens_size,bdnc,25,predictand,predictorz[p],MLR_PRED,FC=True,sig_val = 0.1) # hier wil ik naartoe
            
                    
    # Merge all output files
    #cdo.mergetime(input=bdnc+'pred_v2_'+predictand+'_*.nc',output = bdnc+'pred_v2_'+predictand+'.nc')
    #cdo.mergetime(input=bdnc+'beta_v2_'+predictand+'_*.nc',output = bdnc+'beta_v2_'+predictand+'.nc')
    #for f1,f2 in zip(glob.glob(bdnc+'pred_v2_'+predictand+'_*.nc'),glob.glob(bdnc+'beta_v2_'+predictand+'_*.nc')):
    #    os.remove(f1)
    #    os.remove(f2)
    
    # Sort all data by time dimension
    filn = ['pred_v2_','beta_v2_','predictors_v2_','predictors_fit_v2_']
    for fil in filn:
        tmp = xr.open_dataset(bdnc+fil+predictand+'.nc')
        tmp.sortby('time').to_netcdf(bdnc+fil+predictand+'2.nc','w')
        os.remove(bdnc+fil+predictand+'.nc')
        os.rename(bdnc+fil+predictand+'2.nc',bdnc+fil+predictand+'.nc')
        
    
    
    if VALIDATION:
        print 'Calculating skill scores' 
        
        data_fit = xr.open_dataset(bdnc+'pred_v2_'+predictand+'.nc')#,chunks={'lat':1})
        for m in mon_range:
            t0 = time.time()
            timez = pd.DatetimeIndex(data_fit['time'].sel(time=data_fit['time.month']==m+1).values)
            scorez = xr.Dataset(coords={'lat': data_fit['lat'],
                    'lon': data_fit['lon'],
                    'time': [timez[-1]]})

            print 'month = ',str(m+1)
            # Create correlation scores and its significance
            cor,sig = linregrez(data_fit.kprep.mean('ens').sel(time=timez[:-1]).values,data_fit.obs.sel(time=timez[:-1]).values,COR=True)

            scorez['cor'] = (('time','lat','lon'),cor[np.newaxis,:,:])
            scorez['cor_sig'] = (('time','lat','lon'),sig[np.newaxis,:,:])
            
            scorez['rmsess'] = (('time','lat','lon'),                            f_rmse(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['obs'].sel(time=timez[:-1]).values,ref=data_fit['clim'].sel(time=timez[:-1]).values,SS=True)[np.newaxis,:,:])
         
            scorez['crpss'] = (('time','lat','lon'),                        f_crps2(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['obs'].sel(time=timez[:-1]).values,SS=True,ref=data_fit['clim'].sel(time=timez[:-1]).values)[np.newaxis,:,:])
            
            scorez['tercile'] = (('time','lat','lon'), tercile_category(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['kprep'].sel(time=timez[-1]).values)[np.newaxis,:,:])
            
            scorez['for_anom'] = (('time','lat','lon'),data_fit['kprep'].sel(time=timez[-1]).mean(dim='ens').values[np.newaxis,:,:])
            
            to_nc2(scorez,bdnc + 'scores_v2_'+predictand)
            print 'Calculated skill scores for '+months[m]+', total time is '+str(np.round(time.time()-t0,2))+' seconds'
        if os.path.isfile(bdnc+'scores_v2_'+predictand+'.nc'):
            tmp = xr.open_dataset(bdnc+'scores_v2_'+predictand+'.nc')
            tmp.sortby('time').to_netcdf(bdnc+'scores_v2_'+predictand+'2.nc','w')
            os.remove(bdnc+'scores_v2_'+predictand+'.nc')
            os.rename(bdnc+'scores_v2_'+predictand+'2.nc',bdnc+'scores_v2_'+predictand+'.nc')

    if VALIDATION:
        #predictand = 'GCEcom' 
        print 'Start validation for the last year in mon_range'
        data_fit = xr.open_dataset(bdnc+'pred_v2_'+predictand+'.nc')
        scores = xr.open_dataset(bdnc+'scores_v2_'+predictand+'.nc')

            
        if predictand == 'GCEcom':      
            var = 'Surface air temperature'
            clevz = np.array((-2.,-1.,-0.5,-0.2,0.2,0.5,1.,2.))
            cmap1 = matplotlib.colors.ListedColormap(['#000099','#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400','#cc0022'])
            cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
            cmap_under = '#000099'
            cmap_over = '#cc0022'
        elif predictand == 'GPCCcom':   
            var = 'Surface precipitation' 
            clevz = np.array((-200.,-100.,-50.,-20.,20.,50.,100.,200.))
            cmap1 = matplotlib.colors.ListedColormap(['#993300','#cc8800','#ffcc00','#ffee99','#ffffff','#ccff66','#33ff00','#009933','#006666'])
            cmap2 = matplotlib.colors.ListedColormap(['#cc8800','#ffcc00','#ffee99','#ffffff','#ccff66','#33ff00','#009933'])
            cmap_under = '#993300'
            cmap_over = '#006666'
            
        elif predictand == '20CRslp':
            var = 'Mean sea level pressure'
            clevz=np.array((-4.,-2.,-1.,-0.5,0.5,1.,2.,4.))
            cmap1 = matplotlib.colors.ListedColormap(['#000099','#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400','#cc0022'])
            cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
            cmap_under = '#000099'
            cmap_over = '#cc0022'

        
         
        #for m in (np.unique(timez[0,:])-1).astype(int):
        for m in mon_range:
        #for m in range(12):
                mon = str(m+1).zfill(2)
                timez = pd.DatetimeIndex(data_fit['time'].sel(time=data_fit['time.month']==m+1).values)
                year = str(timez[-1].year)
                season = monthzz[m+1:m+4]
                print 'validation for '+season+' '+year
                
                bdpo = bdp+predictand+'/'+str(resolution)+'/'+year+mon+'/'
                if not os.path.exists(bdpo):
                    os.makedirs(bdpo)
                
               
                # Calculate significance of ensemble mean
                tmp = data_fit.kprep.sel(time=timez[-1]).mean('ens').values
                posneg = tmp > 0.
                above = 1.-(np.sum(data_fit.kprep.sel(time=timez[-1]).values>0,axis=0)/51.)
                below = 1.-(np.sum(data_fit.kprep.sel(time=timez[-1]).values<0,axis=0)/51.)
                sig_ensmean = np.ones((len(data_fit.lat),len(data_fit.lon)))
                sig_ensmean[posneg] = above[posneg]
                sig_ensmean[~posneg] = below[~posneg]
                
                
                plot_climexp(scores.rmsess.sel(time=timez[-1]).values,
                             'RMSESS hindcasts, climatology as reference (1961-current)',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                             predictand = predictand,
                             fname=bdpo+predictand+'_rmsess_'+year+mon+'.png',
                             clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                             cmap=cmap2,
                             cmap_under = cmap_under,
                             cmap_over = cmap_over,
                             PLOT=False,
                             )
                plot_climexp(scores.crpss.sel(time=timez[-1]).values,
                             'CRPSS hindcasts, reference: climatology (1961-'+str(int(year)-1)+')',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                             predictand = predictand,
                             cmap=cmap2,
                             cmap_under = cmap_under,
                             cmap_over = cmap_over,
                             fname=bdpo+predictand+'_crpss_'+year+mon+'.png',
                             clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                             )    
                #plot_climexp(crps_pred_co2,
                             #'CRPSS hindcasts, reference: hindcasts with only CO2 as predictor (1961-'+str(year_nc-1)+')',
                             #'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             #'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                             #predictand = predictand,
                             #cmap = cmap2,
                             #fname=bdpo+predictand+'_crpss_detrended_clim_'+year+mon+'.png',
                             #clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                             #cmap_under = cmap_under,
                             #cmap_over = cmap_over,
                             #)    
                plot_climexp(data_fit.kprep.sel(time=timez[-1]).mean('ens').values,
                             'Ensemble mean anomaly (wrt 1980-2010)',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+'  |  Stippled where NOT significant at 10% level'+' | base time: '+months[m]+' '+year,
                             sig=sig_ensmean,
                             cmap=cmap2,
                             predictand = predictand,
                             cmap_under = cmap_under,
                             cmap_over = cmap_over,
                             fname=bdpo+predictand+'_ensmean_'+year+mon+'.png', 
                             clevs = clevz,
                             MEAN=True,
                             )
                plot_climexp(scores.cor.sel(time=timez[-1]).values,
                             'Correlation between hindcast anomaly and observations (1961-'+str(int(year)-1)+'',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+'  |  Stippled where signficant at 5% level'+' | base time: '+months[m]+' '+year,
                             sig = scores.cor_sig.sel(time=timez[-1]).values,
                             predictand = predictand,
                             fname=bdpo+predictand+'_correlation_'+year+mon+'.png', 
                             clevs = np.arange(-1.,1.01,0.2),
                             )
                plot_climexp(scores.tercile.sel(time=timez[-1]).values,
                             'Probabilty (most likely tercile of '+var+'), based on 1961-'+str(int(year)-1)+' hindcasts',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                             cmap = cmap1,
                             predictand = predictand,
                             fname=bdpo+predictand+'_tercile_'+year+mon+'.png', 
                             clevs = np.array((-100,-70,-60,-50,-40,40,50,60,70,100)),
                             barticks = ['100%','70%','60%', '50%', '40%', '40%', '50%','60%', '70%', '100%'],
                             TERCILE=True,
                             )
                             #plt.annotate('<---- below lower tercile        

                
            
                
    import time        
    print 'Total time taken is: ',np.int((time.time()-start0)//60),' minutes and ',np.int((time.time()-start0)%60), 'seconds'


## Regression loop function
TEST = False
if TEST:
    
    # Data from 1981 onwards..
    # S5 data are note anomalies, hence either make them anomalies or add climatology to our data..
    # First try to add climatology (1980-2010) to our data..
    
    obs_seas = nans_like(predodata[p,-682:,:])
    for m in range(12):
        tmp = seazon_prad(predadata[p,-682:,:]+clim_gcecom[-682:,:],m)
        obs_seas[m::12,:][:tmp.shape[0],:] = tmp
    clim_seas = clim(obs_seas,1980,2010,1961,keepdims=True)    
    
    nc1 = Dataset('ncfiles/pred_v2_GCEcom_causal.nc')  # Baseline forecast
    fc1 = nc1.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    ref = nc1.variables['GCEcom_ref'][:] #+ clim_seas[:,None,:,:]
    obs = nc1.variables['GCEcom_obs'][:] #+ clim_seas
    lons = nc1.variables['longitude'][:]
    lats = nc1.variables['latitude'][:]
    nc1.close()

    nc2 = Dataset('ncfiles/pred_v2_GCEcom_l3o.nc') # Fit trend and mean of predictor on predictand
    fc2 = nc2.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    ref2 = nc2.variables['GCEcom_ref'][:] #+ clim_seas[:,None,:,:]
    #obs2 = nc2.variables['GCEcom_obs'][:]
    nc2.close()

    nc3 = Dataset('ncfiles/pred_v2_GCEcom.nc') # Fit trend and mean of predictor on predictor (Seems physically better)
    fc3 = nc3.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    ref3 = nc3.variables['GCEcom_ref'][:]
    #obs3 = nc3.variables['GCEcom_obs'][:]
    nc3.close()

    #nc4 = Dataset('ncfiles/pred_v2_GCEcom_l3o_mlr_noPERS.nc') # Use trend and mean as separate predictors
    #fc4 = nc4.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    #ref4 = nc4.variables['GCEcom_ref'][:]
    #nc4.close()

    #nc5 = Dataset('ncfiles/pred_v2_GCEcom_l3o_mlr_noPERS_ens.nc') # Use trend and mean as separate predictors
    #fc5 = nc5.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    #ref4 = nc4.variables['GCEcom_ref'][:]
    #nc5.close()

    #nc5 = Dataset('ncfiles/pred_GCEcom_dmavg.nc') # Use either 1, 3 or 5 monthly average for predictors
    #fc5 = nc5.variables['GCEcom_fc'][:] 
    #nc5.close()

    #nc6 = Dataset('ncfiles/pred_GCEcom_dmavg_trend.nc') # Use either 1, 3 or 5 monthly average for predictors
    #fc6 = nc6.variables['GCEcom_fc'][:] 
    #nc6.close()

    for m in [1,4,7,10]:
        # Load ecmwf data
        s5 = load_ecmwf2(var='t2m',m=m,anom=True)
        #bias = np.nanmean(np.nanmean(s5 - clim_seas[m::12,None,:,:][20:-1,:],axis=0),axis=0)
        
        
        #crps1 = f_crps(fc1[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref[m::12,:][-37:-1,:])
        #crps2 = f_crps(fc2[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref2[m::12,:][-37:-1,:])
        #crps3 = f_crps(fc3[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref2[m::12,:][-37:-1,:])
        #crps4 = f_crps(fc4[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref3[m::12,:][-37:-1,:])
        #crps5 = f_crps(fc5[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=ref[m::12,:][:-1,:])
        #crps6 = f_crps(fc6[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=ref[m::12,:][:-1,:])
        #crps_s5 = f_crps(s5,obs[m::12,:][-37:-1,:],SS=True,ref=ref2[m::12,:][-37:-1,:])
        #crps1_2 = f_crps(fc1[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=fc2[m::12,:][:-1,:])
        #crps1_3 = f_crps(fc1[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=fc3[m::12,:][:-1,:])
        #crps5_4 = f_crps(fc5[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=fc4[m::12,:][:-1,:])
        crps3_s5 = f_crps(fc3[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=s5[:])
        #crpss5 = f_crps(s5,obs[m::12,:],SS=True,ref=ref3[m::12,:][-37:-1,:])
        #crps2_s5 = f_crps(fc2[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=s5[:])
        #crps2_1 = f_crps(fc2[m::12,:][:-1,:],obs[m::12,:][59:-1,:],SS=True,ref=fc1[m::12,:][59:-1,:])
        #crps3_2 = f_crps(fc3[m::12,:][:-1,:],obs[m::12,:][59:-1,:],SS=True,ref=fc2[m::12,:][:-1,:])
        cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
        cmap_u = '#000099'
        cmap_o = '#cc0022'
        clev1 = np.array((-0.5,-0.35,-0.2,-0.1, 0.1, 0.2,0.35,0.5))
        clev2 = np.array((-0.2,-0.1, -0.05,-0.025,0.025,0.05,0.1, 0.2))
        #plotdata(crps1,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='causal',fname=str(m)+'crps1.png',PLOT=False,extend=True)
        #plotdata(crps2,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='leave-3-out',fname=str(m)+'crps2.png',PLOT=False,extend=True)
        #plotdata(crps3,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='leave-3-out mlr',fname=str(m)+'crps3.png',PLOT=False,extend=True)
        #plotdata(crps4,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='l3o mlr noPERS',fname=str(m)+'crps4.png',PLOT=False,extend=True)
        #plotdata(crps_s5,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='s5',fname=str(m)+'crps_s5.png',PLOT=False,extend=True)
        #plotdata(crps6,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='dmavg_pred',fname=str(m)+'crps6.png',PLOT=False,extend=True)

        #plotdata(crps2_1,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps2_1',fname=str(m)+'crps1_2.png',PLOT=False,extend=True)
        #plotdata(crps3_2,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps3_2',fname=str(m)+'crps2_3.png',PLOT=False,extend=True)
        #plotdata(crps2_s5,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps2_s5',fname=str(m)+'crps2_s5.png',PLOT=False,extend=True)
        plotdata(crps3_s5,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps3_s5',fname=str(m)+'crps3_s5.png',PLOT=False,extend=True)
        #plotdata(crps1_6,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps1_6',fname=str(m)+'crps1_6.png',PLOT=False,extend=True)
        #plotdata(crps5_4,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps5_4',fname=str(m)+'crps5_4.png',PLOT=False,extend=True)


