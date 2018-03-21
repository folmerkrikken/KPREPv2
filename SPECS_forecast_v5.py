# -*- coding: utf-8 -*-
#!/usr/bin/env python


import os, sys, glob, re, pickle, time
import numpy as np
import numpy.ma as ma
import scipy
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from netCDF4 import Dataset
from scipy import stats
from scipy.stats import linregress,pearsonr
from sklearn import  linear_model
from sklearn.preprocessing import Imputer
import urllib.request, urllib.error, urllib.parse
import zipfile
from SPECS_forecast_v5_tools import *
from cdo import *
cdo = Cdo()
import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xarray as xr
import pandas as pd

## TODO
# Check for valid values of climate indices, if the timestap is updated does not indicate the value is updated..
## Fix causal hindcast

dt = datetime.date.today()
date_list = [dt.year, dt.month, dt.day]
start0 = time.time()

predictands = ["GCEcom","20CRslp","GPCCcom"]

predictors = ['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS']


# Load these predictors, this does not mean that these are neceserally used.. see predictorz for those


# NAMELIST

## Resolution, currently only 25 or 50 is supported..
resolution = 25             # 10, 25 or 50

## Redo full hindcast period and remove original nc output file?
overwrite = False

## Redo a specific month / year?
overwrite_m = False         # Overwrite only the month specified with overw_m and overw_y
overw_m = 5                 # Jan = 1, Feb = 2.. etc
overw_y = 2017


## Update indices if possible
UPDATE = True

## Save a figure with the correlation between predictors and predictand
PLOT_PREDCOR = True

##
VALIDATION = True           # Validates and makes figures of predicted values 

DYN_MONAVG = False          # Include the dynamical monthly averaging in the predictors
MLR_PRED = True             # Include the trend of the last 3 month as predictor

FORECAST = True             # Do forecast for given date?
HINDCAST = True             # Validate forecast using leave n-out cross validation?
CROSVAL = True              # Use cross-validation for validation
CAUSAL = False              # Use causal method for validation
cv_years = 3                # Leave n out cross validation

## Validation period is 1961 - current

ens_size =      51
styear  =       1901    # Use data from this year until current
stvalyear =     1961    # Start validation from this year until previous year
endyear =       dt.year
endmonth =      dt.month-1  # -1 as numpy arrays start with 0

# Set working directories
bd = '/nobackup_1/users/krikken/KPREP/'
bdid = bd+'inputdata/'
bdp = bd+'plots/'
bdnc = bd+'ncfiles/'


# Defining some arrays used for writing labels and loading data
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthzz = 'JFMAMJJASONDJFMAMJJASOND'

print('Data from Jan '+str(styear)+' up to '+str(months[dt.month-2])+' '+str(dt.year))

print('Predictands = ',predictands)
print('Predictors = ',predictors)
print('Horizontal resolution is ',str(resolution/10.),' degrees')


# Create list of dates used
times = pd.date_range('1901-01',str(dt.year)+'-'+str(dt.month),freq='M')


# ************************************************************************
# Read in predictand data for fitting 
# ************************************************************************
start1 = time.time()
print('-- Read in predictand data for fitting --')
predictorz = []     # Predefine empty array, fill with specified predictors for predictand

UPDATE_INDICES = check_updates2()
if UPDATE_INDICES and UPDATE:
    import subprocess
    print("start updating monthly observations")
    subprocess.check_call(["./update_indices.sh",str(resolution)])
    print("done updating monthly opbserations")


for p,predictand in enumerate(predictands):
    
        
    if predictand == 'GISTEMP':
        print('not done yet..')
        
    
    elif predictand == 'GCEcom':
        ## These predictors are selelected for GCEcom in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS'])
        gcecom = xr.open_dataset(bdid+'gcecom_r'+str(resolution)+'.nc').squeeze()
        gcecom = gcecom.drop('lev')
        gcecom = gcecom.rename({'sst':'tas'})
        gcecom.time.values = rewrite_time(gcecom)
        
        # Create mask where no data / sea ice / too many missing values etc.
        mask = np.sum((gcecom.tas.round(3) == 271.350),axis=0)>100. 
        mask[61:,:] = True
        
        # Create anomalies with 1980-2010 as baseline climatology
        
        hadcrucw = xr.open_dataset(bdid+'had4_krig_v2_0_0_r'+str(resolution)+'.nc')
        hadcrucw = hadcrucw.rename({'temperature_anomaly':'tas'})
        hadcrucw = hadcrucw.drop('year')
        hadcrucw = hadcrucw.drop('month')
        hadcrucw.time.values = rewrite_time(hadcrucw) # TODO, make this simpler..
        
        # Create anomalies with 1980-2010 as baseline climatology
        gcecom.tas.values = anom_df(gcecom.tas,1980,2010,1948)
        hadcrucw.tas.values = anom_df(hadcrucw.tas,1980,2010,styear)
         
        # Hadcrucw until 1957, and gcecom from 1958
        com = xr.concat([hadcrucw.tas[:684,:],gcecom.tas[120:,:]],dim='time')
        com.values[:,mask] = np.nan
        #com = com.rename('GCEcom')
        if p == 0: predadata = xr.Dataset({'GCEcom':com.astype('float32')}) 
        else: predadata = xr.merge([predadata,com.astype('float32')])
        

    elif predictand == 'HadCRU4CW':
        # to do
        print('to do')

    elif predictand == 'GPCCcom':
        # These predictors are selelected for GPCCcom in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','AMO','IOD','PERS'])
        gpcccom = xr.open_dataset(bdid+'gpcc_10_combined_r'+str(resolution)+'.nc')
        # Create anomalies with 1980-2010 as baseline climatology
        gpcccom.prcp.values = anom_df(gpcccom.prcp,1980,2010,styear)
        gpcccom.time.values = rewrite_time(gpcccom)

        gpcccom = gpcccom.prcp.rename('GPCCcom')
        if p == 0: predadata = xr.Dataset({'GPCCcom':gpcccom.astype('float32')})
        else: predadata = xr.merge([predadata,gpcccom.astype('float32')])


    elif predictand == '20CRslp':
        # These predictors are selelected for 20CRslp in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS'])
        slp = xr.open_dataset(bdid+'slp_mon_mean_1901-current_r25.nc')
        slp.prmsl.values = anom_df(slp.prmsl,1980,2010,styear)
        slp = slp.prmsl.rename('20CRslp')
        slp.time.values = rewrite_time(slp)
        if p == 0: predadata = xr.Dataset({'20CRslp':slp.astype('float32')})
        else: predadata = xr.merge([predadata,slp.astype('float32')])

    else:
        print('predictand not yet known.. exiting!')
        sys.exit()

 
print('-- Done reading in predictand data for fitting, time = ',str(np.int(time.time()-start1)),' seconds --') 

# ************************************************************************
# Read in predictor data for fitting 
# ************************************************************************
start1 = time.time()
print('-- Read in predictor data for fitting --')

indices_locs = {'CO2EQ':bdid+'RCP45_CO2EQ_mo.dat',
                'NINO34':bdid+'ersst_nino3.4a.dat',
                'QBO':bdid+'qbo?.dat',
                'IOD':bdid+'dmi_ersst.dat',
                'PDO':bdid+'pdo_ersst.dat',
                'AMO':bdid+'amo_ersst_ts.dat',
                'PERS':'not needed',
                'PERS_TREND':'not needed'}

# Load predictors (time,lat,lon)
for i,pred in enumerate(predictors):
    print(i,pred)
    if pred in ['PERS','PERS_TREND']:
        continue

    elif pred == 'CPREC':    # Cum precip [time,lat,lon] - 1901 -current
        gpcccom = xr.open_dataset(bdid+'gpcc_10_combined_r'+str(resolution)+'.nc')
        gpcccom.prcp.values = anom_df(gpcccom.prcp,1980,2010,styear)
        gpcccom = gpcccom.prcp.rename('CPREC')
        gpcccom.time.values = rewrite_time(gpcccom)
        if i == 0: predodata = df
        else: predodata = xr.merge([predodata,gpcccom.astype('float32')])

    else:
        df = load_clim_index(indices_locs[pred],styear,endyear,endmonth,pred)
        if i==0: predodata = df
        else: predodata = xr.merge([predodata,df.astype('float32')])

        
print('-- Done reading in predictor data for fitting, time = ',str(np.int(time.time()-start1)),' seconds --') 



# *************************************************************************   
# Now start the predictor selection and the forecasting / hindcasting loop
# *************************************************************************

for p,predictand in enumerate(predictands):
    # Fill persistence predictor with predictand data
    if 'PERS' in predictorz[p]:
        if 'PERS' in predodata:  predodata = predodata.drop('PERS')
        predodata = xr.merge([predodata,predadata[predictand].rename('PERS')])
    
    print('Predictand: ',predictand)
    print('Predictors: ',predictorz[p])

    if overwrite:
        print('overwrite is True, so remove all data and redo complete forecast cycle')
        mon_range = list(range(12))
        filez = glob.glob(bdnc+'*'+predictand+'.nc')
        for fil in filez: os.remove(fil)
    elif overwrite_m:
        mon_range = [overw_m]
    else:
        try:    
            datanc = xr.open_dataset(bdnc+'pred_v2_'+predictand+'.nc')
            # Create list of dates which are needed to update the data
            monrange=pd.date_range(pd.to_datetime(datanc['time'].values[-1]),pd.to_datetime(predadata['time'].values[-1])+pd.DateOffset(months=1),freq='MS')[1:]
            if len(monrange) == 0:
                print('Data already up to date... continue to next in loop')
                continue
            mon_range = list(range(monrange[0].month-1,monrange[-1].month))
            #print mon_range
        except IOError:  # If file does not exist do full hindcast
            mon_range = list(range(12))
            print('no previous output, do full hindcast!')

    
    # Rolling 3-month mean, first and last timestep become nans, so remove last timestep (slice(None,-1))
    predodata_3m  = predodata[predictorz[p]].rolling(time=3,center=True).mean().isel(time=slice(None,-1))
    predadata_3m = predadata[predictand].rolling(time=3,center=True).mean().isel(time=slice(None,-1))
    
    # Change time values of predictor and predictand data for simplicity. Add 2 months for predictor data and subtract 2 months for predicand data. This means the dates do not represent the exact time off the data anymore, but is much easier for selecting the right training / testing data etc.
    predodata_3m['time'].values = pd.DatetimeIndex(predodata_3m['time'].values) + pd.DateOffset(months=2)
    predadata_3m['time'].values = pd.DatetimeIndex(predadata_3m['time'].values) - pd.DateOffset(months=2)
    
    if MLR_PRED: 
        print('MLR_PRED is True, calculating trend over previous 3 months for all predictors')
        predodata_3m_trend = pred_trend(predodata[predictorz[p]]).isel(time=slice(None,-1))
        predodata_3m_trend['time'].values = pd.DatetimeIndex(predodata_3m_trend['time'].values) + pd.DateOffset(months=2)


    # Start loop over the months to update, normally this is just 1 month
    for m in mon_range:
        
        print('prediction month = ',months[m])
        print('predictor season = ', '-'.join([months[m-3],months[m-2],months[m-1]]))
        try: print('predictor season = ', '-'.join([months[m+1],months[m+2],months[m+3]]))
        except IndexError: print('predictor season = ', '-'.join([months[m-11],months[m-10],months[m-9]]))
        
        if HINDCAST and CAUSAL: #TODO > fix causal hindcast
            print('Hindcasting mode, causal ',str(stvalyear),'-current')
            timez = pd.DatetimeIndex(predodata['time'].sel(time=predodata['time.month']==m+1).values)[1:]
            if timez[-1].month + 2 >= times[-1].month: timez = timez[:-1]
            
            
            for y in range(np.where(timez.year == stvalyear)[0],len(timez)):
                train = timez[:y]
                test = timez[y]

                if predictand == 'GPCCcom':
                    train = train[50:]
                
                print('test years: ',test)

                regr_loop(predodata_3m,predodata_3m_trend,predadata_3m,timez,train,test,ens_size,bdnc,25,predictand,predictorz[p], MLR_PRED,FC=False,sig_val = 0.1) # hier wil ik naartoe

            
        elif HINDCAST and CROSVAL:
            print('Hindcasting mode, leave ',str(cv_years),' out cross-validation')
            timez = pd.DatetimeIndex(predodata['time'].sel(time=predodata['time.month']==m+1).values)[1:-1]

            for y in range(np.array(np.where(timez.year == stvalyear)).squeeze(),len(timez),cv_years):
                test = timez[y:y+3]
                train = timez.drop(test)
                
                if predictand == 'GPCCcom':
                    train = train[50:]
                
                print('test years: ',test)
                #sys.exit()
                regr_loop(predodata_3m,predodata_3m_trend,predadata_3m,timez,train,test,ens_size,bdnc,25,predictand,predictorz[p], MLR_PRED,FC=False,sig_val = 0.1) # hier wil ik naartoe
                

        elif HINDCAST:
            print('either CROSVAL or CAUSAL should be set to true')
            sys.exit()

        if FORECAST:
            print('Forecasting mode')
            timez = pd.DatetimeIndex(predodata_3m['time'].sel(time=predodata_3m['time.month']==m+1).values)
            if timez[0].year == 1901: timez = timez[1:]

            # In forecast mode, the last timestep is the test data, the rest is training data.
            train = timez[:-1]
            test = timez[-1]
            if predictand == 'GPCCcom':
                train = train[50:]

            print('test years: ',test)

            regr_loop(predodata_3m,predodata_3m_trend,predadata_3m,timez,train,test,ens_size,bdnc,25,predictand,predictorz[p],MLR_PRED,FC=True,sig_val = 0.1) # hier wil ik naartoe
            
                    
    # Sort all data by time dimension
    filn = ['pred_v2_','beta_v2_','predictors_v2_','predictors_fit_v2_']
    for fil in filn:
        tmp = xr.open_dataset(bdnc+fil+predictand+'.nc')
        tmp.sortby('time').to_netcdf(bdnc+fil+predictand+'2.nc','w')
        os.remove(bdnc+fil+predictand+'.nc')
        os.rename(bdnc+fil+predictand+'2.nc',bdnc+fil+predictand+'.nc')
        
    
    
    if VALIDATION:
        print('Calculating skill scores') 
        
        data_fit = xr.open_dataset(bdnc+'pred_v2_'+predictand+'.nc')#,chunks={'lat':1})
        for m in mon_range:
            t0 = time.time()
            timez = pd.DatetimeIndex(data_fit['time'].sel(time=data_fit['time.month']==m+1).values)
            scorez = xr.Dataset(coords={'lat': data_fit['lat'],
                    'lon': data_fit['lon'],
                    'time': [timez[-1]]})

            print('month = ',str(m+1))
            # Create correlation scores and its significance
            cor,sig = linregrez(data_fit.kprep.mean('ens').sel(time=timez[:-1]).values,data_fit.obs.sel(time=timez[:-1]).values,COR=True)

            scorez['cor'] = (('time','lat','lon'),cor[np.newaxis,:,:])
            scorez['cor_sig'] = (('time','lat','lon'),sig[np.newaxis,:,:])
            
            scorez['rmsess'] = (('time','lat','lon'),                            f_rmse(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['obs'].sel(time=timez[:-1]).values,ref=data_fit['clim'].sel(time=timez[:-1]).values,SS=True)[np.newaxis,:,:])
         
            scorez['crpss'] = (('time','lat','lon'),                        f_crps2(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['obs'].sel(time=timez[:-1]).values,SS=True,ref=data_fit['clim'].sel(time=timez[:-1]).values)[np.newaxis,:,:])
            
            scorez['crpss_co2'] = (('time','lat','lon'),
            f_crps2(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['obs'].sel(time=timez[:-1]).values,SS=True,ref=data_fit['trend'].sel(time=timez[:-1]).values)[np.newaxis,:,:])                                   
            
            scorez['tercile'] = (('time','lat','lon'), tercile_category(data_fit['kprep'].sel(time=timez[:-1]).values,data_fit['kprep'].sel(time=timez[-1]).values)[np.newaxis,:,:])
            
            scorez['for_anom'] = (('time','lat','lon'),data_fit['kprep'].sel(time=timez[-1]).mean(dim='ens').values[np.newaxis,:,:])
            
            to_nc2(scorez,bdnc + 'scores_v2_'+predictand)
            print('Calculated skill scores for '+months[m]+', total time is '+str(np.round(time.time()-t0,2))+' seconds')
        if os.path.isfile(bdnc+'scores_v2_'+predictand+'.nc'):
            tmp = xr.open_dataset(bdnc+'scores_v2_'+predictand+'.nc')
            tmp.sortby('time').to_netcdf(bdnc+'scores_v2_'+predictand+'2.nc','w')
            os.remove(bdnc+'scores_v2_'+predictand+'.nc')
            os.rename(bdnc+'scores_v2_'+predictand+'2.nc',bdnc+'scores_v2_'+predictand+'.nc')

    if VALIDATION:
        #predictand = 'GCEcom' 
        print('Start validation for the last year in mon_range')
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

        
         
        
        for m in mon_range:
        #for m in range(12):
                mon = str(m+1).zfill(2)
                timez = pd.DatetimeIndex(data_fit['time'].sel(time=data_fit['time.month']==m+1).values)
                year = str(timez[-1].year)
                season = monthzz[m+1:m+4]
                print('validation for '+season+' '+year)
                
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
                
                plot_climexp(scores.rmsess.sel(time=timez[-1]),
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
                plot_climexp(scores.crpss.sel(time=timez[-1]),
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
                plot_climexp(scores.crpss_co2.sel(time=timez[-1]),
                             'CRPSS hindcasts, reference: hindcasts with only CO2 as predictor (1961-'+str(int(year)-1)+')',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                             predictand = predictand,
                             cmap = cmap2,
                             cmap_under = cmap_under,
                             cmap_over = cmap_over,
                             fname=bdpo+predictand+'_crpss_detrended_clim_'+year+mon+'.png',
                             clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                             )    
                plot_climexp(data_fit.kprep.sel(time=timez[-1]).mean('ens'),
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
                             )
                plot_climexp(scores.cor.sel(time=timez[-1]),
                             'Correlation between hindcast anomaly and observations (1961-'+str(int(year)-1)+'',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+'  |  Stippled where signficant at 5% level'+' | base time: '+months[m]+' '+year,
                             sig = scores.cor_sig.sel(time=timez[-1]).values,
                             predictand = predictand,
                             fname=bdpo+predictand+'_correlation_'+year+mon+'.png', 
                             clevs = np.arange(-1.,1.01,0.2),
                             )
                plot_climexp(scores.tercile.sel(time=timez[-1]),
                             'Probabilty (most likely tercile of '+var+'), based on 1961-'+str(int(year)-1)+' hindcasts',
                             'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                             'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                             cmap = cmap1,
                             predictand = predictand,
                             fname=bdpo+predictand+'_tercile_'+year+mon+'.png', 
                             clevs = np.array((-100,-70,-60,-50,-40,40,50,60,70,100)),
                             barticks = ['100%','70%','60%', '50%', '40%', '40%', '50%','60%', '70%', '100%'],
                             )
                             #plt.annotate('<---- below lower tercile        
                
              
            
                
    import time        
    print('Total time taken is: ',np.int((time.time()-start0)//60),' minutes and ',np.int((time.time()-start0)%60), 'seconds')
    
os.system('rsync_climexp')
