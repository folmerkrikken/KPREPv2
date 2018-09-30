
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np           
from scipy.io import netcdf 
from mpl_toolkits.basemap import Basemap
import json
import plotly.plotly as py
from plotly.graph_objs import *
import xarray as xr
from textwrap import dedent as ded
import copy
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import datetime
from flask_caching import Cache
import numpy as np
import os
import pandas as pd
import time
import plotly.graph_objs as go
import plotly.tools as tls
import datetime
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs

#from SPECS_forecast_v5_tools import plot_climexp2

#dt = datetime.date.today()


# Make shortcut to Basemap object, 
# not specifying projection type for this example
m = Basemap() 

monthzz = 'JFMAMJJASONDJFMAMJJASOND'

# Predefine clickdata
clickData = dict({u'points': [{u'y': 0., u'x': 0., u'pointNumber': 6, u'curveNumber': 632}]})

def plot_climexp2(data,line1,line2,line3,cmap=[],cmap_under=[],cmap_over=[],predictand=[],sig=[],fname=[],clevs=[],barticks=[]):
    #print('plotting '+data.name)
    proj = ccrs.PlateCarree()
    figg = plt.figure(figsize=(12,8))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.set_facecolor('red')
    ax.set_global()
    ax.set_extent([-180,180,90,-90], ccrs.PlateCarree())
    lons,lats = data.lon.values,data.lat.values
    lon2d, lat2d = np.meshgrid(lons, lats)
    #ax.annotate(line1, xy=(0, 1.10), xycoords='axes fraction',fontsize=14,ha='left',va='center')
    #ax.annotate(line2, xy=(0, 1.15), xycoords='axes fraction',fontsize=18,ha='left',va='center')
    #ax.annotate(line3, xy=(0, 1.05), xycoords='axes fraction',fontsize=10,ha='left',va='center')
    if clevs == []:
        clevs = np.arange(np.nanmin(data),np.nanmax(data), 0.1)#[-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5]
    if cmap == []: cmap = plt.cm.RdYlBu_r
    if cmap_under != []: cmap.set_under(cmap_under)
    if cmap_over != []: cmap.set_over(cmap_over)
    norm = matplotlib.colors.BoundaryNorm(clevs, cmap.N)

    if data.name in ['tercile','cor']:
        cs = ax.contourf(lons,lats,data.values,clevs,norm=norm,cmap=cmap)#,clevs,norm,cmap)
    elif data.name in ['crpss','rmsess','kprep','crpss_co2']:
        cs = ax.contourf(lons,lats,data.values,clevs,norm=norm,cmap=cmap,extend='both')
    ax.coastlines()
    if sig != []:
       if data.name == 'kprep': sigvals = np.where(np.logical_and(sig[:,:]>0.1,sig[:,:]<1.))
       else: sigvals = np.where(sig[:,:]<0.05)
       ax.scatter(lon2d[sigvals],lat2d[sigvals],marker='.',c='k',s=5.,lw=0.)
                    
    ax.background_patch.set_facecolor('lightgray')
        
    #cbar = fig.colorbar(cs,cmap=cmap,norm=norm,boundaries=clevs,location='bottom',pad="10%")
    cax = matplotlib.axes.Axes(fig,[0.1, 0.15, 0.8, 0.25])
    cbar = figg.colorbar(cs,ax=cax,cmap=cmap,norm=norm,boundaries=clevs,orientation='horizontal',shrink=0.8)
    
    #fig.colorbar(pcont,ax=cax)  
    if barticks != []: cbar.ax.set_xticklabels(barticks)
    #if fname != []: plt.savefig(fname)
    return figg
    #if PLOT: plt.show()
    #else: plt.close('all')    



# Make trace-generating function (return a Scatter object)
def make_scatter(x,y):
    return Scatter(
        x=x,
        y=y,
        mode='lines',
        line=Line(color="black"),
        name=' '  # no name on hover
    )

# Functions converting coastline/country polygons to lon/lat traces
def polygons_to_traces(poly_paths, N_poly):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    traces = []  # init. plotting list 

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)
        
        # add plot.ly plotting options
        traces.append(make_scatter(lon_cc,lat_cc))
     
    return traces

# Function generating coastline lon/lat traces
def get_coastline_traces():
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces(poly_paths, N_poly)

# Function generating country lon/lat traces
def get_country_traces():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces(poly_paths, N_poly)

# Get list of of coastline and country lon/lat traces
traces_cc = get_coastline_traces()+get_country_traces()


bd = '/nobackup_1/users/krikken/KPREP/'
bdnc = bd+'ncfiles/'

scores = xr.open_dataset(bdnc+'scores_v2_GCEcom.nc')
timez = scores.time[-12:].values
months12 = pd.to_datetime(scores.time[-12:].values).strftime('%Y-%m')[::-1]
dict_times = dict(zip(months12,range(1,13)))



app = dash.Dash()


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

plottypes={'Correlation':'cor','RMSESS':'rmsess','CRPSS':'crpss','Tercile summary plot':'tercile','Forecast anomalies':'for_anom'}
variables={'Temperature':'GCEcom','Precipitation':'GPCCcom','Sea-level pressure':'20CRslp'}

anno_text = "Data courtesy of ME"

axis_style = dict(
    zeroline=False,
    showline=False,
    showgrid=False,
    ticks='',
    showticklabels=False,
)

## Start app layout

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    # Create dropdown menu to choose variable
    html.Div([
            dcc.Dropdown(
                id='variable',
                options=[{'label': i,'value': i} for i in variables.keys()],
                value='Temperature',
            ),

        ],
        style={'width': '20%', 'display': 'inline-block'}),
    # Create dropdown menu to choose plot type
    html.Div([
            dcc.Dropdown(
                id='plot_type',
                options=[{'label': i,'value': i} for i in plottypes.keys()],
                value='Forecast anomalies',
            ),

        ],
        style={'width': '20%', 'display': 'inline-block'}),

    # Create dropdown menu to choose time step
    html.Div([
            dcc.Dropdown(
                id='fc_time',
                options=[{'label': i,'value': i} for i in dict_times.keys()],
                value=list(dict_times.keys())[0],
            ),

        ],
        style={'width': '30%', 
               'display': 'inline-block',
               'margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}
                }),
    # Create     
    html.Div([
        dcc.Graph(id='basemap_plot')],
        style={'width':'55%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    
    html.Div([
        dcc.Graph(id='predictor_plot')],
        style={'width':'40%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    html.Div([
        dcc.Graph(id='predictand_plot')],
        style={'width':'60%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    html.Div([
        dcc.Graph(id='bar_plot')],
        style={'width':'40%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}})
    

    ])
      
## End app layout

## Start plotting functions
      
def create_map(clickData,plot_type,variable,fc_time):
    year = fc_time[:4]
    month = fc_time[5:]
    season = monthzz[np.int(month)+1:np.int(month)+4]
    print(year,month)
    dt = datetime.datetime.today()
    scores = xr.open_dataset(bdnc+'scores_v2_'+variables[variable]+'.nc')
    #scores_1t = scores.isel(time=-dict_times[fc_time])
    titel = u"variable = "+variable+", plot type = "+plot_type
    if variables[variable] == 'GCEcom':      
        var = 'Surface air temperature'
        clevz = np.array((-2.,-1.,-0.5,-0.2,0.2,0.5,1.,2.))
        cmap1 = matplotlib.colors.ListedColormap(['#000099','#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400','#cc0022'])
        cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
        cmap_under = '#000099'
        cmap_over = '#cc0022'
    elif variables[variable] == 'GPCCcom':   
        var = 'Surface precipitation' 
        clevz = np.array((-200.,-100.,-50.,-20.,20.,50.,100.,200.))
        cmap1 = matplotlib.colors.ListedColormap(['#993300','#cc8800','#ffcc00','#ffee99','#ffffff','#ccff66','#33ff00','#009933','#006666'])
        cmap2 = matplotlib.colors.ListedColormap(['#cc8800','#ffcc00','#ffee99','#ffffff','#ccff66','#33ff00','#009933'])
        cmap_under = '#993300'
        cmap_over = '#006666'
        
    elif variables[variable] == '20CRslp':
        var = 'Mean sea level pressure'
        clevz=np.array((-4.,-2.,-1.,-0.5,0.5,1.,2.,4.))
        cmap1 = matplotlib.colors.ListedColormap(['#000099','#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400','#cc0022'])
        cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
        cmap_under = '#000099'
        cmap_over = '#cc0022'  
    fig = plot_climexp2(scores.rmsess.sel(time=timez[-1]),
                'test1',
                'test2',
                'test3',
                predictand = variables[variable],
                clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                cmap = cmap2,
                cmap_under = cmap_under,
                cmap_over = cmap_over,
                ) 
    
    #import pprint
    #pp = pprint.PrettyPrinter(indent=4)
    #print(pp.pprint(plotly_fig['layout']))
    return(tls.mpl_to_plotly(fig))
    
#def create_map(clickData,plot_type,variable,fc_time):
    ##print('hoiiii')
    ##print(clickData)
    #if clickData == None:
        #lat_click = 0
        #lon_click = 0
    #else:
        #lat_click=clickData['points'][0]['y']
        #lon_click=clickData['points'][0]['x']
    #print(fc_time)
    #scores = xr.open_dataset(bdnc+'scores_v2_'+variables[variable]+'.nc')
    ##scores_1t = scores.isel(time=-dict_times[fc_time])
    #titel = u"variable = "+variable+", plot type = "+plot_type

    
    #return( 
            #go.Figure(
            #data=
                #Data([Contour(z=scores[plottypes[plot_type]].isel(time=-dict_times[fc_time]).values,x=scores.lon.values,y=scores.lat.values,colorscale=[[0, '#000099'], [0.2, '#3355ff'], [0.3, '#66aaff'], [0.4, '77ffff'], [0.5, '#ffffff'], [0.6, '#ffff33'], [0.7, '#ffaa00'], [0.8, '#ff4400'], [1,'#cc0022']],contours=dict(start=-2,end=2))]
                #+[Scatter(x=lon_click,y=lat_click)]+traces_cc),
            #layout = Layout(
                #title=titel,
                #showlegend=False,
                #hovermode="closest",        # highlight closest point on hover
                ##colorscale=[[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], [0.45, 'rgb(178,223,138)'], [0.65, 'rgb(51,160,44)'], [0.85, 'rgb(251,154,153)'], [1, 'rgb(227,26,28)']],
                #xaxis=XAxis(
                    #axis_style,
                    ##range=[lon[0],lon[-1]]  # restrict y-axis to range of lon
                    #range=[-180,180]
                #),
                #yaxis=YAxis(
                    #axis_style,
                #),
                #annotations=Annotations([
                    #Annotation(
                        #text=anno_text,
                        #xref='paper',
                        #yref='paper',
                        #x=0,
                        #y=1,
                        #yanchor='bottom',
                        #showarrow=False
                    #)
                #]),
                #autosize=False,
                #width=1000,
                #height=500,)
            #))      
   
   
def create_time_series(clickData,variable,fc_time):
    if clickData == None:
        lat_click = 0
        lon_click = 0
    else:
        lat_click=clickData['points'][0]['y']
        lon_click=clickData['points'][0]['x']
    #print lat_click
    tt = dict_times[fc_time]
    pred = xr.open_dataset(bdnc+'pred_v2_'+variables[variable]+'.nc')
    # Select right location and time slice
    pred1d = pred.sel(lon=lon_click,lat=lat_click,method=str('nearest')).isel(time=slice(None,-tt))
    time_pd = pred1d.time.to_pandas()
    kprep_mean = pred1d['kprep'].mean(dim='ens').values
    kprep_std = pred1d['kprep'].std(dim='ens').values * 2.
    clim_mean = pred1d['clim'].mean(dim='ens').values
    clim_std = pred1d['clim'].std(dim='ens').values * 2.
    
    
    return(
        go.Figure(
        data=Data(
            [Scatter(x=time_pd,y=(kprep_mean+kprep_std)+(kprep_mean-kprep_std),mode='lines',fill='tozeroz',fillcolor='rgba(0,100,80,0.2)',line=Line(color='transparent'))]+
            [Scatter(x=time_pd,y=kprep_mean,mode='lines',name='Forecast',line=dict(color='rgba(0,100,80)'))]+[Scatter(x=time_pd,y=clim_mean,mode='lines',name='Climatology',line=dict(color='green'))]+
            [Scatter(x=time_pd,y=pred1d['obs'].values,mode='lines',name='Observations',line=dict(color='black'))]
            ),
        layout = Layout(
            title = 'Time series of the forecast, climatology and observations',
            #height =  225,
            margin = {'l': 20, 'b': 30, 'r': 10, 't': 10},
            autosize=False,
            width=1000.,
            height=400.,
            xaxis=dict(
                rangeselector=dict(
                buttons=list([
                dict(count=1,
                     label='12m',
                     step='year',
                     stepmode='backward'),
                dict(count=6,
                     label='120m',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
                ])
            ),
            rangeslider=dict(),
            type='date'
            )
            )
        ))   
      
def create_po_timeseries(clickData,variable,fc_time):
    if clickData == None:
        lat_click = 0
        lon_click = 0
    else:
        lat_click=clickData['points'][0]['y']
        lon_click=clickData['points'][0]['x']    
    tt = dict_times[fc_time]
    pred_1d = xr.open_dataset(bdnc+'pred_v2_'+variables[variable]+'.nc').sel(lon=lon_click,lat=lat_click,method=str('nearest')).isel(time=-tt)
    for_anom = pred_1d['kprep'].mean(dim='ens').values
    co2_anom = pred_1d['trend'].mean(dim='ens').values
    beta = xr.open_dataset(bdnc+'beta_v2_'+variables[variable]+'.nc')
    beta_1d = beta.sel(lon=lon_click,lat=lat_click,method=str('nearest')).isel(time=-tt)
    predo = xr.open_dataset(bdnc+'predictors_fit_v2_'+variables[variable]+'.nc')
    predo_1d = predo.sel(lon=lon_click,lat=lat_click,method=str('nearest'))
    predos = list(predo_1d.data_vars)
    # Which predictors are significant
    #sigp =~np.isnan(beta_1d.beta.values).any(axis=0)
    sigp = ~np.isnan(beta_1d.beta.values)
    nr_sigp = np.sum(sigp)
    time_pd = predo_1d.time.to_pandas()
    traces=[]
    xaxs = ['x1','x2','x3','x4','x5','x6','x7','x8']
    yaxs = ['y1','y2','y3','y4','y5','y6','y7','y8']
    if nr_sigp == 0:
        print('no significant predictors..')
    else:
        for ii in range(nr_sigp):
            traces.append(go.Scatter(
                x=time_pd,
                y=predo_1d[np.asarray(predos)[sigp][ii]][:-tt].values,
                xaxis=xaxs[ii],
                yaxis=yaxs[ii],
                text=np.asarray(predos)[sigp][ii],
                textposition='top',
                name=np.asarray(predos)[sigp][ii]
                ))
        vals = np.asarray(np.append((predo_1d.isel(time=-tt).to_array(dim='predictors')*beta_1d.beta.values)[sigp].values,for_anom))
        if 'CO2EQ' in np.asarray(predos)[sigp]:
            vals[0]=co2_anom    
        traces.append(go.Bar(
            x=np.append(np.asarray(predos)[sigp],np.asarray('Total')),
            y=vals,
            xaxis=xaxs[ii+1],
            yaxis=yaxs[ii+1]
            ))
                
    fig = tls.make_subplots(rows=nr_sigp+1,cols=1)
    for ii in range(nr_sigp+1):
        fig.append_trace(traces[ii],ii+1,1)
    fig['layout'].update(   height=600,
                            width=1000.,
                            autosize=False,
                            title='Predictor data..',
                         )
    
    return(fig)
    
## End plotting functions

## Start callbacks

# Update predictand map      
@app.callback(
    dash.dependencies.Output('basemap_plot', 'figure'),
    [dash.dependencies.Input('basemap_plot','clickData'),
     dash.dependencies.Input('plot_type', 'value'),
     dash.dependencies.Input('variable', 'value'),
     dash.dependencies.Input('fc_time','value')])
def update_map(clickData,plot_type,variable,fc_time):
    return create_map(clickData,plot_type,variable,fc_time)

# Update predictand timeseries                    
@app.callback(
    dash.dependencies.Output('predictand_plot', 'figure'),
    [dash.dependencies.Input('basemap_plot', 'clickData'),
     dash.dependencies.Input('variable','value'),
     dash.dependencies.Input('fc_time','value')])
def update_time_series(clickData,variable,fc_time):
    return create_time_series(clickData,variable,fc_time)


# Update predictor timeseries
@app.callback(
    dash.dependencies.Output('predictor_plot', 'figure'),
    [dash.dependencies.Input('basemap_plot', 'clickData'),
     dash.dependencies.Input('variable','value'),
     dash.dependencies.Input('fc_time','value')])
def update_po_timeseries(clickData,variable,fc_time):
    return create_po_timeseries(clickData,variable,fc_time)




if __name__ == '__main__':
    app.run_server(debug=True)
