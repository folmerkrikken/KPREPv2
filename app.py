
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np           
from scipy.io import netcdf 
import os,conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
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

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import plotly.graph_objs

#from SPECS_forecast_v5_tools import plot_climexp2

#dt = datetime.date.today()


# Make shortcut to Basemap object, 
# not specifying projection type for this example
m = Basemap() 

monthzz = 'JFMAMJJASONDJFMAMJJASOND'

# Predefine clickdata
clickData = dict({u'points': [{u'y': 0., u'x': 0., u'pointNumber': 6, u'curveNumber': 632}]})


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
traces_cc = get_coastline_traces()#+get_country_traces()


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

anno_text = "Data courtesy of Folmer Krikken"

axis_style = dict(
    zeroline=False,
    showline=False,
    showgrid=False,
    ticks='',
    showticklabels=False,
)

## Start app layout

app.layout = html.Div(children=[
    html.H1(children='Sources of predictability - KPREP empirical forecast system'),

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
        style={'width':'65%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    html.Div([
        dcc.Graph(id='bar_plot')],
        style={'width':'25%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    
    html.Div([
        dcc.Graph(id='predictor_plot')],
        style={'width':'55%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    html.Div([
        dcc.Graph(id='predictand_plot')],
        style={'width':'60%','display': 'inline-block','margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}}),
    

    ])
      
## End app layout

## Start plotting functions
      

    
def create_map(clickData,plot_type,variable,fc_time):
    month = np.int(fc_time[5:])
    season = monthzz[month:month+3]
    year = np.int(fc_time[:4])
    #print('hoiiii')
    #print(clickData)
    if clickData == None:
        lat_click = 0
        lon_click = 0
    else:
        lat_click=clickData['points'][0]['y']
        lon_click=clickData['points'][0]['x']
    print(lat_click,lon_click)
    print(fc_time)
    scores = xr.open_dataset(bdnc+'scores_v2_'+variables[variable]+'.nc')
    times_m = scores['time.month']
    data_xr = scores[plottypes[plot_type]].isel(time=-dict_times[fc_time]).values
    #data = scores[plottypes[plot_type]].sel(times_m == m).values
    #scores_1t = scores.isel(time=-dict_times[fc_time])
    titel = u"variable = "+variable+", plot type = "+plot_type+', valid for: '+season+' '+str(year)
    #[[0, '#000099'], [0.2, '#3355ff'], [0.35, '#66aaff'], [0.45, '#77ffff'], [0.55, '#ffffff'], [0.65, '#ffff33'], [0.7, '#ffaa00'], [0.8, '#ff4400'], [1,'#cc0022']]
    #+[Scatter(x=lon_click,y=lat_click)]
    colorsceel=[[0, '#000099'], [0.2, '#3355ff'], [0.35, '#66aaff'], [0.45, '#77ffff'], [0.55, '#ffffff'], [0.65, '#ffff33'], [0.8, '#ff4400'], [1,'#cc0022']]
    #,colorscale=colorsceel,contours=dict(start=-maxval,end=maxval),
    #fig = go.Contour(
    maxval = np.max(np.abs(data_xr))
    return( 
            #go.contour(z=data_xr,x=scores.lon.values,y=scores.lat.values,contours=dict(start=-2,end=2))
            go.Figure(
            data=
                Data(traces_cc+[Contour(z=data_xr,x=scores.lon.values,y=scores.lat.values,zmin=-maxval,zmax=maxval,colorscale=colorsceel,opacity=1.)]),#+traces_cc),
            layout = Layout(
                title=titel,
                showlegend=False,
                #clickmode="event",
                hovermode='closest',        # highlight closest point on hover
                #colorscale=[[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], [0.45, 'rgb(178,223,138)'], [0.65, 'rgb(51,160,44)'], [0.85, 'rgb(251,154,153)'], [1, 'rgb(227,26,28)']],
                #colorscale=colorsceel,
                margin=go.Margin(
                    l=50,
                    r=50,
                    b=10,
                    t=70,
                    pad=4
                    ),
                xaxis=XAxis(
                    axis_style,
                       range=[-180,180]
                ),
                yaxis=YAxis(
                    axis_style,
                ),
                annotations=Annotations([
                    Annotation(
                        text=anno_text,
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=1,
                        yanchor='bottom',
                        showarrow=False
                    )
                ]),

                autosize=False,
                width=1000,
                height=500,)
            ))      
   
   
def create_time_series(clickData,variable,fc_time):
    if clickData == None:
        lat_click = 0
        lon_click = 0
    else:
        lat_click=clickData['points'][0]['y']
        lon_click=clickData['points'][0]['x']
    print('Hello2!!')
    print(lat_click,lon_click)
    print(fc_time)           
    #print lat_click
    tt = dict_times[fc_time]
    pred = xr.open_dataset(bdnc+'pred_v2_'+variables[variable]+'.nc')
    # Select right location and time slice
    #pred1d = pred.sel(lon=lon_click,lat=lat_click,method=str('nearest')).isel(time=slice(None,-tt))
    pred1d = pred.sel(lon=lon_click,lat=lat_click,method=str('nearest')).sel(time=(pred['time.month']==np.int(fc_time[5:])))
    #print(pred1d)
    time_pd = pred1d.time.to_pandas()
    kprep_mean = pred1d['kprep'].mean(dim='ens').values
    print('hi..',kprep_mean[-1])
    kprep_std = pred1d['kprep'].std(dim='ens').values * 2.
    clim_mean = pred1d['clim'].mean(dim='ens').values
    clim_std = pred1d['clim'].std(dim='ens').values * 2.
    trend = pred1d['trend'].mean(dim='ens').values
    
    
    return(
        go.Figure(
        data=Data(
            #[Scatter(x=time_pd,y=kprep_mean+kprep_std,mode='lines',fillcolor='rgba(0,100,80,0.2)',line=Line(color='gray'))]+
            #[Scatter(x=time_pd,y=kprep_mean-kprep_std,mode='lines',fill='tonexty',fillcolor='rgba(0,100,80,0.2)',line=Line(color='gray'))]+
            [Scatter(x=time_pd,y=kprep_mean,mode='lines',name='Forecast',line=dict(color='blue'))]
            +[Scatter(x=time_pd,y=clim_mean,mode='lines',name='Climatology',line=dict(color='green'))]
            +[Scatter(x=time_pd,y=trend,mode='lines',name='Trend CO2',line=dict(color='red'))]
            +[Scatter(x=time_pd,y=pred1d['obs'].values,mode='lines',name='Observations',line=dict(color='black'))]
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
      
def create_bar_plot(clickData,variable,fc_time):
    
    if clickData == None:
        lat_click = 0
        lon_click = 0
    else:
        lat_click=clickData['points'][0]['y']
        lon_click=clickData['points'][0]['x']        
    
    # Load data
    pred_1d = xr.open_dataset(bdnc+'pred_v2_'+variables[variable]+'.nc').sel(lon=lon_click,lat=lat_click,method=str('nearest'),time=fc_time+'-01')
    for_anom = pred_1d['kprep'].mean(dim='ens')
    #for_anom = pred_1d['kprep'].isel(ens=0)
    co2_anom = pred_1d['trend'].mean(dim='ens')
    beta_1d = xr.open_dataset(bdnc+'beta_v2_'+variables[variable]+'.nc').sel(lon=lon_click,lat=lat_click,method=str('nearest'),time=fc_time+'-01')
    predo_1d = xr.open_dataset(bdnc+'predictors_fit_v2_'+variables[variable]+'.nc').sel(lon=lon_click,lat=lat_click,method=str('nearest'),time=fc_time+'-01')
    predos = list(predo_1d.data_vars)
    sigp = ~np.isnan(beta_1d.beta.values)
    nr_sigp = np.sum(sigp)
    traces=[]
    fig = tls.make_subplots(rows=1,cols=1)
    if nr_sigp == 0:
        print('no significant predictors..')
    else:
        print(predos,sigp)
        vals = np.asarray(np.append((predo_1d.to_array(dim='predictors')*beta_1d.beta.values)[sigp].values,for_anom))
        
        if 'CO2EQ' in np.asarray(predos)[sigp]:
                vals[0]=co2_anom  
        dif = for_anom-np.sum(vals[:-1])
        vals = np.append(vals,dif)   
        trace = Bar(
                x=np.append(np.asarray(predos)[sigp],np.asarray(['Total','dif'])),
                y=vals
                )
        layout = go.Layout(
            height=500,
            width=500.,
            autosize=False,
            title='Individual contribution predictors (lat='+str(lat_click)+', lon='+str(lon_click)+')',
            )
   
    fig = go.Figure(data=Data([trace]), layout=layout)     
    return(fig)
        
def create_po_timeseries(clickData,variable,fc_time):
    if clickData == None:
        lat_click = 0
        lon_click = 0
    else:
        lat_click=clickData['points'][0]['y']
        lon_click=clickData['points'][0]['x']   
    print('Hello!!')
    print(lat_click,lon_click)
    print(fc_time)        
    tt = dict_times[fc_time]
    mo = np.int(fc_time[5:])
    
    # Load monthly data for predictions
    pred = xr.open_dataset(bdnc+'pred_v2_'+variables[variable]+'.nc')#.sel(lon=lon_click,lat=lat_click,method=str('nearest'))
    pred_1d = pred.sel(lon=lon_click,lat=lat_click,method=str('nearest'),time=(pred['time.month']==mo))
    for_anom = pred_1d['kprep'].mean(dim='ens').isel(time=-1).values
    co2_anom = pred_1d['trend'].mean(dim='ens').isel(time=-1).values
    beta_1d = xr.open_dataset(bdnc+'beta_v2_'+variables[variable]+'.nc').sel(lon=lon_click,lat=lat_click,method=str('nearest'),time=fc_time+'-01')
    # Load predictor data (fitted)
    predo = xr.open_dataset(bdnc+'predictors_fit_v2_'+variables[variable]+'.nc')
    predo_1d = predo.sel(time=(predo['time.month']==mo),lon=lon_click,lat=lat_click,method='nearest')
    predos = list(predo_1d.data_vars)
    
    sigp = ~np.isnan(beta_1d.beta.values)
    nr_sigp = np.sum(sigp)
    time_pd = predo_1d.time.to_pandas()
    traces=[]
    xaxs = ['x1','x2','x3','x4','x5','x6','x7','x8']
    yaxs = ['y1','y2','y3','y4','y5','y6','y7','y8']
    if nr_sigp == 0:
        print('no significant predictors..')
    else:
        print('sig predictors: ',np.asarray(predos)[sigp])
        print(sigp)
        for ii in range(nr_sigp):
            traces.append(go.Scatter(
                x=time_pd,
                y=predo_1d[np.asarray(predos)[sigp][ii]].values,
                xaxis=xaxs[ii],
                yaxis=yaxs[ii],
                text=np.asarray(predos)[sigp][ii],
                textposition='top center',
                name=np.asarray(predos)[sigp][ii]
                ))
                
    print('nr sigp2 = ',nr_sigp)
    fig = tls.make_subplots(rows=np.int(nr_sigp),cols=1)
    for ii in range(nr_sigp):
        fig.append_trace(traces[ii],ii+1,1)
    fig['layout'].update(   height=600,
                            width=1000.,
                            autosize=False,
                            title='Time series of (fitted) predictor data',
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

# Update barplot
@app.callback(
    dash.dependencies.Output('bar_plot', 'figure'),
    [dash.dependencies.Input('basemap_plot', 'clickData'),
     dash.dependencies.Input('variable','value'),
     dash.dependencies.Input('fc_time','value')])
def update_bar_plot(clickData,variable,fc_time):
    return create_bar_plot(clickData,variable,fc_time)
    
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
