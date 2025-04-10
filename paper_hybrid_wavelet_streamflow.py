#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


from pyextremes import EVA
import numpy as np
import pandas as pd
from pandas import read_csv
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from numpy import dstack
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict,TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import warnings
import glob
import matplotlib
import os
from math import sqrt
import pywt
import seaborn as sns
from collections.abc import Sequence
from wavelet_forecast import wavelet_forecast_ann

import sys   
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
pd.set_option('display.max_columns', None)



def set_style():

    """
Sets the plotting style for seaborn.

    This method configures seaborn with a specific color palette, 
    context settings (font size and family), and overall style to 
    ensure consistent and visually appealing plots.

    Returns:
        None
    """

    flatui = [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["amber"]]    
    sns.set_palette("colorblind")
    sns.set_context("paper", font_scale=1.0, 
        rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20,
            'xtick.labelsize':20,'ytick.labelsize':20,
            'legend.fontsize': 20,
            'font.family':"Times New Roman", }) 
   
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style(style="whitegrid", rc={
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    

def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

 
def make_instances_from_dataframe(data, ts, window_size, output_size=1):
    """
Creates instances from a dataframe for time series forecasting.

    Args:
        data: The input dataframe containing the time series data.
        ts: The target time series data.
        window_size: The size of the sliding window.
        output_size: The number of steps to predict (default is 1).

    Returns:
        tuple: A tuple containing the input features (X), target values (y), and datetime indices, all as NumPy arrays.
    """
    datetime = data.index    
    data = np.asarray(data)
    assert 0 < window_size+output_size < data.shape[0]
    X = np.atleast_3d(np.array([data[start:start + window_size] for start in range(0, data.shape[0] - window_size-output_size)]))
    X = np.asanyarray([ x.T.ravel() for x in X])    
    X = np.atleast_2d(X).astype(float)
 
    y = np.zeros((X.shape[0],1))
   
    for i in range(y.shape[1]):
        y[:,i] =ts[window_size+output_size-1:window_size+output_size-1+y.shape[0]]
         
    y = np.atleast_2d(y).astype(float)  
    idx = ~np.isnan(np.asarray(np.c_[X, y])).any(axis=1)
    dt=datetime[:datetime.shape[0] - window_size-output_size]
    return X[idx], y[idx], dt[idx]                                               
    
def MAPE(forecasts, original):
    
    
    
    
    """
Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        forecasts: The forecasted values.
        original: The original or actual values.

    Returns:
        float: The calculated MAPE value.
    """
    
    
    
    
    mape_all=[]
    for i in range(len(forecasts)):
        mape_all.append(100*(abs(original[i]-forecasts[i])/original[i]))
    mape_all=np.array(mape_all)
    
    mape = mape_all.mean()
    
    print("MAPE is: ", mape)
    
    return mape

def MAPE_all(forecasts, original):
    
    
    
    
    """
Calculates the Mean Absolute Percentage Error (MAPE) for all forecast values.

    Args:
        forecasts: The forecasted values.
        original: The original or actual values.

    Returns:
        numpy.ndarray: A NumPy array containing the MAPE for each corresponding 
                       forecast and original value.
    """
    
    
    
    
    mape_all=[]
    for i in range(len(forecasts)):
        mape_all.append(100*(abs(original[i]-forecasts[i])/original[i]))
    mape_all=np.array(mape_all)
    
    
    
    
    
    return mape_all

def create_dataset_modified_wann(data, coeffs, lookback, forecast):
    '''
    For a given time series 
    
    data = [t0, t1, t2, ..., t8]
    
    And its corresponding coefficients of the wavelet decomposition
    
    coeffs = [(array([cA0_0, cA0_1, cA0_2, ..., cA0_8]), 
                array([cD0_0, cD0_1, cD0_2, ..., cD0_8])), # first decomposition level
              (array([cA1_0, cA1_1, cA1_2, ..., cA1_8]), 
                array([cD1_0, cD1_1, cD1_2, ..., cD1_8])), # second decomposition level
              ..., 
    ]
    
    Supposing lookback = 3 days and forecast = 2 days ahead
    
    Returns
    
    X = [
        array([t0, t1, t2, cA0_0, cA0_1, cA0_2, cD0_0, ...]), 
        array([t1, t2, t3, cA0_1, cA0_2, cA0_3, ...]), 
        array([t2, t3, t4, cA0_2, cA0_3, cA0_4, ...]), 
        array([t3, t4, t5, cA0_3, cA0_4, cA0_5, ...])
    ]
    
    y = [t4, t5, t6, t7]
    '''
    
    X = []
    y = []

    n_total = len(coeffs[0][0]) # total number of days
    n_days = n_total-lookback-forecast
    
    for i in range(n_days):
        aux = list(data[i:i+lookback])
        for c in coeffs: # levels
            for k in range(2): # approximation and detail
                aux.extend(c[k][i:i+lookback])
        X.append(np.array(aux))
        y.append(data[i+lookback+forecast-1])
    
    return X, y


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def fit_model(model, trainX, trainy, param_grid):
        
        
    """
Calculates the Mean Absolute Percentage Error.

    Args:
        y_true: The true values.
        y_pred: The predicted values.

    Returns:
        float: The Mean Absolute Percentage Error as a percentage.
    """
        
        
    my_cv = [(traincv,testcv) for traincv, testcv in TimeSeriesSplit(n_splits=20).split(trainX)]
    gcv = GridSearchCV(estimator=model, param_grid= param_grid, cv=my_cv, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error')
           
    gcv.fit(trainX, trainy)        
    
    best = gcv.best_estimator_
    print(gcv.best_score_)
    print(gcv.best_params_)
   
    
    return best,  dict(gcv.best_params_)     


    
def make_data(input_file):
    
    
    
    """
Fits a model using GridSearchCV with time series cross-validation.

    Args:
        model: The model to be fitted.
        trainX: The training data features.
        trainy: The training data target values.
        param_grid: A dictionary of parameters to tune for the model.

    Returns:
        tuple: A tuple containing the best fitted estimator and a dictionary 
               of the best parameters found during grid search.
    """
    
    
    
    Est = input_file[-12:-4]
    
    ts1= read_csv(input_file,  delimiter=';')
    ts1.index = pd.DatetimeIndex(ts1['NivelConsistencia'].values)
    ts2 = pd.DataFrame(columns=ts1.columns, index=pd.date_range(min(ts1.index), max(ts1.index)))
    
    ts2 = ts1.filter(['NivelConsistencia','MediaAnualStatus', 'Vazao01', 'Vazao02', 'Vazao03',
           'Vazao04', 'Vazao05', 'Vazao06', 'Vazao07', 'Vazao08', 'Vazao09',
           'Vazao10', 'Vazao11', 'Vazao12', 'Vazao13', 'Vazao14', 'Vazao15',
           'Vazao16', 'Vazao17', 'Vazao18', 'Vazao19', 'Vazao20', 'Vazao21',
           'Vazao22', 'Vazao23', 'Vazao24', 'Vazao25', 'Vazao26', 'Vazao27',
           'Vazao28', 'Vazao29', 'Vazao30'], axis=1)
    ts2_newname=['Data','01', '02', '03',
           '04', '05', '06', '07', '08', '09',
           '10', '11', '12', '13', '14', '15',
           '16', '17', '18', '19', '20', '21',
           '22', '23', '24', '25', '26', '27',
           '28', '29', '30', '31']
    ts2.columns=ts2_newname
    
    ts2.iloc[0:1,1:32]
    ts2.iloc[0:1,:] # first row
    ts2.iloc[0:1,:].columns.values[1] #name_column = '01'
    
    ts2.iloc[0:1,:]['Data'].values[0]
    ts2.iloc[0:1,:]['Data'].values[0][2:10] #'/01/2015'
    
    ts2.iloc[1:2,:].columns.values[2]+ts2.iloc[0:1,:]['Data'].values[0][2:10] #'02'+'/01/2015'
    
    
    l=[]  
    for  index, row in ts2.iloc[:,1:32].iterrows():
        """
Creates a dataset from an input file and returns it along with the station estimate.

    Args:
        input_file: The path to the input CSV file.

    Returns:
        tuple: A tuple containing the processed dataset (numpy array) and the station estimate (string).
    """
        #print(index, row)
        for i in range(len(row.index)):
           # print(index.day)
            l.append([str(index.year)+'/'+ str(index.day)+'/'+ str(row.index[i]), float(str(row[i]).replace(',','.'))])

    aux=pd.DataFrame(l,columns=['data', 'vazao'])
    
    
    
    aux1=aux.dropna() 
    aux1.index = pd.DatetimeIndex(aux1['data'].values)
            
    ts = aux1[(aux1['data']>= '1990-01-01') ] 
    ts = ts[(ts['data']<=  '2012-12-31') ] 
    
    ts1 = ts.sort_index()
    
#    plt.plot(ts1['vazao'], label='True')
#    plt.title(Est)
#    plt.show()
    
    all_y = ts1['vazao'].values
    dataset=all_y.reshape(-1, 1)
    return dataset, Est


def param_config(): 

    filters = ['db1']
    look_backs = [10]
    horizontes= [7]
    dec_level = [3]
    
    
    configs_ann = []
    for k in filters:
        for i in look_backs:
            for j in horizontes:
                for t in dec_level:
                    cfg = [k, i, j, t]
                    configs_ann.append(cfg)
    print('Total configs: %d' % len(configs_ann))
    return configs_ann

   
'''----------------main part------------------------------'''


path_pkl = 'pkl/'

if not os.path.exists(path_pkl):
    """
Generates a list of parameter configurations.

    This method creates all possible combinations of filter, lookback, 
    horizon, and decision level parameters and returns them as a list of lists.

    Args:
        None

    Returns:
        list[list]: A list of lists, where each inner list represents a unique 
            parameter configuration with the format [filter, lookback, horizon, decision_level].
    """
    os.makedirs(path_pkl)

estimators=[


('ANN',
 MLPRegressor(),
[
	{
	  'learning_rate': ['constant'],
	  'solver' : [ 'lbfgs', 'adam' ],
      'learning_rate_init':[0.01, 0.001],
	  'hidden_layer_sizes':[100, 200],
      'activation': ['relu'],
       
	  },
      ]
),
 
 


   
]



set_style()  
run=30
for run in range(run):
    np.random.seed(run)
     
    path_dados='dados/10_est/'
        
    csv_list = glob.glob(path_dados + '*.txt')
    csv_list.sort()
    
    lista=[]
    X=[]
    for csv in csv_list:
       
        
        Est = csv[-12:-4]
        
        ts1= read_csv(csv,  delimiter=';')
        ts1.index = pd.DatetimeIndex(ts1['NivelConsistencia'].values)
        ts2 = pd.DataFrame(columns=ts1.columns, index=pd.date_range(min(ts1.index), max(ts1.index)))
        
        ts2 = ts1.filter(['NivelConsistencia','MediaAnualStatus', 'Vazao01', 'Vazao02', 'Vazao03',
               'Vazao04', 'Vazao05', 'Vazao06', 'Vazao07', 'Vazao08', 'Vazao09',
               'Vazao10', 'Vazao11', 'Vazao12', 'Vazao13', 'Vazao14', 'Vazao15',
               'Vazao16', 'Vazao17', 'Vazao18', 'Vazao19', 'Vazao20', 'Vazao21',
               'Vazao22', 'Vazao23', 'Vazao24', 'Vazao25', 'Vazao26', 'Vazao27',
               'Vazao28', 'Vazao29', 'Vazao30'], axis=1)
        ts2_newname=['Data','01', '02', '03',
               '04', '05', '06', '07', '08', '09',
               '10', '11', '12', '13', '14', '15',
               '16', '17', '18', '19', '20', '21',
               '22', '23', '24', '25', '26', '27',
               '28', '29', '30', '31']
        ts2.columns=ts2_newname
        
        ts2.iloc[0:1,1:32]
        ts2.iloc[0:1,:] # first row
        ts2.iloc[0:1,:].columns.values[1] #name_column = '01'
        
        ts2.iloc[0:1,:]['Data'].values[0]
        ts2.iloc[0:1,:]['Data'].values[0][2:10] #'/01/2015'
        
        ts2.iloc[1:2,:].columns.values[2]+ts2.iloc[0:1,:]['Data'].values[0][2:10] #'02'+'/01/2015'
        
        
        l=[]  
        for  index, row in ts2.iloc[:,1:32].iterrows():
            #print(index, row)
            for i in range(len(row.index)):
               # print(index.day)
                l.append([str(index.year)+'/'+ str(index.day)+'/'+ str(row.index[i]), float(str(row[i]).replace(',','.'))])
       
        aux=pd.DataFrame(l,columns=['data', 'vazao'])
        
        
        
        aux1=aux.dropna() 
        aux1.index = pd.DatetimeIndex(aux1['data'].values)
                      
        ts = aux1[(aux1['data']>= '1973-01-01') ] #1973
        ts = ts[(ts['data']<= '2020-12-31') ] #'2019-05-31'
        ts[u'Estacao']=Est
        ts1 = ts.sort_index()
        
        plt.rcParams["figure.figsize"] = (20,8)
        plt.xlabel("Intervalo de tempo (dias)")
        plt.ylabel("Vazão (m³/s)") 
        plt.plot(ts1['vazao'], '.',markersize=5,linewidth=None, label='True')
        plt.title(Est)
        
        plt.show()
        vazao = ts1['vazao'].values
 
        
        
        
        
        X.append(ts1)
             
    X = pd.concat(X) 

    V=[]       
    for i in range(len(X)):
        df=X.iloc[i]
        v = df['Estacao']
        V.append({'Date': df['data'], 'Gauge':v, 'value':df['vazao'], 'type':'flow'})
     
    V=pd.DataFrame(V)
 
    U=pd.crosstab(index=V['Date'], columns=V['Gauge'], values=V['value'], aggfunc=sum) 
    

    gauge_list=[] 
    g=np.array(U.columns)           
    for i in range(len(g)):
        gauge_list.append(('mod'+str(i+1),g[i], (g[i],),))


    for entrada, e, gauges in gauge_list:
            
            #make list of parametres   
            cfg_list1=param_config()
            
            
            print (e, entrada)
            W=U[[str(i) for i in gauges]] #x train, ts = estação modelada
            
            
            print(W.shape, W.dropna().shape)
            W.dropna(inplace=True)#fig1
           
            aux=W
         
            # split into train and test sets, 80% test data, 20% training data
            train_size = int(aux.shape[0]*0.80)
            test_size = int(aux.shape[0] - train_size)
            
            data_ts1, test_ts1 = aux[0:train_size], aux[train_size:aux.shape[0]]
            len_train=data_ts1.shape[0]
            print('Lenth of train = ',len_train)
            len_test=test_ts1.shape[0]
            print('Lenth of test = ',len_test)
            
            for cfg_list in cfg_list1:
                coefs_pred = []
                results_list = []
                l= cfg_list[3] #decomposition level
                   
                ##new train and test size because of IWT
                train_size = (2**l)*int(len_train/(2**l))
                test_size =  (2**l)*int(len_test/(2**l))
                
                train, test = aux[0:train_size], aux[train_size:train_size+test_size]
               
                        
            
                """
                WANN
                """   
                for col in test.columns: 
                    print('COLUNA: ',col) 
                
                
                results=[]   
                
                results.append([wavelet_forecast_ann(aux , cfg_list,  train, test, estimators, entrada,e)] )
 
                df = pd.DataFrame(results[0], columns=['mape_mean_he','mape_mean_le','mape_extremes','est', 'entrada', 'hor', 'modelo', 'wav', 'level', 'lb', 'Lj', 'mape', 'r2', 'rmse','pred', 'obs', 'date'])
                df.to_pickle(path_pkl+str(run)+'_'+str(e)+'_hor_'+str(cfg_list[2])+'_wann_one_'+'_wf_'+str(cfg_list[0])+'_l_'+str(cfg_list[3])+'_lb_'+str(cfg_list[1]) +'_'+time.strftime("_%Hh_%Mm_%S")+ 
                                                           '.pkl') 
            
    

