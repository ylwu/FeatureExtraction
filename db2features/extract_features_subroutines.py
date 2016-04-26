import datetime,time
import numpy as np
import os,sys
import tempfile
import copy
import pandas as pd
import complex_features as cf
import multiprocessing as mp
import random
import csv
import logging
import json
import gc
import utils

def extractSimpleCatVariableToDF(raw_df,field_id,table='time_varying_values', project_ids=None):
    #print 'start extracting field_id ', field_id
    df = raw_df[raw_df['channel_id'] == field_id]
    df.drop('channel_id', axis=1, inplace=True)
    df.drop_duplicates(subset = ['date_index'], inplace=True)
    #df.dropna(axis=0, subset=['date_index'], inplace=True)
    df.set_index(['date_index'],inplace=True)
    df.sort_index(inplace = True)
    return df

def extractSimpleNumVariableToDF(raw_df,field_id,table='time_varying_values', project_ids=None):
    df = raw_df[raw_df['channel_id'] == field_id]
    df.drop('channel_id', axis=1, inplace=True)
    df.drop_duplicates(subset = ['date_index'], inplace=True)
    #df.dropna(axis=0, subset=['date_index'], inplace=True)
    df.set_index(['date_index'],inplace=True)
    df.sort_index(inplace = True)
    return df

def fieldIDsOfType(feature_type, table='time_varying_values'):
    base_json_path = 'intermediate/'
    if feature_type == 'num':
        with open(base_json_path + 'num_field_ids','rb') as f:
            field_ids = [int(x) for x in json.load(f)]
    elif feature_type == 'cat':
        with open(base_json_path + 'cat_field_ids','rb') as f:
            field_ids = [int(x) for x in json.load(f)]
    return field_ids

def metadataFieldNamesOfType(conn, feature_type):
    query = '''
    SELECT distinct(field_name)
    FROM metadata
    WHERE field_value_%s is NOT NULL
    ''' % (feature_type)
    featureData = [x[0] for x in sql.executeAndReturnData(conn,query)]
    return featureData

def extractVariablesAndInsertChunk(field_ids, extractionFunc, insertionFunc, table, project_ids, have_features,
                            metadata, csv_output_dir,cutoffs,csv_path):
    #works for all categorical and numeric

    print 'read', csv_path

    if csv_path == 'intermediate/num_signals.csv':
        raw_df = pd.read_csv(csv_path, names=['date_index','channel_id','value'],dtype = {'date_index':np.int64, 'channel_id':np.int32, 'value': np.float64},error_bad_lines = False)
    else:
        raw_df = pd.read_csv(csv_path, names=['date_index','channel_id','value'],dtype = {'date_index':np.int64, 'channel_id':np.int32, 'value': np.int64},error_bad_lines = False)


    csv_file = 'intermediate/temp_csvs/' + str(os.getpid())+'_tmp_data.csv'
    
    if not field_ids:
        print csv_file, 'no field id'
    else:
        print csv_file, 'total:', len(field_ids), ' first:',field_ids[0], 'last: ', field_ids[-1]
    #print "EXTRACTING FIELD_IDS:", field_ids
    for i,field_id in enumerate(field_ids):
        t0 = time.time()
        print extractionFunc
        df = extractionFunc(raw_df, field_id,table=table, project_ids=project_ids)
        if df.empty:
            continue
        df['feature_desc'] = str(field_id)
        insertionFunc(df, metadata, have_features, csv_file, cutoffs)
        #print 'finished extracting field id ', field_id
        #print "done"
    raw_df = None
    gc.collect()
    return csv_file

def catFeatureInsertChunk(dfList, metadata, have_features):
    csv_file = utils.csvPath(str(os.getpid())+'_tmp_data.csv')
    for i,df in enumerate(dfList):
        catFeatureInsert(df, metadata, have_features, csv_file)
    return csv_file
def catFeatureInsert(df, metadata, have_features, csv_file,cutoffs):
    if df.empty:
        print "empty"
    if 'ohe' in df.columns:
        ohe = df['ohe'].iat[0]
        df.drop('ohe', axis=1, inplace=True)
        isNumber = True #doesn't matter for ohe case, for non-ohe case these are all still numbers inside strings
    else:
        ohe = 0
        try:
            int(df['value'].iat[0])
        except:
            isNumber = False
        else:
            isNumber = True
    catFeatures = cf.CategoricalFeature(df, ohe, csv_file,cutoffs, nonint = not isNumber)
    #print "initialized features"

    dont_have_features = not have_features
    catFeatures.extract(create_new_features=dont_have_features)

def numFeatureInsertChunk(dfList, metadata, have_features):
    csv_file = utils.csvPath(str(os.getpid())+'_tmp_data.csv')

    for i,df in enumerate(dfList):
        if df.empty:
            continue
        numFeatureInsert(df, metadata, have_features, csv_file)
    return collective_stats, csv_file
    
def numFeatureInsert(df, metadata, have_features, csv_file, cutoffs):
    if not metadata:
        numFeatures = cf.NumericalFeature(df,csv_file,cutoffs)
    else:
        numFeatures = cf.MetaNumericalFeature(df,csv_file)
    #print "finished num init"
    dont_have_features = not have_features
    numFeatures.extract(create_new_features=dont_have_features)
