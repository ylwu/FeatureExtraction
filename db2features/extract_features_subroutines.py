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

def extractSimpleCatVariableToDF(raw_df,conn, field_id,table='time_varying_values', project_ids=None):
    #print 'start extracting field_id ', field_id
    df = raw_df[raw_df['channel_id'] == field_id]
    df.drop('channel_id', axis=1, inplace=True)
    df.drop_duplicates(subset = ['date_index'], inplace=True)
    #df.dropna(axis=0, subset=['date_index'], inplace=True)
    df.set_index(['date_index'],inplace=True)
    df.sort_index(inplace = True)
    return df

def extractSimpleNumVariableToDF(raw_df,conn, field_id,table='time_varying_values', project_ids=None):
    df = raw_df[raw_df['channel_id'] == field_id]
    df.drop('channel_id', axis=1, inplace=True)
    df.drop_duplicates(subset = ['date_index'], inplace=True)
    #df.dropna(axis=0, subset=['date_index'], inplace=True)
    df.set_index(['date_index'],inplace=True)
    df.sort_index(inplace = True)
    return df

def extractVariablesForProjectID(conn, project_id, field_ids, table='time_varying_values'):
    #extract all variables in field_ids for a single project_id
    #used in image extraction because we want a multichanneled image where each project contains an image with all features
    #this is opposed to the non-image paradigm of extracting a single variable for all project ids
    extractFeatures = '''
    SELECT project_id, field_id, date, field_value_float, field_value_int,field_value_bool,field_value_date,
                             field_value_varchar, field_value_text
    FROM %s
    WHERE project_id = %s
    AND field_id in (%s)
    AND (field_value_float is not NULL OR
         field_value_int is not NULL OR
         field_value_bool is not NULL OR
         field_value_date is not NULL OR
         field_value_text is not NULL OR
         field_value_varchar is not NULL)
    ''' % (table, project_id, str(field_ids)[1:-1])
    rawFeatureData = sql.executeAndReturnData(conn,extractFeatures)
    featureData = []
    for row in rawFeatureData:
        for i in xrange(6):
            if row[i+3] != None:
                featureData.append((row[0],row[1],row[2],row[i+3]))

    columns = ['project_id','field_id','date','value']
    df = pd.DataFrame.from_records(featureData, columns=columns)

    df.set_index(['project_id', 'field_id', 'date'], inplace=True)
    #remove rows with null dates (except for nonexistant field_ids, which we'll add in next line)
    df = df[df.index.labels[2] != -1]

    #add in field_ids with null values
    nonnull_field_ids = set(df.index.get_level_values('field_id').unique())
    empty_field_ids = pd.DataFrame(columns=columns)
    empty_field_ids['date'] = empty_field_ids['date'].astype('datetime64[ns]')
    field_ids_not_included = []
    for field_id in field_ids:
        if field_id not in nonnull_field_ids:
            field_ids_not_included.append(field_id)
            new = pd.DataFrame([[project_id, field_id, pd.NaT, None]], columns = columns)
            empty_field_ids = empty_field_ids.append(new, ignore_index=True)

    empty_field_ids.set_index(['project_id', 'field_id', 'date'], inplace=True)

    df = pd.concat([df, empty_field_ids])
    df.sortlevel(level=0,inplace = True)

    return df

def fieldIDsOfType(conn,feature_type, table='time_varying_values'):
    base_json_path = '/media/ylwu/DATA/alfad7/alfa/data/JLR/signal_all_converted/'
    if feature_type == 'num':
        with open(base_json_path + 'num_field_ids','rb') as f:
            field_ids = [int(x) for x in json.load(f)]
    elif feature_type == 'cat':
        with open(base_json_path + 'cat_field_ids','rb') as f:
            field_ids = [int(x) for x in json.load(f)]
    # featureData = [x[0] for x in sql.executeAndReturnData(conn,query)]

    return field_ids

def metadataFieldNamesOfType(conn, feature_type):
    query = '''
    SELECT distinct(field_name)
    FROM metadata
    WHERE field_value_%s is NOT NULL
    ''' % (feature_type)
    featureData = [x[0] for x in sql.executeAndReturnData(conn,query)]
    return featureData

def checkCertified(conn,person_hash, date):
    sql = '''
    SELECT date_certified
    FROM people
    WHERE person_hash = '%s'
    ''' % person_hash
    date_certified = sql.executeAndReturnData(conn,sql)
    if date_certified:
        date_certified = date_certified[0][0]
        if date_certified:
            if date_certified <= date:
                return True
    return False

def extractCertifiedFeatures(conn,feature_df, person_field_name, project_start_dates):
    #not using this right now- preliminary results suggest its meaningless
    cert_field_name = person_field_name + '_certified'
    certified_feature = pd.DataFrame(columns=[cert_field_name])
    for project_id in feature_df.index:
        if project_id in project_start_dates.index:
            start_date = project_start_dates.loc[project_id]['start_date']
            person_hash = feature_df[person_field_name].loc[project_id]
            cert = checkCertified(conn, person_hash, start_date)
            certified_feature.loc[project_id] = [cert]
    feature_df = pd.concat([feature_df,certified_feature],axis=1)
    return feature_df, person_field_name+"_certified"

def consolidateCSVImages(csv_files,field_ids, img_sizes):
    #since images are not currently saved to db,
    #we consolidate the csv files generated by each thread
    csvsByThreshold = {}
    for thresh in csv_files[0]:
        csvsByThreshold[thresh] = [csv_files[0][thresh]]
    for pid in csv_files[1:]:
        for thresh in pid:
            csvsByThreshold[thresh].append(pid[thresh])
    for thresh in csvsByThreshold:
        new_filename = utils.csvPath(thresh+'_images.csv')
        with open(new_filename, 'wb') as new_csv:
            writer = csv.writer(new_csv)
            writer.writerow([img_sizes[thresh]])
            writer.writerow(field_ids[thresh])

            for filename in csvsByThreshold[thresh]:
                with open(filename, 'rb') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        writer.writerow(row)
    for csv_file in csv_files:
        try:
            os.remove(csv_file)
        except:
            print "couldn't remove %s" % in_file
    return new_filename

def extractVariablesAndInsertChunk(field_ids, extractionFunc, insertionFunc, table, project_ids, have_features,
                            metadata, csv_output_dir,cutoffs,csv_path):
    #works for all categorical and numeric
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)

    print 'read', csv_path

    if csv_path == '/media/ylwu/DATA/alfad7/alfa/data/JLR/signal_all_converted/num_signals.csv':
        raw_df = pd.read_csv(csv_path, names=['date_index','channel_id','value'],dtype = {'date_index':np.int64, 'channel_id':np.int32, 'value': np.float64},error_bad_lines = False)
    else:
        raw_df = pd.read_csv(csv_path, names=['date_index','channel_id','value'],dtype = {'date_index':np.int64, 'channel_id':np.int32, 'value': np.int64},error_bad_lines = False)

    if csv_output_dir:
        csv_file = os.path.join(csv_output_dir, str(os.getpid())+'_tmp_data.csv')
    else:
        csv_file = utils.csvPath(str(os.getpid())+'_tmp_data.csv')
    
    if not field_ids:
        print csv_file, 'no field id'
    else:
        print csv_file, 'total:', len(field_ids), ' first:',field_ids[0], 'last: ', field_ids[-1]
    #print "EXTRACTING FIELD_IDS:", field_ids
    for i,field_id in enumerate(field_ids):
        t0 = time.time()
        df = extractionFunc(raw_df,conn, field_id,table=table, project_ids=project_ids)
        if df.empty:
            continue
        df['feature_desc'] = str(field_id)
        insertionFunc(df, metadata, have_features, csv_file, conn,cutoffs)
        #print 'finished extracting field id ', field_id
        #print "done"
    raw_df = None
    gc.collect()
    return csv_file

def extractVariablesChunk(field_ids, extractionFunc, table, project_ids):
    #for text features
    utils.initialize_logging()
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)

    dfList = []
    for field_id in field_ids:
        df = extractionFunc(conn, field_id, table=table, project_ids=project_ids)
        if not df.empty:
            df['feature_desc'] = str(field_id)
            dfList.append(df)

    return dfList

def extractVariablesAndInsertByProjectIDChunk(project_ids, extractionFunc, insertionFunc, table, field_ids, cutoffs,csv_output_dir):
    #for image creation
    utils.initialize_logging()
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)

    img_sizes = imageSizes()
    if csv_output_dir:
        csv_file_base = os.path.join(csv_output_dir, str(os.getpid())+'_images_')
    else:
        print "SHOULDNT BE HERE"
        sys.exit()
        csv_file_base = utils.csvPath(str(os.getpid())+'_images_')

    csv_files = imageCSVFiles(csv_file_base, img_sizes)

    collective_stats = {}

    field_ids = []
    for project_id in project_ids:
        df = extractionFunc(conn, project_id, field_ids, table=table)
        if not df.empty:
            new_field_ids = insertionFunc(df, cutoffs, csv_file_base, conn, collective_stats)
            field_ids.extend(new_field_ids)
    return collective_stats, csv_files, img_sizes, field_ids

def catFeatureInsertChunk(dfList, metadata, have_features):
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)
    csv_file = utils.csvPath(str(os.getpid())+'_tmp_data.csv')
    for i,df in enumerate(dfList):
        catFeatureInsert(df, metadata, have_features, csv_file, conn)
    return csv_file
def catFeatureInsert(df, metadata, have_features, csv_file, conn,cutoffs):
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
    catFeatures = cf.CategoricalFeature(df, ohe,conn, csv_file,cutoffs, nonint = not isNumber)
    #print "initialized features"

    dont_have_features = not have_features
    catFeatures.extract(create_new_features=dont_have_features)

def numFeatureInsertChunk(dfList, metadata, have_features):
    utils.initialize_logging()
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)
    csv_file = utils.csvPath(str(os.getpid())+'_tmp_data.csv')

    for i,df in enumerate(dfList):
        if df.empty:
            continue
        numFeatureInsert(df, metadata, have_features, csv_file, conn)
    conn.close()
    return collective_stats, csv_file
def numFeatureInsert(df, metadata, have_features, csv_file, conn,cutoffs):
    if not metadata:
        numFeatures = cf.NumericalFeature(df, conn, csv_file,cutoffs)
    else:
        numFeatures = cf.MetaNumericalFeature(df, conn, csv_file)
    #print "finished num init"
    dont_have_features = not have_features
    numFeatures.extract(create_new_features=dont_have_features)

def imageSizes():
    return {
            '0.1': 8,
            '0.5': 16,
            '0.75': 32
            }
def imageCSVFiles(csv_file_base, img_sizes):
    csv_files = {}
    for thresh in img_sizes:
        csv_files[thresh] = csv_file_base + thresh + '.csv'
    return csv_files

def imageFeatureInsertChunk(dfList, cutoffs):
    utils.initialize_logging()
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)

    csv_file_base = utils.csvPath(str(os.getpid())+'_images_')
    csv_files = imageCSVFiles(csv_file_base, img_sizes)

    collective_stats = {}
    field_ids = []
    for i,df in enumerate(dfList):
        if df.empty:
            continue
        new_field_ids = imageFeatureInsert(df, cutoffs,csv_file_base, conn, collective_stats=collective_stats)
        field_ids.extend(new_field_ids)
    conn.close()
    return collective_stats, csv_files, img_sizes, field_ids


def imageFeatureInsert(df, cutoffs, csv_file_base, conn, collective_stats={}):
        #access project_id
        print "accessing project_id"
        level_values = df.index.get_level_values(0)
        if len(level_values) < 1:
            return
        else:
            project_id = level_values[0]
        cByProjectID = [cutoffs[x][project_id] for x in xrange(len(cutoffs))]
        #create an image for a single project_id (each df in this case is one project_id)
        numFeatures = cf.Image(df, project_id, conn, csv_file_base,img_sizes, cByProjectID)
        print "passed img init"
        field_ids, stats = numFeatures.extract()

        #collect stats about distribution of variables and lengths of time series
        for key in stats:
            if key in collective_stats:
                for op in stats[key]:
                    if op in collective_stats[key]:
                        collective_stats[key][op].append(stats[key][op])
                    else:
                        collective_stats[key][op] = [stats[key][op]]
            else:
                collective_stats[key] = {}
                for op in stats[key]:
                    collective_stats[key][op]= [stats[key][op]]
        return field_ids
def textFeatureInsertChunk(dfList, nlpType, model, have_features, csv_output_dir):
    utils.initialize_logging()
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)

    if csv_output_dir:
        csv_file = os.path.join(csv_output_dir, str(os.getpid())+'_tmp_data.csv')
    else:
        print "SHOULDNT BE HERE"
        sys.exit()
        csv_file = utils.csvPath(str(os.getpid())+'_tmp_data.csv')

    dont_have_features = not have_features
    for i,df in enumerate(dfList):
        if df.empty:
            continue
        print os.getpid(), "working on df", i
        if 'topic' in nlpType:
            topicFeature = cf.TopicFeature(df, conn, model['topic'], csv_file)
            print os.getpid(), "finished topic init"
            topicFeature.extract(create_new_features=dont_have_features)
            print os.getpid(), "finished topic df"
        if 'vector' in nlpType:
            vectorFeature = cf.VectorFeature(df, conn, model['vector'], csv_file)
            print os.getpid(), "finished vector init"
            vectorFeature.extract(create_new_features=dont_have_features)
            print os.getpid(), "finished vector df"

    #column_names = ['project_id', 'date', 'value', 'feature_desc', 'feature_id']
    #print os.getpid(), 'writing to db'
    #utils.insertCSVIntoDB(conn, 'generic_time_varying_features', column_names, csv_file, lock=utils.lock)
    #os.remove(csv_file)
    #print os.getpid(), 'finished writing to db'
    conn.close()
    return csv_file
