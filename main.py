# from util.functions import forFileInDir, readLiteTable
# from readDataDir_v2_3 import readChannelDict
import json
import os
import multiprocessing as mp
import subprocess
import sys
import ConfigParser
# sys.path.insert(0,'/scratch')
# from endtoend_jaguar.raw2db import files_2_tvv_db as r2db
from db2features import main as db2features
# from endtoend_jaguar.db2features import extract_features
import time
# import shutil
import gc

#from endtoend_jaguar.raw2db import r2db_main

#channel_meta_dict: {channel_id:[min,max,isCat]}

def checkfilesindir():
    path = "/media/ylwu/DATA/alfad7/alfa/data/JLR/signal_all/"
    l = os.listdir(path)
    for i in range(1,7082):
        if 'signal_schema_' + str(i) + '.csv' not in l:
            print i

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def convert_csv_format(in_file_path,base_path):
    with open(in_file_path,'rb') as csvfile:
        trip_id = in_file_path.split("_")[-1][:-4]
        all_rows_raw = csvfile.readlines()
        all_rows = filter(lambda x: len(x.split(',')) ==4 and isfloat(x.split(',')[2]),all_rows_raw)
        distinct_timestamps = []
        distinct_timestamps = list(set([float(row.split(",")[1]) for row in all_rows]))

        distinct_timestamps.sort()
        timestamp_to_index = {}
        for i in range(len(distinct_timestamps)):
            timestamp_to_index[distinct_timestamps[i]] = i
        
        cutoff_timestamps = findCutoffs(distinct_timestamps,10)
        cutoff_indices = [timestamp_to_index[t] for t in cutoff_timestamps]

        index_to_low,index_to_low_high = findCutoffsIndex_low_high(cutoff_indices)

        with open(base_path + 'cutoffs.json', 'w') as outfile2:
            json.dump(cutoff_indices, outfile2)

        print 'finished meta', in_file_path

        t = 0
        cat_field_ids = set()
        num_field_ids = set()

        cat_filelines = []
        num_filelines = []

        cat_file = open(base_path + 'cat_signals.csv',"w")
        num_file = open(base_path + 'num_signals.csv',"w")

        for row_line in all_rows:
            row = row_line.rstrip().split(",")
            if int(row[0]) in text_channels:
                continue
            elif int(row[0]) in all_num_channels:
                num_file.write("%s,%s,%s"%(timestamp_to_index[float(row[1])],row[0],float(row[2])))
                num_file.write('\r\n')
                num_field_ids.add(row[0])
            elif row[0] in cat_channel_meta_dict:
                cat_file.write("%s,%s,%d"%(timestamp_to_index[float(row[1])],row[0],int(float(row[2]))))
                cat_file.write('\r\n')
                cat_field_ids.add(row[0])
            else:
                print "error, wrong id " + row[0]

        cat_file.close()
        num_file.close()
        with open(base_path + 'cat_field_ids',"w") as f:
            json.dump(list(cat_field_ids),f)
        with open(base_path + 'num_field_ids',"w") as f:
            json.dump(list(num_field_ids),f)
        all_rows = None
        gc.collect()

def findCutoffs(distinct_timestamps,num_cuts):
    low = distinct_timestamps[0]
    high = distinct_timestamps[-1]
    if high == low:
        return [high] * num_cuts
    interval = (high - low) / float(num_cuts)
    print interval
    cutoffs = []
    cutoff_index = 1
    i = 0
    while i < len(distinct_timestamps):
        if distinct_timestamps[i+1] <= low + cutoff_index * interval:
            i += 1
            continue
        else:
            cutoffs.append(distinct_timestamps[i])
            cutoff_index += 1
            if cutoff_index == num_cuts:
                break
    cutoffs.append(high)
    return cutoffs

def findCutoffsIndex_low_high(cutoff_indices):
    start = 0
    t = 1
    index = cutoff_indices[0]
    index_to_low = {}
    index_to_low[index] = 1
    index_to_low_high = {}
    for t in range(len(cutoff_indices)):
        if cutoff_indices[t] != index:
            index_to_low[index] = start + 1
            if start +1 != t:
                index_to_low_high[index] = [start+1,t]
            index = cutoff_indices[t]
            start = t
    index_to_low[index] = start +1
    if start+1 != len(cutoff_indices):
        index_to_low_high[index] = [start+1, len(cutoff_indices)]
    return index_to_low, index_to_low_high


def output_features_at_cutoffs(intermediate_path,final_file_path,cutoff_index_path):
    with open(cutoff_index_path,'rb') as f:
        cutoff_indices = set(json.load(cutoff_index_path))
    all_files = os.lisdir(intermediate_path)
    all_files_path = [intermediate_path + fi for fi in all_files]
    for f in all_files_path:
        lines = []
        with open(f,'rb') as csvfile:
            all_rows = csv.readlines()
            for row in all_rows:
                time_index = row.split(',')
                if time_index in cutoff_indices:
                    lines.append(row)
        with open(final_file_path,'a') as newfile:
            newfile.write('\r\n'.join(lines))

def convert_signal_to_features(signal_file_path):
    
    print 'start converting', signal_file_path

    intermediate_csv_path = 'intermediate/'
    if not os.path.exists(intermediate_csv_path):
        os.makedirs(intermediate_csv_path)
    file_name = signal_file_path.split('/')[-1]
    fid = file_name.split('.')[0].split('_')[-1]

    tic = time.time()
    convert_csv_format(signal_file_path,intermediate_csv_path)
    tic_2 = time.time()
    print 'conversion time: ' , tic_2 - tic

    print 'extracting features', signal_file_path

    db2features.extractFeatures()
    tic_3 = time.time()
    print 'extraction time:', tic_3 - tic_2
    output_features_path = '/media/ylwu/DATA/alfad7/alfa/data/JLR/features_all/intermediate/'
    output_path = '/media/ylwu/DATA/alfad7/alfa/data/JLR/features_all/final_v2/'
    mergeFiles(output_path,fid,output_features_path,feature_desc_map)
    clearFolder(output_features_path)


def matchCutoffs(indices, cutoffs):
    newCutoffs = []
    k = 0
    if len(indices) == 0:
        return []
    while cutoffs[k] < indices[0]:
        k += 1
        newCutoffs.append(None)
    i = 0
    while i < len(indices) - 1:
        if indices[i] == cutoffs[k] or indices[i+1] > cutoffs[k]:
            newCutoffs.append(indices[i])
            k += 1
        else:
            i += 1
    while k < len(cutoffs):
        newCutoffs.append(indices[len(indices)-1])
        k += 1

def clearFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

def mergeFiles(output_path,tid, intermediate_path,feature_desc_map):
    
    base_path = '/media/ylwu/DATA/alfad7/alfa/data/JLR/signal_all_converted/'
    with open(base_path + 'cat_field_ids',"rb") as f:
        cat_field_ids = json.load(f)
    with open(base_path + 'num_field_ids',"rb") as f:
        num_field_ids = json.load(f)

    all_field_ids = set(cat_field_ids+num_field_ids)

    file_list = sorted(os.listdir(intermediate_path))
    file_list_full_path = [intermediate_path + fn for fn in file_list]
    with open(output_path+str(tid) + ".csv",'w') as writefile:
        for file_path in file_list_full_path:
            with open(file_path,'rb') as infile:
                for row_line in infile.readlines():
                    row = row_line.rstrip().split(",")
                    cid = row[2].split('__')[0]
                    all_field_ids.discard(cid)
                    writefile.write("%s,%s,%d"%(row[0],row[1],feature_desc_map[row[2]]))
                    writefile.write("\r\n")
    if len(all_field_ids) != 0:
        print tid, 'error: not all fields are transformed'

def convert_all_signals(base_path):
    all_files = os.listdir(base_path)
    for i in range(6958,7082):
        f_name = 'signal_schema_' + str(i) + '.csv'
        if f_name in all_files:
            try:
                convert_signal_to_features(base_path+f_name)
            except Exception,e:
                print e
                print 'error in ==============', i
                output_features_path = '/media/ylwu/DATA/alfad7/alfa/data/JLR/features_all/intermediate/'
                clearFolder(output_features_path)

if __name__ == "__main__":
    config = ConfigParser.RawConfigParser()
    config.read('config.ini')

    data_base_path = config.get('Data','data_base_path')
    metadata_base_path = config.get('Data','metadata_base_path')

    with open(metadata_base_path + 'all_cat_fields_min_max_dict','r') as in_file:
        cat_channel_meta_dict = json.load(in_file)
    with open(metadata_base_path + 'all_text_fields', 'r') as outfile:
        text_channels = set(json.load(outfile))
    with open(metadata_base_path + 'all_num_fields', 'r') as outfile2:
        all_num_channels = set(json.load(outfile2))
    with open(metadata_base_path + 'all_features_dict.json', 'rb') as infile:
        feature_desc_map = json.load(infile)

    #convert_signal_to_features(OLD_DATA_FILES_DIRECTORY + "signal_schema_5417.csv")
    convert_signal_to_features(data_base_path + "signal_schema_27.csv")
    #convert_signal_to_features(OLD_DATA_FILES_DIRECTORY + "signal_schema_2.csv")
    #convert_all_signals(OLD_DATA_FILES_DIRECTORY)
    


    

