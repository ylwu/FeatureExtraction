import json
import os
import multiprocessing as mp
import subprocess
import sys
import ConfigParser
from db2features import main as db2features
import time
import shutil
import gc
import csv
import numpy as np

#from endtoend_jaguar.raw2db import r2db_main

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
            if row[0] in all_num_fields:
                num_file.write("%s,%s,%s"%(timestamp_to_index[float(row[1])],row[0],float(row[2])))
                num_file.write('\r\n')
                num_field_ids.add(row[0])
            elif row[0] in cat_fields_meta_dict:
                cat_file.write("%s,%s,%d"%(timestamp_to_index[float(row[1])],row[0],int(float(row[2]))))
                cat_file.write('\r\n')
                cat_field_ids.add(row[0])
            else:
                continue

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

def convert_signal_to_features(signal_file_path,fid):
    
    print 'start converting', signal_file_path

    if not os.path.exists(temp_csv_path):
        os.makedirs(temp_csv_path)

    tic = time.time()
    convert_csv_format(signal_file_path,intermediate_path)
    tic_2 = time.time()
    print 'conversion time: ' , tic_2 - tic

    print 'extracting features', signal_file_path

    db2features.extractAllFeatures()
    tic_3 = time.time()
    print 'extraction time:', tic_3 - tic_2
    
    mergeFiles(output_path,fid,temp_csv_path,all_features_dict)
    clearFolder(intermediate_path)
    clearFolder(temp_csv_path)


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

def mergeFiles(output_path,tid, temp_csv_path,all_features_dict):
    
    base_path = 'intermediate/'
    with open(base_path + 'cat_field_ids',"rb") as f:
        cat_field_ids = json.load(f)
    with open(base_path + 'num_field_ids',"rb") as f:
        num_field_ids = json.load(f)

    all_field_ids = set(cat_field_ids+num_field_ids)

    file_list = sorted(os.listdir(temp_csv_path))
    file_list_full_path = [temp_csv_path + fn for fn in file_list]
    with open(output_path+str(tid) + ".csv",'w') as writefile:
        for file_path in file_list_full_path:
            with open(file_path,'rb') as infile:
                for row_line in infile.readlines():
                    row = row_line.rstrip().split(",")
                    cid = row[2].split('__')[0]
                    all_field_ids.discard(cid)
                    writefile.write("%s,%s,%d"%(row[0],row[1],all_features_dict[row[2]]))
                    writefile.write("\r\n")
    if len(all_field_ids) != 0:
        print tid, 'error: not all fields are transformed'

def convert_all_signals(base_path):
    all_files = os.listdir(base_path)
    fid = 1
    for fname in all_files:
        try:
            convert_signal_to_features(base_path+fname,fid)
            fid += 1
        except Exception,e:
            print e
            print 'error in ==============', i
            clearFolder(intermediate_path)
            clearFolder(temp_csv_path)

def generate_all_features():
    global all_features_dict, all_features

    num_feature_suffix = ['absmeandiff',"absmaxdiff","maxdiff","mindiff","absmindiff","absmeandiffdiff","absmaxdiffdiff","absmindiffdiff"
    ,"lastminusfirst","sum","sum_abs","mean","mean_abs","max","min","max_abs","min_abs","var"]
    cat_feature_suffix = ["mode","jitter","stability"]

    all_features = []
    all_features_dict = {}
    for i in range(len(fields_dict)):
        if str(i) in all_num_fields: #this is a numerical feature
            all_features.append(str(i))
            all_features.extend(map(lambda x: str(i) + '__' + x,num_feature_suffix))
        elif str(i) in cat_fields_meta_dict:
            all_features.extend(map(lambda x: str(i) + '__' + x,cat_feature_suffix))
            min_max = cat_fields_meta_dict[str(i)]
            all_features.extend(map(lambda x: str(i) + '__percent__' + str(x),range(min_max[0],min_max[1]+1)))
            all_features.extend(map(lambda x: str(i) + '__total__' + str(x),range(min_max[0],min_max[1]+1)))
    
    for i in range(len(all_features)):
        all_features_dict[all_features[i]] = i
    with open(metadata_output_path + 'all_features_list.json', 'w') as outfile:
        json.dump(all_features, outfile)
    with open(metadata_output_path + 'all_features_dict.json', 'w') as outfile:
        json.dump(all_features_dict, outfile)
    
    return len(all_features)

def build_feature_matrix(tid,feature_matrix,has_feature_matrix): #tid is one off from file name
    global row_id
    filename = output_path + str(tid+1) + '.csv'
    if not os.path.isfile(filename):
        return
    with open(filename,'rb') as csvfile:
        all_rows = csvfile.readlines()
    for i,row_line in enumerate(all_rows):
        row = row_line.rstrip().split(",")
        time_index = int(float(row[0]))
        feature_index = int(float(row[2]))
        if not (np.isinf(np.float16(float(row[1]))) or np.isnan(np.float16(float(row[1])))):
            feature_matrix[row_id,(time_index-1) * feature_num + feature_index] = np.float16(float(row[1]))
            has_feature_matrix[row_id,feature_index] = 1
    row_id += 1

def write_matrix_to_file():
    num_entities = len(os.listdir(output_path))

    feature_matrix = np.zeros([num_entities,feature_num * 10],dtype=np.float16)
    has_feature_matrix = np.zeros([num_entities,feature_num],dtype=np.int8)
    global row_id
    row_id = 0
    for tid in range(num_entities):
        print tid
        build_feature_matrix(tid,feature_matrix,has_feature_matrix)
    print 'rows', row_id
    feature_matrix = feature_matrix[:row_id,:]
    has_feature_matrix = has_feature_matrix[:row_id,:]
    np.savetxt(feature_matrix_path+'feature_matrix.csv',feature_matrix,fmt='%.10g')
    np.savetxt(feature_matrix_path+'has_feature_matrix.csv',has_feature_matrix,fmt='%.3g')

def make_dirs(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def revert_dict(d):
    new_d = {}
    for key in d:
        new_d[d[key]] = key
    return new_d

if __name__ == "__main__":
    config = ConfigParser.RawConfigParser()
    config.read('config.ini')

    data_base_path = config.get('Data','data_base_path')
    metadata_base_path = config.get('Data','metadata_base_path')
    metadata_output_path = config.get('Data', 'metadata_output_path')
    feature_matrix_path = config.get('Data', 'feature_matrix_path')

    #directorys to store intermediate results
    intermediate_path = 'intermediate/'
    temp_csv_path = 'intermediate/temp_csvs/'
    output_path = 'output/'

    make_dirs([intermediate_path,temp_csv_path,output_path,metadata_output_path,feature_matrix_path])

    with open(metadata_base_path + 'all_cat_fields_min_max_dict.json','r') as in_file:
        cat_fields_meta_dict = json.load(in_file)
    with open(metadata_base_path + 'all_num_fields.json', 'r') as outfile2:
        all_num_fields = set(json.load(outfile2))
    with open(metadata_base_path + 'fields_dict.json', 'r') as infile:
        fields_dict = json.load(infile)

    reverse_fields_dict = revert_dict(fields_dict)
    with open(metadata_output_path + 'fields_reverse_dict.json', 'w') as outfile:
        json.dump(reverse_fields_dict,outfile)

    shutil.copy(metadata_base_path + 'all_cat_fields_min_max_dict.json',metadata_output_path)
    shutil.copy(metadata_base_path + 'all_num_fields.json',metadata_output_path)
    shutil.copy(metadata_base_path + 'fields_dict.json',metadata_output_path)


    feature_num = generate_all_features()
    convert_all_signals(data_base_path)
    write_matrix_to_file()
    clearFolder(output_path)
    


    

