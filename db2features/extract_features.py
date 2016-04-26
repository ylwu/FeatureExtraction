import datetime,time
import numpy as np
import os,sys
import tempfile
import copy
import pandas as pd
import extract_features_subroutines as efs
import complex_features as cf
import multiprocessing as mp
import random
import csv
import logging
import json
import gc
import utils

def extractGenericNumericalFeatures(new_project_ids=None, images=False, csv_output_dir=None):
    #for images, only extract 3 versions of the features, early, mid and late which
    #will eventually correspond to different sized images
    #for non-images, extract versions of features for every cutoff date
    print 'extractGenericNumericalFeatures'
    cutoffs = None


    have_features = True #if we're passed new_project_ids, we assume we have all the features already,
                    #and we're calculating for new projects the features we already have
    if not new_project_ids:
        have_features = False


    field_ids = efs.fieldIDsOfType('num')

    print 'numeical field ids',len(field_ids)

    if len(field_ids) == 0:
        return []

    with open('intermediate/cutoffs.json', 'rb') as outfile2:
        cutoffs = json.load(outfile2)

    csv_path = 'intermediate/num_signals.csv'
    file_size = os.path.getsize(csv_path)

    if file_size > 1000:
        ncores = min(6000000000 /file_size,10)
    else:
        ncores = 2

    print 'ncores', ncores
    chunk_size = len(field_ids)/ncores
    chunks = field_ids
    chunkExtractionFunc =  efs.extractVariablesAndInsertChunk
    csv_path = 'intermediate/num_signals.csv'
    args = [efs.extractSimpleNumVariableToDF,
            efs.numFeatureInsert,
            'time_varying_values',
            new_project_ids,
            have_features,
            False,
            csv_output_dir,
            cutoffs,
            csv_path]

    iter_args = [[]]
    for chunk in xrange(ncores-1):
        iter_args[0].append(chunks[chunk*chunk_size:(chunk+1)*chunk_size])
    iter_args[0].append(chunks[(ncores-1)*chunk_size:])

    outputs = utils.runFunctionInParallel(chunkExtractionFunc,iter_args, args,ncores)

    return outputs

def extractGenericCategoricalFeatures(new_project_ids=None, images=False, csv_output_dir=None):
    print 'start extractGenericCategoricalFeatures ============================='

    #only need pickle file if we're generating all features
    #for specific project_ids we re-extract every time
    have_features = True #if we're passed new_project_ids, we assume we have all the features already,
                    #and we're calculating for new projects the features we already have
    if not new_project_ids:
        have_features = False

    print 'extract cat field ids'
    cat_field_ids = efs.fieldIDsOfType('cat')

    print 'cat field_ids ', len(cat_field_ids)

    if len(cat_field_ids) == 0:
        return []

    with open('intermediate/cutoffs.json', 'rb') as outfile2:
        cutoffs = json.load(outfile2)

    csv_path = 'intermediate/cat_signals.csv'
    file_size = os.path.getsize(csv_path)

    if file_size > 1000:
        ncores = min(6000000000 /file_size,10)
    else:
        ncores = 2

    print 'ncores'
    print ncores
    iter_args1 = []
    chunk_size = len(cat_field_ids)/ncores#save 2 processors for varchar and bool
    for chunk in xrange(ncores-1):
        iter_args1.append(cat_field_ids[chunk*chunk_size:(chunk+1)*chunk_size])
    iter_args1.append(cat_field_ids[(ncores-1)*chunk_size:])
    iter_args2 = [efs.extractSimpleCatVariableToDF]*ncores
    iter_args = [iter_args1,iter_args2]
    
    
    args = [efs.catFeatureInsert,
            'time_varying_values',
            new_project_ids,
            have_features,
            False,
            csv_output_dir,
            cutoffs,
            csv_path]
    csv_files = utils.runFunctionInParallel(efs.extractVariablesAndInsertChunk,
                                        iter_args,args,ncores)

    gc.collect()
    conn.close()
    return csv_files
