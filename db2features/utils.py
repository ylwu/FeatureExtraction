import os,sys
import datetime
import time
import numpy as np
import xlrd
import re
import dill as pickle
import multiprocessing as mp
import csv
import random

def parallelize(ncores):
    pool = mp.Pool(processes=ncores)
    return pool

def runFunctionInParallel(function, iter_args, args, ncores):
    pool = parallelize(ncores)
    funclist = []
    for i in range(len(iter_args)):
        args.insert(0,None)
    for chunk in xrange(ncores):
        for i,arg_type in enumerate(iter_args): #iter_args = [[[1,2],[3,4],[5,6]],[extractSimpleVariableToDF,extractSimpleVariableToDF,extractSimpleVariableToDF]]
            args[i] = iter_args[i][chunk]
        f = pool.apply_async(function, args[:])
        funclist.append(f)
    print 'nparallel', len(funclist)
    result = [f.get() for f in funclist]
    pool.close()
    pool.terminate()
    return result