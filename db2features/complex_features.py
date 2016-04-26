import os,sys
import csv
from itertools import chain
import numpy as np
import copy
import pandas as pd
from scipy.stats import mstats
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import filters
from scipy.misc import toimage, fromimage, imresize
from PIL import Image as pilImage
import time
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
import complex_features_c_optimized as copt

def prepareDBAndCreateCSV(df,featureType,csv_file, create_new_features = True):
    df.to_csv(csv_file, header=False, index=False, line_terminator = '\r\n',mode='a')
    #print '----finished writing to csv', csv_file

######Categorical Feature Functions############
class CategoricalFeature(object):
    def __init__(self, df,ohe,csv_file,cutoffs, nonint = False, max_uniques=50):
        self.df = df
        self.ohe = ohe
        self.csv_file = csv_file
        self.nonint = nonint
        self.max_uniques = max_uniques
        self.newCutoffs = self.get_new_cutoffs(cutoffs)

    def matchCutoffs(self,indices, cutoffs):
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
        return newCutoffs

    def get_new_cutoffs(self,cutoffs): #cutoffs for this particular signal, as it doesn't have value at each time index
        #print 'new =================================================================='
        if self.df.empty:
            print "is Empty!!!!"
            return []
        all_indices = sorted(self.df.index.values)
        return self.matchCutoffs(all_indices,cutoffs)


    def filter_cutoff_index_and_translate(self,df):
        df.reset_index(inplace = True)
        filtered_df = df[df['date_index'].isin(self.newCutoffs)]
        new_df = pd.DataFrame(columns=('cutoff_index', 'value', 'feature_desc'))
        k = 0
        for i in range(len(self.newCutoffs)):
            if self.newCutoffs[i] != None:
                row_num = filtered_df[filtered_df['date_index'] == self.newCutoffs[i]].index[0]
                new_df.loc[k] = [i+1,filtered_df.at[row_num,'value'],filtered_df.at[row_num,'feature_desc']]
                k += 1
        return new_df

    def extract(self, create_new_features=True):

        #TODO: try doing these all in a window, as opposed to the whole sequence length
        features = [(self.featureMode, 'categorical', 'mode'),
                    (self.featureStability, 'numeric', 'stability'),
                    (self.featureJitter, 'numeric', 'jitter'),
                    ]
        for feature, feature_type, feature_name in features:
            if feature == self.featureStability and self.ohe:
                continue
            #print "EXTRACTING FEATURE", feature_name
            prepareDBAndCreateCSV(self.filter_cutoff_index_and_translate(feature()),
                                feature_type,
                                self.csv_file,
                                create_new_features=create_new_features)

        if len(self.getUnique()) <= self.max_uniques:
            #print 'EXTRACTING FEATURE PERCENT/TOTAL'
            percent_features,total_features = self.featureTotalAndPercentageEachValue()
            [prepareDBAndCreateCSV(self.filter_cutoff_index_and_translate(feature), 'numeric', self.csv_file, create_new_features=create_new_features) for feature in percent_features+total_features]

    def getUnique(self):
        unique = self.df['value'].unique()
        if self.ohe:
            uniqueOHE = unique
            unique = set()
            for v in uniqueOHE:
                for w in v.split(','):
                    unique.add(w)
            unique = list(unique)
        return unique
    def oheValues(self,values):
        valuesSet = set()
        [[valuesSet.add(av) for av in v.split(',')] for v in values]
        return list(valuesSet)
    def mapStrToNumbers(self, values):
        fmapping = {}
        bmapping = {}
        numbers = np.zeros((len(values),), dtype=np.int)
        cmap = 0
        for i,v in enumerate(values):
            if v not in fmapping:
                fmapping[v] = cmap
                bmapping[cmap] = v
                cmap += 1
            numbers[i] = fmapping[v]
        return fmapping, bmapping, numbers
    def featureMode(self):
        def coptWrapper(x):
            if not self.ohe:
                if self.nonint:
                    fmapping, bmapping, numbers = self.mapStrToNumbers(x.values)
                    values = list(copt.expanding_mode(numbers))
                    for i,v in enumerate(values):
                        values[i] = bmapping[values[i]]
                else:
                    values = copt.expanding_mode(x.values.astype(int))
            else:
                values = copt.expanding_mode_ohe(x.values)
            return values

        self.df['value'].fillna(0,inplace=True)
        mode = coptWrapper(self.df['value'])

        mode_df = self.df.copy()
        mode_df['value']=mode
        mode_df['feature_desc'] = self.df['feature_desc']+"__"+"mode"
        return mode_df

    def featureJitter(self):
        #number of times value has changed divided by total number of values seen so far
        def coptWrapper(x):
            if not self.ohe:
                if self.nonint:
                    fmapping, bmapping, numbers = self.mapStrToNumbers(x.values)
                    values = copt.expanding_jitter(numbers)
                else:
                    values = copt.expanding_jitter(x.values.astype(int))
            else:
                values = copt.expanding_jitter_ohe(x.values)
            return values

        self.df['value'].fillna(0,inplace=True)
        mode = coptWrapper(self.df['value'])

        mode_df = self.df.copy()
        mode_df['value']=mode
        mode_df['feature_desc'] = self.df['feature_desc']+"__"+"jitter"
        return mode_df

    def featureStability(self):
        #number of times most common value has been seen divided by total number of values seen so far
        def coptWrapper(x):
            if not self.ohe:
                if self.nonint:
                    fmapping, bmapping, numbers = self.mapStrToNumbers(x.values)
                    values = copt.expanding_stability(numbers)
                else:
                    values = copt.expanding_stability(x.values.astype(int))
            else:
                print "stability ohe not implemented yet, and doesn't make sense to me to implement"
                raise Exception
            return values

        self.df['value'].fillna(0,inplace=True)
        mode = coptWrapper(self.df['value'])

        mode_df = self.df.copy()
        mode_df['value']=mode
        mode_df['feature_desc'] = self.df['feature_desc']+"__"+"stability"
        return mode_df

    def featureTotalAndPercentageEachValue(self):
        feature_desc = self.df['feature_desc']+"__"
        percent_features = []
        total_features = []

        def coptWrapper(x):
            if not self.ohe:
                if self.nonint:
                    fmapping, bmapping, numbers = self.mapStrToNumbers(x['value'].values)
                    total, percentage, mapping = copt.expanding_total_and_percentage(numbers)
                    mapping = list(mapping)
                    for i,v in enumerate(mapping):
                        mapping[i] = bmapping[mapping[i]]
                else:
                    total, percentage, mapping = copt.expanding_total_and_percentage(x['value'].values.astype(int))
            else:
                total, percentage, mapping = copt.expanding_total_and_percentage_ohe(list(x['value'].values))
            for i, value in enumerate(mapping):
                x[str(feature_desc.iat[0])+"total__"+str(value)] = total[i,:]
                x[str(feature_desc.iat[0])+"percent__"+str(value)] = percentage[i,:]
            return x

        all_values = self.df.copy()

        unique = self.getUnique()


        for value in unique:
            all_values[feature_desc.iat[0]+"total__"+str(value)] = 0
            all_values[feature_desc.iat[0]+"percent__"+str(value)] = 0

        all_values = coptWrapper(all_values)

        for value in unique:
            current_total = self.df.copy()
            total_feature_name = feature_desc.iat[0]+"total__"+str(value)
            if total_feature_name in all_values.columns:
                current_total['value'] = all_values[total_feature_name]
                current_total['feature_desc'] = total_feature_name
                total_features.append(current_total)
            else:
                print "ERROR: feature name %s not in all_values" % total_feature_name
                raise Exception
            percent_feature_name = feature_desc.iat[0]+"percent__"+str(value)
            current_percent = self.df.copy()
            current_percent['value'] = all_values[percent_feature_name]
            current_percent['feature_desc'] = percent_feature_name
            percent_features.append(current_percent)
        return percent_features, total_features

class NumericalFeature(object):
    def __init__(self, df,csv_file,cutoffs,timeSeriesChunks=3):
        self.df = df
        self.timeSeriesChunks = timeSeriesChunks
        self.functions = [
            self.featureIdentity,
            self.featureAbsMeanDiff,
            self.featureAbsMaxDiff,
            self.featureMaxDiff,
            self.featureMinDiff,
            self.featureAbsMinDiff,
            self.featureAbsMeanDiffDiff,
            self.featureAbsMaxDiffDiff,
            self.featureAbsMinDiffDiff,
            self.featureLastMinusFirst,
            self.featureSum,
            self.featureAbsSum,
            self.featureMean,
            self.featureMeanAbs,
            self.featureMax,
            self.featureMaxAbs,
            self.featureMin,
            self.featureMinAbs,
            self.featureVar
        ]
        self.csv_file = csv_file
        self.newCutoffs = self.get_new_cutoffs(cutoffs)

    def matchCutoffs(self,indices, cutoffs):
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
        return newCutoffs

    def get_new_cutoffs(self,cutoffs): #cutoffs for this particular signal, as it doesn't have value at each time index
        #print 'new =================================================================='
        if self.df.empty:
            print "is Empty!!!!"
            return []
        all_indices = sorted(self.df.index.values)
        return self.matchCutoffs(all_indices,cutoffs)


    def filter_cutoff_index_and_translate(self,df):
        df.reset_index(inplace = True)
        filtered_df = df[df['date_index'].isin(self.newCutoffs)]
        new_df = pd.DataFrame(columns=('cutoff_index', 'value', 'feature_desc'))
        k = 0
        for i in range(len(self.newCutoffs)):
            if self.newCutoffs[i] != None:
                row_num = filtered_df[filtered_df['date_index'] == self.newCutoffs[i]].index[0]
                new_df.loc[k] = [i+1,filtered_df.at[row_num,'value'],filtered_df.at[row_num,'feature_desc']]
                k += 1
        return new_df

    def extract(self, create_new_features=True):
        #print "extracting"
        for f in self.functions:
            prepareDBAndCreateCSV(self.filter_cutoff_index_and_translate(f()), 'numeric', self.csv_file, create_new_features=create_new_features)


    def featureIdentity(self):
        return self.df.copy()
    def featureAbsMeanDiff(self):
        mean_diff = copt.expanding_abs_mean_diff(self.df['value'].values.astype(np.float64))

        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"absmeandiff"
        return diff_df

    def featureAbsMaxDiff(self):
        mean_diff = copt.expanding_abs_max_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"absmaxdiff"
        return diff_df

    def featureMaxDiff(self):
        mean_diff = copt.expanding_max_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"maxdiff"
        return diff_df

    def featureMinDiff(self):
        mean_diff = copt.expanding_min_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"mindiff"
        return diff_df

    def featureAbsMinDiff(self):
        mean_diff = copt.expanding_abs_min_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"absmindiff"
        return diff_df

    #could also do non abs versions of these in future
    def featureAbsMeanDiffDiff(self):
        mean_diff = copt.expanding_abs_mean_diff_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"absmeandiffdiff"
        return diff_df

    def featureAbsMaxDiffDiff(self):
        mean_diff = copt.expanding_abs_max_diff_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"absmaxdiffdiff"
        return diff_df

    def featureAbsMinDiffDiff(self):
        mean_diff = copt.expanding_abs_min_diff_diff(self.df['value'].values.astype(np.float64))
        diff_df = self.df.copy()
        diff_df['value'] = mean_diff
        diff_df['value'].fillna(0, inplace=True)
        diff_df['feature_desc'] = self.df['feature_desc']+"__"+"absmindiffdiff"
        return diff_df

    def featureLastMinusFirst(self):
        last_first_diff = self.df['value'] - self.df['value'].iloc[0]
        last_first_df = self.df.copy()
        last_first_df['value'] = last_first_diff
        last_first_df['value'].fillna(0, inplace=True)
        last_first_df['feature_desc'] = self.df['feature_desc']+"__"+"lastminusfirst"
        return last_first_df

    def featureSum(self):
        just_sum = copt.expanding_sum(self.df['value'].values.astype(np.float64), 0)
        sum_df = self.df.copy()
        sum_df['value'] = just_sum
        sum_df['value'].fillna(0, inplace=True)
        sum_df['feature_desc'] = self.df['feature_desc']+"__"+"sum"
        return sum_df

    def featureAbsSum(self):
        just_sum = copt.expanding_sum(self.df['value'].values.astype(np.float64), 1)
        sum_df = self.df.copy()
        sum_df['value'] = just_sum
        sum_df['value'].fillna(0, inplace=True)
        sum_df['feature_desc'] = self.df['feature_desc']+"__"+"sum_abs"
        return sum_df

    def featureMean(self):
        just_mean = copt.expanding_mean(self.df['value'].values.astype(np.float64),0)
        mean_df = self.df.copy()
        mean_df['value'] = just_mean
        mean_df['value'].fillna(0, inplace=True)
        mean_df['feature_desc'] = self.df['feature_desc']+"__"+"mean"
        return mean_df

    def featureMeanAbs(self):
        just_mean = copt.expanding_mean(self.df['value'].values.astype(np.float64),1)
        mean_df = self.df.copy()
        mean_df['value'] = just_mean
        mean_df['value'].fillna(0, inplace=True)
        mean_df['feature_desc'] = self.df['feature_desc']+"__"+"mean_abs"
        return mean_df

    def featureMax(self):
        just_max = copt.expanding_max(self.df['value'].values.astype(np.float64), 0)
        max_df = self.df.copy()
        max_df['value'] = just_max
        max_df['value'].fillna(0, inplace=True)
        max_df['feature_desc'] = self.df['feature_desc']+"__"+"max"
        return max_df

    def featureMin(self):
        just_min = copt.expanding_min(self.df['value'].values.astype(np.float64), 0)
        min_df = self.df.copy()
        min_df['value'] = just_min
        min_df['value'].fillna(0, inplace=True)
        min_df['feature_desc'] = self.df['feature_desc']+"__"+"min"
        return min_df

    def featureMaxAbs(self):
        just_max = copt.expanding_max(self.df['value'].values.astype(np.float64), 1)
        max_df = self.df.copy()
        max_df['value'] = just_max
        max_df['value'].fillna(0, inplace=True)
        max_df['feature_desc'] = self.df['feature_desc']+"__"+"max_abs"
        return max_df

    def featureMinAbs(self):
        just_min = copt.expanding_min(self.df['value'].values.astype(np.float64), 1)
        min_df = self.df.copy()
        min_df['value'] = just_min
        min_df['value'].fillna(0, inplace=True)
        min_df['feature_desc'] = self.df['feature_desc']+"__"+"min_abs"
        return min_df

    def featureVar(self):
        just_var = copt.expanding_var(self.df['value'].values.astype(np.float64))
        min_var = self.df.copy()
        min_var['value'] = just_var
        min_var['value'].fillna(0, inplace=True)
        min_var['feature_desc'] = self.df['feature_desc']+"__"+"var"
        return min_var
