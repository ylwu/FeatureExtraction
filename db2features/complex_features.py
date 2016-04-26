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

def maxFeatureID(conn,table):
    query = '''
    select max(feature_id)
    from %s
    ''' % table
    maxF = sql.executeAndReturnData(conn,query)[0][0]
    if maxF != None:
        return maxF
    else:
        return -1

def prepareDBAndCreateCSV(conn,df,featureType,csv_file, create_new_features = True):
    df.to_csv(csv_file, header=False, index=False, line_terminator = '\r\n',mode='a')
    #print '----finished writing to csv', csv_file

def findFeatureIDFromDesc(conn, desc):
    findInSQL = '''
    SELECT feature_id
    FROM feature_types
    WHERE feature_desc = '%s'
    LIMIT 1
    ''' % desc
    try:
        fID = sql.executeAndReturnData(conn, findInSQL)[0][0]
    except:
        return None
    else:
        return fID


######Categorical Feature Functions############
class CategoricalFeature(object):
    def __init__(self, df,ohe,conn,csv_file,cutoffs, nonint = False, max_uniques=global_vars.MAX_CATEGORICAL_UNIQUE_VALUES_FOR_TP_FEATS):
        self.df = df
        self.ohe = ohe
        self.conn = conn
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
            prepareDBAndCreateCSV(self.conn,self.filter_cutoff_index_and_translate(feature()),
                                feature_type,
                                self.csv_file,
                                create_new_features=create_new_features)

        if len(self.getUnique()) <= self.max_uniques:
            #print 'EXTRACTING FEATURE PERCENT/TOTAL'
            percent_features,total_features = self.featureTotalAndPercentageEachValue()
            [prepareDBAndCreateCSV(self.conn,self.filter_cutoff_index_and_translate(feature), 'numeric', self.csv_file, create_new_features=create_new_features) for feature in percent_features+total_features]

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

    # def featureLongestCycle(self):
        # #largest number of values seen in a 'cycle' so far, cycle defined as the first
        # #time the sequence returns to a value previously seen
        # #e.g. [1,2,3,1,4,5,4,5,6,7,8,9,4]:
        # #cycles are 1-2-3-1, 4-5-4, 5-4-5, 4-5-6-7-8-9-4
        # #values of this feature would be [0,0,0,2, 2,2,2,2,2,2,2,2,5]
        # def coptWrapper(x):
            # if not self.ohe:
                # if self.nonint:
                    # fmapping, bmapping, numbers = self.mapStrToNumbers(x.values)
                    # values = list(copt.expanding_stability(numbers))
                    # for i,v in enumerate(values):
                        # values[i] = bmapping[values[i]]
                # else:
                    # values = copt.expanding_longest_cycle(x.values.astype(int))
            # else:
                # values = copt.expanding_longest_cycle_ohe(x.values)
            # return values

        # self.df['value'].fillna(0,inplace=True)
        # mode = self.df.groupby(level='project_id',sort=False)['value'].transform(coptWrapper)

        # mode_df = self.df.copy()
        # mode_df['value']=mode
        # mode_df['feature_desc'] = self.df['feature_desc']+"__"+"expanding_longest_cycle"
        # return mode_df

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


class Image(object):
    #TODO: also use categorical features
    def __init__(self, df,project_id,conn, csv_file_base,thresholds, projectCutoffs):
        self.df = df
        self.conn = conn
        self.csv_file_base = csv_file_base
        self.projectCutoffs = projectCutoffs
        self.project_id = project_id
        self.thresholds = thresholds
    def extract(self):
        image_data,stats = self.genGAFImage()
        field_ids = image_data['field_ids']
        self.appendImagesToCSV(image_data)
        return field_ids, stats
    def appendImagesToCSV(self, image_data):
        project_id = image_data['project_id']
        images = image_data['image']
        for t in self.thresholds:
            image = images[t]
            csv_file = self.csv_file_base + t + '.csv'
            row = [project_id]+list(image)
            with open(csv_file, 'ab') as c:
                writer = csv.writer(c)
                writer.writerow(row)

    def downSample(self,img, targetSize):
        im = toimage(img)
        imnew = im.resize((targetSize, targetSize), resample=pilImage.ANTIALIAS)
        ds = fromimage(imnew)
        return ds

    def upSample(self,img, targetSize):
        us = imresize(img, (targetSize,targetSize), interp='bicubic')
        return us
    def GAF(self, cos_phi, size):
        if len(cos_phi) == 0:
            return [np.zeros((size,size)),np.zeros((size,size))]
        cos_phi = cos_phi[~np.isnan(cos_phi)]
        n = len(cos_phi)
        maxElmt = cos_phi.max()
        minElmt = cos_phi.min()
        if maxElmt == minElmt:
            return [np.zeros((size,size)),np.zeros((size,size))]
        scaler = 1./(maxElmt-minElmt)
        cos_phi = (cos_phi*scaler) - (minElmt*scaler)
        sin_phi = np.sqrt(np.abs(1-np.square(cos_phi)))
        cos_phi = np.matrix(cos_phi)
        sin_phi = np.matrix(sin_phi)
        gasf = np.array(cos_phi.T*cos_phi - sin_phi.T*sin_phi)
        gadf = np.array(sin_phi.T*cos_phi - cos_phi.T*sin_phi)
        if n > size:
            gasf = self.downSample(gasf, size)
            gadf = self.downSample(gadf, size)
        elif n < size:
            gasf = self.upSample(gasf, size)
            gadf = self.upSample(gadf, size)
        return gasf,gadf

    def genGAFImage(self):
        lengths = copy.deepcopy(self.thresholds)
        for t in lengths:
            lengths[t] = []

        thresholds = self.thresholds.keys()
        image_data = {
                'project_id': self.project_id,
                'image': {},
                'field_ids': {}
                }
        for i,threshold in enumerate(thresholds):
            cutoff = self.projectCutoffs[i]
            size = self.thresholds[threshold]
            image_data['field_ids'][threshold] = []

            grouped = self.df.groupby(level='field_id',sort=False)
            #2*(len(grouped) is number of channels: gasf, gadf for each field_id
            image = np.zeros((2*len(grouped), size,size))
            print "threshold = ",threshold, "; size = ",image.shape
            for j,groupObject in enumerate(grouped):
                fieldId, group = groupObject
                dates = group.index.get_level_values('date')
                cutoffIndex  = len(dates[dates < cutoff])

                lengths[threshold].append(cutoffIndex)

                gasf,gadf = self.GAF(group['value'][:cutoffIndex], size)
                image[j,:,:] = gasf
                image[len(grouped)+j,:,:] = gadf
                image_data['field_ids'][threshold].append(fieldId)

            image_data['image'][threshold] = image.ravel()

        stats = {}
        keys = lengths.keys()
        for key in keys:
            stats[key] = {'mean': np.mean(lengths[key]),
                          'median': np.median(lengths[key]),
                          'max': np.max(lengths[key]),
                          'std': np.std(lengths[key])}
        return image_data, stats


    def featureMTF(self):
        #TODO: incorporate as a channel into image
        Q = 4
        def MTF(x):
            scaler = 1./(x.max()-x.min())
            x *= scaler
            x -= x.min()*scaler
            q = pd.qcut(np.unique(x),Q)
            dic = dict(zip(np.unique(x), q.labels))
            Mkv= np.zeros([Q,Q])
            labels = [dic[y] for y in x]
            for i,l in enumerate(labels[:-1]):
                Mkv[l, labels[i+1]] += 1
            Mkv = np.nan_to_num(np.divide(
                            Mkv, np.sum(
                                Mkv, axis=1).reshape(
                                    (Mkv.shape[0],1))))
            n = len(x)
            MTF = [Mkv[labels[p],labels[q]] for p in xrange(n) for q in xrange(n)]
            return np.array(MTF).reshape((n,n))

class NumericalFeature(object):
    def __init__(self, df,conn, csv_file,cutoffs,timeSeriesChunks=3):
        self.df = df
        self.conn = conn
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
            prepareDBAndCreateCSV(self.conn,self.filter_cutoff_index_and_translate(f()), 'numeric', self.csv_file, create_new_features=create_new_features)


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

class TopicFeature(object):
    #change to only do corpus over comments already seen?
    def __init__(self, df,conn, topicModel, csv_file):
        print os.getpid(), "starting init"
        self.df = df
        self.conn = conn
        self.topicModel = topicModel
        self.csv_file = csv_file

    def extract(self, create_new_features=True):
        print os.getpid(), "extracting"
        topicDistributionFeatures = self.featuresTopicDistribution()
        [prepareDBAndCreateCSV(self.conn,f, 'numeric',self.csv_file, create_new_features=create_new_features) for f in topicDistributionFeatures]
    def featuresTopicDistribution(self):
        print os.getpid(), "getting topic dist"
        featureList = []
        featuresDF = self.df.groupby(level='project_id',sort=False).apply(lambda x: self.expandingApplyTopicDist(x))
        print "starting rearrange cols"
        for col in featuresDF.columns:
            featureDF = featuresDF.loc[:,[col]]
            feature_desc = featureDF.columns[0]
            featureDF.columns = ['value']
            featureDF['feature_desc'] = feature_desc
            featureList.append(featureDF)
        print os.getpid(), "finished getting topic dist"
        return featureList
    def expandingApplyTopicDist(self,frame):
        def func(x):
            dist = self.topicModel.get_topic_distribution(x['value'].values)
            return dist
        values = np.array([func(frame.iloc[0:i+1]) for i in xrange(len(frame))])
        feature_desc = frame['feature_desc'].iloc[0]
        frame.drop(['value','feature_desc'],axis=1,inplace=True)
        for column in xrange(len(values[0])):
            frame[feature_desc+"topic_"+str(column)] = values[:,column]
        #values is a matrix like:
        #[[date1topic1dist date1topic2dist ... date1topicNdist]
        #...
        # [dateNtopic1dist dateNtopic2dist ... dateNtopicNdist]]
        return frame

class VectorFeature(object):
    #train word2vec on corpus over all comments (or all already seen)
    #cluster with MiniBatchKMeans
    #choose features to be number of words per cluster
    #so a single comment variable will yield N features, where N is # of clusters
    #, with integer values for each feature
    def __init__(self, df,conn, clusterDict, csv_file, ngrams=False):
        print os.getpid(), "starting init"
        self.df = df
        self.conn = conn
        #clusterDict is keyed by the words and valued by cluster id
        self.clusterDict = clusterDict
        self.numClusters = np.max(clusterDict.values())
        self.ngrams = ngrams
        self.csv_file = csv_file

    def extract(self, create_new_features=True):
        print os.getpid(), "extracting"
        vectorClusterFeatures = self.clusterFeatures()
        [prepareDBAndCreateCSV(self.conn,f, 'numeric',self.csv_file, create_new_features=create_new_features) for f in vectorClusterFeatures]
    def clusterFeatures(self):
        print os.getpid(), "getting topic dist"
        featureList = []
        featuresDF = self.df.groupby(level='project_id',sort=False).apply(lambda x: self.expandingApplyCluster(x))
        print "starting rearrange cols"
        for col in featuresDF.columns:
            featureDF = featuresDF.loc[:,[col]]
            feature_desc = featureDF.columns[0]
            featureDF.columns = ['value']
            featureDF['feature_desc'] = feature_desc
            featureList.append(featureDF)
        print os.getpid(), "finished getting topic dist"
        return featureList
    def expandingApplyCluster(self,frame):
        def func(x):
            featureSeries = []
            words = vm.splitterFunc(self.ngrams)(''.join(list(x['value'].values)))
            clusterBinCount = np.bincount([self.clusterDict[w] for w in words if w in self.clusterDict], minlength = 1+self.numClusters)
            return clusterBinCount
        values = np.array([func(frame.iloc[0:i+1]) for i in xrange(len(frame))])
        feature_desc = frame['feature_desc'].iloc[0]
        frame.drop(['value','feature_desc'],axis=1,inplace=True)
        for column in xrange(len(values[0])):
            frame[feature_desc+"_clustercount_"+str(column)] = values[:,column]
        #values is a matrix like:
        #[[date1cluster1count date1cluster2count ... date1clusterNcount]
        #...
        # [dateNcluster1count dateNcluster2count ... dateNclusterNcount]]
        return frame

def buildTopicModel(dfList, numTopics = 20, numPasses = 10):
    pf = utils.picklePath("topic_model_topics_%s_passes_%s.pickle" %\
                                       (numTopics, numPasses))
    utils.checkPickleFileExistsAndCreate(pf)
    topicModel = utils.loadObjectsFromPickleFile(['topicModel'],pf)[0]
    if not topicModel:
        topicModel = tm.TopicModel(numTopics, numPasses)
        docList = []
        for df in dfList:
            docList.extend(list(df['value'].values))
        topicModel.build(docList)
        utils.saveObjectsToPickleFile({'topicModel':topicModel},pf)
    return topicModel

class DocIterator(object):
    def __init__(self, dfList):
        self.dfList = dfList
    def __iter__(self):
        for df in self.dfList:
            for comment in list(df['value'].values):
                yield comment

def buildVectorModel(dfList,clusters = 100, ngrams = False):
    pf = utils.picklePath("vector_model_clust_%s_ngrams_%s.pickle" % (clusters, ngrams))
    utils.checkPickleFileExistsAndCreate(pf)
    vectorModel,clusterDict = utils.loadObjectsFromPickleFile(['vectorModel', 'clusterDict'],pf)
    if not vectorModel:
        docList = DocIterator(dfList)
        vectorModel = vm.buildVectorModel(docList,ngram=ngrams)
        utils.saveObjectsToPickleFile({'vectorModel':vectorModel},pf)

    if not clusterDict:
        word2vec_dict = {}
        for i in vectorModel.vocab.keys():
            try:
                word2vec_dict[i]=vectorModel[i]
            except:
                pass
        clusters = MiniBatchKMeans(n_clusters=clusters, max_iter = 10,
                                    batch_size=200,n_init=3,init_size=2000)
        X = np.array([i.T for i in word2vec_dict.itervalues()])
        y = [i for i in word2vec_dict.iterkeys()]
        clusters.fit(X)
        from collections import defaultdict
        clusterDict=defaultdict(list)
        for word,label in zip(y,clusters.labels_):
            clusterDict[word].append(label)
        utils.saveObjectsToPickleFile({'clusterDict':clusterDict},pf)
    return vectorModel, clusterDict

def timeSeriesStatistics():
    conn = sql.openSQLConnectionP(global_vars.DATABASE_NEW, global_vars.USERNAME, global_vars.PASSWORD, global_vars.HOST, global_vars.PORT)
    query = '''
    select distinct(project_id) from generic_time_varying_features limit 50
    '''
    pIds = sql.executeAndReturnData(conn, query)
    pIDList = [x[0] for x in pIds]
    avgNumberDates = '''
    select count(date) as s
    from generic_time_varying_features
    where project_id in (%s)
    group by project_id, feature_id
    ''' % (str(pIDList)[1:-1])
    counts = sql.executeAndReturnData(conn, avgNumberDates)
    counts = np.array([x[0] for x in counts])
    print np.median(counts)
    print np.std(counts)
    conn.close()
if __name__ == '__main__':
    global_vars.init()
    topicModel = tm.TopicModel(10, 10)
    pf = utils.picklePath('test_topic.p')
    utils.checkPickleFileExistsAndCreate(pf)
    utils.saveObjectsToPickleFile({'topic': topicModel},pf)
