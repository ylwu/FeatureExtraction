import extract_features as exF
import complex_features as cf
import sys
import json

def extractAllFeatures(config_file = None, insertIntoDB=False, deleteCSVSOnFinish=False):
    cat_files = exF.extractGenericCategoricalFeatures(images=False)
    print 'extract cat done'
    num_files = exF.extractGenericNumericalFeatures(images=False)
    print 'extract num done'

