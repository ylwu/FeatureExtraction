import extract_features as exF
import complex_features as cf
import sys
import json

def extractAllFeatures(config_file = None, insertIntoDB=False, deleteCSVSOnFinish=False):
    csv_output_files = []

    if extractCat:
        cat_files = exF.extractGenericCategoricalFeatures(images=extractAsImages)
        csv_output_files.extend(cat_files)
        print 'extract cat done'
    if extractNum:
        num_files = exF.extractGenericNumericalFeatures(images=extractAsImages)
        csv_output_files.extend(num_files)
        print 'extract num done'

if __name__ == '__main__':
    insertIntoDB = True
    deleteCSVSOnFinish = False
    extractAllFeatures(config_file = config_file, insertIntoDB=insertIntoDB, deleteCSVSOnFinish=deleteCSVSOnFinish)

