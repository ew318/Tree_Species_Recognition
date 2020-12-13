import pickle
import os
import pandas as pd


def collate_results(infolder):
    results = []
    for subdir, dirs, files in os.walk(infolder):
        results = files
        break
    evaluation_files = [x for x in results if 'evaluation' in x]
    data_dict = {'model_type': [],
                 'loss': [],
                 'accuracy': [],
                 'epochs': []
                 }
    for file in evaluation_files:
        evaluation = pickle.load(open(infolder + file, "rb"))
        data_dict['model_type'].append(file)
        data_dict['loss'].append(evaluation[0])
        data_dict['accuracy'].append(evaluation[1])
        history_file = file.replace('evaluation', 'history')
        history = pickle.load(open(infolder + history_file, "rb"))
        data_dict['epochs'].append(len(history['loss']) - 10)
    summary_df = pd.DataFrame(data_dict, columns=['model_type', 'loss', 'accuracy', 'epochs'])
    return summary_df


collate_results('/scratch/emiwat01/projects/final/Tree_Species_Recognition/mymodels/')


def collate_kfold_results(infolder='kfold/'):
    results = []
    for subdir, dirs, files in os.walk(infolder):
        results = files
        break
    data_dict = {'model_type': [],
                 'accuracy': [],
                 'recall': [],
                 'precision': []
                 }
    for file in results:
        evaluation = pickle.load(open(infolder + file, "rb"))
        data_dict['model_type'].extend([file] * 10)
        data_dict['accuracy'] += evaluation['accuracy']
        data_dict['recall'] += evaluation['avg_recall']
        data_dict['precision'] += evaluation['avg_precision']
    summary_df = pd.DataFrame(data_dict, columns=['model_type', 'accuracy', 'recall', 'precision'])
    return summary_df
