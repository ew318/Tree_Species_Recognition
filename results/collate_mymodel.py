import pickle
import os
import pandas as pd


def collate_results(infolder, outfile):
    results = []
    for subdir, dirs, files in os.walk(infolder):
        results = files
        break

    evaluation_files = [x for x in results if 'evaluation' in x]

    data_dict = {'model_type': [],
                 'loss': [],
                 'accuracy': []
                 }

    for file in evaluation_files:
        evaluation = pickle.load(open(infolder + file, "rb"))
        data_dict['model_type'].append(file)
        data_dict['loss'].append(evaluation[0])
        data_dict['accuracy'].append(evaluation[1])

    summary_df = pd.DataFrame(data_dict, columns=['model_type', 'loss', 'accuracy'])
    summary_df.to_csv(outfile, index=False)


collate_results('/scratch/emiwat01/projects/camden/mymodel_gen/', 'model_comparison_summary.csv')
