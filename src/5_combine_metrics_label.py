import os, sys, requests, itertools
import pandas as pd
from tqdm import tqdm
from glob import glob
sys.path.append(".")
sys.path.append("..")
pd.set_option('display.max_columns', None)
from config import config_global


def run_projects_thresholds():
    files = glob("/home/20cy3/topic1/clone2api/data/group_metrics/*_group_metric_merged.csv")
    projects = set([os.path.basename(file).split("_")[0] for file in files])
    projects = ['spock', 'PocketHub']
    print(len(projects))
    for project in tqdm(projects, desc='projects running...'):
        values = [0.3, 0.4, 0.5, 0.6, 0.7]
        combinations = list(itertools.product(values, repeat=3))
        combinations = [(0.3, 0.3, 0.3), (0.4, 0.4, 0.4), (0.5, 0.5, 0.5), (0.6, 0.6, 0.6), (0.7, 0.7, 0.7)]
        for combo in combinations:
            print(project, combo)
            lifecycle_threshold, prevalence_threshold, quality_threshold = combo[0], combo[1], combo[2]
            raw_dataset_path = os.path.join(config_global.DATASET_PATH, f'20230912_{project}_raw_dataset_{lifecycle_threshold}_{prevalence_threshold}_{quality_threshold}.csv')
            if os.path.exists(raw_dataset_path):
                continue
            
            # read in metrics
            merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
            metric_on_group_df = pd.read_csv(os.path.normpath(merged_metric_on_group_path))
            print("group metric: ", metric_on_group_df.shape)
            for col_name in metric_on_group_df.columns:
                metric_on_group_df = metric_on_group_df.fillna({col_name: 0})
            
            # combine with label
            # label_path = os.path.join(config_global.LABEL_PATH, '%s_3label.csv' % project)
        
            # label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), f'{project}_3label_20230808_{config_global.threshold}.csv')
            label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), f'20230912_{project}_3label_{lifecycle_threshold}_{prevalence_threshold}_{quality_threshold}.csv')
        
            label_df = pd.read_csv(os.path.normpath(label_path))
            # print("label: ", label_df.shape, label_df.columns)
        
            dataset = pd.merge(metric_on_group_df, label_df, how='inner', on='clone_group_tuple')#.drop_duplicates()
            # print(dataset.columns)
            print("dataset: ", dataset.shape)
        
            raw_dataset_path = os.path.join(config_global.DATASET_PATH, f'20230912_{project}_raw_dataset_{lifecycle_threshold}_{prevalence_threshold}_{quality_threshold}.csv')
            print(raw_dataset_path)
            dataset.to_csv(raw_dataset_path, index=False)


def main():
    project = 'glide'
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)
    
    files = glob("/home/20cy3/topic1/clone2api/data/group_metrics/*_group_metric_merged.csv")
    projects = [os.path.basename(file).split("_")[1] for file in files]
    projects = list(config_global.SUBJECT_SYSTEMS_YOUNG.keys()) + list(config_global.SUBJECT_SYSTEMS_MIDDLE.keys()) + list(config_global.SUBJECT_SYSTEMS_OLD.keys())
    # projects = ['jabref']
    projects = ['spock', 'PocketHub']
    print(len(projects))
    for project in projects:
        # read in metrics
        merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
        metric_on_group_df = pd.read_csv(os.path.normpath(merged_metric_on_group_path))
        print("group metric: ", metric_on_group_df.shape)
        for col_name in metric_on_group_df.columns:
            metric_on_group_df = metric_on_group_df.fillna({col_name: 0})
    
        # combine with label
        # label_path = os.path.join(config_global.LABEL_PATH, '%s_3label.csv' % project)
    
        # label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), f'{project}_3label_20230808_{config_global.threshold}.csv')
        label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), f'20230912_{project}_3label_{config_global.lifecycle_threshold}_{config_global.prevalence_threshold}_{config_global.quality_threshold}.csv')
    
        label_df = pd.read_csv(os.path.normpath(label_path))
        # print("label: ", label_df.shape, label_df.columns)
    
        dataset = pd.merge(metric_on_group_df, label_df, how='inner', on='clone_group_tuple')#.drop_duplicates()
        # print(dataset.columns)
        print("dataset: ", dataset.shape)
    
        raw_dataset_path = os.path.join(config_global.DATASET_PATH, f'20230912_{project}_raw_dataset_{config_global.lifecycle_threshold}_{config_global.prevalence_threshold}_{config_global.quality_threshold}.csv')
        print(raw_dataset_path)
        dataset.to_csv(raw_dataset_path, index=False)
        # dataset = dataset.drop(['clone_group_tuple'], axis=1)
    

if __name__ == '__main__':
    # main()
    run_projects_thresholds()