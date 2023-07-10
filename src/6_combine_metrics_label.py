import os, sys, requests
import pandas as pd
sys.path.append("..")
pd.set_option('display.max_columns', None)
from config import config_global


if __name__ == '__main__':
    project = config_global.PROJECT
    project = 'lxc'
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)

    # read in metrics
    merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
    metric_on_group_df = pd.read_csv(os.path.normpath(merged_metric_on_group_path))
    print("group metric: ", metric_on_group_df.shape)
    for col_name in metric_on_group_df.columns:
        metric_on_group_df = metric_on_group_df.fillna({col_name: 0})

    # combine with label
    # label_path = os.path.join(config_global.LABEL_PATH, '%s_3label.csv' % project)

    label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), '%s_3label_20230118_%s.csv' % (project, config_global.threshold))
    label_df = pd.read_csv(os.path.normpath(label_path))
    print("label: ", label_df.shape, label_df.columns)

    dataset = pd.merge(metric_on_group_df, label_df, how='inner', on='clone_group_tuple')#.drop_duplicates()
    print(dataset.columns)
    print("dataset: ", dataset.shape)

    raw_dataset_path = os.path.join(config_global.DATASET_PATH, '%s_raw_dataset_20230118_%s.csv' % (project, config_global.threshold))
    print(raw_dataset_path)
    dataset.to_csv(raw_dataset_path, index=False)
    # dataset = dataset.drop(['clone_group_tuple'], axis=1)
