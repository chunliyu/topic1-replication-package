import sys, os
sys.path.append("..")
from config import config_global
from pydriller import Git
import pandas as pd
import math, pickle
pd.set_option("Display.max_columns", None)
pd.set_option("Display.max_colwidth", None)


def anatomize_genealogy(genealogy_df, commit_time_dict, commit_author_dict):
    genealogy_df['n_siblings_start'] = genealogy_df['clone_group_tuple'].map(lambda x: len(x.split('|')))

    genealogy_df['genealogy_list'] = genealogy_df['genealogy'].map(lambda gen: gen.split('|'))
    genealogy_df['commit_list'] = genealogy_df['genealogy_list'].map(lambda gen: [x.split(":")[0] for x in gen])
    genealogy_df['churn'] = genealogy_df['genealogy_list'].map(lambda gen: sum([int(x.split(":")[1]) for x in gen]))
    #print("================", genealogy_df['churn'])
    genealogy_df.drop(['genealogy'], axis=1, inplace=True)
    genealogy_df['n_genealogy'] = genealogy_df['commit_list'].map(lambda x: len(x))

    # each commit in genealogy: commit_id: #updates: clone_group_sig
    genealogy_df['end_commit'] = genealogy_df['commit_list'].map(lambda x: x[-1])
    #genealogy_df['end_clone_group_tuple'] = genealogy_df['genealogy_list'].map(lambda x: x[-1].split(':', 2)[2])
    #genealogy_df['n_siblings_end'] = genealogy_df['end_clone_group_tuple'].map(lambda x: len(x.split(',')))


    # calculate duration in terms of days
    genealogy_df['n_days'] = list(map(lambda x, y: (commit_time_dict[y] - commit_time_dict[x]).days,
                                      genealogy_df['start_commit'], genealogy_df['end_commit']
                                      )
                                  )  # git_repo.get_commit(commit_id).committer.email for commit_id in gen.split('-')
    genealogy_df['start_timestamp'] = genealogy_df['start_commit'].map(lambda x: commit_time_dict[x])  # get timestamp for time-series training
    print("----------------------------------------------------------------")
    #print("start time xxxxxxxxxxxxxxxxxxxxxxxxxxx: ", genealogy_df.start_timestamp)
    # calucate #authors

    # genealogy_df['genealogy_list'].to_csv("bad.txt")
    genealogy_df['author_list'] = genealogy_df['genealogy_list'].map(
            lambda gen: set([commit_author_dict[commit.split(':', 2)[0]] for commit in gen])
        )

    genealogy_df['n_authors'] = genealogy_df['author_list'].map(lambda x: len(x))
    genealogy_df['cnt_siblings'] = genealogy_df['genealogy_list'].map(lambda gen: max([int(x.split(":")[2]) for x in gen]))
    print(list(genealogy_df['cnt_siblings']))
    # reorder the columns
    #print(genealogy_df.columns)
    print("before drop duplicates: ", genealogy_df.shape)

    '''
    # calucate #updators
    genealogy_df['updator_list'] = genealogy_df['genealogy_list'].map(
        lambda gen: set([commit_author_dict[commit.split(':', 2)[0]] for commit in
                         (filter(lambda c: c.split(':', 2)[1] != '0', gen))
                         ])
    )
    genealogy_df['n_updators'] = genealogy_df['updator_list'].map(lambda x: len(x))
    genealogy_df['n_genealogy_updated'] = genealogy_df['genealogy_list'].map(
        lambda gen: len(list(filter(lambda c: c.split(':', 2)[1] != '0', gen)))
    )
    '''
    # drop duplicates
    genealogy_df['clone_group_tuple'].drop_duplicates(inplace=True)
    print("after drop duplicates: ", genealogy_df.shape)
    print("==================================")
    return genealogy_df


# rank by longevity: if longevous then 1 else 0
def rank_by_lifecycle(genealogy_df, is_by_genealogy, threshold=0.5): # if is_by_genealogy, rank by length of commit genealogy, otherwise by number of days
    '''
        genealogy_df['rank_by_n_genealogy'] = genealogy_df['n_genealogy'].map(
                lambda x: 1 if x > genealogy_df.n_genealogy.quantile(0.75) else (
                    0 if x < genealogy_df.n_genealogy.quantile(0.25) else -1))
    '''

    genealogy_df['rank_by_n_genealogy'] = genealogy_df['n_genealogy'].map(
        lambda x: 1 if x > genealogy_df.n_genealogy.quantile(threshold) else 0)

    genealogy_df['rank_by_n_days'] = genealogy_df['n_days'].map(
        lambda x: 1 if x > genealogy_df['n_days'].quantile(threshold) else 0)

    if is_by_genealogy:
        genealogy_df['rank_by_lifecycle'] = genealogy_df['rank_by_n_genealogy']
    else:
        genealogy_df['rank_by_lifecycle'] = genealogy_df['rank_by_n_days']


def rank_by_prevalence(genealogy_df, threshold=0.5):
    #genealogy_df['rank_by_prevalence'] = genealogy_df['n_siblings_start'].map(lambda x: 1 if x > genealogy_df.n_siblings_start.quantile(0.5) else 0)

    genealogy_df['rank_by_prevalence'] = genealogy_df['cnt_siblings'].map(
        lambda x: 1 if x > genealogy_df.cnt_siblings.quantile(threshold) else 0)
    print(list(genealogy_df['rank_by_prevalence']))
    print("rank_by_prevalence: ", genealogy_df['rank_by_prevalence'].value_counts())
    print("----------------")
    #genealogy_df['rank_by_n_authors'] = genealogy_df['rank_by_n_authors'].map({1: 'high', 0: 'low'})  # 'volvo':0 , 'bmw':1, 'audi':2} )


# to retrieve the bug-proneness info: # bug-inducing commits / # normal commits
def rank_by_bugproneness(genealogy_df, buggy_commit_list, threshold=0.5):
    # look into the distribution of n_genealogy
    segments = pd.cut(genealogy_df['n_genealogy'], bins=[0,2,5,100,1000])
    counts = pd.value_counts(segments, sort=True)
    #print(genealogy_df['n_genealogy'].value_counts())
    print(counts)

    buggy_commit_list = list(set(buggy_commit_list))
    print("genealogy columns: ", genealogy_df.columns)

    genealogy_df['buggy_genealogy'] = genealogy_df['commit_list'].map(lambda gen: list(set(gen).intersection(set(buggy_commit_list))))
    genealogy_df['n_buggy_genealogy'] = genealogy_df['buggy_genealogy'].map(lambda gen: len(gen))

    genealogy_df['bug_proneness'] = genealogy_df.apply(lambda row:  (row['n_genealogy'] - row['n_buggy_genealogy']) / (row['n_buggy_genealogy'] if row['n_buggy_genealogy'] != 0 else 1), axis=1)
    genealogy_df['rank_by_bugproneness'] = genealogy_df['bug_proneness'].map(
        lambda x: 1 if x > genealogy_df.bug_proneness.quantile(threshold) else 0)

    # look into the distribution of bug_proneness
    segments = pd.cut(genealogy_df['bug_proneness'], bins=[0, 2, 5, 10, 100, 10000])
    #print(genealogy_df['rank_by_bugproneness'].value_counts())
    #counts = pd.value_counts(segments, sort=True)
    #print(counts)


def decide_final_label(genealogy_df, is_longevous, is_prevalent, is_buggy):
    if is_longevous and is_prevalent and is_buggy: # decide the label by all the three dimensions
        genealogy_df['is_reusable'] = genealogy_df.apply(
            lambda row: math.floor((row.rank_by_lifecycle + row.rank_by_prevalence + row.rank_by_bugproneness) / 3),
            axis=1
        ).astype(int)
    elif is_longevous and is_buggy:
        genealogy_df['is_reusable'] = genealogy_df.apply(
            lambda row: math.floor((row.rank_by_lifecycle + row.rank_by_bugproneness) / 2),
            axis=1
        ).astype(int)
    elif is_longevous:
        genealogy_df['is_reusable'] = genealogy_df['rank_by_lifecycle']

    #genealogy_anatomized_df = genealogy_anatomized_df[genealogy_anatomized_df.is_reusable != -1]
    #genealogy_df['rank1'] = (
        #genealogy_df.apply(lambda row: math.floor(row.rank_by_n_genealogy + row.rank_by_n_authors), axis=1)).astype(int)


# map to human readable
def map_label(col):
    genealogy_df[col] = genealogy_df[col].map({1: 'high', 0: 'low'})


if __name__ == "__main__":
    project = config_global.PROJECT
    project = 'redis'
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)

    is_by_genealogy = True
    is_longevous = True
    is_prevalent = True
    is_buggy = True

    # loading genealogy file
    genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH, '%s_group_genealogy_distinct.csv' % (project))
    genealogy_df = pd.read_csv(genealogy_path)

    # generating commit_author_dict and commit_timestamp dict
    commits_log_clean_path = os.path.join(config_global.COMMIT_LOG_CLEAN_PATH, '%s_logs.txt' % project)
    #commits_log_df = pd.read_csv(commits_log_clean_path, header=None, names=['commit_id', 'committer', 'timestamp'], encoding= 'unicode_escape')
    commits_log_df = pd.read_csv(commits_log_clean_path, header=None, names=['commit_id', 'committer', 'timestamp'], encoding= 'unicode_escape')

    commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['timestamp'], infer_datetime_format=True, errors='coerce')

    #commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['timestamp'], infer_datetime_format=True)
    commit_author_dict = dict(zip(commits_log_df.commit_id, commits_log_df.committer))
    commit_time_dict = dict(zip(commits_log_df.commit_id, commits_log_df.timestamp))

    genealogy_anatomized_df = anatomize_genealogy(genealogy_df, commit_time_dict, commit_author_dict)

    # rank by 1rd dimension - Clone LifeCycle
    rank_by_lifecycle(genealogy_anatomized_df, is_by_genealogy, config_global.threshold)

    # rank by 1rd dimension - Clone Prevalence
    rank_by_prevalence(genealogy_anatomized_df, config_global.threshold)

    # rank by 3rd dimension - Bug Proneness
    buggy_commit_list_path = os.path.join(config_global.LABEL_PATH, "buggy_commits", "%s_buggy_commits.pkl" % project)
    with open(buggy_commit_list_path, 'rb') as fp:
        buggy_commit_list = pickle.load(fp)
    #print("length of buggy_commit_list: ", len(buggy_commit_list))

    rank_by_bugproneness(genealogy_anatomized_df, buggy_commit_list, config_global.threshold)

    # combine all the three dimensions
    decide_final_label(genealogy_anatomized_df, is_longevous, is_prevalent, is_buggy)

    # save labels info to file
    # label_df = genealogy_anatomized_df[['clone_group_tuple', 'start_timestamp', 'start_commit', 'churn', 'is_reusable']]
    print(genealogy_anatomized_df.columns)

    label_df = genealogy_anatomized_df[['clone_group_tuple', 'start_commit', 'n_siblings_start',
       'churn', 'n_genealogy', 'end_commit', 'n_days', 'start_timestamp', 'n_authors', 'cnt_siblings',
       'rank_by_n_genealogy', 'rank_by_n_days', 'rank_by_lifecycle',
       'rank_by_prevalence', 'n_buggy_genealogy',
       'bug_proneness', 'rank_by_bugproneness', 'is_reusable']]
    label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), '%s_3label_20230118_%s.csv' % (project, config_global.threshold))
    label_df.to_csv(label_path, index=False)

    # EDA
    print(genealogy_anatomized_df['is_reusable'].value_counts())
    #genealogy_df.drop(genealogy_df[genealogy_df['rank1'] == 1].index, inplace=True)






