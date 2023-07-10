from __future__ import division
import json, os, subprocess, re, csv, sys, gc, whatthepatch, logging, pickle, pickle
import networkx as nx
from collections import OrderedDict
from tqdm import tqdm
import networkx as nx
from tqdm import tqdm
sys.path.append("..")
from config import config_global

# Run shell command from a string
import pandas as pd

# Run shell command from a string
from pydriller import Git
import getopt
from git import Repo, GitCommandError


# map line number from old commit to new commit
def get_mapped_clone(clone_start_old, clone_end_old, line_mapping):
    clone_start_new = clone_start_old
    clone_end_new = clone_end_old

    churn = 0
    begin_to_count = False

    for line_pair in line_mapping:
        old_line = line_pair[0]
        new_line = line_pair[1]

        if old_line is None:
            old_line = 0

        if old_line > clone_end_old:
            return clone_start_new, clone_end_new, churn

        if old_line and new_line:
            # find the new start line
            if old_line <= clone_start_old:
                clone_start_new = (new_line - old_line) + clone_start_old
                clone_end_new = (new_line - old_line) + clone_end_old

                # calculate the new end line
            elif old_line <= clone_end_old:
                begin_to_count = True
                clone_end_new = (new_line - old_line) + clone_end_old
        else:
            if old_line >= clone_start_old:
                begin_to_count = True

            # if last line deleted in the clone boundary
            if begin_to_count:
                if new_line is None:
                    clone_end_new -= 1
                churn += 1
    return clone_start_new, clone_end_new, churn


# 有问题: group_tuple实际存储的是list,需要看下class_clone_dict, 或者调整下,添加一个change list to tuple函数
# 通过method的startline和endline找到method, 然后找到method的new startline和endline
def get_mapped_group(clone_group_tuple, commit_modified_files):
    group_modified_list = list()
    churn_all = 0
    #breakpoint()
    #modified_files = filter(lambda x: x.old_path and x.old_path.endswith('.java'), commit_modified_files)
    for clone in clone_group_tuple:
        clone_path = clone.split(":")[0]
        clone_range = clone.split(":")[1]
        clone_start = int(clone_range.split('-')[0])
        clone_end = int(clone_range.split('-')[1])
        churn = 0

        for modified_file in commit_modified_files:
            # check if the changed file is related to the clones in the clone group
            if clone_path == modified_file[0]:  # old path有调整
                # 获取new_path
                if modified_file[1] is None: # new path == None, file being deleted, 当前clone已经不存在
                    clone_end = -1
                    churn = (clone_end - clone_start) + 1
                else: # 只是单纯的修改
                    # 可以优化为modified_file.diff_parsed
                    clone_path = modified_file[1]  # new path

                    for diff1 in whatthepatch.parse_patch(modified_file[2]):  # only one element in the generator
                        line_mapping = diff1[1]
                        clone_start, clone_end, churn = get_mapped_clone(clone_start, clone_end, line_mapping)
                    #line_mapping = next(whatthepatch.parse_patch(modified_file[2]))[1] # only one element in the generator
                    #clone_start, clone_end, churn = get_mapped_clone(clone_start, clone_end, line_mapping)

                break  # 继续查看下一个clone, 该clone消失

        # clone_start, clone_end 如果变化，记录变化后的
        if clone_start <= clone_end:
            group_modified_list.append("%s:%d-%d" % (clone_path, clone_start, clone_end))
            churn_all += churn
    return tuple(group_modified_list), churn_all


def calculate_churn_added(clone_group):
    churn = 0
    for clone in clone_group:
        churn += abs(eval(clone.split(":")[1]))
    return churn


# apply dfs searching for the same group
def retrieve_clone_group_genealogy(clone_group_tuple, commit_list_sliced, clone_class_dict, commit_modification_dict):
    clone_group_genealogy_list = list()
    for commit_id in commit_list_sliced:  # consider the start commit_id
        churn_all = 0

        if commit_modification_dict[commit_id]:
            clone_group_tuple, churn_all = get_mapped_group(clone_group_tuple, commit_modification_dict[commit_id])

        for group in clone_class_dict[commit_id]:
            #breakpoint()
            if set(group).intersection(set(clone_group_tuple)):  # is_clone_group_matched(group, clone_group_tuple):
                clone_class_dict[commit_id].remove(group)
                #breakpoint()
                churn_all += calculate_churn_added(set(group) - set(clone_group_tuple))
                cnt_siblings = len(set(group))
                #clone_group_genealogy_list.append("%s:%d:%s" % (commit_id, churn_all, group))
                clone_group_genealogy_list.append("%s:%d:%d" % (commit_id, churn_all, cnt_siblings))
                clone_group_tuple = group
                break  # stop when matched

    return clone_group_genealogy_list


def get_all_commit_modifications(commit_list_sliced, git_repo):
    commit_modification_dict = dict()
    for commit_id in commit_list_sliced:

        #if '8f1f229' in commit_id:
            #breakpoint()
        commit_modification_dict[commit_id] = list()
        try:
            commit_modified_files = git_repo.get_commit(commit_id).modified_files
            commit_modified_files_java = list(filter(lambda x: (x.old_path and x.old_path.endswith('.java')) or
                                                           (x.new_path and x.new_path.endswith('.java')),
                                                 commit_modified_files))
            for modified_java_file in commit_modified_files_java:
                # 需要检查diff的类型
                commit_modification_dict[commit_id].append([modified_java_file.old_path, modified_java_file.new_path, modified_java_file.diff])

        except (ValueError, GitCommandError) as e:
            print("error commit_id: ", commit_id)
            print(f"An error occurred: {e}")
    return commit_modification_dict


# some clone groups detected are acturally the same clone group
def merge_groups(clone_groups):
    #L = [['a', 'b', 'c'], ['b', 'd', 'e'], ['k'], ['o', 'p'], ['e', 'f'], ['p', 'a'], ['d', 'g']]

    graph = nx.Graph()
    # Add nodes to Graph
    graph.add_nodes_from(sum(clone_groups, []))
    # Create edges from list of nodes
    q = [[(group[i], group[i + 1]) for i in range(len(group) - 1)] for group in clone_groups]

    for i in q:
        # Add edges to Graph
        graph.add_edges_from(i)

    # Find all connnected components in graph and list nodes for each component
    return [list(i) for i in nx.connected_components(graph)]


def filter_merge_tuplize(clone_class_dict):
    clone_class_dict_tuplized = dict()

    for commit_id in clone_class_dict:
        commit_groups = list()

        #filter out test functions
        for clone_group in clone_class_dict[commit_id]:
            clone_group_list = list()
            for clone in clone_group:
                clone[0] = os.path.normpath(clone[0])
                if clone[0].lower().find('test') == -1: # remove the test functions
                    clone_str = ':'.join(clone[:2])  # clone[2] is the function name
                    clone_group_list.append(clone_str)

            # if len(clone_group_list) > 1: # exclude the empty clone groups and the longly group
            if clone_group_list:
                commit_groups.append(clone_group_list)

        commit_groups_merged = merge_groups(commit_groups)

        # tuplize the clone groups at a certain commit
        commit_groups_merged_tuplized = list()
        for clone_group in commit_groups_merged:
            commit_groups_merged_tuplized.append(tuple(clone_group))
        clone_class_dict_tuplized[commit_id] = commit_groups_merged_tuplized

    return clone_class_dict_tuplized


def remove_duplicates(genealogy_df):
    genealogy_df_distinct = genealogy_df.groupby('clone_group_tuple', as_index=False).agg(
        {'start_commit': list, 'genealogy': list})
    genealogy_df_distinct['genealogy'] = genealogy_df_distinct['genealogy'].apply(lambda x: "|".join(x))
    genealogy_df_distinct['start_commit'] = genealogy_df_distinct['start_commit'].apply(lambda x: x[0])

    print("after distinct: ", genealogy_df_distinct.shape, '\n', genealogy_df_distinct.columns)
    group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH,
                                                 '%s_group_genealogy_distinct.csv' % (project))
    print("distinct: ", group_genealogy_distinct_path)
    genealogy_df_distinct.to_csv(group_genealogy_distinct_path, index=False)


if __name__ == '__main__':
    project = config_global.PROJECT

    project = 'FreeRDP'
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)

    group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, '%s_group_genealogy.csv' % (project))
    print("genealogy path: ", group_genealogy_path)
    with open(group_genealogy_path, 'w') as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(['clone_group_tuple', 'start_commit', 'genealogy'])

        # load clone classes
        clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH,
                                                  '%s_clone_result_purified_with_paratype.json' % project)
        print("purified path: ", clone_result_purified_path)

        with open(clone_result_purified_path, 'r') as clone_jsonfile:
            clone_result_json = json.load(clone_jsonfile, object_pairs_hook=OrderedDict)
            print("clone_jsonfile load successfully")
            clone_class_dict = filter_merge_tuplize(clone_result_json)
            print("len(clone_class_dict): ", len(clone_class_dict))
            #clone_class_dict_nonzero = dict((k, v) for k, v in clone_class_dict.items() if v) # filter out commits with empty clone groups

            nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
            project_repo = os.path.join(nicad_workdir, project)
            git_repo = Git(project_repo)
            commit_list = list(clone_class_dict)[::-1]        #[:10146] 时间从前往后看
            commit_list_sliced = list(clone_class_dict)[::-1]    #[:10146] # initially all commits base on time ascending

            # no need to read modifications again
            commit_modification_dict = dict()
            commit_modification_file = os.path.join(config_global.MODIFICATION_PATH, '%s_commit_modifications.pickle' % project)
            if os.path.exists(commit_modification_file):  # no need to read again
                print("extracting existing modifications")
                commit_modification_dict = pickle.load(open(commit_modification_file, 'rb'))  # if file already exists
            else:
                print("generating modifications")
                nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
                project_repo = os.path.join(nicad_workdir, project)

                git_repo = Git(project_repo)
                commit_modification_dict = get_all_commit_modifications(commit_list_sliced, git_repo)
                pickle.dump(commit_modification_dict, open(commit_modification_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            print("len(commit_modification_dict): ", len(commit_modification_dict))
            #for commit_id in tqdm(clone_class_dict_nonzero):
            # build up group genealogies
            for commit_id in tqdm(commit_list):
                # dfs
                # slice from the start_commit
                commit_list_sliced.remove(commit_id) # start commit will not be taken into account
                for clone_group_tuple in clone_class_dict[commit_id]:
                     genealogy_list = retrieve_clone_group_genealogy(clone_group_tuple, commit_list_sliced, clone_class_dict, commit_modification_dict)
                     genealogy_list.insert(0, "%s:%d:%d" % (commit_id, 0, 0))

                     if genealogy_list:
                         output_writer.writerow(["|".join(clone_group_tuple), commit_id, '|'.join(genealogy_list)])

    # loading genealogy file
    genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, '%s_group_genealogy.csv' % (project))
    print("redundant genealogy: ", genealogy_path)
    genealogy_df = pd.read_csv(genealogy_path)
    print("before: ", genealogy_df.shape, '\n', genealogy_df.columns)

    remove_duplicates(genealogy_df)