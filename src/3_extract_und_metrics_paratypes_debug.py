'''
add file to understand path

'''

import logging
import subprocess
import sys,os
import time
import platform

from tqdm import tqdm
import pandas as pd
import shutil
pd.set_option('display.max_columns', None)
sys.path.append("..")
sys.path.append("")
#sys.path.append('C:/Program Files/SciTools/bin/pc-win64/Python')
sys.path.append('/home/chunliyu/apps/scitools/bin/linux64/Python')

from config import config_global
from platform import python_version

if "3.8" in python_version() and platform.system() == 'Windows':
    os.add_dll_directory(r"C:\Program Files\Scitools\bin\pc-win64")
# import understand


# Run shell command from a string
def shellCommand(command_str):
    cmd_str = " ".join(command_str)
    # print(cmd_str)
    cmd = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    return cmd_out, cmd_err


def build_und_project_db(commit_id, metric_columns, project, clone_files):
    # print("commit_id: ", commit_id)
    # run understand cli to construct the project understand db
    # create db
    und_commit_db = os.path.join(config_global.DATA_PATH, 'udb', "%s" %project, '%s.und' % commit_id)

    if platform.system() == 'Windows':
        und_commit_db = os.path.join(config_global.DATA_PATH, 'udb', "%s" % project, '%s.und' % commit_id)
    elif platform.system() == "Linux":
        # und_commit_db = os.path.join(config_global.DATA_PATH, 'udb', "%s" % project, '%s' % commit_id)
        und_commit_db = os.path.join(config_global.DATA_PATH, 'udb', "%s" % project, '%s.und' % commit_id)
    elif platform.system() == "Darwin":
        pass

    # debug
    #if os.path.exists(und_commit_db):
        #shutil.rmtree(und_commit_db, ignore_errors=True)

    # print("und_commit_db: ",und_commit_db)

    # create und db
    # cmd_create_und_db = ['und', '-db', und_commit_db, 'create', '-languages', 'Java']
    if not os.path.exists(und_commit_db):
        cmd_create_und_db = ['und', '-db', und_commit_db, 'create', '-languages', 'C++']
        shellCommand(cmd_create_und_db)

    # add all files into db corresponding to the current commit
    # und_commit_db += '.udb'
    for clone_file in clone_files:
        #cmd_add_file = ['und', 'add', clone_file, und_commit_db]
        # 变更
        if platform.system() == "Linux":
            clone_file = clone_file.replace("\\", "/")
        cmd_add_file = ['und', 'add', clone_file, und_commit_db]
        shellCommand(cmd_add_file)

    shellCommand(cmd_add_file)

    # settings and analyze udb to retrieve functions with parameters
    #cmd_setting_analyze = ['und', '-db', und_commit_db, 'settings', '-metrics', 'all', '-ReportDisplayParameters', 'on', 'analyze', 'metrics']
    cmd_setting_analyze = ['und', '-db', und_commit_db, 'settings', '-metrics']
    cmd_setting_analyze.extend(metric_columns)
    cmd_setting_analyze.extend(['-MetricShowFunctionParameterTypes', 'on'])
    # cmd_setting_analyze.extend(['-MetricFileNameDisplayMode', 'FullPath'])
    cmd_setting_analyze.extend(['-MetricFileNameDisplayMode', 'RelativePath'])
    cmd_setting_analyze.extend(['-MetricDeclaredInFileDisplayMode', 'RelativePath'])
    cmd_setting_analyze.extend(['-MetricShowDeclaredInFile', 'on'])
    cmd_setting_analyze.extend(['-MetricAddUniqueNameColumn', 'off'])
    cmd_setting_analyze.extend(['-ReportDisplayParameters', 'on'])
    cmd_setting_analyze.extend(['-ReportFileNameDisplayMode', 'RelativePath'])
    cmd_setting_analyze.extend(['analyze', 'metrics'])
    # print("cmd: ", cmd_setting_analyze)
    #print(" ".join(cmd_setting_analyze))
    shellCommand(cmd_setting_analyze)


def get_files_by_commit(commit_id, genealogy_df):
    groups_by_commit = genealogy_df.loc[genealogy_df['start_commit'] == commit_id]['clone_group_tuple']

    files_by_commit = set()
    for group in groups_by_commit:
        for clone in group.split("|"):
            # print("clone: ", clone)
            clone_path = os.path.normpath(clone.replace("'", "").strip().split(":")[0])

            # debug
            # print("clone_path: ", clone_path)
            #clone_str = eval(repr(clone.split("-")[0].replace("'", "").strip())).replace('\\\\', '\\')
            #clone_path = clone_path.replace("%s\\" % project, "") # 去掉前面的project 名称
            # clone = os.path.normpath(clone.replace(".java", "")).replace(os.path.sep, ".")
            # clone_path = os.path.normpath(clone.split(":")[0])
            if len(clone):
                files_by_commit.add(clone_path)
    #print("len set : ", len(files_by_commit))
    return files_by_commit


if __name__ == '__main__':
    #logging.info(understand.version())
    project = config_global.PROJECT

    project = 'ompi' # 'ompi', 'systemd'
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)


    # read in commits only related to clone groups
    # group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, '%s_group_genealogy.csv' % (project))
    group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH,
                                                 '%s_group_genealogy_distinct.csv' % (project))
    genealogy_df = pd.read_csv(group_genealogy_distinct_path)
    print(genealogy_df.shape, genealogy_df.columns)

    # traverse and checkout commits
    current_dir = os.getcwd()
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    project_repo = os.path.join(nicad_workdir, project)

    print(project_repo)
    os.chdir(project_repo)
    print("cwd:", os.getcwd())

    cmd_config_git = ['git', 'config', 'core.protectNTFS', 'false']
    shellCommand(cmd_config_git)

    metric_columns = config_global.METRIC_COLUMNS
    cols = ['commit_id', 'clone_signiture']
    cols.extend(metric_columns)
    metrics_all_df = pd.DataFrame(columns=cols)

    # traverse all the start commits for the clone_group_tuple
    #commit_list = ['de55860'] #'1a17ebc','256bf8f','8f66c7f','c32a501','7660a41','4741552','fda9aa2','1b07465','c22f209','911fc14','5d76c0b'

    commit_list = ['0cb3e3a6d'
                   ] #['5908750', 'f317c7a', '082ffa3', '81ca4ad']
    #for commit_id in tqdm(commit_list):
    #print("# start commits: ", list(genealogy_df['start_commit'].drop_duplicates()))
    #for commit_id in tqdm(list(genealogy_df['start_commit'].drop_duplicates())):
    for commit_id in tqdm(genealogy_df['start_commit'].unique().tolist()):
    # debug
    #for commit_id in ['418de21d8', '5dd15443a']:
        # print("commit_id: ", commit_id)
        # for commit_id in tqdm(commit_list):
        if len(commit_id) <= 0:
            continue

        # debug
        #if commit_id != '5dec0f9': # 81ca4ad, 082ffa3, f317c7a, 5908750, 5d55fe0
            #continue
        #commit_id = 'b4d3365' # '655a7db' # 'ac62351'
        # check if the corresponding metrics have been retrieved
        metrics_path = os.path.join(os.path.normpath(config_global.UDB_PATH), "%s" % project, '%s.csv' % commit_id)
        #print(metrics_path)

        if os.path.exists(metrics_path):
            continue

        # check out project repo at a specified commit to update the source repo
        cmd_checkout = ['git', 'checkout', '-f', commit_id]  # 'git checkout %s' % commit_id
        #print(cmd_checkout)
        shellCommand(cmd_checkout)  # optimize: can be checked out using pydriller.Git().checkout(hash)

        # 检查checkout 是否成功
        curr_commit_id = os.popen('git rev-parse --short HEAD').read()
        #n = 0
        while curr_commit_id[:len(commit_id)] != commit_id:
            print(curr_commit_id[:len(commit_id)], commit_id)
            time.sleep(5)
            sys.exit(-1)
            #n = n + 1
            #if n == 100:
                #

        clone_files = list(get_files_by_commit(commit_id, genealogy_df))

        # debug
        # print("clone_files: ", clone_files)
        #变更
        #for i, clone_file in enumerate(clone_files):
            #clone_files[i] = clone_file[10:].replace("\\main", "").replace("\\java", "")

        # print("clone_files: ", clone_files)
        #clone_files = ['src\\com\\itmill\\toolkit\\demo\\demo\\Calc.java']
        files_to_analyze_path = os.path.join(config_global.UDB_PATH, project, '%s_clone_files.txt' % commit_id)
        if not os.path.exists(files_to_analyze_path):
            with open(files_to_analyze_path, 'w') as fp:
                fp.write("\n".join(clone_files))

        # get metrics with understand tool
        build_und_project_db(commit_id, metric_columns, project, clone_files)

        # debug

    os.chdir(current_dir)
