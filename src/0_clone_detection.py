import os, sys
import re
import shutil
import subprocess
from sys import stderr
import platform
from tqdm import tqdm
from numba import jit, cuda
from timeit import default_timer as timer
import pandas as pd
import json
import pprint
import pickle
import timeit
# Run shell command from a string
sys.path.append("..")
from config import config_global
#from pydriller import Git
#from pydriller import Repository


# Run shell command from a string
def shellCommand(command_str):
    #cd print("command_str: ", command_str)
    cmd = subprocess.Popen(command_str.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    return cmd_out, cmd_err

'''
def get_commits(project_repo):
    n = 0
    git_repo = Repository(project_repo, only_in_branch="master")
    for commit in git_repo.traverse_commits():
        short_sha = git_repo.git.rev_parse(commit.hash, short=8)
        print(short_sha)
        n = n + 1
    print(n)

    # git_repo.get_list_commits(rev='HEAD')
'''


def get_all_commits(project_path):
    command_str = 'git log --pretty=format:"%h, %ae, %ai, %s" > ' + commits_log_path
    shellCommand(command_str)


def generate_commits_log(commits_log_path):
    current_dir = os.getcwd()
    project_repo = os.path.join(config_global.REPO_PATH, project)
    os.chdir(project_repo)
    cmd_git_log = 'git log --pretty=format:"%h, %ae, %ai, %s" > ' % commits_log_path
    # git log --pretty=format:"%h,%ce,%ci" > ~/clone2api/data/commit_logs_clean/presto_logs.txt
    # git log --pretty=format:"%h,%ce,%ci,%s" > ~/clone2api/data/commit_logs/presto_logs.txt

    # git log --pretty=format:"%h,%ce,%ci" > ~/scratch/clone2api/data/commit_logs_clean/Terasology_logs.txt
    # git log --pretty=format:"%h,%ce,%ci" > ~/clone2api/data/commit_logs_clean/netty_logs.txt
    # git log --pretty=format:"%h,%ce,%ci,%s" > ~/clone2api/data/commit_logs/netty_logs.txt
    shellCommand(cmd_git_log)
    os.chdir(current_dir)

# function optimized to run on gpu
#@jit(target_backend='cuda')
def detect_clones_on_commits(commits_to_run):
    for commit_id in tqdm(commits_to_run):
        # checkout a specific commit
        # shellCommand('git checkout %s' % commit_id) # optimize: can be checked out using pydriller.Git().checkout(hash)
        try:
            os.chdir(project_repo)

            cmd_checkout_commit = 'git checkout -f %s' % commit_id
            shellCommand(cmd_checkout_commit)

            # clone detection
            # perform clone detection by NiCad
            os.chdir(nicad_workdir)
            shellCommand(cmd_nicad)

            # clear
            dest_path = '%s/%s/%s.xml' % (config_global.CLONE_RESULT_PATH, project, commit_id)
            if os.path.exists(dest_path):
                shellCommand('rm -rf %s' % dest_path)
            if os.path.exists(src_path):
                # move the results to the result folder
                shutil.move(src_path, dest_path)
                # delete NiCad output files

            child = subprocess.Popen('rm -rf %s/%s_functions*' % (nicad_workdir, project), shell=True)
            child.poll()
            # clean memory
            # shellCommand(cmd_clear_cache)
        except Exception as e:
            print("err: ", commit_id, e)
            sys.exit(1)


if __name__ == "__main__":
    start = timer()
    project = config_global.PROJECT
    language = 'c' # 'java'


    if len(sys.argv) > 1:
        project = sys.argv[1]

    print("project: ", project)
    #for project in config_global.PROJECTS:
    commits_log_path = os.path.join(config_global.COMMIT_LOG_CLEAN_PATH, '%s_logs.txt' % project)
    if not os.path.exists(commits_log_path):
        generate_commits_log(commits_log_path)

    # clean previous clone results and make the results' directory
    subprocess.Popen('rm -rf %s/%s_functions*' % (config_global.REPO_PATH, project), shell=True)
    cmd_mkdir_clone_result = 'mkdir -p %s/%s' % (config_global.CLONE_RESULT_PATH, project)
    shellCommand(cmd_mkdir_clone_result)

    # read from clean commit log
    commits_log_df = pd.read_csv(commits_log_path, header=None, names=['commit_id', 'committer', 'timestamp'])
    #commits_log_df.sort_values('timestamp', inplace=True)
    print(commits_log_df.shape)

    current_dir = os.getcwd()
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    print("nicad_workdir: ", nicad_workdir)
    project_repo = os.path.join(nicad_workdir, project)

    # git_repo = Git(project_repo)

    cmd_nicad = 'nicad6 functions %s %s' % (language, project_repo)
    print(cmd_nicad)
    cmd_clear_cache = 'sudo sysctl -w vm.drop_caches=3'
    src_path = os.path.join(nicad_workdir, '%s_functions-blind-clones' % project,
                            '%s_functions-blind-clones-0.30-classes-withsource.xml' % project)
    #commit_list = ['1bdb28e']
    #for index, commit_id in tqdm(commits_log_df['commit_id'].iteritems()):
    for i, j, k in os.walk(os.path.join(config_global.CLONE_RESULT_PATH, project)):
        clone_result_extracted = list(map(lambda x: x.split(".")[0], k))  # remove the posix

    # for index, commit_id in tqdm(commits_log_df['commit_id'].iteritems()):
    commits_to_run = list(set(commits_log_df['commit_id']) - set(clone_result_extracted))
    #for commit_id in tqdm(list(commits_log_df['commit_id'])):
    detect_clones_on_commits(commits_to_run)
    print("with GPU:", timer() - start)
