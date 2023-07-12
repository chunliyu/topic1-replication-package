import os, sys, re, shutil, subprocess, re, platform, json, pickle, timeit, multiprocessing
from sys import stderr
from tqdm import tqdm
from numba import jit, cuda
from timeit import default_timer as timer
import pandas as pd
from pydriller import Git

# Run shell command from a string
sys.path.append(os.getcwd())
sys.path.append("..")
from config import config_global
from utils import Git_repo
import numpy as np
#from pydriller import Git
#from pydriller import Repository


def detect_clones_on_commit(project, commit_id, nicad_workdir, cmd_nicad, nicad_clone_file_path):
    #nicad_workdir_commit = os.path.join(config_global.REPO_PATH, nicad_workdir, "%s_%s" % (nicad_workdir, commit_id))

    child = subprocess.Popen('rm -rf %s/%s_functions*' % (nicad_workdir, project), shell=True)
    child.poll()

    project_repo = os.path.join(nicad_workdir, project)
    # shellCommand('git checkout %s' % commit_id) # optimize: can be checked out using pydriller.Git().checkout(hash)
    try:
        # cmd_checkout_commit = 'git checkout -f %s' % commit_id
        # shellCommand(cmd_checkout_commit) # we use subprocess.run to do it
        subprocess.run(['git', 'checkout', '-f', commit_id], cwd=project_repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # check out local repo to the newest commit
        
        # perform clone detection by NiCad
        subprocess.run(cmd_nicad, cwd=nicad_workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # check out local repo to the newest commit

        # clear working space
        dest_path = '%s/%s/%s.xml' % (config_global.CLONE_RESULT_PATH, project, commit_id)
        if os.path.exists(dest_path):
            shellCommand('rm -rf %s' % dest_path)

        if os.path.exists(nicad_clone_file_path):
            # move the results to the result folder
            shutil.move(nicad_clone_file_path, dest_path)
            # delete NiCad output files
        
        # clean memory
        # shellCommand(cmd_clear_cache) cmd_clear_cache = 'sudo sysctl -w vm.drop_caches=3' # no privileges on gpu1
    except Exception as e:
        print("err: ", commit_id, e)
        sys.exit(-1)


# function optimized to run on gpu
#@jit(target_backend='cuda')
def detect_clones_on_commits(project):
    print("project: ", project)
    # clean previous clone results and make the results' directory

    # git_repo = Git(project_repo)
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    project_repo = os.path.join(nicad_workdir, project)
    project_language = Git_repo.get_project_programming_language(project).lower()
    # cmd_nicad_str = 'nicad6 functions %s %s' % (project_language.lower(), project_repo)
    cmd_nicad = ['nicad6', 'functions', project_language, project_repo]
    nicad_clone_file_path = os.path.join(nicad_workdir, '%s_functions-blind-clones' % project,
                            '%s_functions-blind-clones-0.30-classes-withsource.xml' % project)

    # 优化1：7000/1000 commits change java file,  => apollo: (2843, 3) 925
    # only run commits that have changes, i.e., commit_id is in commit_modifications_dict
    commit_log_df = Git_repo.load_commit_log_df(project)
    commit_modifications_dict = Git_repo.load_commit_modifications_dict(project)
    commits_modified = commit_modifications_dict.keys()
    indices_commits_modified = commit_log_df[commit_log_df['commit_id'].isin(commits_modified)].index
    expanded_indices = np.concatenate([indices_commits_modified, indices_commits_modified + 1]) # Expand the indices to include index+1
    commits = commit_log_df.loc[expanded_indices]['commit_id']

    # resume from the clone detection breakpoint
    clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH, project)
    os.makedirs(clone_result_path, exist_ok=True) # cmd_mkdir_clone_result = 'mkdir -p %s/%s' % (config_global.CLONE_RESULT_PATH, project)
    clone_result_extracted = [x.split(".")[0] for _, _, files in os.walk(clone_result_path) for x in files] # remove the posix 
    commits_to_run = list(set(commits) - set(clone_result_extracted)) # commits_to_run = list(set(commit_log_df['commit_id']) - set(clone_result_extracted) - set()) 

    for commit_id in tqdm(commits_to_run): 
        '''
        优化1：7000/1000 commits change java file,  => apollo: (2843, 3) 925
        优化2: for a hash, if not java mofification, ignore; else generate new dir, 对于每个commit，创建单独的工作目录
        优化3: python multi-processing
        '''
        detect_clones_on_commit(project, commit_id, nicad_workdir, cmd_nicad, nicad_clone_file_path)


if __name__ == "__main__":
    start = timer()

    total_cores = multiprocessing.cpu_count()
    multiprocessing_cores = int(total_cores * 0.7)
    pool = multiprocessing.Pool(processes=min(len(config_global.SUBJECT_SYSTEMS_TEST.keys()), multiprocessing_cores))  # Adjust the number of processes as needed
    pool.map(detect_clones_on_commits, config_global.SUBJECT_SYSTEMS_TEST.keys())
    pool.close()
    pool.join()

    print("with GPU:", timer() - start)
