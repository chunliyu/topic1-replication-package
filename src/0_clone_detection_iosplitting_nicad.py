import os, sys, re, shutil, subprocess, re, platform, json, pickle, timeit, multiprocessing, traceback
from sys import stderr
from tqdm import tqdm
from numba import jit, cuda
from timeit import default_timer as timer
import pandas as pd
from pydriller import Git
from git import Repo
from glob import glob

# Run shell command from a string
sys.path.append(os.getcwd())
sys.path.append("..")
from config import config_global
from utils import Git_repo
import numpy as np
from functools import partial
#from pydriller import Git
#from pydriller import Repository


'''
Step 2: git clone projects to the server in individual folders
    - git clone <proj_1_url> experiment/proj_1/
    - git clone <proj_2_url> experiment/proj_2/
    - finish the loop to git clone all n projects into their folders (e.g., experiment/proj_i)
    - note that this loop needs to be in sequence because we cannot git clone two projects at the same time due to API limit

'''
def gitclone_repos(projects): 
    for project in projects:
        print("project: ", project)
        Git_repo.gitclone_repo(project)


'''
Step 3: for each project, create subfolders to host the code per commit
    - for each project (e.g., experiment/proj_i/), get the list of commits in time order, also get the commit count (num_commits_proj_i)
    - copy and paste the project folder to another folder (experiment/proj_i/checkout/), which is used to iterate all commits to create local copies for all {project-commit} pairs
    - run git checkout for experiment/proj_i/checkout/, with commit #1, after this, experiment/proj_i/checkout/ will be in a state where the code becomes when commit #1 happens, get the commit hash id (hash_1), then copy and paste the current experiment/proj_i/checkout/ into experiment/proj_i/commits/hash_1/
    - in a loop, iterate the above step for all commits in project i, when this step is done, we get multiple folders (e.g., experiment/proj_i/commits/hash_j/) to host the code at commit hash j for project i, this loop is sequential because each git checkout will change the state of experiment/proj_i/checkout/
    - in an outer loop, iterate for different projects, so at the project level, the git checkout process can be paralleled
        - in this outer loop, we use a table (table_status) to record the state, each row looks like: proj_i, commit_j, [ready|running], proj_i_LOC_at_commit_j, proj_i_commit_count
        - note that in the above proj_i_LOC_at_commit_j, it is the LOC for project i at the commit j
        - [ready|running]: this column has a value of either ready or running, when the process of git checkout and copy & paste to the folder experiment/proj_i/commits/hash_j/ is finished, we update the value to ready, else we assign the value to running
        - note that cp is slow, use rsync to copy (https://www.zylk.net/en/web-2-0/blog/-/blogs/how-to-copy-files-in-linux-faster-and-safer-than-cp)
        - the parallel program (using the multiprocessing package) allocates N processes (a good choice is 75% of the core count, so for 40 cores, N can be 30)
        - we allocate each {project-commit} pair to one of the N processes, the allocation is based on the sorted order of {proj_i_LOC, proj_i_commit_count}, for example, for 5 projects with LOC: 10, 50, 100, 100, 202, and commit counts: 5, 10, 20, 30, 40, and 3 processes, we assign the project with LOC 10 and commit counts 5 to process #1, the project with LOC 50 and commit counts 10 to process #2, the project with LOC 50 and commit counts 20 to process #3, when process #1 finishes, then we assign the project with LOC 100 and commit counts 30 to process #1, in this way, we prioritize the execution of smaller LOC projects and smaller commits count
'''


# debug
def detect_clones_on_commit(project, programming_lang, commit_id):
    print("callback running: ", project, commit_id)
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    # note that cp is slow, use rsync to copy (https://www.zylk.net/en/web-2-0/blog/-/blogs/how-to-copy-files-in-linux-faster-and-safer-than-cp)
    project_commit_dir = os.path.normpath(os.path.join(nicad_workdir, f'{project}_{commit_id}'))

    # clean previous clone results and make the results' directory
    
    clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH, project)
    if not os.path.exists(clone_result_path): # check if project clone_result path exists
        os.makedirs(clone_result_path, exist_ok=True) # cmd_mkdir_clone_result = 'mkdir -p %s/%s' % (config_global.CLONE_RESULT_PATH, project)
    # remove the dest file if exists
    clone_file_dest_path = f'{config_global.CLONE_RESULT_PATH}/{project}/{commit_id}.xml'
    print("clone_file_dest_path: ", clone_file_dest_path)
    if os.path.exists(clone_file_dest_path):
        os.remove(clone_file_dest_path) #shellCommand('rm -rf %s' % dest_path)
    # perform clone detection by NiCad
    # cmd_nicad_str = 'nicad6 functions %s %s' % (project_language.lower(), project_repo)
    os.chdir(project_commit_dir) # to make nicad6 run under this directory
    cmd_detect_clones = ['nicad6', 'functions', programming_lang, project_commit_dir]
    print("project_commit_dir: ", project_commit_dir)
    subprocess.run(cmd_detect_clones, cwd=project_commit_dir+"/", stdout=subprocess.PIPE, stderr=subprocess.PIPE) # the make nicad6 detect source code on the cwd
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    nicad_clone_file_path = os.path.join(f'{project_commit_dir}_functions-blind-clones',
                            f'{project}_{commit_id}_functions-blind-clones-0.30-classes-withsource.xml') # check
    print("clone_file_src_path: ", nicad_clone_file_path)
    if os.path.exists(nicad_clone_file_path):
        # move the results to the result folder
        shutil.move(nicad_clone_file_path, clone_file_dest_path)

    # todo
    # clean the nicad working directory or use nicad cleanall
    '''
    nicad_workdir_output_pattern = f'{project}_{commit_id}_functions*'
    nicad_workdir_output_pattern_path = os.path.join(nicad_workdir, nicad_workdir_output_pattern)
    print("nicad_workdir_output: ", nicad_workdir_output_pattern_path)
    globed_results = glob(nicad_workdir_output_pattern_path)
    print("golo: ", globed_results)
    try:
        for clone_intermediate_result in glob(os.path.join(nicad_workdir, nicad_workdir_output_pattern)):
            print("clone_intermediate_result: ", clone_intermediate_result)
            if os.path.isfile(clone_intermediate_result):
                os.remove(clone_intermediate_result)
            elif os.path.isdir(clone_intermediate_result):
                os.rmdir(clone_intermediate_result)
    except:
        print(e)
        sys.exit(-1)
    '''
    

    # subprocess.check_call(['rm', '--'] + glob(nicad_workdir_output))
    # cmd_clean_nicad_workdir = ['rm', '-rf',  nicad_workdir_output] # child = subprocess.Popen('rm -rf %s/%s_functions*' % (nicad_workdir, project), shell=True)
    # print("cmd_clean_nicad_workdir: ", " ".join(cmd_clean_nicad_workdir))
    # subprocess.run(cmd_clean_nicad_workdir, cwd=nicad_workdir, shell=True) # wildcard is not recognized
    # clean memory
    # shellCommand(cmd_clear_cache) cmd_clear_cache = 'sudo sysctl -w vm.drop_caches=3' # no privileges on gpu1


# debug
def detect_clones_on_commit_status(project_commit_info, df):
    project, commit_id = project_commit_info
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    # note that cp is slow, use rsync to copy (https://www.zylk.net/en/web-2-0/blog/-/blogs/how-to-copy-files-in-linux-faster-and-safer-than-cp)
    project_commit_dir = os.path.normpath(os.path.join(nicad_workdir, f'{project}_{commit_id}'))

    # clean previous clone results and make the results' directory
    
    clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH, project)
    if not os.path.exists(clone_result_path): # check if project clone_result path exists
        os.makedirs(clone_result_path, exist_ok=True) # cmd_mkdir_clone_result = 'mkdir -p %s/%s' % (config_global.CLONE_RESULT_PATH, project)
    # remove the dest file if exists
    clone_file_dest_path = f'{config_global.CLONE_RESULT_PATH}/{project}/{commit_id}.xml'
    print("clone_file_dest_path: ", clone_file_dest_path)
    if os.path.exists(clone_file_dest_path):
        os.remove(clone_file_dest_path) #shellCommand('rm -rf %s' % dest_path)
    # perform clone detection by NiCad
    # cmd_nicad_str = 'nicad6 functions %s %s' % (project_language.lower(), project_repo)
    os.chdir(project_commit_dir) # to make nicad6 run under this directory
    cmd_detect_clones = ['nicad6', 'functions', programming_lang, project_commit_dir]
    print("project_commit_dir: ", project_commit_dir)
    subprocess.run(cmd_detect_clones, cwd=project_commit_dir+"/", stdout=subprocess.PIPE, stderr=subprocess.PIPE) # the make nicad6 detect source code on the cwd
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    nicad_clone_file_path = os.path.join(f'{project_commit_dir}_functions-blind-clones',
                            f'{project}_{commit_id}_functions-blind-clones-0.30-classes-withsource.xml') # check
    print("clone_file_src_path: ", nicad_clone_file_path)
    if os.path.exists(nicad_clone_file_path):
        # move the results to the result folder
        shutil.move(nicad_clone_file_path, clone_file_dest_path)

    # todo
    # clean the nicad working directory or use nicad cleanall
    '''
    nicad_workdir_output_pattern = f'{project}_{commit_id}_functions*'
    nicad_workdir_output_pattern_path = os.path.join(nicad_workdir, nicad_workdir_output_pattern)
    print("nicad_workdir_output: ", nicad_workdir_output_pattern_path)
    globed_results = glob(nicad_workdir_output_pattern_path)
    print("golo: ", globed_results)
    try:
        for clone_intermediate_result in glob(os.path.join(nicad_workdir, nicad_workdir_output_pattern)):
            print("clone_intermediate_result: ", clone_intermediate_result)
            if os.path.isfile(clone_intermediate_result):
                os.remove(clone_intermediate_result)
            elif os.path.isdir(clone_intermediate_result):
                os.rmdir(clone_intermediate_result)
    except:
        print(e)
        sys.exit(-1)
    '''
    

    # subprocess.check_call(['rm', '--'] + glob(nicad_workdir_output))
    # cmd_clean_nicad_workdir = ['rm', '-rf',  nicad_workdir_output] # child = subprocess.Popen('rm -rf %s/%s_functions*' % (nicad_workdir, project), shell=True)
    # print("cmd_clean_nicad_workdir: ", " ".join(cmd_clean_nicad_workdir))
    # subprocess.run(cmd_clean_nicad_workdir, cwd=nicad_workdir, shell=True) # wildcard is not recognized
    # clean memory
    # shellCommand(cmd_clear_cache) cmd_clear_cache = 'sudo sysctl -w vm.drop_caches=3' # no privileges on gpu1


# function optimized to run on gpu
#@jit(target_backend='cuda')
def detect_clones_on_commits(project):
    print("project: ", project)
    # clean previous clone results and make the results' directory

    # git_repo = Git(project_repo)
    nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
    project_repo = os.path.join(nicad_workdir, project)
    os.makedirs(nicad_workdir, exist_ok=True)
    project_language = Git_repo.get_project_programming_language(project).lower()
    # cmd_nicad_str = 'nicad6 functions %s %s' % (project_language.lower(), project_repo)
    cmd_nicad = ['nicad6', 'functions', project_language, project_repo]
    nicad_clone_file_path = os.path.join(nicad_workdir, '%s_functions-blind-clones' % project,
                            '%s_functions-blind-clones-0.30-classes-withsource.xml' % project)


    for commit_id in tqdm(commits_to_run): 
        '''
        优化1：7000/1000 commits change java file,  => apollo: (2843, 3) 925
        优化2: for a hash, if not java mofification, ignore; else generate new dir, 对于每个commit，创建单独的工作目录
        优化3: python multi-processing
        '''
        detect_clones_on_commit(project, commit_id, nicad_workdir, cmd_nicad, nicad_clone_file_path)


def get_commits_to_run(project):
    # 优化1：7000/1000 commits change java file,  => apollo: (2843, 3) 925
    # only run commits that have changes, i.e., commit_id is in commit_modifications_dict
    commit_log_df = Git_repo.load_commit_log_df(project)
    commit_modifications_dict = Git_repo.load_commit_modifications_dict(project)
    commits_modified = commit_modifications_dict.keys()
    indices_commits_modified = commit_log_df[commit_log_df['commit_id'].isin(commits_modified)].index

    indices_commits_previous = indices_commits_modified + 1
    indices_commits_previous_valid = indices_commits_previous[indices_commits_previous.isin(commit_log_df.index)]

    indices_commits_to_run = np.concatenate([indices_commits_modified, indices_commits_previous_valid])
    indices_commits_to_run = np.unique(indices_commits_to_run)

    commit_ids = commit_log_df.loc[indices_commits_to_run]['commit_id']

    # resume from the clone detection breakpoint
    clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH, project)
    os.makedirs(clone_result_path, exist_ok=True) # cmd_mkdir_clone_result = 'mkdir -p %s/%s' % (config_global.CLONE_RESULT_PATH, project)
    clone_result_extracted = [x.split(".")[0] for _, _, files in os.walk(clone_result_path) for x in files] # remove the posix 

    commits_to_run = list(set(commit_ids) - set(clone_result_extracted)) # commits_to_run = list(set(commit_log_df['commit_id']) - set(clone_result_extracted) - set()) 
    return commits_to_run


def get_project_commit_status_df():
    project_commit_status_df_path = os.path.join(config_global.LOG_PATH, 'project_commit_process_status_df.csv')
    project_commit_status_df = pd.read_csv(project_commit_status_df_path)
    return project_commit_status_df


def detect_clones_parallely():
    '''
    step 3
    '''
    project_commit_status_df = get_project_commit_status_df()
    project_commit_ready_df = project_commit_status_df.loc[project_commit_status_df['status'] == 'ready'] # make sure project_commit_ready_df has the same memory location as that in project_commit_status_df
    total_cores = multiprocessing.cpu_count()
    multiprocessing_cores = int(total_cores * 0.7)
    pool = multiprocessing.Pool(processes=min(len(projects), multiprocessing_cores))  # Adjust the number of processes as needed
    pool.map(partial(detect_clones_on_commit_status, df=project_commit_status_df), project_commit_ready_df.iterrows())

    pool.close()
    pool.join()


if __name__ == "__main__":
    start = timer()

    '''
    Step 1: get the list of n github projects to study
    - list_proj = [proj_1, proj_2, ..., proj_n]
    '''
    projects = config_global.SUBJECT_SYSTEMS.keys()

    '''
    Step 2: git clone projects to the server in individual folders
    - git clone <proj_1_url> experiment/proj_1/
    - git clone <proj_2_url> experiment/proj_2/
    - finish the loop to git clone all n projects into their folders (e.g., experiment/proj_i)
    - note that this loop needs to be in sequence because we cannot git clone two projects at the same time due to API limit
    '''
    gitclone_repos(projects)

    detect_clones_parallely()
    
    print("Time elapsed:", timer() - start)
