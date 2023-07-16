import os, sys, re, shutil, subprocess, re, platform, json, pickle, timeit, multiprocessing, traceback, argparse, threading
from sys import stderr
from tqdm import tqdm
from numba import jit, cuda
from timeit import default_timer as timer
import pandas as pd
import concurrent.futures as cf
from pydriller import Git
from git import Repo
from glob import glob
from time import time, sleep

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
producer is to read from git repo related to I/O, so multi-threading is applied
'''
class Producer:

    def __init__(self):
        self.projects = config_global.SUBJECT_SYSTEMS.keys()

        os.makedirs(config_global.LOG_PATH, exist_ok=True)
        self.project_commit_status_all_df_path = os.path.join(config_global.LOG_PATH, 'project_commit_process_status_all_df.csv')
        # self.project_commit_status_all_df_lock = threading.Lock()
        # self.project_commit_status_all_df = pd.DataFrame()
        self.project_commit_status_list = []

        self.max_workers = 5
        self.executor = cf.ThreadPoolExecutor(max_workers=self.max_workers)

        # Start the thread that saves the dataframe to disk every hour
    

    def gitclone_projects(self, projects):
        for project in self.projects:
            print("project: ", project)
            Git_repo.gitclone_repo(project)

    
    def git_checkout_commits(self, project): # should not get_checkout_commit, since each commit will affact the working env of next commit
        project_commit_status_df = self.init_project_commit_status_df(project)
        nicad_workdir = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}')
        project_repo_path = os.path.join(nicad_workdir, project)
        for commit_id in tqdm(project_commit_status_df['commit_id']):
            cmd_git_checkout_commit = ['git', 'checkout', '-f', commit_id]
            execution = subprocess.run(cmd_git_checkout_commit, cwd=project_repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Git_repo.git_checkout_commit(project, commit_id)
            if execution.returncode == 0:
                project_commit_dir = os.path.join(nicad_workdir, f'{project}_{commit_id}')
                cmd_rsync = ['rsync', '-arp', project_repo_path, project_commit_dir] # note that cp is slow, use rsync to copy (https://www.zylk.net/en/web-2-0/blog/-/blogs/how-to-copy-files-in-linux-faster-and-safer-than-cp)
                subprocess.run(cmd_rsync, cwd=nicad_workdir) # Git_repo.git_checkout_commit(project, commit_id)
                self.update_project_commit_status(project_commit_status_df, project, commit_id, "ready")
            else:
                print(f"checkout {project} {commit_id} error")
                sys.exit(-100)

        return project_commit_status_df


    def init_project_commit_status_df(self, project):
        # Initializing status dataframe with 'idle' status
        commit_log_df = Git_repo.get_commit_log_df(project)[:10]
        # Convert the map result to a DataFrame
        project_commit_status_list = map(lambda commit_id: (project, commit_id, 'idle', 'java', 0, time()), commit_log_df['commit_id'])
        project_commit_status_df = pd.DataFrame(project_commit_status_list, columns=['project', 'commit_id', 'status', 'lang', 'size', 'timestamp'])
        # print("**************", project_commit_status_df)
        return project_commit_status_df


    def update_project_commit_status(self, project_commit_status_df, project, commit_id, status='idle'):
        project_commit_status_df.loc[(project_commit_status_df['project'] == project) & (project_commit_status_df['commit_id'] == commit_id), 'status'] = 'ready'


    def save_periodically(self, future):
        # self.save_thread = threading.Thread(target=self.save_periodically, daemon=True)
        # self.save_thread.start()
        # while True:
            #sleep(100)
            #self.project_commit_status_df.to_csv(self.project_commit_status_df_path, index=False)
        project_commit_status_df = future.result()
        self.project_commit_status_list.append(project_commit_status_df)
        project_commit_status_all_df = pd.concat(self.project_commit_status_list)
        project_commit_status_all_df.to_csv(self.project_commit_status_all_df_path, index=False) # save the final state


    def run(self):
        self.gitclone_projects(self.projects)

        for project in self.projects:
            print("project: ", project)
            future = self.executor.submit(self.git_checkout_commits, project)
            future.add_done_callback(self.save_periodically) #for future in cf.as_completed(futures)
        
        # Ensure all tasks are completed before shutting down the executor
        self.executor.shutdown(wait=True)


'''
consumer is for nicad clone detection, which involves a lot of code parsing costing CPU, so multi-processing is applied
'''
class Consumer:
    
    def __init__(self):
        self.project_commit_status_all_df_path = os.path.join(config_global.LOG_PATH, 'project_commit_process_status_all_df.csv')
        self.num_workers = 3 # int(os.cpu_count() / 2)
        self.completed_futures_result = []
        self.num_processes = 0


    def get_project_commit_status_ready_df(self):
        project_commit_status_all_df = pd.read_csv(self.project_commit_status_all_df_path)
        project_commit_ready_all_df = project_commit_status_all_df.loc[project_commit_status_all_df['status'] == 'ready']
        return project_commit_ready_all_df


    def detect_project_commit(self, project, commit, lang='java'):
        sleep(7)
        print("project-commit: ", project, commit)
        return project, commit, 'done', time()
    

    def detect_clones_on_commit(self, project, commit_id, programming_lang='java'):
        print("callback running: ", project, commit_id)
        nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
        project_commit_dir = os.path.normpath(os.path.join(nicad_workdir, f'{project}_{commit_id}'))
    
        clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH, project)
        if not os.path.exists(clone_result_path): # check if project clone_result path exists
            os.makedirs(clone_result_path, exist_ok=True) # cmd_mkdir_clone_result = 'mkdir -p %s/%s' % (config_global.CLONE_RESULT_PATH, project)
        # remove the dest file if exists
        clone_file_dest_path = os.path.join(config_global.CLONE_RESULT_PATH, project, f'{commit_id}.xml')
        print("clone_file_dest_path: ", clone_file_dest_path)
        
        if os.path.exists(clone_file_dest_path):
            os.remove(clone_file_dest_path) #shellCommand('rm -rf %s' % dest_path)
        
        # perform clone detection by NiCad
        print("project_commit_dir: ", project_commit_dir)
        # os.chdir(project_commit_dir) # to make nicad6 run under this directory
        cmd_detect_clones = ['nicad6', 'functions', programming_lang, project_commit_dir]
        subprocess.run(cmd_detect_clones, cwd=project_commit_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # the make nicad6 detect source code on the cwd
        nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
        nicad_clone_file_path = os.path.join(f'{project_commit_dir}_functions-blind-clones', f'{project}_{commit_id}_functions-blind-clones-0.30-classes-withsource.xml') # check
        print("clone_file_src_path: ", nicad_clone_file_path)
        if os.path.exists(nicad_clone_file_path):
            # move the results to the result folder
            shutil.move(nicad_clone_file_path, clone_file_dest_path)
        
        # cleanup the nicad working directory or use nicad cleanall, be care of the wildcard
        nicad_workdir_output_pattern = f'{project}_{commit_id}_functions*'
        nicad_workdir_output_pattern_path = os.path.join(nicad_workdir, nicad_workdir_output_pattern)
        globbed_paths = glob(nicad_workdir_output_pattern_path)
        for path in globbed_paths:
            os.remove(path) if os.path.isfile(path) else shutil.rmtree(path)


        return project, commit_id, 'done', time()
        
        # clean memory
        # shellCommand(cmd_clear_cache) cmd_clear_cache = 'sudo sysctl -w vm.drop_caches=3' # no privileges on gpu1


    def collect_process_status(self):
        print("*************** len futures: ", len(self.completed_futures_result))
        
        while True: 
            print("okay ", len(self.completed_futures_result))
            # print("*************** len futures: ", len(self.futures))
            sleep(10)  # Sleep for 10 seconds
            
            print(self.completed_futures_result)
            for project_commit_process_status in self.completed_futures_result:
                (project, commit_id, status, timestamp) = project_commit_process_status
                self.project_commit_ready_all_df.loc[(self.project_commit_ready_all_df['project'] == project) & (self.project_commit_ready_all_df['commit_id'] == commit_id), 'status'] = status
                self.project_commit_ready_all_df.loc[(self.project_commit_ready_all_df['project'] == project) & (self.project_commit_ready_all_df['commit_id'] == commit_id), 'timestamp'] = timestamp   
            
            self.project_commit_ready_all_df.to_csv(os.path.join(config_global.LOG_PATH, 'test.csv'), mode='a', index=False)

            if len(self.completed_futures_result) == self.num_processes:
                break


    def run(self):
        self.project_commit_ready_all_df = self.get_project_commit_status_ready_df()
        self.num_processes = self.project_commit_ready_all_df.shape[0]
        print("num processes: ", self.num_processes)
        for project in self.project_commit_ready_all_df['project'].unique(): # make sure the clone result folder exists
            os.makedirs(os.path.join(config_global.CLONE_RESULT_PATH, project), exist_ok=True)

        executor = cf.ProcessPoolExecutor(max_workers=self.num_workers)
        futures = []
        for row in self.project_commit_ready_all_df.itertuples(index=False):
            future = executor.submit(self.detect_project_commit, row.project, row.commit_id, row.lang)
            futures.append(future)
            print("================ len futures: ", len(futures))

        # Start the thread for gathering results, should start before the future result is yielding
        gather_thread = threading.Thread(target=self.collect_process_status, daemon=True)
        gather_thread.start()

        for future in cf.as_completed(futures):
            self.completed_futures_result.append(future.result())
        
        gather_thread.join()  # Wait for the gathering thread to finish
        executor.shutdown()
        


if __name__ == "__main__":
    '''
    producer = Producer()
    project = 'kafka'
    commits = Git_repo.get_commit_log_df(project)[:10]['commit_id']
    # producer.git_checkout_commits(project)
    #print(commits)
    producer.run()
    
    '''

    # project = 'kafka'
    consumer = Consumer()
    # consumer.detect_clones_on_commit(project, 'ea0bb00126', 'java')
    consumer.run()
    

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['producer', 'consumer'], help='Run the script as a producer or consumer')
    args = parser.parse_args()

    if args.role == 'producer':
        producer = Producer()
        producer.run()
    else:
        consumer = Consumer()
        consumer.run()
    '''
