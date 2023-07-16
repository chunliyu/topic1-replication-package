import os, sys, subprocess, requests
from git import Repo
import pandas as pd


sys.path.append(".")
sys.path.append("..")
from config import config_global
from timeit import default_timer as timer


class Git_repo:
    
    def __init__(self, project):
        self.project = project
        self.owner, self.repo_name = config_global.SUBJECT_SYSTEMS[project].split("/")
        self.repo_url = f'https://github.com/{owner}/{repo_name}.git'   
        self.repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
    
    
    @staticmethod
    def gitclone_repo(project):
        owner, repo_name = config_global.SUBJECT_SYSTEMS[project].split("/")
        repo_url = f'https://github.com/{owner}/{repo_name}.git'   
        repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
        
        if not (os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, '.git'))): # project repo not exists
            try:
                Repo.clone_from(repo_url, repo_path)
            except Exception as e:
                print(f"An error occurred while cloning the repository: {str(e)}")
        else:
            Git_repo.git_pull(project)
        
    
    @staticmethod
    def git_checkout_commit(project, commit):
        repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
        print("repo_path: ", repo_path)
        project_repo = Repo(repo_path)
        # checkout the commit
        project_repo.git.checkout(commit)


    @staticmethod
    def get_default_branch(project):
        owner, repo_name = config_global.SUBJECT_SYSTEMS[project].split("/")
        url = f'https://api.github.com/repos/{owner}/{repo_name}'
        response = requests.get(url, headers=config_global.GIT_HEADERS)
        if response.status_code == 200:
            data = response.json()
            default_branch = data['default_branch']
            return default_branch
        else:
            response.raise_for_status()
        

    @staticmethod
    def git_pull(project):
        default_branch = Git_repo.get_default_branch(project)
        cmd_gitpull = ['git', 'pull', 'origin', default_branch]
        project_repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
        subprocess.run(cmd_gitpull, cwd=project_repo_path, capture_output=True)
    

    @staticmethod
    def get_commit_log_df(project):
        commit_log_df_path = os.path.join(config_global.COMMIT_LOG_PATH, f'{project}_commit_log_df.csv')
        if os.path.exists(commit_log_df_path):
            commit_log_df = pd.read_csv(commit_log_df_path, header=0)
        else:
            os.makedirs(config_global.COMMIT_LOG_PATH, exist_ok=True)

            # change to repo directory
            Git_repo.git_pull(project)
            
            # get commit logs
            cmd_gitlog = ["git", "log", "--pretty=format:%h,%ce,%ci"]
            project_repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
            execution_output = subprocess.check_output(cmd_gitlog, cwd = project_repo_path, universal_newlines=True)
            commit_log_lines = list(map(lambda line: tuple(line.split(',')), filter(None, execution_output.split("\n"))))
            commit_log_df = pd.DataFrame(commit_log_lines, columns=['commit_id', 'email', 'timestamp'])
            commit_log_df.to_csv(commit_log_df_path, index=False)

        return commit_log_df
            


if __name__ == "__main__":
    start = timer()
    project = "nacos"

    Git_repo.gitclone_repo(project)
    branch = Git_repo.get_default_branch(project)
    print("branch: ", branch)

    #commit_id_old = '6252a88'
    #Git_repo.git_checkout_commit(project, commit_id_old)
    #commit_id_newer = '8fa83ce'
    #Git_repo.git_checkout_commit(project, commit_id_newer)
    Git_repo.get_commit_log_df(project)
    print("Time elapsed:", timer() - start)
