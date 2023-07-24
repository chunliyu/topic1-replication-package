import os, sys, subprocess, requests, pickle, json, logging
from git import Repo, GitCommandError
from github import Auth, Github
import pandas as pd

sys.path.append(".")
sys.path.append("..")
from config import config_global
from timeit import default_timer as timer
from tqdm import tqdm

logging.basicConfig(filename=os.path.join(config_global.LOG_PATH, config_global.SERVER_NAME, 'utils.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class Git_repo:
    
    def __init__(self, project):
        self.project = project
        self.owner, self.repo_name = config_global.SUBJECT_SYSTEMS_ALL[project].split("/")
        self.repo_url = f'https://github.com/{owner}/{repo_name}.git'   
        # self.repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
    
    
    @staticmethod
    def gitclone_repo(project, repo_path=None):
        logging.info(f"git clone to repo_path: {repo_path}")
        if not repo_path:
            repo_path = os.path.join(config_global.REPO_PATH, f"nicad_workdir_{project}", project)
            
        if os.path.isdir(repo_path) and os.path.exists(os.path.join(repo_path, '.git')):
            Git_repo.git_pull(project, repo_path)
        else: # project repo not exists
            try:
                os.makedirs(repo_path, exist_ok=True)
                owner, repo_name = config_global.SUBJECT_SYSTEMS_ALL[project].split("/")
                repo_url = f'https://github.com/{owner}/{repo_name}.git'   
                Repo.clone_from(repo_url, repo_path)
            except Exception as e:
                logging.error(f"An error occurred while cloning the repository: {str(e)}")  
        
    
    @staticmethod
    def git_checkout_commit(project, commit):
        repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
        logging.info(f"repo_path: {repo_path}")
        project_repo = Repo(repo_path)
        # checkout the commit
        project_repo.git.checkout(commit)


    @staticmethod
    def get_default_branch(project):
        owner, repo_name = config_global.SUBJECT_SYSTEMS_ALL[project].split("/")
        url = f'https://api.github.com/repos/{owner}/{repo_name}'
        response = requests.get(url, headers=config_global.GIT_HEADERS)
        if response.status_code == 200:
            data = response.json()
            default_branch = data['default_branch']
            return default_branch
        else:
            response.raise_for_status()

    
    @staticmethod
    def get_programming_language(project):
        owner, repo_name = config_global.SUBJECT_SYSTEMS_ALL[project].split("/")
        url = f'https://api.github.com/repos/{owner}/{repo_name}'
        response = requests.get(url, headers=config_global.GIT_HEADERS)
        if response.status_code == 200:
            data = response.json()
            programming_lang = data.get("language")
            return programming_lang.lower()
        else:
            response.raise_for_status()
            sys.exit(-1)
        

    @staticmethod
    def git_pull(project, repo_path=None):
        if not repo_path:
            repo_path = os.path.join(config_global.REPO_PATH, f"nicad_workdir_{project}", project)
        default_branch = Git_repo.get_default_branch(project)
        cmd_gitpull = ['git', 'pull', 'origin', default_branch]
        subprocess.run(cmd_gitpull, cwd=repo_path, capture_output=True)
    

    @staticmethod
    def get_commit_log_df(project, repo_path=None):
        commit_log_df_path = os.path.join(config_global.COMMIT_LOG_PATH, f'{project}_commit_log_df.csv')
        if os.path.exists(commit_log_df_path):
            commit_log_df = pd.read_csv(commit_log_df_path, header=0)
        else:
            os.makedirs(config_global.COMMIT_LOG_PATH, exist_ok=True)

            # change to repo directory
            Git_repo.git_pull(project, repo_path)
            
            # get commit logs
            cmd_gitlog = ["git", "log", "--pretty=format:%h,%ce,%ci"]
            project_repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
            execution_output = subprocess.check_output(cmd_gitlog, cwd = project_repo_path, universal_newlines=True)
            commit_log_lines = list(map(lambda line: tuple(line.split(',')), filter(None, execution_output.split("\n"))))
            commit_log_df = pd.DataFrame(commit_log_lines, columns=['commit_id', 'email', 'timestamp'])
            commit_log_df.to_csv(commit_log_df_path, index=False)

        return commit_log_df
    

    @staticmethod
    def get_commit_log_msg(project, repo_path=None):
        commit_log_path = os.path.join(config_global.COMMIT_LOG_PATH, "%s_gitlog.json" % project)
        if os.path.exists(commit_log_path):
            with open(commit_log_path) as fp:
                commit_log_msg_json = json.loads(fp.read())
                return commit_log_msg_json
        else: # generate log file
            logging.info("generating commit log history ... ")
            if not repo_path:
                project_repo = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project, project)
            else:
                project_repo = repo_path
            #hashes = subprocess.run(['git', 'rev-list', init_commit], cwd=project_repo, stdout=subprocess.PIPE).stdout.decode('ascii').split()
    
            owner, repo = config_global.SUBJECT_SYSTEMS_ALL[project].split("/") # redis/redis
            logging.info(f"repo: {owner} {repo}")
            main_branch = requests.get(f"https://api.github.com/repos/{owner}/{repo}").json()["default_branch"] # get the name of main branch: master/main
            cmd_git_checkout_branch = ['git', 'checkout', '-f', f'origin/{main_branch}']
            output = subprocess.run(cmd_git_checkout_branch, cwd=project_repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout # check out local repo to the newest commit
            
            hashes = subprocess.run(['git', 'rev-list', 'HEAD'], cwd=project_repo, stdout=subprocess.PIPE).stdout.decode('ascii').split()
            commit_log_msg_json = []
            for hash in tqdm(hashes, desc=f"getting commit messages for {project}"):
                entry = subprocess.run(['git', 'show', '--quiet', '--date=iso', hash],
                                       cwd=project_repo, stdout=subprocess.PIPE).stdout.decode(errors='replace')
                commit_log_msg_json.append(entry)

            if len(commit_log_msg_json) >= 500:
                with open(commit_log_path, 'w') as fp:
                    fp.write(json.dumps(commit_log_msg_json))
                return commit_log_msg_json
            else:
                logging.error("commit messages are None")
                return None
    

    @staticmethod
    def get_commits_with_modifications(project, programming_lang='c'):
        commit_modifications_dict = dict()
        os.makedirs(config_global.MODIFICATION_PATH, exist_ok=True)
        commit_modifications_dict_path = os.path.join(config_global.MODIFICATION_PATH, f'{project}_commit_modifications_dict.pickle')
        if os.path.exists(commit_modifications_dict_path):  # no need to read again
            logging.info("extracting existing modifications")
            commit_modifications_dict = pickle.load(open(commit_modifications_dict_path, 'rb'))  # if file already exists
        else:
            logging.info(f"generating modifications ... for project {project}")
            project_repo_path = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)

            from pydriller import Git
            project_repo = Git(project_repo_path)
            commit_log_df = Git_repo.get_commit_log_df(project)
            logging.info("shape: ", commit_log_df.shape)
            for commit_id in tqdm(commit_log_df['commit_id'], desc=f"getting commit modifications for {project}"):
                commit_modifications_dict[commit_id] = list()
                try:
                    commit_modified_files = project_repo.get_commit(commit_id).modified_files
                    commit_modified_files_java = list(filter(lambda x: (x.old_path and x.old_path.endswith(f'.{programming_lang}')) or
                                                                   (x.new_path and x.new_path.endswith(f'.{programming_lang}')),
                                                         commit_modified_files))
                    for modified_java_file in commit_modified_files_java:
                        # 需要检查diff的类型
                        commit_modifications_dict[commit_id].append([modified_java_file.old_path, modified_java_file.new_path, modified_java_file.diff])
                except (ValueError, GitCommandError) as e:
                    logging.error(f"An error occurred on commit {commit_id}: {e}")
                    sys.exit(-1)
            pickle.dump(commit_modifications_dict, open(commit_modifications_dict_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        return commit_modifications_dict 


    @staticmethod
    def get_github_issues(project):
        issue_path = os.path.join(config_global.DATA_PATH, 'issues', '%s_issues.csv' % project)
        if os.path.exists(issue_path):
            issues_df = pd.read_csv(issue_path)
            return issues_df
        else:
            os.makedirs(os.path.join(config_global.DATA_PATH, 'issues'), exist_ok=True)

            for token in config_global.GIT_ACCESS_TOKEN_LIST:
                try:
                    auth = Auth.Token(token)
                    g = Github(auth=auth)
                    # git_repo = g.get_repo("redis/redis") 
                    git_repo = g.get_repo(config_global.SUBJECT_SYSTEMS_ALL[project]) # repo = g.get_repo("redis/redis")
                    # project_topics = git_repo.get_topics()
                    # project_labels = git_repo.get_labels()
                    issues = git_repo.get_issues(state='closed') # git_repo.get_issues(state='open') state='closed' # doc: https://pygithub.readthedocs.io/en/stable/examples/Issue.html
                    issues_dict = [{'issue_number': int(issue.number), 'issue_title': issue.title} for issue in issues] # print(issue.title, issue.number)
                    issues_df = pd.DataFrame(issues_dict)
                    issues_df.to_csv(issue_path, index=False)
                    break
                except Exception as err:
                    print(f"error happend to {project} when get issues in utils.py: {err}")
                    continue
            return issues_df


if __name__ == "__main__":
    start = timer()
    project = "betaflight"

    Git_repo.gitclone_repo(project)
    Git_repo.get_commits_with_modifications(project, 'c')
    

    #commit_id_old = '6252a88'
    #Git_repo.git_checkout_commit(project, commit_id_old)
    #commit_id_newer = '8fa83ce'
    #Git_repo.git_checkout_commit(project, commit_id_newer)
    #Git_repo.get_commit_log_df(project)
    print("Time elapsed:", timer() - start)
