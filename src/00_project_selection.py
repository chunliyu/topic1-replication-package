
import os, sys, json, requests, pickle
from github import Auth, Github
from datetime import datetime, timedelta
from tqdm import tqdm
from git import Repo
from pydriller import Repository, Git # , RepositoryMining
from time import time, sleep
from glob import glob



class Github_filtering:
    def __init__(self):
        self.git_access_token = "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"
        self.git_access_token2 = "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"
        self.session = requests.session()
        self.session.keep_alive = True
        self.headers = {"Authorization": "token " + self.git_access_token, "Connection": "close", "User-Agent": "FansCount-GitHub"}
        self.auth = Auth.Token(self.git_access_token)
        self.gh = Github(auth=self.auth)

        #  -is:archived'#  stars:>289' # size:>10000' # size>10MB # java 647
        # self.search_query = 'created:<2022-07-01 language:java is:public -is:fork pushed:2022-07-01..2023-07-01 forks:>0 size:>10000' # 11k
        # self.search_query = 'created:<2022-07-01 language:java is:public -is:fork pushed:2022-07-01..2023-07-01 forks:>0 size:>10000 stars:>289' => 1.6k
        # self.sesarch_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork forks:>0 size:>10000 archived:false ' => 8.8k
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork forks:>0 size:>10000 archived:false stars:>289 ' => 1.5k

        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:mit ' => 765
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:GPL-3.0 ' => 867
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>1 size:>10000 archived:false license:apache-2.0 ' => 1.1k
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:bsd-2-clause ' => 1.1k
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:bsd-3-clause' => 186
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:bsd-3-clause-clear' => 186
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:bsl-1.0' => 1
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:cc0-1.0' => 40
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:wtfpl' => 5
        # self.search_qeury = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:unlicense' => 44
        # self.search_qeury = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork  forks:>0 size:>10000 archived:false license:zlib' => 4

        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:mit ' => 616
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:GPL-3.0 ' => 503
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>1 size:>10000 archived:false license:apache-2.0 ' => 849
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-2-clause ' => 88
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-3-clause' => 186
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-3-clause-clear' => 20
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsl-1.0' => 1
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:cc0-1.0' => 40
        # self.search_query = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:wtfpl' => 5
        # self.search_qeury = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:unlicense' => 44
        # self.search_qeury = 'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:zlib' => 4

        self.search_query = 'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false stars:>1000'
        self.search_java_qeury_list = [
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:mit ',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:GPL-3.0 ',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>1 size:>10000 archived:false license:apache-2.0 ',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-2-clause ',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-3-clause',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-3-clause-clear',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsl-1.0',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:cc0-1.0',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:wtfpl',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:unlicense',
            'language:java created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:zlib'
        ]

        self.search_c_qeury_list = [
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:mit ',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:GPL-3.0 ',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>1 size:>10000 archived:false license:apache-2.0 ',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-2-clause ',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-3-clause',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsd-3-clause-clear',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:bsl-1.0',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:cc0-1.0',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:wtfpl',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:unlicense',
            'language:C created:<2022-07-01 pushed:2022-07-01..2023-07-01 is:public -is:fork mirror:false  forks:>0 size:>10000 archived:false license:zlib'
        ]


    def traverse_returned_repos(self, language):
        all_repos_filtered_by_commits = []
        repo_filenames = glob(f'../data/project_selection/{language}_projects_1000stars/*_requests_repos_page_*.pkl')
        print("len files: ", len(repo_filenames))
        for repos_part_file in tqdm(repo_filenames):
            with open(repos_part_file, 'rb') as fp:
                # Load the pickled object
                response_json = pickle.load(fp)
                # print(response_json)

                # print("number of repos: ", response_json['total_count']) # number of repos:  56749
                repositories = response_json['items']
                repositories_filtered_by_commits = []
                for repo in tqdm(repositories):
                    #print("keys: ", repo.keys())
                    owner, repo_name = repo['full_name'].split("/")
                    git_url = repo['git_url'].replace("git:", "https:")

                    # get total number of commits, filter projects with #commits < 500
                    try:
                        commits_count = len(list(Repository(git_url).traverse_commits()))
                        # gr = Git(repo['full_name'])
                        #commits_count2 = gr.total_commits()

                        if commits_count >= 500: # contributors_count >= 10  and commits_count >= 500 and prs_count >= 50
                            repositories_filtered_by_commits.append(repo)
                            # print("html url: ", repo['full_name'], commits_count)
                    except Exception as e:
                        print("error: ", e)
                        url = f'https://api.github.com/repos/{owner}/{repo_name}/commits'
                        response = requests.get(url, headers=self.headers)
                        commits_count = len(response.json())
                        if commits_count >= 500: # contributors_count >= 10  and commits_count >= 500 and prs_count >= 50
                            repositories_filtered_by_commits.append(repo)
                        sleep(10)

                all_repos_filtered_by_commits.extend(repositories_filtered_by_commits)

                print("number of repos filtered by commits: ", len(repositories_filtered_by_commits))


                # get total number of contributors, filter projects with #contributors < 10
                #mining = RepositoryMining(git_url)
                #break

        with open('all_repos_filtered_by_commits.pkl', 'wb') as fp:
            pickle.dump(all_repos_filtered_by_commits, fp)

    def filter_by_contributors_and_pr(self):
        with open('all_repos_filtered_by_commits.pkl', 'rb') as fp:
            all_repos_filtered_by_commits = pickle.dump(fp)
            print("len repos filtered by commits: ", len(all_repos_filtered_by_commits))
            # contributors_count >= 10 and commits_count >= 500 and prs_count >= 50:

    def search_github_repositories_by_requests_v3(self, language):
        url = 'https://api.github.com/search/repositories'
        git_access_token = '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'
        headers = {'Authorization': f'token {git_access_token}'}
        # Construct the search query
        # query += ' stars:>=10'
        # query += ' merged:>=50'
        params = {
            'q': self.search_query, # 54131 repositories
            'per_page': 100, #number of results per page
            'page': 1 # start with the first page
                  }
        # Send the GET request to the GitHub API
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            total_counts = data['total_count']
            print("number of repos: ", data['total_count'])
            per_page = params['per_page']
            total_pages = (total_counts + per_page - 1) // per_page

            # Loop through the pages and retrieve the results
            for page in range(1, total_pages + 1):
                params['page'] = page  # Update the page parameter
                try:
                    response = requests.get(url, headers=headers, params=params)
                    data = response.json()

                    with open(f"{language}_requests_repos_page_{page}.pkl", "wb") as fp:
                        pickle.dump(data, fp)

                    sleep(30)

                except Exception as e:
                    sleep(300)
        else:
            print('Request failed with status code:', response.status_code)


    def search_all_github_repositories_by_requests_v3(self, language):
        today = datetime.now().strftime("%Y-%m-%d")
        search_query = f'created:<2022-07-01 language:{language} is:public -is:fork pushed:2022-07-01..{today} forks:>0'
        url = 'https://api.github.com/search/repositories'
        git_access_token = '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'
        headers = {'Authorization': f'token {git_access_token}'}

        search_list = (self.search_java_qeury_list if language == 'java' else self.search_c_qeury_list)
        for search_query_id in range(len(search_list)):
            print("search string: ", search_list[search_query_id])
            params = {
                'q': search_list[search_query_id], # 54131 repositories
                'per_page': 100, #number of results per page
                'page': 1 # start with the first page
            }
            # Send the GET request to the GitHub API
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                total_counts = data['total_count']
                print("number of repos: ", data['total_count'])
                per_page = params['per_page']
                total_pages = (total_counts + per_page - 1) // per_page

                # Loop through the pages and retrieve the results
                for page in range(1, total_pages + 1):
                    params['page'] = page  # Update the page parameter
                    try:
                        response = requests.get(url, headers=headers, params=params)
                        data = response.json()

                        with open(f"{search_query_id}_requests_repos_page_{page}.pkl", "wb") as fp:
                            pickle.dump(data, fp)

                        sleep(10)

                    except Exception as e:
                        sleep(300)
            else:
                print('Request failed with status code:', response.status_code)
                sleep(1000)

    def search_github_repositories_by_requests_v4(self):
        url = 'https://api.github.com/search/repositories'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        # Construct the search query
        # query += ' stars:>=10'
        # query += ' merged:>=50'
        params = {
            'q': self.search_query, # 54131 repositories
            'per_page': 100, #number of results per page
            'page': 1 # start with the first page
                  }
        # Send the GET request to the GitHub API
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            total_counts = data['total_count']
            print("number of repos: ", data['total_count'])
            per_page = params['per_page']
            total_pages = (total_counts + per_page - 1) // per_page

            # Loop through the pages and retrieve the results
            for page in range(1, total_pages + 1):
                params['page'] = page  # Update the page parameter
                try:
                    response = requests.get(url, headers=headers, params=params)
                    data = response.json()

                    with open(f"requests_repos_page_{page}.pkl", "wb") as fp:
                        pickle.dump(data, fp)

                    sleep(10)

                except Exception as e:
                    sleep(300)
        else:
            print('Request failed with status code:', response.status_code)


    def search_github_repositories_by_GithubSearchAPI(self):
        #repositories = self.gh.search_repositories(query='good-first-issues:>3 language:java commits:>5000')
        repositories = self.gh.search_repositories(query=self.search_query)
        repositories_list = list(repositories)
        print("len repo: ", len(repositories))


    def filter_repositories(self):

        cutoff_date = datetime.strptime("2022-05-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
        start_date = datetime(2022, 5, 1)
        end_date = datetime(2023, 5, 31)

        repos = []
        projects = []

        print("type repositories:",  type(repositories))
        for repo in tqdm(repositories):
            # Get the repository details
            print("type repo: ", type(repo))
            contributors_count = repo.get_contributors().totalCount
            created_at = repo.created_at
            commits_count = repo.get_commits().totalCount
            last_update = repo.pushed_at
            prs_count = repo.get_pulls(state='closed').totalCount

            # filtered the forked projects
            if contributors_count >= 10 and commits_count >= 500 and prs_count >= 50:
                # and created_at < cutoff_date \ # created:<2022-07-01
                # and (start_date <= last_update <= end_date):# pushed:2022-07-01..2023-07-01
                # print("repo: ", repo, contributors_count, created_at, commits_count, prs_count)
                # (not repo.fork) \ # -is:fork
                projects.append(repo.html_url)
                repos.append(repo)

        with open('repo.pkl', 'wb') as fp:
            pickle.dump(repos, fp)

        print("len projects: ", projects)

    # At least ten contributors and more than one year of history, to exclude toy/personal projects. according to [Why Do Developers Reject Refactorings in Open-Source Projects?]
    def apply_criteria_1(self):
        pass


    # At least 500 commits and 50 closed PRs, to exclude projects having a small change history and that are unlikely to provide useful PRs for our analysis. according to [Why Do Developers Reject Refactorings in Open-Source Projects?]
    def apply_criteria_2(self):
        pass


    # Modified at least once in the period from May 2019 to May 2020, to filter out inactive projects. This criterion is also key for our survey, as we invite developers of the inspected projects to participate in our study. according to [Why Do Developers Reject Refactorings in Open-Source Projects?]
    def apply_criteria_3(self):
        pass


    def search_java_projects(self):
        search_url = 'https://api.github.com/search/repositories'
        headers = {'Accept': 'application/vnd.github.v3+json'}

        query_params = {
            'q': 'language:java is:public',
            #"q": "created:<2022-05-01",
            'stars': '>0',
            'forks': '>0',
            'pushed': '>=2022-05-01',
            'pushed': '<=2023-05-31',
            'size': '>0'
        }

        response = requests.get(search_url, headers=headers, params=query_params)

        if response.status_code == 200:
            results = response.json()

            filtered_projects = []
            for project in results['items']:
                print("project: ", project, '\n')
                if project["contributors"] >= 10 and project["created_at"] < "YYYY-MM-DDT00:00:00Z":
                    filtered_projects.append(project["html_url"])

            print("len projects : ", len(filtered_projects))
            return filtered_projects

        else:
            print(f"Request failed with status code: {response.status_code}")
            return []

    def get_contributors_count(url):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return len(data)
        else:
            return 0

    def get_commits_count(url):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return len(data)
        else:
            return 0

    def get_closed_prs_count(url):
        response = requests.get(url, params={'state': 'closed'})
        if response.status_code == 200:
            data = response.json()
            return len(data)
        else:
            return 0
    def __init__(self):
        pass

    def get_all(self):
        #GitHub GraphQL API URL

        base_url = "https://api.github.com/graphql"
        git_access_token = "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"
        # git_access_token = "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"
        # Add your GitHub token here
        headers = {'Authorization': f'token {git_access_token}', 'Accept': "application/vnd.github.vixen-preview+json"}
        # forks { totalCount }
        # Define the GraphQL query with a variable
        search_query = """
            query($queryString: String!, $numRepos: Int!, $after: String) {
              search(query:$queryString, type: REPOSITORY, first: $numRepos, , after: $after) {
                repositoryCount
                edges {
                  node {
                    ... on Repository {
                       name
                       url
                       createdAt
                       pushedAt
                       forkCount
                       stargazers { totalCount }
                       issues {totalCount}
                       pullRequests {totalCount}
                       licenseInfo {
                          spdxId
                          name
                       }
                       defaultBranchRef {
                          target {
                              ... on Commit {
                              history {
                                  totalCount
                              }
                            }
                         }
                      }
                    }
                  }
                }
                pageInfo {
                  endCursor
                  hasNextPage
                }
              }
            }
            """

        # Define the variable
        variables = {
            # "queryString": "language:java is:public stars:>1000 archived:false",
            # "queryString": "created:<2016-02-01 language:java is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # java 941
            # "queryString": "created:2016-02-01..2022-07-01 language:java is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # java 981
            # "queryString": "created:<2016-02-01 language:C is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # C 755
            "queryString": "created:2016-02-01..2022-07-01 language:C is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000",
            # C 577
            "numRepos": 50,
            "after": None
        }

        from time import sleep
        from tqdm import tqdm
        repositories = []

        while True:
            # Send a POST request
            response = run_query(search_query, variables)
            # print("response: ", response)
            # Handle the response

            if response.status_code == 200:
                result = response.json()
                # print(result)
                # Process the results here
                repositoryCount = result['data']['search']['repositoryCount']
                print("rest repositories: ", repositoryCount - len(repositories), variables['after'])
                for edge in tqdm(result['data']['search']['edges']):
                    repo = edge['node']
                    # print(repo)
                    # print(f"Name: {repo['name']}, URL: {repo['url']}, Created At: {repo['createdAt']}, Last Pushed: {repo['pushedAt']}, Forks: {repo['forkCount']}, Stars: {repo['stargazers']['totalCount']}")
                    repositories.append(repo)

                # Get the pageInfo and check if there is a next page
                if result['data']['search']['pageInfo']['hasNextPage']:
                    # Update the 'after' variable with the endCursor value for the next page
                    variables['after'] = result['data']['search']['pageInfo']['endCursor']
                else:
                    # If there is no next page, break the loop
                    break
            else:
                print(f"Query failed with status code {r.status_code}.")
                break

            sleep(10)

        print("len repos: ", len(repositories))

        with open("../data/project_selection/c_projects/before2016-02-01_c.pkl",
                  "wb") as fb:  # "../data/project_selection/c_projects/between2016-02-01and2022-07-01_java.pkl"
            pickle.dump(repositories, fb)




class Github_filtering_v4:
    
    def _run_query(self, search_query, variables): # A simple function to use requests.post to make the API call. Note the json= section.
        try:
            # Make the GraphQL request
            response = requests.post(base_url, headers=headers, json={'query': search_query, 'variables': variables})
            if response.status_code == 200:
                return response
            else:
                raise Exception("Query failed to run by returning code of {}. {}".format(response.status_code, query))
        except requests.exceptions.HTTPError as err:
            if response.status_code == 502:
                print("502 Bad Gateway error occurred", err)
            if response.status_code == 401:
                print('HTTP error occurred: Unauthorized -', err)
            sys.exit(-1)
        except requests.exceptions.RequestException as err:
            print(f"Request exception occurred: {err}")
            sys.exit(-1)
        pass
    
    def request_num_commits():
        # GitHub GraphQL API URL
        base_url = "https://api.github.com/graphql"
        git_access_token = "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"
        # git_access_token = "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"
        # Add your GitHub token here
        headers = {'Authorization': f'token {git_access_token}', 'Accept': "application/vnd.github.vixen-preview+json"}
        #forks { totalCount }
        # Define the GraphQL query with a variable
        search_query = """
            query($queryString: String!, $numRepos: Int!, $after: String) {
                search(query:$queryString, type: REPOSITORY, first: $numRepos, , after: $after) {
                    repositoryCount
                    edges {
                        node {
                          ... on Repository {
                             name
                             url
                             createdAt
                             pushedAt
                             forkCount
                             stargazers { totalCount }
                             issues {totalCount}
                             pullRequests {totalCount}
                             licenseInfo {
                                spdxId
                                name
                             }
                             defaultBranchRef {
                                target {
                                    ... on Commit {
                                    history {
                                        totalCount
                                    }
                                  }
                               }
                            }
                          }
                        }
                    }
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                } 
            }
            """

        # Define the variable
        variables = {
            #"queryString": "language:java is:public stars:>1000 archived:false",
            #"queryString": "created:<2016-02-01 language:java is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # java 941
            #"queryString": "created:2016-02-01..2022-07-01 language:java is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # java 981
            #"queryString": "created:<2016-02-01 language:C is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # C 755
            "queryString": "created:2016-02-01..2022-07-01 language:C is:public -is:fork archived:false pushed:2022-07-01..2023-07-08 forks:>0 stars:>1000", # C 577
            "numRepos": 50,
            "after": None
        }
        
        repositories = []
        
        while True:
            # Send a POST request
            response = run_query(search_query, variables)
            #print("response: ", response)
            # Handle the response
            
            if response.status_code == 200:
                result = response.json()
                # print(result)
                # Process the results here
                repositoryCount = result['data']['search']['repositoryCount']
                print("rest repositories: ", repositoryCount - len(repositories), variables['after'])
                for edge in tqdm(result['data']['search']['edges']):
                    repo = edge['node']
                    #print(repo)
                    # print(f"Name: {repo['name']}, URL: {repo['url']}, Created At: {repo['createdAt']}, Last Pushed: {repo['pushedAt']}, Forks: {repo['forkCount']}, Stars: {repo['stargazers']['totalCount']}")
                    repositories.append(repo)
        
                # Get the pageInfo and check if there is a next page
                if result['data']['search']['pageInfo']['hasNextPage']:
                    # Update the 'after' variable with the endCursor value for the next page
                    variables['after'] = result['data']['search']['pageInfo']['endCursor']
                else:
                    # If there is no next page, break the loop
                    break
            else:
                print(f"Query failed with status code {r.status_code}.")
                break
            
            sleep(10)
            
        print("len repos: ", len(repositories))
        
        from config import config_global, model_config
        project_selection_datapath = os.path.join(config_global.DATA_PATH, "project_selection", "%s_projects"%language, "before2016-02-01_c.pkl")
        with open(project_selection_datapath, "wb") as fb: # "../data/project_selection/c_projects/between2016-02-01and2022-07-01_java.pkl"
            pickle.dump(repositories, fb)


if __name__ == "__main__":
    # Search Java projects based on the criteria
    github_filtering = Github_filtering()
    # github_filtering.search_github_repositories_by_requests()
    # github_filtering.search_all_github_repositories_by_requests_v3('c')
    #java_projects = search_github_repositories_requests()
    #github_filtering.search_github_repositories_by_requests_v3('c')
    #print("java_projects: ", java_projects)
    #print("len java_projects: ", len(java_projects))
    print("hello")

    github_filtering.traverse_returned_repos('java')
