
import os, sys, json, requests


class Github_filtering:
    def __init__(self):
        pass

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
            'q': 'language:java',
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


if __name__ == "__main__":
    # Search Java projects based on the criteria
    gf = Github_filtering()
    java_projects = gf.search_java_projects()
    