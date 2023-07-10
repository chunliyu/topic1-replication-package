import logging, subprocess, sys, os, time, re
from bs4 import BeautifulSoup
from posixpath import join
from collections import defaultdict, OrderedDict
import requests

sys.path.append("..")
from config import config_global
from tqdm import tqdm
import pandas as pd
import pickle
import shutil
import uvicorn as uvicorn
from fastapi import FastAPI
import requests, json, os
from tqdm import tqdm

git_token = "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"
s = requests.session()
s.keep_alive = True
headers = {"Authorization": "token " + git_token, "Connection": "close", "User-Agent": "FansCount-GitHub"}
user_followers = dict()


def get_authorid_by_commit(commit_id):  # cannot get result
    url = os.path.join(f'https://api.github.com', 'netty', 'commits', commit_id)
    response = s.get(url, headers=headers)


def get_loginname_by_email(email):
    url = f"https://api.github.com/search/users?q=" + email
    response = s.get(url, headers=headers)
    json_response = json.loads(response.text)

    if json_response['total_count'] > 0:
        loginname = json_response['items'][0]['login']
        return loginname
    return ""


def get_no_followers_by_email(email):
    loginname = get_loginname_by_email(email)
    print("loginname: ", loginname)

    status_code = 200
    message = "success"
    followers = 0

    if not loginname:
        return -1
    followers = get_no_followers_by_userid(loginname)
    print(email, loginname, followers)
    '''
    response = s.get(f"https://api.github.com/users/"+loginname, headers=headers)
    json_response = json.loads(response.text)

    message = json_response.get("message")

    if message == None:
        status_code = 200
        message = "success"
        followers = json_response.get("followers")
    else:
        status_code = -400
        followers = -1

    print("==================")
    print(status_code, message, followers)
    '''
    return followers


def get_no_followers_by_userid(userid):
    response = s.get(f"https://api.github.com/users" + userid, headers=headers)
    json_response = json.loads(response.text)
    message = json_response.get("message")
    if message == None:
        status_code = 200
        message = "success"
        followers = json_response.get("followers")
    else:
        status_code = -400
        followers = -1
    return int(followers)
    '''
    #access_token = "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"
    http_url = f'https://github.com' + userid
    # github_html = requests.get(f'https://github.com/{"strkkk"}').text
    github_html = requests.get(http_url).text
    soup = BeautifulSoup(github_html, "html.parser")
    cnt = soup.find('span', {"class": "text-bold color-fg-default"})
    return str(cnt.text)
    '''


# html
def get_contributors_by_clone(github_base_url, commit_id, clone_file_path):
    from posixpath import join
    clone_contributor_url = join(github_base_url, "contributors-list", commit_id, clone_file_path.replace('\\', '/'))
    # debug

    github_html = requests.get(clone_contributor_url).text
    soup = BeautifulSoup(github_html, "html.parser")
    contributors_block = soup.find_all('a', {"class": "Link--primary no-underline"})

    contributors = set()
    for contributor in contributors_block:
        contributor_id = contributor.get('href')
        contributors.add(contributor_id)

    #print("clone_contributor_url: ", clone_contributor_url, contributors)
    # debug
    #sys.exit(-2)
    return contributors


def get_no_followers_by_clone(github_base_url, commit_id, clone_file_path):
    from posixpath import join
    clone_contributor_url = join(github_base_url, "contributors-list", commit_id, clone_file_path.replace('\\', '/'))
    # http_url = 'https://github.com/prestodb/presto/blame/184fb504592d8ba432d6904ae0dd36ec34aa9457/presto-atop/src/main/java/com/facebook/presto/atop/AtopConnector.java#L39'
    github_html = requests.get(clone_contributor_url).text
    soup = BeautifulSoup(github_html, "html.parser")
    # contributors = soup.find_all('a', {"rel": "contributor"})
    contributors_block = soup.find_all('a', {"class": "Link--primary no-underline"})
    contributors = set()
    for contributor in contributors_block:
        contributor_id = contributor.get('href')
        contributors.add(contributor_id)

    for contributor_id in contributors:
        if contributor_id not in user_followers:
            cnt_followers = get_no_followers_by_userid(contributor_id)
            user_followers[contributor_id] = cnt_followers
            print(contributor_id, cnt_followers)

    return contributors
    # cnt = soup.find('a', {"rel": "contributor"})
    # cnt = soup.find('div', {"class": "AvatarStack-body"})
    '''
    user_list = soup.find_all('a', {"class": "avatar avatar-user"})
    print(set(user_list))

    userid_list = set()
    for user in set(user_list):
        userid_list.add(user.get('href'))
        cnt_followers = get_no_followers_by_http(user.get('href'))
        print(user, cnt_followers)
    print(userid_list)
    '''
    # cnt = soup.find('rel' "contributor")
    # print("login: ", cnt, type(cnt))
    # print(cnt.get('href'))


def get_common_path(clone_siblings):
    common_prefix = os.path.commonprefix(clone_siblings)
    longest_common_directory = os.path.dirname(common_prefix)
    return len(longest_common_directory.split("\\"))


def get_group_contributors(project_github_url, genealogy_df):
    group_contributors_path = os.path.join(config_global.OTHER_METRIC_PATH,
                                           "%s_group_contributors.pkl" % project)
    if os.path.exists(group_contributors_path):
        with open(group_contributors_path, 'rb') as fp:
            return pickle.load(fp)

    print("retrieving contributors ...")
    group_contributors_dict = dict()
    for index, row in tqdm(genealogy_df[['clone_group_tuple', 'start_commit']].iterrows()):
        try:
            start_commit = row['start_commit']
            clone_siblings = row['clone_group_tuple'].split("|")
            group_contributors = set()
            for clone in clone_siblings:
                clone_file_path = clone.split(":")[0]
                # clone_contributors = get_no_followers_by_clone(project_github_url, start_commit, clone_file_path)
                clone_contributors = get_contributors_by_clone(project_github_url, start_commit, clone_file_path)
                group_contributors.update(clone_contributors)

            group_contributors_dict[row['clone_group_tuple']] = list(group_contributors)
        except:
            with open(group_contributors_path, 'wb') as fp:
                pickle.dump(group_contributors_dict, fp)
                sys.exit(-1)

    with open(group_contributors_path, 'wb') as fp:
        pickle.dump(group_contributors_dict, fp)
    return group_contributors_dict


def get_user_followers(group_contributors_dict):
    group_followers_path = os.path.join(config_global.OTHER_METRIC_PATH,
                                           "%s_group_followers.pkl" % project)
    if os.path.exists(group_followers_path):
        with open(group_followers_path, 'rb') as fp:
            return pickle.load(fp)

    print("retrieving followers ...")
    all_contributors_distinct = set()
    for group_contributors in tqdm(group_contributors_dict.values()):
        all_contributors_distinct.update(group_contributors)

    user_followers_dict = dict()
    # with open('contributors.pkl', 'rb') as f:
    # all_contributors = pickle.load(f)

    for contributor in all_contributors_distinct:
        cnt_followers = get_no_followers_by_userid(contributor)
        user_followers_dict[contributor] = cnt_followers

    with open(group_followers_path, 'wb') as fp:
        pickle.dump(user_followers_dict, fp)

    return user_followers_dict


def get_clone_class(project):
    clone_class_dict_4_clone = defaultdict(defaultdict)
    project_clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH,
                                                      '%s_clone_result_purified_with_paratype.json' % project)
    # load clone classes
    with open(project_clone_result_purified_path, 'r') as clone_jsonfile:
        clone_result_json = json.load(clone_jsonfile, object_pairs_hook=OrderedDict)

        for commit_id in clone_result_json:
            # filter out test functions
            # clone group:  [['dist/tools/zep_dispatch/main.c', '181-197', '_print_help(const)'], ['dist/tools/benchmark_udp/main.c', '193-206', '_print_help(const)']]
            for clone_group in clone_result_json[commit_id]:
                for clone in clone_group:
                    #dot_java_idx = clone[0].rfind(".java")
                    #print("clone: ", clone)
                    #clone[0] = clone[0][0:dot_java_idx] + clone[0][dot_java_idx:].replace(".java", "")
                    # clone[0] = os.path.normpath(clone[0]).replace(os.path.sep, ".")
                    #clone[0] = os.path.normpath(clone[0].split(".java")[0]).replace(os.path.sep, '.')  # remove .java and replace / with .
                    if clone[0].lower().find('test') == -1:  # filter out test methods
                        clone_signiture = ':'.join(clone[:2])  # clone[2] is the function name

                        clone_class_dict_4_clone[commit_id][clone_signiture] = clone[2]

    return clone_class_dict_4_clone



# given clone_path and clone_range, retrieve clone_name
def load_clone_class_dict_4_clone():
    clone_class_dict_4_clone = defaultdict(defaultdict)
    clone_class_dict_4_clone_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH,
                                                 "%s_clone_class_dict_4_clone.pkl" % project)
    print("clone_class_dict_4_clone_path", clone_class_dict_4_clone_path)
    if os.path.exists(clone_class_dict_4_clone_path):
        with open(clone_class_dict_4_clone_path, 'rb') as handle:
            clone_class_dict_4_clone = pickle.load(handle)
    else:
        print("not exists clone_class_dict_4_clone_path")
        clone_class_dict_4_clone = get_clone_class(project)
        with open(clone_class_dict_4_clone_path, 'wb') as handle:
            pickle.dump(clone_class_dict_4_clone, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return clone_class_dict_4_clone


if __name__ == '__main__':
    project = config_global.PROJECT
    project = 'FreeRDP' # 'radare2'
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)

    project_github_url = join('https://github.com', config_global.SUBJECT_SYSTEMS[project])
    print(project_github_url)

    clone_class_dict_4_clone = load_clone_class_dict_4_clone()

    ## read in commits only related to clone groups
    group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH,
                                        '%s_group_genealogy_distinct.csv' % (project))
    genealogy_df = pd.read_csv(group_genealogy_path)
    print(genealogy_df.shape, genealogy_df.columns)

    group_contributors_dict = get_group_contributors(project_github_url, genealogy_df)
    # group_contributors_dict = dict()
    # with open('group_contributors.pkl', 'rb') as f:
    # group_contributors_dict = pickle.load(f)

    user_followers_dict = get_user_followers(group_contributors_dict)
    # user_followers_dict = dict()
    # with open('user_followers.pkl', 'rb') as f:
    # user_followers_dict = pickle.load(f)

    for index, row in tqdm(genealogy_df[['clone_group_tuple', 'start_commit']].iterrows()):
        clone_siblings = row['clone_group_tuple'].split("|")
        cnt_clone_siblings = len(clone_siblings)

        cnt_group_paras = 0
        # print(clone_siblings)
        for clone in clone_siblings:
            # dot_java_idx = clone.rfind(".java")
            # clone = clone[0:dot_java_idx] + clone[dot_java_idx:].replace(".java", "")
            # clone = os.path.normpath(clone).replace(os.path.sep, ".")
            if len(clone) < 3:
                continue

            func_name = clone_class_dict_4_clone[row['start_commit']][clone]
            func_paras = re.findall(r"[(](.*?)[)]", func_name)[0]
            cnt_func_paras = len(list(filter(None, func_paras.split(","))))
            cnt_group_paras += cnt_func_paras

        genealogy_df.loc[index, 'cnt_group_paras'] = int(cnt_group_paras / cnt_clone_siblings)

        genealogy_df.loc[index, 'cnt_clone_siblings'] = cnt_clone_siblings
        len_common_path = get_common_path(clone_siblings)
        genealogy_df.loc[index, 'len_common_path'] = len_common_path
        # print(genealogy_df.loc[index])

        # get followers of developers for group
        cnt_group_followers = 0
        group_contributors = group_contributors_dict[row['clone_group_tuple']]
        genealogy_df.loc[index, 'cnt_distinct_contributors'] = len(group_contributors)

        for contributor in group_contributors:
            cnt_group_followers += user_followers_dict[contributor]
        genealogy_df.loc[index, 'cnt_group_followers'] = int(cnt_group_followers / cnt_clone_siblings)

    genealogy_df.drop(['start_commit', 'genealogy'], axis=1, inplace=True)
    other_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_other_metric.csv' % project)
    genealogy_df.to_csv(other_metric_on_group_path, index=False)

    '''
    # based on distinct contributors,
    '''
    # genealogy_df.to_csv('netty_genealogy_df.csv')

    # commit_author_path = os.path.join(config_global.DATA_PATH, 'commit_author_logs', '%s_author_logs.txt'%project)
    # print(commit_author_path)
    # commit_author_df = pd.read_csv(commit_author_path, names=['commit_id', 'author_email', 'timestamp'])
    # print(commit_author_df.shape)
    # print(commit_author_df.head(5))

    # df = pd.merge(genealogy_df, commit_author_df, left_on='start_commit', right_on='commit_id', how='left')
    # df = df[['clone_group_tuple', 'start_commit', 'author_email']]

    # 通过git blame
    # email = 't@motd.kr'
    # developer = 'normanmaurer'  # developer = 'zzxzzk115'
    # commit_id = 'aef2ab4'
    # print(get_no_followers_by_api(email))

    # print(len(df['author_email'].drop_duplicates()))
    # for email in df['author_email'].drop_duplicates():
    #    cnt_followers = get_no_followers_by_api(email)
