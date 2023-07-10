# relate metrics to clone group
import os, sys, json, pickle, re, requests
import pandas as pd
from time import time, sleep
from collections import Counter, OrderedDict, defaultdict
from bs4 import BeautifulSoup
tree = lambda: defaultdict(tree)

sys.path.append("..")
from config import config_global
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
import networkx as nx
from tqdm import tqdm
from difflib import SequenceMatcher


git_token = "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"
s = requests.session()
s.keep_alive = True
headers = {"Authorization": "token " + git_token, "Connection": "close", "User-Agent": "FansCount-GitHub"}
user_followers = dict()

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)


def get_averages(list_of_methodmetrics_in_samefile):
    # Create a dictionary to store the sums
    sums = {}
    for d in list_of_methodmetrics_in_samefile:
        for key, value in d.items():
            # Add the value to the running total for this key
            if key in sums:
                sums[key] += value
            else:
                sums[key] = value

    # Calculate averages and store in a new dictionary
    averages = {}
    for key, value in sums.items():
        averages[key] = value / len(list_of_methodmetrics_in_samefile)

    return averages


'''
[NiCad] based on commit_id and clone_signiture, identify function name， 
'''
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


'''
[Understand Tool] based on commit_id and clone_signiture, identify function name， 
'''
def get_metrics_by_commit(project, commit_id):
    # commit_metric_df = metric_df.loc[metric_df['commit_id'] == commit_id].drop_duplicates()
    commit_metric_path = os.path.join(config_global.UDB_PATH, "%s" % project, '%s.csv' % commit_id)
    if not os.path.exists(commit_metric_path):
        print("commit metrics not exist: ", commit_id)

    commit_metric_df = pd.read_csv(commit_metric_path)
    # print("commit_metric_df: ", commit_metric_df.shape, commit_metric_df.columns)

    # filter out non-methods
    commit_metric_df = commit_metric_df[commit_metric_df['Kind'].str.contains('method', case=False) |
                                        commit_metric_df['Kind'].str.contains('function', case=False) |
                                        commit_metric_df['Kind'].str.contains('procedure', case=False) |
                                        commit_metric_df['Kind'].str.contains('constructor', case=False)
                                        ]

    if 'Kind' in commit_metric_df.columns:
        commit_metric_df.drop(['Kind'], axis=1, inplace=True)

    #print('commit_metric_df after: ', commit_metric_df.shape)

    # process the function signiture column
    pattern = "\\(.*?\\)"
    #commit_metric_df['Name'] = commit_metric_df['Name'].str.replace(pattern, '')

    #commit_metric_df['Name'] = commit_metric_df['Name'].str.replace('\.[a-zA-Z0-9_]+\.\.', '.') # remove (Anon_1)
    commit_metric_df['Name'] = commit_metric_df['Name'].str.replace('.\(Anon_[0-9]+\).', '.')  # remove (Anon_1)

    # filter out empty methods
    commit_metric_df = commit_metric_df[commit_metric_df['CountLine'] > 0]

    # debug
    # commit_metric_df['File'] = commit_metric_df['File'].str.replace('/', '.')
    commit_metric_df['Name'] = commit_metric_df['File'] + '/' + commit_metric_df['Name']
    if 'File' in commit_metric_df.columns:
        commit_metric_df.drop(['File'], axis=1, inplace=True)
    # print("columns: ", commit_metric_df.columns)
    # filter out duplicates
    commit_metric_df.drop_duplicates(inplace=True)

    # add range to the clone signiture
    #commit_metric_df['Name'] = commit_metric_df['Name'].str.cat(commit_metric_df['CountLine'].astype(str), sep=':')

    # adjust the path to make it consistent with the clone path in genealogy
    # map to the relative path
    # commit_metric_df['clone_signiture'] = commit_metric_df['clone_signiture'].map(
    # lambda path: os.path.relpath(path, start=os.path.normpath(r'C:\Users\Cleo\Dropbox\876\subject_systems\zaproxy')))

    # since all commit are the same
    # commit_metric_df.drop(['commit_id'], axis=1, inplace=True)

    # convert it to dict with key on clone_signiture
    # overriding functions: retrieve the first match
    commit_metric_df = commit_metric_df.drop_duplicates(subset='Name')
    commit_metric_df.set_index(['Name'], inplace=True)
    commit_metric_dict = commit_metric_df.to_dict('index')

    return commit_metric_dict


# some path has lone prefix while not exists in understand metrics
def search_clone(commit_metric_dict, clone_str):
    # print("==================", commit_metric_dict)
    # print("clone_str: ", clone_str)
    candidate_keys = []
    methods_in_samefile = []
    for key in commit_metric_dict:
        key_path, key_funcname_paras = key.rsplit("/", 1)

        try:
            clone_str_path_funcname, clone_str_paras = clone_str.split("(")
        except:
            print("\nexcept: ", clone_str, "|", clone_str_paras)
            sys.exit(-2)
        # clone_str_path, clone_str_funcname_paras = clone_str.rsplit(os.path.sep, 1)  # src/apache.c/config_set_boolean(int,/*)
        clone_str_path, clone_str_funcname = clone_str_path_funcname.rsplit(os.path.sep, 1)  # src/apache.c/config_set_boolean(int,/*)
        if key_path in clone_str_path or clone_str_path in key_path:
            methods_in_samefile.append(key)
            clone_str_funcname_paras = "(".join([clone_str_funcname, clone_str_paras])
            if key_funcname_paras in clone_str_funcname_paras:
                return commit_metric_dict.get(key)

            key_funcname, key_paras = key_funcname_paras.split("(", 1)


            #clone_str_funcname = clone_str_funcname_paras.split("(")[0]
            if key_funcname in clone_str_funcname:
                return commit_metric_dict.get(key)

            if SequenceMatcher(None, key_funcname_paras, clone_str_funcname_paras).ratio() >= 0.8:
                candidate_keys.append(key)

    if len(candidate_keys):
        return commit_metric_dict.get(candidate_keys[0])

    if len(methods_in_samefile):
        average_metric = get_averages([commit_metric_dict.get(key) for key in methods_in_samefile])
        # print("average metrics: ", average_metric, '|', clone_str, '|', commit_metric_dict)
        return average_metric
        # print("methods_in_samefile: ", methods_in_samefile)

        '''
        print("key: ", key)
        if key in clone_str or clone_str[:-1] in key: # clone_str:  cpu.cc26x0.periph.i2c.c.i2c_read_bytes(i2c_t,uint16_t,void,size_t,uint8_t) vs. periph.i2c.c.i2c_read_bytes(i2c_t,uint16_t,void *,size_t,uint8_t)
            return commit_metric_dict.get(key)

        # compare by path without parameters
        idx = key.find('(')
        key_without_para_info = key[:idx]
        if key_without_para_info in clone_str:
            return commit_metric_dict.get(key)

        # compare by function name and parameters
        key_without_path_info = key.split(".")[-1]
        clone_without_path_info = clone_str.split(".")[-1].split()[-1] # uint32_t \n SetupGetTrimForRadcExtCfg(uint32_t)
        if clone_without_path_info in key_without_path_info:
            return commit_metric_dict.get(key)

        # compare merely by function name
        key_without_path_para_info = key_without_path_info.split("(")[0]
        clone_without_path_para_info  = clone_without_path_info.split("(")[0]
        if clone_without_path_para_info in key_without_path_para_info:
            return commit_metric_dict.get(key)

        # compare by
        # key_without_class_info = key_without_class_info
        # key_without_class_info_str = '.'.join(key_without_class_info)
        # print(" key no class: ", key_without_class_info_str)
        # some method from understand tool has class info
        # if key_without_class_info_str in clone_str:
            # return commit_metric_dict.get(key)

        #key_without_class_info.pop(-2) # 调整

        # key_without_class_info_str = '.'.join(key_without_class_info)

        # if key_without_class_info_str in clone_str:
            # return commit_metric_dict.get(key)
        '''


    return None


# given clone_path and clone_range, retrieve clone_name
def load_clone_class_dict_4_clone(project):
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


def combine_und_other_metrics(und_metric_on_group_df, other_metric_on_group_df):
    print("\n\ngroup metric und: ", und_metric_on_group_df.shape)
    print("\n\ngroup metric other: ", other_metric_on_group_df.shape)
    merged_df = pd.merge(und_metric_on_group_df, other_metric_on_group_df, on='clone_group_tuple', how='inner')
    print("merged: ", merged_df.shape, merged_df.columns)
    merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
    merged_df.to_csv(merged_metric_on_group_path, index=False)


# html
def get_contributors_by_clone(github_base_url, commit_id, clone_file_path):
    from posixpath import join
    clone_contributor_url = join(github_base_url, "contributors-list", commit_id, clone_file_path.replace('\\', '/'))
    # debug

    '''
    'Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), 
    请求过于频繁，导致请求被拒绝 =》 每次请求设置一个休眠时间，例如time.sleep(1)
    接口的反爬虫机制 =》 在请求头中设置user-agent绕过验证 =》headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko', "Content-Type": "application/json"}
    '''
    max_retries, retry_delay = 5, 2
    # default_header = requests.sessions.Session().default_headers
    header = headers
    for attempt in range(max_retries):
        try:
            github_html = requests.get(clone_contributor_url, headers=header).text
            soup = BeautifulSoup(github_html, "html.parser")
            contributors_block = soup.find_all('a', {"class": "Link--primary no-underline"})

            contributors = set()
            for contributor in contributors_block:
                contributor_id = contributor.get('href')
                contributors.add(contributor_id)
            return contributors
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.TooManyRedirects) as e:
            # Exception resolution logic for ConnectionError, Timeout, and TooManyRedirects
            print("An error occurred:", str(e))
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
            elif header == headers: # try add headers
                print("Maximum number of retries reached. Exiting.")
                header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                       "Content-Type": "application/json"}
                attempt -= 1 # give another retry with different header, be careful, this might cause unlimited loop
            else:
                return None


def get_group_contributors(project_github_url, genealogy_df):
    group_contributors_path = os.path.join(config_global.GROUP_METRIC_PATH,
                                           "%s_group_contributors.pkl" % project)
    if os.path.exists(group_contributors_path):
        with open(group_contributors_path, 'rb') as fp:
            return pickle.load(fp)
    else:
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
            except Exception as e:
                # 保存断点
                group_contributors_part_path = os.path.join(config_global.GROUP_METRIC_PATH,
                                                       "%s_group_contributors_part.pkl" % project)
                with open(group_contributors_part_path, 'wb') as fp:
                    pickle.dump(group_contributors_dict, fp)
                print("error: ", e)
                sys.exit(-1)

        with open(group_contributors_path, 'wb') as fp:
            pickle.dump(group_contributors_dict, fp)
        return group_contributors_dict


def get_no_followers_by_userid(userid):
    response = s.get(f"https://api.github.com/users/{userid}", headers=headers)
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


def get_user_followers(group_contributors_dict):
    group_followers_path = os.path.join(config_global.GROUP_METRIC_PATH,
                                           "%s_group_followers.pkl" % project)
    if os.path.exists(group_followers_path):
        with open(group_followers_path, 'rb') as fp:
            return pickle.load(fp)
    else:
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


def get_common_path(clone_siblings):
    common_prefix = os.path.commonprefix(clone_siblings)
    longest_common_directory = os.path.dirname(common_prefix)
    return len(longest_common_directory.split("\\"))


def generate_other_metrics(project):
    from posixpath import join
    project_github_url = join('https://github.com', config_global.SUBJECT_SYSTEMS[project])
    print(project_github_url)

    clone_class_dict_4_clone = load_clone_class_dict_4_clone(project)

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


def load_other_metrics(project):
    # loading other metrics file
    other_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_other_metric.csv' % project)
    if os.path.exists(other_metric_on_group_path):
        other_metric_on_group_df = pd.read_csv(os.path.normpath(other_metric_on_group_path))
    else:
        other_metric_on_group_df = generate_other_metrics(project)

    print("group metric other: ", other_metric_on_group_df.shape)
    return other_metric_on_group_df


def load_undstand_metrics(project):
    group_metric_und_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric.csv' % project)
    # print("group_metric_path: ", group_metric_und_path)
    if os.path.exists(group_metric_und_path):
        und_metric_on_group_df = pd.read_csv(group_metric_und_path)
    else:
        clone_class_dict_4_clone = load_clone_class_dict_4_clone(project)
        und_metric_on_group_df = pd.DataFrame(
            columns=['clone_group_tuple', 'CountInput', 'CountLine', 'CountLineCode', 'CountLineCodeDecl',
                     'CountLineCodeExe',
                     'CountOutput', 'CountPath', 'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
                     'Cyclomatic', 'CyclomaticModified', 'CyclomaticStrict', 'Essential', 'MaxNesting'])

        # loading genealogy file
        group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH,
                                                     '%s_group_genealogy_distinct.csv' % (project))
        genealogy_df = pd.read_csv(group_genealogy_distinct_path)
        print(genealogy_df.shape, '\n', genealogy_df.columns)

        for commit_id in tqdm(genealogy_df['start_commit'].unique().tolist()):
            commit_metric_dict = get_metrics_by_commit(project, commit_id)  # metrics from understand tool

            # print(commit_metric_dict.keys())

            commit_groups = genealogy_df[genealogy_df['start_commit'] == commit_id]['clone_group_tuple'].tolist()
            for group in commit_groups:
                metric_on_group = Counter()

                clone_siblings = group.split("|")
                clone_count = len(clone_siblings)
                for clone in clone_siblings:
                    # dot_java_idx = clone.rfind(".java")
                    # dot_java_idx = clone.rfind(".c.")
                    # clone = clone[0:dot_java_idx] + clone[dot_java_idx:].replace(".java", "")
                    # clone = os.path.normpath(clone).replace(os.path.sep, ".")

                    if len(clone) < 3:
                        continue
                    clone_path = os.path.normpath(clone.split(":")[0])

                    func_name = ""
                    try:
                        func_name = clone_class_dict_4_clone[commit_id][clone]
                        # func_name = clone_class_dict_4_clone[commit_id][eval(clone)]
                    except SyntaxError as err:
                        print("error clone syntax: ", clone)
                        sys.exit(-1)
                    except KeyError as err:
                        # print(clone_class_dict_4_clone)
                        print("error clone key: ", clone, commit_id)
                        sys.exit(-1)

                    func_name = re.sub(' +', ' ', func_name.replace("\n", " "))
                    # clone_str = "/".join([clone_path, func_name]).strip()  # [5:] # there might be spaces
                    clone_str = os.path.join(clone_path, func_name).strip()  # [5:] # there might be spaces

                    # clone_str = eval(repr(clone.split("-")[0].replace("'", "").strip())).replace('\\\\', '\\')

                    # clone_str: org.zaproxy.zap.extension.encoder2.EncodeDecodeDialog.addField
                    # if commit_metric_dict.get(clone_str, None) is None:
                    clone_metrics = search_clone(commit_metric_dict, clone_str)

                    if clone_metrics is not None:
                        # only need the metrics on method level
                        clone_metrics = {key: val for key, val in clone_metrics.items() if
                                         key in config_global.METRIC_COLUMNS}
                        metric_on_group += Counter(clone_metrics)  # aggregate the metrics for clone group

                        # not_missing_output_writer.writerow([commit_id, clone_str])
                        # not_missing_commit_clone = not_missing_commit_clone.append(
                        # pd.DataFrame({'commit_id': [commit_id], 'clone_str': [clone_str]}), ignore_index=True)
                    else:
                        print("not in dict: ", commit_id, "|", clone_str, "|", commit_metric_dict.keys())
                        # print(commit_metric_dict)
                        # sys.exit(-1)

                        # missing_output_writer.writerow([commit_id, clone_str])
                        # missing_commit_clone = missing_commit_clone.append(
                        # pd.DataFrame({'commit_id': [commit_id], 'clone_str': [clone_str]}), ignore_index=True)
                    # clone_count += 1

                if clone_count:
                    # print("clone_count", clone_count)
                    metric_on_group_dict = dict(metric_on_group)
                    if metric_on_group_dict:
                        # print("metric_on_group_dict: ", metric_on_group_dict)
                        # print("cols: ", metric_on_group_dict.keys())
                        # get the average metric value
                        metric_on_group_dict = {k: v / clone_count for k, v in metric_on_group_dict.items()}
                        metric_on_group_dict.update({'clone_group_tuple': group})
                        # print("lallala 0: ", metric_on_group_dict)
                        metric_on_group_dict_df = pd.DataFrame(metric_on_group_dict, index=[0])
                        # metric_on_group_df = metric_on_group_df.append(metric_on_group_dict, ignore_index=True)
                        metric_on_group_df = pd.concat([metric_on_group_df, metric_on_group_dict_df], ignore_index=True)

        und_metric_on_group_df.to_csv(group_metric_und_path, index=False)
    return und_metric_on_group_df


def main(project):
    und_metric_on_group_df = load_undstand_metrics(project)
    other_metric_on_group_df = load_other_metrics(project)
    combine_und_other_metrics(und_metric_on_group_df, other_metric_on_group_df)


if __name__ == '__main__':
    start_time = time()

    project = config_global.PROJECT
    project = 'systemd' # 'FreeRDP' # problematic projects: ['systemd']
    if len(sys.argv) > 1:
        project = sys.argv[1]
    print("project: ", project)

    main(project)

    print(f'Time taken = {time() - start_time} seconds')  # print(f"Execution time: {elapsed_time:.4f} seconds")

