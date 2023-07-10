import os, subprocess, re, json, sys
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

sys.path.append("..")
from config import config_global


# Run shell command from a string
def shellCommand(command_str):
    cmd = subprocess.Popen(command_str.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    return cmd_out


def extract_para_types(function_parameters):
    function_parameters = function_parameters.strip()
    if not function_parameters:
        return ""

    function_parameters = function_parameters.replace("final", " ")
    function_parameters = re.sub('\s+', ' ', function_parameters.strip())
    parts = []
    bracket_level = 0
    current = []
    # trick to remove special-case of trailing chars
    for c in (function_parameters + ","):
        if c == "," and bracket_level == 0:
            parts.append("".join(current))
            current = []
        else:
            if c == "<":
                bracket_level += 1
            elif c == ">":
                bracket_level -= 1
            current.append(c)
    parts = [part.strip().split(" ")[0] for part in parts]
    return ",".join(parts)


# Parse NiCad clone pair strings
def extract_clone_signiture(project, clone_info_str):
    clone_info_str = re.sub('\n+', " ", clone_info_str)
    clone_info_str = re.sub('\)\s+\{', "){", clone_info_str)
    # clone_info = re.findall(r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"', info_str)
    #clone_info = re.findall(r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\(', clone_info_str)
    idx_right_parenthesis = clone_info_str.find('){')
    '''
    clone_info = re.findall(
        r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\(',
        clone_info_str[:idx_left_parenthesis+1], re.S)
    '''
    clone_info = re.findall(
        r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\)',
        clone_info_str[:idx_right_parenthesis + 1], re.S
    )

    if len(clone_info):
        file_path = clone_info[0][0].split('/' + project + '/', 1)[-1]
        # 变更
        #file_path = clone_info[0][0].split(project + '/', 1)[-1]
        startline = clone_info[0][1]
        endline = clone_info[0][2]
        pcid = clone_info[0][3]

        # get function name without parameters
        #func_name = clone_info[0][4].strip().split(' ')[-1]

        # get function name and parameters
        func_header = clone_info[0][4].split('(')
        func_name = func_header[0].strip().split(' ')[-1]
        #breakpoint()
        func_paras = extract_para_types(func_header[1].strip())

            #print(func_header)
            #breakpoint()
        '''
        if func_header[1]:
            func_paras = func_header[1].split(',')
            try:
                func_para_str = ','.join(list(map(lambda para: para.split()[-2], func_paras)))
                func_name += '(' + func_para_str + ')'
            except IndexError:
                if func_header[1] == 'Map<String, Map<String, T>> vars, String scriptName, String key, T value':
                    func_name += '(Map<String, Map<String, T>>,String,String,T)'
                print(commit_id, func_header[1])

        else:
            func_name += '()'
        '''

        return [file_path, startline + '-' + endline, func_name + "(" + func_paras + ")"]
    else:
        #breakpoint()
        #print("no clone classes find: ", commit_id)
        return None


# Parse clone result files
def parse_clone_result(project, commit_id):
    result_list = list()
    commit_clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH, project, '%s.xml' % commit_id)
    # commit_clone_result_path = os.path.join(r"C:\Users\Cleo\Downloads\clone_detection_result_from_jzlarge_1220", project, '%s.xml' % commit_id)
    print(commit_clone_result_path)
    # breakpoint()
    if os.path.exists(commit_clone_result_path):
        with open(commit_clone_result_path, 'r', encoding="ISO-8859-1") as f:
        #with open(commit_clone_result_path, 'r', encoding='unicode_escape') as f:
            reader = f.read()
            # print(reader)

            # extract a pair of clones
            group_list = re.findall(
                r'<class classid=\"[0-9]+\" nclones=\"[0-9]+\" nlines=\"[0-9]+\" similarity=\"[0-9]+\">(.+?)</class>',
                reader, re.DOTALL)
            for group in group_list:  # 一个class
                # extract clone pair strings
                clone_group = list()
                clone_info = re.findall(r'<source (.+?)</source>', group, re.DOTALL)
                clone_signiture = ""
                for snippet in clone_info:
                    try:
                        clone_signiture = extract_clone_signiture(project, snippet)
                    except Exception as e:
                        print("error commit_id: ", commit_id)
                        print(f"An error occurred: {e}")
                        print("=======================")
                        print(snippet)
                        print("=======================")
                        sys.exit(-1)

                    if clone_signiture:
                        clone_group.append(clone_signiture)
                if len(clone_group):
                    result_list.append(clone_group)
    else:
        print("file not found")
    return result_list


if __name__ == '__main__':
    project = config_global.PROJECT
    project = 'systemd'
    if len(sys.argv) > 1:
        project = sys.argv[1]

    #project = 'che'  # func_paras = extract_para_types(func_header[1].strip()) IndexError: list index out of range
    print("project: ", project)

    # extract commit sequence
    commits_log_clean_path = os.path.join(config_global.COMMIT_LOG_CLEAN_PATH, '%s_logs.txt' % project)
    commits_log_df = pd.read_csv(commits_log_clean_path, names=['commit_id', 'committer', 'timestamp'], encoding="ISO-8859-1")
    print(commits_log_df.shape, commits_log_df.columns)
    #commits_log_df['timestamp'] = commits_log_df['timestamp'].map(lambda x: x[:20])
    #commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['timestamp'],  format='%Y-%m-%d %H:%M:%S', errors='coerce') # infer_datetime_format=True,
    # commits_log_df.sort_values('timestamp', inplace=True, ignore_index=True)  # 按commit时间升序 the ai timestamp is the author date

    # extract clone results
    print('Extracting clone results ...')
    # subprocess.Popen('mkdir -p %s' %config_global.COMMIT_LOG_CLEAN_PATH, shell=True)
    # parse each commit's clone results
    clone_dict = OrderedDict()

    for commit_id in tqdm(list(commits_log_df['commit_id'])): # [17510:]): #[::-1]):
        commit_clones_list = parse_clone_result(project, commit_id)
        clone_dict[commit_id] = commit_clones_list

    # output results
    clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH, '%s_clone_result_purified_with_paratype.json' % project)
    print("clone_result_purified_path: ", clone_result_purified_path)

    with open(clone_result_purified_path, 'w') as jsonfile:
            json.dump(clone_dict, jsonfile)
