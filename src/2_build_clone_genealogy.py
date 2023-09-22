import common


class Clone_genealogy_extractor:
    
    def __init__(self, project):
        self._config_logger()

        self.project = project
        os.makedirs(config_global.CLONE_RESULT_PURIFIED_PATH, exist_ok=True)
        self.clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH, f'{self.project}_clone_result_purified_with_paratype.json')

        os.makedirs(config_global.GROUP_GENEALOGY_PATH, exist_ok=True)
        self.group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, '%s_group_genealogy.csv' % (project))

        self.programming_lang = Git_repo.get_programming_language(self.project)


    def _config_logger(self):
        # configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        log_dir = os.path.join(config_global.LOG_PATH, config_global.SERVER_NAME)
        os.makedirs(log_dir, exist_ok=True)
        now = strftime('%Y-%m-%d-%H:%M:%S', localtime(time()))
        log_file_path = os.path.join(log_dir, f'{Clone_genealogy_extractor.__name__}_running_{now}.log')
        log_file_path = os.path.join(log_dir, f'{Clone_genealogy_extractor.__name__}_running.log')
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        
        # Create a logging format
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(handler)


    def filter_merge_tuplize(self, clone_class_dict):
        clone_class_dict_tuplized = dict()
    
        for commit_id in clone_class_dict:
            commit_groups = list()
    
            #filter out test functions
            for clone_group in clone_class_dict[commit_id]:
                clone_group_list = list()
                for clone in clone_group:
                    clone[0] = os.path.normpath(clone[0])
                    if clone[0].lower().find('test') == -1: # remove the test functions
                        clone_str = ':'.join(clone[:2])  # clone[2] is the function name
                        clone_group_list.append(clone_str)
    
                # if len(clone_group_list) > 1: # exclude the empty clone groups and the longly group
                if clone_group_list:
                    commit_groups.append(clone_group_list)
    
            commit_groups_merged = self.merge_groups(commit_groups)
    
            # tuplize the clone groups at a certain commit
            commit_groups_merged_tuplized = list()
            for clone_group in commit_groups_merged:
                commit_groups_merged_tuplized.append(tuple(clone_group))
            clone_class_dict_tuplized[commit_id] = commit_groups_merged_tuplized
    
        return clone_class_dict_tuplized
    

    # some clone groups detected are acturally the same clone group
    def merge_groups(self, clone_groups):
        graph = nx.Graph()
        graph.add_nodes_from(sum(clone_groups, []))
        
        q = [[(group[i], group[i + 1]) for i in range(len(group) - 1)] for group in clone_groups]
    
        for i in q:
            graph.add_edges_from(i)
    
        # Find all connnected components in graph and list nodes for each component
        return [list(i) for i in nx.connected_components(graph)]

    
    # map line number from old commit to new commit
    def get_mapped_clone(self, clone_start_old, clone_end_old, line_mapping):
        clone_start_new = clone_start_old
        clone_end_new = clone_end_old
    
        churn = 0
        begin_to_count = False
    
        for line_pair in line_mapping:
            old_line = line_pair[0]
            new_line = line_pair[1]
    
            if old_line is None:
                old_line = 0
    
            if old_line > clone_end_old:
                return clone_start_new, clone_end_new, churn
    
            if old_line and new_line:
                # find the new start line
                if old_line <= clone_start_old:
                    clone_start_new = (new_line - old_line) + clone_start_old
                    clone_end_new = (new_line - old_line) + clone_end_old
    
                    # calculate the new end line
                elif old_line <= clone_end_old:
                    begin_to_count = True
                    clone_end_new = (new_line - old_line) + clone_end_old
            else:
                if old_line >= clone_start_old:
                    begin_to_count = True
    
                # if last line deleted in the clone boundary
                if begin_to_count:
                    if new_line is None:
                        clone_end_new -= 1
                    churn += 1
        return clone_start_new, clone_end_new, churn


    def get_mapped_group(self, clone_group_tuple, commit_modified_files):
        group_modified_list = list()
        churn_all = 0
        #breakpoint()
        
        for clone in clone_group_tuple:
            clone_path = clone.split(":")[0]
            clone_range = clone.split(":")[1]
            clone_start = int(clone_range.split('-')[0])
            clone_end = int(clone_range.split('-')[1])
            churn = 0
    
            for modified_file in commit_modified_files:
                # check if the changed file is related to the clones in the clone group
                if clone_path == modified_file[0]:  # old path有调整
                    # 获取new_path
                    if modified_file[1] is None: # new path == None, file being deleted, 当前clone已经不存在
                        clone_end = -1
                        churn = (clone_end - clone_start) + 1
                    else: # 只是单纯的修改
                        # 可以优化为modified_file.diff_parsed
                        clone_path = modified_file[1]  # new path
    
                        for diff1 in whatthepatch.parse_patch(modified_file[2]):  # only one element in the generator
                            line_mapping = diff1[1]
                            clone_start, clone_end, churn = self.get_mapped_clone(clone_start, clone_end, line_mapping)
                        #line_mapping = next(whatthepatch.parse_patch(modified_file[2]))[1] # only one element in the generator
                        #clone_start, clone_end, churn = get_mapped_clone(clone_start, clone_end, line_mapping)
    
                    break  
    
            # clone_start, clone_end 如果变化，记录变化后的
            if clone_start <= clone_end:
                group_modified_list.append("%s:%d-%d" % (clone_path, clone_start, clone_end))
                churn_all += churn
        return tuple(group_modified_list), churn_all
    

    # apply dfs searching for the same group
    def retrieve_clone_group_genealogy(self, clone_group_tuple, commit_list_sliced, clone_class_dict, commit_modification_dict):
        clone_group_genealogy_list = list()
        for commit_id in commit_list_sliced:  # consider the start commit_id
            churn_all = 0
    
            if commit_modification_dict[commit_id]:
                clone_group_tuple, churn_all = self.get_mapped_group(clone_group_tuple, commit_modification_dict[commit_id])
    
            for group in clone_class_dict[commit_id]:
                #breakpoint()
                if set(group).intersection(set(clone_group_tuple)):  # is_clone_group_matched(group, clone_group_tuple):
                    clone_class_dict[commit_id].remove(group)
                    #breakpoint()
                    #churn_all += self.calculate_churn_added(set(group) - set(clone_group_tuple))
                    cnt_siblings = len(set(group))
                    clone_group_genealogy_list.append("%s:%d:%d:%s" % (commit_id, churn_all, cnt_siblings, "|".join(group)))
                    clone_group_tuple = group
                    break  # stop when matched
    
        return clone_group_genealogy_list


    def calculate_churn_added(self, clone_group):
        churn = 0
        for clone in clone_group:
            churn += abs(eval(clone.split(":")[1]))
        return churn


    def build_group_genealogy(self, project):
        #self.logger.info("build_group_genealogy")
        if not os.path.exists(self.clone_result_purified_path):
            #self.logger.info(f"{self.clone_result_purified_path} not existed, generating...")
            self.purify_project_clones(project)

        if os.path.exists(self.group_genealogy_path):
            with open(self.group_genealogy_path, 'r') as output_file:
                genealogy_df = pd.read_csv(self.group_genealogy_path)
                print(f"{project} before: {genealogy_df.shape}, {genealogy_df.columns}")
                self.remove_duplicates(project, genealogy_df)
                return
        else:
            try:
                clone_jsonfile = open(self.clone_result_purified_path, 'r')
                clone_result_json = json.load(clone_jsonfile, object_pairs_hook=OrderedDict)

                # no need to read modifications againc
                commit_modifications_dict = Git_repo.get_commits_with_modifications(self.project, self.programming_lang)

                clone_class_dict = self.filter_merge_tuplize(clone_result_json)
    
                # Collect rows to write to CSV
                csv_rows = [['clone_group_tuple', 'start_commit', 'genealogy']]
                    
                commit_list = list(clone_class_dict)[::-1]        
                commit_list_sliced = list(clone_class_dict)[::-1]    
        
                # build up group genealogies
                for commit_id in tqdm(commit_list, desc=f'Building genealogies for {project}'):
                    # dfs, slice from the start_commit
                    commit_list_sliced.remove(commit_id) # start commit will not be taken into account
                    for clone_group_tuple in clone_class_dict[commit_id]:
                        genealogy_list = self.retrieve_clone_group_genealogy(clone_group_tuple, commit_list_sliced, clone_class_dict, commit_modifications_dict)
                        if genealogy_list:
                            genealogy_list.insert(0, "%s:%d:%d:%s" % (commit_id, 0, 0, '|'.join(clone_group_tuple))) # insert clone_group in the start commit
                            csv_rows.append(["|".join(clone_group_tuple), commit_id, ';'.join(genealogy_list)])

                # Write rows to CSV file in one operation
                with open(self.group_genealogy_path, 'w') as output_file:
                    output_writer = csv.writer(output_file)
                    output_writer.writerows(csv_rows)

                # loading genealogy file
                genealogy_df = pd.read_csv(self.group_genealogy_path)
                print("before: ", genealogy_df.shape, '\n', genealogy_df.columns)
                self.remove_duplicates(project, genealogy_df)
    
            finally:
                clone_jsonfile.close()


    def purify_project_clones(self, project):
        print(f"purify clones for {project}")
        # extract commit sequence
        # commits_log_clean_path = os.path.join(config_global.COMMIT_LOG_CLEAN_PATH, '%s_logs.txt' % project)
        # commits_log_df = pd.read_csv(commits_log_clean_path, names=['commit_id', 'committer', 'timestamp'], encoding="ISO-8859-1")
        commits_log_df = Git_repo.get_commit_log_df(self.project)
        
        # self.logger.info(f'Extracting clone results ... for {self.project}: {commits_log_df.shape}, {commits_log_df.columns}')
        commit_clones_dict = OrderedDict()

        with cf.ProcessPoolExecutor(max_workers=30) as executor:
            # Map get_commit_value_list to all commit_ids in commit_list, get a list of Future instances
            #future_list = [executor.submit(self.parse_clone_result, project, commit_id) for commit_id in list(commits_log_df['commit_id'])]
            #future_list = [executor.submit(self.parse_clone_result, project, commit_id) for commit_id in ['3a4a261c5']]
            future_list = list(tqdm((executor.submit(self.parse_clone_result, project, commit_id) for commit_id in list(commits_log_df['commit_id'])), 
                                    total=len(list(commits_log_df['commit_id']))
                                    )
                                )
    
        # Create a dictionary from the results of the Future instances
        commit_clones_dict = {future.result()[0]: future.result()[1] for future in future_list}

        with open(self.clone_result_purified_path, 'w') as jsonfile:
            json.dump(commit_clones_dict, jsonfile)
        # self.logger.info(f'clone_result_purified_path {self.clone_result_purified_path}')


    def extract_para_types(self, function_parameters):
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
    def extract_clone_signiture(self, project, clone_info_str):
        clone_info_str = re.sub('\n+', " ", clone_info_str)
        
        clone_info_str = re.sub('\)\s+\{', "){", clone_info_str)
        idx_right_parenthesis = clone_info_str.find('){')
        

        clone_info = re.findall(
            r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\)',
            clone_info_str[:idx_right_parenthesis + 1], re.S
        )

    
        if len(clone_info):
            
            file_path = clone_info[0][0].split('/' + project + '/', 1)[-1]
            startline = clone_info[0][1]
            endline = clone_info[0][2]
            pcid = clone_info[0][3]
    
            # get function name without parameters
            #func_name = clone_info[0][4].strip().split(' ')[-1]
    
            # get function name and parameters
            func_header = clone_info[0][4].split('(')
            func_name = func_header[0].strip().split(' ')[-1]
            
            func_paras = self.extract_para_types(func_header[1].strip())
            return [file_path, startline + '-' + endline, func_name + "(" + func_paras + ")"]
        else:
            
            #print("no clone classes find: ", commit_id)
            return None
    

    # Parse clone result files
    def parse_clone_result(self, project, commit_id):
        result_list = list()
        commit_clone_result_path = os.path.join(config_global.CLONE_RESULT_PATH_DATADIR, project, '%s.xml' % commit_id)
        
        if os.path.exists(commit_clone_result_path):
            with open(commit_clone_result_path, 'r', encoding="ISO-8859-1") as fp: # encoding='unicode_escape'
                reader = fp.read()
    
                # extract a pair of clones
                group_list = re.findall(
                    r'<class classid=\"[0-9]+\" nclones=\"[0-9]+\" nlines=\"[0-9]+\" similarity=\"[0-9]+\">(.+?)</class>',
                    reader, re.DOTALL)
                # self.logger.info(f'\t\t {group_list} to find startline and endline from <class classid= etc.>')
                for group in group_list:  # 一个class
                    # extract clone pair strings
                    clone_group = list()
                    clone_info = re.findall(r'<source (.+?)</source>', group, re.DOTALL)
                    clone_signiture = ""
                    for snippet in clone_info:
                        #self.logger.info(f'\t\t\t snippet: {snippet}')
                        try:
                            clone_signiture = self.extract_clone_signiture(project, snippet)
                        except Exception as e:
                            print(f'\t failed: {project} - {commit_id} on clone snippet {snippet}')
                            # self.logger.error(f'\t failed: {project} - {commit_id} on clone snippet {snippet}')
                            sys.exit(-1)
    
                        if clone_signiture:
                            clone_group.append(clone_signiture)
                    if len(clone_group):
                        result_list.append(clone_group)
        else:
            print(f'\t file path not existed: {commit_clone_result_path}')
            self.logger.warning(f'\t file path not existed: {commit_clone_result_path}')

        return commit_id, result_list
    

    def remove_duplicates(self, project, genealogy_df):
        genealogy_df_distinct = genealogy_df.groupby('clone_group_tuple', as_index=False).agg(
            {'start_commit': list, 'genealogy': list})
        genealogy_df_distinct['genealogy'] = genealogy_df_distinct['genealogy'].apply(lambda x: ";".join(x))
        genealogy_df_distinct['start_commit'] = genealogy_df_distinct['start_commit'].apply(lambda x: x[0])
    
        print(f"{project} after distinct: {genealogy_df_distinct.shape}, {genealogy_df_distinct.columns}")
        group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_PATH,
                                                     '%s_group_genealogy_distinct.csv' % (project))
        print(f"{project} distinct: {group_genealogy_distinct_path}")
        genealogy_df_distinct.to_csv(group_genealogy_distinct_path, index=False)


def worker(project):
    group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_PATH,
                                                     '%s_group_genealogy_distinct.csv' % (project))

    if os.path.exists(group_genealogy_distinct_path):
        return
    
    gen_extractor = Clone_genealogy_extractor(project)
    gen_extractor.build_group_genealogy(project)


if __name__ == '__main__':
    projects = list(config_global.SUBJECT_SYSTEMS_YOUNG.keys()) + list(config_global.SUBJECT_SYSTEMS_MIDDLE.keys()) + list(config_global.SUBJECT_SYSTEMS_OLD.keys()) 
    with cf.ProcessPoolExecutor(max_workers=int(os.cpu_count() * 1 / 5)) as executor:
        executor.map(worker, projects)
    
        
    
    

    