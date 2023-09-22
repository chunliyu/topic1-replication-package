import common

class Understand_metrics_extracter():
    
    def __init__(self, project):
        self._config_logger()

        self.project = project
        # read in commits only related to clone groups
        group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, f'{self.project}_group_genealogy_distinct.csv')
        # group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, f'{self.project}_group_genealogy.csv')
        if not os.path.exists(group_genealogy_distinct_path):
            return

        self.genealogy_df = pd.read_csv(group_genealogy_distinct_path)
        self.logger.info(f'{self.genealogy_df.head(5)}')

        self.metric_columns = config_global.METRIC_COLUMNS
        cols = ['commit_id', 'clone_signiture'] + self.metric_columns
        self.metrics_all_df = pd.DataFrame(columns=cols)

        self.nicad_workdir = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}')
        self.undstand_workdir = os.path.join(config_global.DATA_PATH, 'udb', project)
        self.project_repo_path = os.path.join(self.nicad_workdir, project)
        self._config_understand_scitools(project)
        Git_repo.gitclone_repo(project, self.project_repo_path)


    def _config_understand_scitools(self, project):
        project_udb_path = os.path.join(self.undstand_workdir, f'{project}.und')  # "%s_" % project, '%s.und' % commit_id)
        
        if os.path.exists(project_udb_path):
            shutil.rmtree(project_udb_path)
            
        ### 
        try:
            lang = Git_repo.get_programming_language(project)
            print("lang: ", lang)
            os.makedirs(self.undstand_workdir, exist_ok=True)
            udb_lang = ('Java' if lang == 'java' else 'C++')
            # run understand cli to construct the project understand db
            cmd_create_udb = ['und', 'create', '-db', f'{project}.und', '-languages', udb_lang]
            subprocess.run(cmd_create_udb, cwd=self.undstand_workdir, shell=False)
            
            # settings and analyze udb to retrieve functions with parameters
            # cmd_setting_analyze = ['und', '-db', und_commit_db, 'settings', '-metrics', 'all', '-ReportDisplayParameters', 'on', 'analyze', 'metrics']
            cmd_setting_udb = ['und', '-db', f'{project}.und', 'settings', '-metrics', 'all']
            # cmd_setting_udb.extend(self.metric_columns)
            cmd_setting_udb.extend(['-MetricShowFunctionParameterTypes', 'on'])
            # cmd_setting_udb.extend(['-MetricFileNameDisplayMode', 'FullPath'])
            cmd_setting_udb.extend(['-MetricFileNameDisplayMode', 'RelativePath'])
            cmd_setting_udb.extend(['-MetricDeclaredInFileDisplayMode', 'RelativePath'])
            cmd_setting_udb.extend(['-MetricShowDeclaredInFile', 'on'])
            cmd_setting_udb.extend(['-ReportDisplayParameters', 'on'])
            # cmd_setting_udb.extend(['-ReportFileNameDisplayMode', 'RelativePath'])
            cmd_setting_udb.extend(['-MetricAddUniqueNameColumn', 'off'])
            execution_setting_udb = subprocess.run(cmd_setting_udb, cwd=self.undstand_workdir, shell=False) # Git_repo.git_checkout_commit(project, commit_id)
            
        except Exception as err:
            logging.fatal("udb creation failed")
            raise Exception


    def _config_logger(self):
        # configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        log_dir = os.path.join(config_global.LOG_PATH, config_global.SERVER_NAME)
        os.makedirs(log_dir, exist_ok=True)
        now = strftime('%Y-%m-%d-%H:%M:%S', localtime(time()))
        log_file_path = os.path.join(log_dir, f'{Understand_metrics_extracter.__name__}_running_{now}.log')
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        
        # Create a logging format
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(handler)


    def get_commits_to_run(self, project):
        return self.genealogy_df['start_commit'].unique().tolist()


    def extract_und_metrics(self, commits):
        # traverse and checkout commits
        for commit_id in tqdm(commits, desc=f'extract und metrics for {self.project}'): 
            if len(commit_id) <= 0:
                continue

            # check if the corresponding metrics have been retrieved
            commit_metrics_path = os.path.join(os.path.normpath(config_global.UDB_PATH), project, f'{project}_{commit_id}.csv')
            
            if os.path.exists(commit_metrics_path):
                continue
            
            # check out project repo at a specified commit to update the source repo
            cmd_git_checkout_commit = ['git', 'checkout', '-f', commit_id]
            execution_checkout = subprocess.run(cmd_git_checkout_commit, cwd=self.project_repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Git_repo.git_checkout_commit(project, commit_id)
            if execution_checkout.returncode == 0:
                commit_files_to_analyze_path = os.path.normpath(os.path.join(config_global.UDB_PATH, project, f'{commit_id}_clone_files.txt'))
                
                if os.path.exists(commit_files_to_analyze_path):
                    os.remove(commit_files_to_analyze_path)
                    
                if not os.path.exists(commit_files_to_analyze_path):
                    clone_files = list(self.get_files_by_commit(commit_id, self.genealogy_df))
                    with open(commit_files_to_analyze_path, 'w') as fp:
                        fp.write("\n".join(clone_files))
                
                self.understand_project_commit(project, commit_id, commit_files_to_analyze_path)
            else:
                print("checkout failed")
                sys.exit(-1)

    
    def understand_project_commit(self, project, commit_id, clones_file_path):
        # add all files into db corresponding to the current commit
        # und_commit_db += '.udb'
        
        #cmd_add_file = ['und', 'add', clone_file, und_commit_db]
        project_udb_path = os.path.join(self.undstand_workdir, f'{project}.und')
        cmd_add_file = ['und', 'add', f'@{clones_file_path}', f'{project_udb_path}']
        self.logger.info(f'cmd_add_file: {" ".join(cmd_add_file)}')
        execution_addfile = subprocess.run(cmd_add_file, cwd=self.project_repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Git_repo.git_checkout_commit(project, commit_id)
        cmd_gen_metrics = ['und', 'settings', '-MetricOutputFile', f'{project}_{commit_id}.csv', 'analyze', 'metrics', f'{project_udb_path}']
        execution_genmetrics = subprocess.run(cmd_gen_metrics, cwd=self.undstand_workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if execution_genmetrics.returncode == 0:
            cmd_rm_file = ['und', 'remove', f'@{clones_file_path}', f'{project_udb_path}']
            execution_rmfile = subprocess.run(cmd_rm_file, cwd=self.project_repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Git_repo.git_checkout_commit(project, commit_id)
        else:
            print("fail to generate clone metrics")
            # sys.exit(-2)

    
    def get_files_by_commit(self, commit_id, genealogy_df):
        groups_by_commit = genealogy_df.loc[genealogy_df['start_commit'] == commit_id]['clone_group_tuple']
    
        files_by_commit = set()
        for group in groups_by_commit:
            for clone in group.split("|"):
                #clone_path = os.path.normpath(clone.replace("'", "").strip().split(":")[0])
                clone_path = os.path.normpath(clone.strip().split(":")[0])
                # clone_str = eval(repr(clone.split("-")[0].replace("'", "").strip())).replace('\\\\', '\\')
                # clone_path = clone_path.replace("%s\\" % project, "") # 去掉前面的project 名称
                # clone = os.path.normpath(clone.replace(".java", "")).replace(os.path.sep, ".")
                # clone_path = os.path.normpath(clone.split(":")[0])
                if len(clone):
                    files_by_commit.add(clone_path)
        
        return files_by_commit


class Understand_metrics_group_extracter():

    def __init__(self, project):
        self.project = project
    

    def get_averages(self, list_of_methodmetrics_in_samefile):
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
    [Understand Tool] based on commit_id and clone_signiture, identify function name， 
    '''
    def get_metrics_by_commit(self, project, commit_id):
        # commit_metric_df = metric_df.loc[metric_df['commit_id'] == commit_id].drop_duplicates()
        #commit_metric_path = os.path.join(config_global.UDB_PATH, "%s" % project, f'{project}_{commit_id}.csv')
        commit_metric_path = os.path.join(os.path.normpath(config_global.UDB_PATH), project, f'{project}_{commit_id}.csv')
        if not os.path.exists(commit_metric_path):
            print("commit metrics not exist: ", commit_id)
            return None
    
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
        commit_metric_df['Name'] = commit_metric_df['File'] + os.sep + commit_metric_df['Name']
        if 'File' in commit_metric_df.columns:
            commit_metric_df.drop(['File'], axis=1, inplace=True)
        # print("columns: ", commit_metric_df.columns)
        # filter out duplicates
        commit_metric_df.drop_duplicates(inplace=True)
    
        commit_metric_df = commit_metric_df.drop_duplicates(subset='Name')
        commit_metric_df.set_index(['Name'], inplace=True)
        commit_metric_dict = commit_metric_df.to_dict('index')
    
        return commit_metric_dict


    # some path has lone prefix while not exists in understand metrics
    def search_clone(self, commit_metric_dict, clone_str):
        candidate_keys = []
        methods_in_samefile = []
        for key in commit_metric_dict:
            # key_path, key_funcname_paras = key.rsplit(os.path.sep, 1)
            last_slash = key.rfind('/', 0, key.find('('))
            key_path, key_funcname_paras = key[:last_slash], key[last_slash+1:]
            try:
                key_funcname, key_paras = key_funcname_paras.split("(", 1)
            except Exception as err:
                print("key: ", key)
                print("key_path: ", key_path, '\t|', 'key_funcname_paras: ', key_funcname_paras)
                print("os sep: ", os.path.sep)
                print(err)
                sys.exit(-1)
    
            try:
                clone_str_path_funcname, clone_str_paras = clone_str.split("(")
            except:
                print("\n failed: ", clone_str, ": ", clone_str_paras)
                sys.exit(-2)
            
            # clone_str_path, clone_str_funcname_paras = clone_str.rsplit(os.path.sep, 1)  # src/apache.c/config_set_boolean(int,/*)
            clone_str_path, clone_str_funcname = clone_str_path_funcname.rsplit(os.path.sep, 1)  # src/apache.c/config_set_boolean(int,/*)
            
            if key_path in clone_str_path or clone_str_path in key_path:
                methods_in_samefile.append(key)
                clone_str_funcname_paras = "(".join([clone_str_funcname, clone_str_paras])
                if (key_funcname_paras in clone_str_funcname_paras) or (clone_str_funcname_paras in key_funcname_paras):
                    return commit_metric_dict.get(key)
    
    
                #clone_str_funcname = clone_str_funcname_paras.split("(")[0]
                if key_funcname in clone_str_funcname:
                    return commit_metric_dict.get(key)
                elif (key_funcname.rsplit('.', 1)[-1] + key_paras) == (clone_str_funcname + clone_str_paras):
                    return commit_metric_dict.get(key)
                elif key_funcname.rsplit('.', 1)[-1] == clone_str_funcname:
                    candidate_keys.append(key)
                elif SequenceMatcher(None, key_funcname_paras, clone_str_funcname_paras).ratio() >= 0.8:
                    candidate_keys.append(key)
            elif (key_funcname.rsplit(".", 1)[-1] + key_paras) == (clone_str_funcname + clone_str_paras):
                return commit_metric_dict.get(key)
            elif key_funcname.rsplit(".", 1)[-1] == clone_str_funcname:
                return commit_metric_dict.get(key)
    
        if len(candidate_keys):
            return commit_metric_dict.get(candidate_keys[0])
    
        if len(methods_in_samefile):
            average_metric = self.get_averages([commit_metric_dict.get(key) for key in methods_in_samefile])
            return average_metric
    
        return None
    

    def load_undstand_metrics(self, project):
        group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, f'{project}_group_genealogy_distinct.csv')
        if not os.path.exists(group_genealogy_distinct_path):
            print("group genealogy not exists!")
            sys.exit(-1)
        else:
            und_metrics_extracter = Understand_metrics_extracter(project)
            commits_to_run = und_metrics_extracter.get_commits_to_run(project)
            und_metrics_extracter.extract_und_metrics(commits_to_run)
        
            group_metric_und_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric.csv' % project)
            # print("group_metric_path: ", group_metric_und_path)
            if os.path.exists(group_metric_und_path):
                print(f"{group_metric_und_path} exists")
                und_metric_on_group_df = pd.read_csv(group_metric_und_path)
            else:
                print(f"understand metrics: {group_metric_und_path} not exists")
                clone_class_dict_4_clone = load_clone_class_dict_4_clone(project)
                und_metric_on_group_df = pd.DataFrame(columns=['clone_group_tuple'] + config_global.METRIC_COLUMNS_ALL)
                #columns=['clone_group_tuple', 'CountInput', 'CountLine', 'CountLineCode', 'CountLineCodeDecl',
                             #'CountLineCodeExe',
                             #'CountOutput', 'CountPath', 'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
                             #'Cyclomatic', 'CyclomaticModified', 'CyclomaticStrict', 'Essential', 'MaxNesting'])
        
                # loading genealogy file
                group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, f'{project}_group_genealogy_distinct.csv')
                genealogy_df = pd.read_csv(group_genealogy_distinct_path)
                print("project genealogy: ", genealogy_df.shape)
                commits_to_run = genealogy_df['start_commit'].unique().tolist()
                commit_metric_files = glob(os.path.join(config_global.UDB_PATH, project, '*.csv'))
                
                if len(commits_to_run) - len(commit_metric_files) > 1:
                    print(f"{project} too many commit_metric_files missing")
                    return None
                
                for commit_id in tqdm(commits_to_run, desc=f'{project}'):
                    commit_metric_dict = self.get_metrics_by_commit(project, commit_id)  # metrics from understand tool
                    if not commit_metric_dict:
                        continue
        
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
                            clone_metrics = self.search_clone(commit_metric_dict, clone_str)
        
                            if clone_metrics is not None:
                                # only need the metrics on method level
                                clone_metrics = {key: val for key, val in clone_metrics.items() if
                                                 key in config_global.METRIC_COLUMNS}
                                metric_on_group += Counter(clone_metrics)  # aggregate the metrics for clone group
        
                                # not_missing_output_writer.writerow([commit_id, clone_str])
                                # not_missing_commit_clone = not_missing_commit_clone.append(
                                # pd.DataFrame({'commit_id': [commit_id], 'clone_str': [clone_str]}), ignore_index=True)
                            else:
                                print("not in dict: ", commit_id, ",", clone_str, ",", commit_metric_dict.keys())
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
                                und_metric_on_group_df = pd.concat([und_metric_on_group_df, metric_on_group_dict_df], ignore_index=True)
        
            und_metric_on_group_df.to_csv(group_metric_und_path, index=False)
            return und_metric_on_group_df


class Other_metrics_extracter():

    def __init__(self, project):
        self.project = project
        self.blame_cache = defaultdict(set)  # Initialize a cache
        self.programming_lang = Git_repo.get_programming_language(self.project)
        self.clone_class_dict_4_clone = load_clone_class_dict_4_clone(project)

        ## read in commits only related to clone groups
        group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, f'{project}_group_genealogy_distinct.csv')
        self.genealogy_df = pd.read_csv(group_genealogy_path)
        print(project, self.genealogy_df.shape, self.genealogy_df.columns)

    
    # html
    def get_contributors_by_clonefile(self, github_base_url, commit_id, clone_file_path):
        from posixpath import join
        clone_contributor_url = join(github_base_url, "contributors-list", commit_id, clone_file_path.replace('\\', '/'))
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


    # PyGithub
    def get_contributors_by_clone(self, repo, commit_id, clone_file_path, start_line, end_line):
        cache_key = f"{commit_id}:{clone_file_path}:{start_line}:{end_line}"
        
        # Check if result is cached
        if cache_key in self.blame_cache:
            return self.blame_cache[cache_key]
        
        authors = set()

        try:
            # Run blame command
            blame_info = repo.git.blame("-e", f"-L {start_line},{end_line}", commit_id, '--', clone_file_path).splitlines()
            for line in blame_info:
                author_email = line.split('(')[1].split(' ')[0].strip()[1:-1].strip()
                authors.add(author_email)
                #authors[author] = authors.get(author, 0) + 1 # the #lines the author made
        except Exception as e:
            return set()

        # Cache the result
        self.blame_cache[cache_key] = authors

        return authors


    def process_commit(self, commit, repo):
        commit_id, _, _, clone_group_variant = commit.split(":", 3)
        clone_siblings = clone_group_variant.split("|")
        contributors_for_commit = set()
    
        for clone in clone_siblings:
            #print("clone_file_path: ", clone)
            clone_file_path = clone.split(":")[0]
            start_line, end_line = clone.split(":")[1].split('-')
            clone_contributors = self.get_contributors_by_clone(repo, commit_id, clone_file_path, start_line, end_line)
            contributors_for_commit.update(clone_contributors)
         
        return contributors_for_commit
    

    def process_row(self, row, repo):
        start_commit = row['start_commit']
        clone_group_tuple = row['clone_group_tuple']
        genealogy = row['genealogy']
    
        group_contributors = set()
        commits = genealogy.split(';')

        # filter commit
        commit_modifications_dict = Git_repo.get_commits_with_modifications(project, self.programming_lang)
        commits = [commit for commit in commits if commit.split(":", 1)[0] in list(commit_modifications_dict.keys())]
        if len(commits) > 1000:
            commits = random.sample(commits, 1000)
    
        with cf.ThreadPoolExecutor(max_workers=100) as executor:
            future_results = {executor.submit(self.process_commit, commit, repo): commit for commit in commits}
    
            for future in cf.as_completed(future_results):
                group_contributors.update(future.result())
    
        return group_contributors


    # refine the #contributors also retrieved from the start commit, we get all the #contributors along the genealogy that updated the clone
    def get_group_contributors(self, project, repo, genealogy_df):
        group_contributors_path = os.path.join(config_global.GROUP_METRIC_PATH,
                                               "%s_group_contributors.pkl" % project)
        if os.path.exists(group_contributors_path):
            with open(group_contributors_path, 'rb') as fp:
                return pickle.load(fp)
        else:
            group_contributors_dict = dict()
            with cf.ProcessPoolExecutor(max_workers=10) as process_executor:
                future_to_row = {process_executor.submit(self.process_row, row, repo): row for index, row in genealogy_df.iterrows()}
        
                group_contributors_dict = {} 
                for future in tqdm(cf.as_completed(future_to_row), total=len(future_to_row), desc=f'getting contributors on groups for {project}'):
                    row = future_to_row[future]
                    try:
                        group_contributors = future.result()
                        clone_group_tuple = row['clone_group_tuple']
                        group_contributors_dict[clone_group_tuple] = list(group_contributors)
                    except Exception as e:
                        print(f"Exception occurred: {e}")
    
            with open(group_contributors_path, 'wb') as fp:
                pickle.dump(group_contributors_dict, fp)
            return group_contributors_dict
        

    def get_path_metric_by_longest_common(self, file_paths):
        common_prefix = os.path.commonprefix(file_paths)
        longest_common_directory = os.path.dirname(common_prefix)
        return len(longest_common_directory.split(os.sep))


    # Calculate the Levenshtein distance between paths to see how similar or different they are. [Code Duplication and Reuse in Jupyter Notebooks]
    def get_path_metric_by_levenshtein_distance(self, file_paths):
        diversity_metric = sum(distance(a, b) for a in file_paths for b in file_paths) / (len(file_paths)**2)
        #print(f"!!!{file_paths} levenshtein {diversity_metric}")
        return diversity_metric


    # Calculate the entropy of the segments of paths. Higher entropy could mean higher diversity.
    def get_path_metric_by_entropy(self, file_paths):
        segments = [segment for file_path in file_paths for segment in file_path.split(os.path.sep)]
        segment_counts = Counter(segments)
        total = sum(segment_counts.values())
    
        entropy = -sum((count/total) * math.log2(count/total) for count in segment_counts.values())
        #print(f"!!!{file_paths} entropy {entropy}")
        return entropy


    # Use Jaccard Similarity to measure how similar paths are to each other. The smaller the Jaccard Similarity, the higher the diversity. 
    # [Code Reviewer Recommendation for Architecture Violations: An Exploratory Study]
    def get_path_metric_by_jaccard_similarity(self, file_paths):
        def jaccard_similarity(a, b):
            a, b = set(a.split(os.path.sep)), set(b.split(os.path.sep))
            return len(a & b) / len(a | b)

        total_similarity = sum(jaccard_similarity(a, b) for a in file_paths for b in file_paths)
        total_count = len(file_paths)**2
        
        average_similarity = total_similarity / total_count
        diversity_metric = 1 - average_similarity
        #print(f"!!!{file_paths} jaccard {diversity_metric}")
        return diversity_metric
    

    def get_path_metric_by_hamming_distance(self, file_paths):
    
        def pad_list(lst, length, padding=None):
            return lst + [padding] * (length - len(lst))
    
        def hamming_distance(a, b):
            return sum(el1 != el2 for el1, el2 in zip(a, b))
        
        # Split the file paths into their individual components
        split_paths = [path.split(os.path.sep) for path in file_paths]
        
        # Find the maximum length among all paths
        max_length = max(len(path) for path in split_paths)
        
        # Pad all paths to the maximum length
        padded_paths = [pad_list(path, max_length) for path in split_paths]
        
        # Calculate the Hamming distance for each pair of paths and sum them up
        total_hamming_distance = sum(
            hamming_distance(a, b) for a in padded_paths for b in padded_paths
        )
        
        # Normalize the result by dividing by the maximum possible Hamming distance
        num_pairs = len(file_paths) ** 2
        max_possible_hamming = max_length * num_pairs
        
        # Compute the diversity metric
        diversity_metric = total_hamming_distance / max_possible_hamming
        #print(f"!!!{file_paths} hamming {diversity_metric}")
        return diversity_metric


    def generate_other_metrics(self, project):
    
        ## read in commits only related to clone groups
        group_genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_PATH, f'{project}_group_genealogy_distinct.csv')
        genealogy_df = pd.read_csv(group_genealogy_path)
        print(project, genealogy_df.shape, genealogy_df.columns)

        # Path to your local repository
        project_local_repo = os.path.join(config_global.REPO_PATH, f'nicad_workdir_{project}', project)
        repo = git.Repo(project_local_repo)
        group_contributors_dict = self.get_group_contributors(project, repo, genealogy_df)
    
        project_contributor_df = Git_repo.get_github_contributors(project)
        project_contributor_df['contributor_createat'] = pd.to_datetime(project_contributor_df['contributor_createat'])
        project_contributor_df['contributor_years_experience'] = (pd.Timestamp('2023-05-01') - project_contributor_df['contributor_createat']).dt.days
        project_contributor_df['contributor_years_experience'] = project_contributor_df['contributor_years_experience'].astype(int)
        project_contributor_df['contributions'] = project_contributor_df['contributions'].astype(int)

        for index, row in tqdm(genealogy_df[['clone_group_tuple', 'start_commit']].iterrows(), total=genealogy_df.shape[0], desc='getting other metrics: '):
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
    
                func_name = self.clone_class_dict_4_clone[row['start_commit']][clone]
                func_paras = re.findall(r"[(](.*?)[)]", func_name)[0]
                cnt_func_paras = len(list(filter(None, func_paras.split(","))))
                cnt_group_paras += cnt_func_paras
    
            genealogy_df.loc[index, 'cnt_group_paras'] = int(cnt_group_paras / cnt_clone_siblings)
            genealogy_df.loc[index, 'cnt_clone_siblings'] = cnt_clone_siblings

            # get common_path of clone siblings
            clone_siblings_paths = [clone_sibling.split(':')[0] for clone_sibling in clone_siblings]
            genealogy_df.loc[index, 'path_longest_common'] = self.get_path_metric_by_longest_common(clone_siblings_paths) # len_common_path
            genealogy_df.loc[index, 'path_levenshtein_distance'] = self.get_path_metric_by_levenshtein_distance(clone_siblings_paths)
            genealogy_df.loc[index, 'path_entropy'] = self.get_path_metric_by_entropy(clone_siblings_paths)
            genealogy_df.loc[index, 'path_jaccard_similarity'] = self.get_path_metric_by_jaccard_similarity(clone_siblings_paths)
            genealogy_df.loc[index, 'path_hamming_distance'] = self.get_path_metric_by_hamming_distance(clone_siblings_paths)
            
            cnt_group_followers, cnt_group_experience, cnt_group_contributions = 0, 0, 0
            group_contributors = group_contributors_dict[row['clone_group_tuple']]
            genealogy_df.loc[index, 'cnt_distinct_contributors'] = len(group_contributors)
    
            for contributor_email in group_contributors:
                try:
                    # Find cnt_followers based on contributor_name
                    matched = project_contributor_df[project_contributor_df['emails'].apply(lambda x: contributor_email in x)]
                    if matched.empty:
                        email_name_map_df = Git_repo.get_github_contributors_email_name(project)
                        contributor_name_matched_df = email_name_map_df[email_name_map_df['email']==contributor_email]
                        if contributor_name_matched_df.empty:
                            print(f"!!! contributor_name_matched_df.empty: {contributor_email}")
                            cnt_group_followers += 0
                            cnt_group_experience += 0
                            cnt_group_contributions += 0
                        else:
                            for contributor_name in contributor_name_matched_df['name']:
                                name_matched = project_contributor_df[project_contributor_df['names'].apply(lambda x: contributor_name in x)]
                                if name_matched.empty:
                                    print(f"!!! name not found: {contributor_email}")
                                    cnt_group_followers += 0
                                    cnt_group_experience += 0
                                    cnt_group_contributions += 0
                                else:
                                    contributor_login = name_matched['contributor_login'].tolist()[0]
                                    cnt_followers = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['cnt_followers'].tolist()[0]
                                    cnt_experience = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributor_years_experience'].tolist()[0]
                                    cnt_contributions = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributions'].tolist()[0]

                                    cnt_group_followers += int(cnt_followers)
                                    cnt_group_experience += int(cnt_experience)
                                    cnt_group_contributions += int(cnt_contributions)

                                    break
                    else:
                        contributor_login = matched['contributor_login'].tolist()[0]

                        cnt_followers = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['cnt_followers'].tolist()[0]
                        cnt_experience = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributor_years_experience'].tolist()[0]
                        cnt_contributions = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributions'].tolist()[0]

                        cnt_group_followers += int(cnt_followers)
                        cnt_group_experience += int(cnt_experience)
                        cnt_group_contributions += int(cnt_contributions)

                except Exception as err:
                    print(f'faild contributor: {contributor_email} because of {err}')
                    sys.exit(-1)

            genealogy_df.loc[index, 'cnt_group_followers'] = int(cnt_group_followers / cnt_clone_siblings)
            genealogy_df.loc[index, 'cnt_group_experience'] = int(cnt_group_experience / cnt_clone_siblings)
            genealogy_df.loc[index, 'cnt_group_contributions'] = int(cnt_group_contributions / cnt_clone_siblings)
    
        genealogy_df.drop(['start_commit', 'genealogy'], axis=1, inplace=True)
        other_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_other_metric.csv' % project)
        genealogy_df.to_csv(other_metric_on_group_path, index=False)
        return genealogy_df
    

    def load_other_metrics(self, project):
        # loading other metrics file
        other_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_other_metric.csv' % project)
        if os.path.exists(other_metric_on_group_path):
            other_metric_on_group_df = pd.read_csv(os.path.normpath(other_metric_on_group_path))
        else:
            other_metric_on_group_df = self.generate_other_metrics(project)
    
        return other_metric_on_group_df


def get_clone_class(project):
    clone_class_dict_4_clone = defaultdict(defaultdict)
    project_clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH,
                                                      '%s_clone_result_purified_with_paratype.json' % project)
    # load clone classes
    with open(project_clone_result_purified_path, 'r') as clone_jsonfile:
        clone_result_json = json.load(clone_jsonfile, object_pairs_hook=OrderedDict)

        for commit_id in clone_result_json:
            for clone_group in clone_result_json[commit_id]:
                for clone in clone_group:
                    if clone[0].lower().find('test') == -1:  # filter out test methods
                        clone_signiture = ':'.join(clone[:2])  # clone[2] is the function name
                        clone_class_dict_4_clone[commit_id][clone_signiture] = clone[2]

    return clone_class_dict_4_clone


# given clone_path and clone_range, retrieve clone_name
def load_clone_class_dict_4_clone(project):
    clone_class_dict_4_clone = defaultdict(defaultdict)
    clone_class_dict_4_clone_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH,
                                                 "%s_clone_class_dict_4_clone.pkl" % project)
    if os.path.exists(clone_class_dict_4_clone_path):
        with open(clone_class_dict_4_clone_path, 'rb') as handle:
            clone_class_dict_4_clone = pickle.load(handle)
    else:
        print("not exists clone_class_dict_4_clone_path")
        clone_class_dict_4_clone = get_clone_class(project)
        with open(clone_class_dict_4_clone_path, 'wb') as handle:
            pickle.dump(clone_class_dict_4_clone, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return clone_class_dict_4_clone


def combine_und_other_metrics(project):
    merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
    
    und_metrics_extracter = Understand_metrics_group_extracter(project)
    und_metric_on_group_df = und_metrics_extracter.load_undstand_metrics(project)
    
    other_metrics_extracter = Other_metrics_extracter(project)
    other_metric_on_group_df = other_metrics_extracter.load_other_metrics(project)

    if (und_metric_on_group_df is not None) and (other_metric_on_group_df is not None):
        merged_df = pd.merge(und_metric_on_group_df, other_metric_on_group_df, on='clone_group_tuple', how='inner')
        merged_df.to_csv(merged_metric_on_group_path, index=False)
       

if __name__ == '__main__':
    projects_all = set(list(config_global.SUBJECT_SYSTEMS_YOUNG.keys()) + list(config_global.SUBJECT_SYSTEMS_MIDDLE.keys()) + list(config_global.SUBJECT_SYSTEMS_OLD.keys())) - set(['shardingsphere-elasticjob'])
    
    for project in projects_all:
        combine_und_other_metrics(project)