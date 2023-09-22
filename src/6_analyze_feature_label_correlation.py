import common

features = ['CountInput', 'CountLine', 'CountLineBlank',
       'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
       'CountLineComment', 
       'CountOutput', 'CountPath', 'CountSemicolon',
       'CountStmt', 'CountStmtDecl', 'CountStmtExe',
       'Cyclomatic', 'CyclomaticModified', 'CyclomaticStrict', 'Essential',
       'Knots',  'MaxEssentialKnots',
        'MaxNesting', 'MinEssentialKnots',
        'RatioCommentToCode', 'SumCyclomatic',
       'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
       'cnt_group_paras', 'cnt_clone_siblings', 'path_longest_common',
       'path_levenshtein_distance', 'path_entropy', 'path_jaccard_similarity',
       'path_hamming_distance', 'cnt_distinct_contributors',
       'cnt_group_followers']

features_toremove = ['CountLineCodeExe', 'CountLineCode', 'CyclomaticModified', 'CyclomaticStrict', 
                     'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'CyclomaticModified', 'CyclomaticStrict', 'CountSemicolon']

features = list(set(features) - set(features_toremove))

features_round2 = ['CountPath', 'Knots', 'path_jaccard_similarity', 'CountLine', 'cnt_distinct_contributors', 'cnt_clone_siblings', 'path_longest_common', 'CountInput', 'cnt_group_paras', 'CountLineBlank', 'MaxNesting', 'path_entropy', 'CountStmt', 'CountOutput', 'CountLineComment', 'CountStmtDecl', 'CountLineCodeDecl', 'Essential', 'cnt_group_followers']
features_toremove = ['MaxEssentialKnots', 'MinEssentialKnots', 'CountStmtExe', 'RatioCommentToCode', 'path_hamming_distance', 
                     'SumEssential', 'Cyclomatic', 'path_levenshtein_distance', 'CountStmt', 'CountLineCodeDecl']

features_round3 = list(set(features_round2) - set(features_toremove))
print(features_round3)

features_toremove = ['MaxEssentialKnots', 'MinEssentialKnots', 'CountStmtExe', 'RatioCommentToCode', 'path_hamming_distance', 
                     'SumEssential', 'Cyclomatic', 'path_levenshtein_distance', 'CountStmt', 'CountLineCodeDecl', 'MaxNesting', 'Knots']
features_round4 = list(set(features_round3) - set(features_toremove))

# Convert DataFrame to HTML and add bold to cells with values > 0.7
def bold_gt_07(val):
    if abs(val) > 0.7:
        return f'<b>{val:.2f}</b>'
    else:
        return f'{val:.2f}'


def get_corr_matrix(round, features):
    print(f'features: {features}')
    corr_df_path = os.path.join(config_global.DATA_PATH, 'feature_correlation', f'corr_df_{round}.csv')
    if os.path.exists(corr_df_path):
        corr_df = pd.read_csv(corr_df_path)
        corr_df['corr-value-abs'] = corr_df['corr-value'].abs()
        print(corr_df.shape)
        return corr_df
    else:
        files = glob("/home/20cy3/topic1/clone2api/data/dataset/20230912_*_raw_dataset_0.5_0.5_0.5.csv")
        print(len(files))
        corr_df_list = []
        for file in files:
            df = pd.read_csv(file)
            # spearman_corr_matrix = df[features].corr(method='spearman')
            spearman_corr_matrix = df[features_round4].corr(method='spearman')
        
            # Generate HTML table
            html_table = spearman_corr_matrix.applymap(bold_gt_07).to_html(escape=False)
            
            # Display HTML table in Jupyter Notebook
            print(HTML(html_table))
            
            # Loop through the upper triangular part of the correlation matrix to find pairs with correlation greater than 0.7
            for i, feature1 in enumerate(spearman_corr_matrix.index):
                for j, feature2 in enumerate(spearman_corr_matrix.columns):
                    if i >= j:  # Skip diagonal and lower triangular part of matrix
                        continue
                    corr_value = spearman_corr_matrix.iloc[i, j]
                    
                    #if abs(corr_value) > 0.7:
                    pair = '-'.join(sorted([feature1, feature2]))
                    project = os.path.basename(file).split("_", 2)[1]
                    tuple = (project, pair, corr_value)
                    corr_df_list.append(tuple)
                    print(f"The correlation between {feature1} and {feature2} is {corr_value:.2f}")
    
        corr_df = pd.DataFrame(corr_df_list, columns=['project', 'feature-pair', 'corr-value'])
        corr_df['corr-value-abs'] = corr_df['corr-value'].abs()
        print(corr_df.head(5))
        corr_df.to_csv(corr_df_path, index=False)
        return corr_df
    

def check_correlation_features(corr_df):
    corr_df_grouped = corr_df.groupby('feature-pair')['corr-value-abs'].mean().reset_index()
    corr_df_grouped_sorted = corr_df_grouped.sort_values(by='corr-value-abs', ascending=False)
    corr_df_grouped_sorted_gt70 = corr_df_grouped_sorted[corr_df_grouped_sorted['corr-value-abs']>=0.7]
    return corr_df_grouped_sorted_gt70


def check_correlation_features_iterately():
    features = ['cnt_distinct_contributors', 'path_longest_common', 'CountPath', 'CountOutput', 'Essential', 'cnt_group_followers', 'CountLineComment', 'CountStmtDecl', 'path_jaccard_similarity', 'CountInput', 'cnt_group_paras', 'CountLine', 'CountLineBlank']
    round = 0
    corr_df = get_corr_matrix(5, features)
    corr_df_grouped_sorted_gt70 = check_correlation_features(corr_df)
    while not corr_df_grouped_sorted_gt70.empty:
        round += 1
        print(f"round: {round} \n features: {features}")
        corr_df = get_corr_matrix(5, features)
        corr_df_grouped_sorted_gt70 = check_correlation_features(corr_df)
        user_input = input("Please enter features to remove, splitted by a space: ")
        features_toremove = user_input.split()
        print(f"features to remove: {features_toremove}")
        features = list(set(features) - set(features_toremove))


def check_correlation_dimensions():
    def highlight_cells(val):
        if abs(val) > 0.7:
            return 'font-weight: bold'
        else:
            return ''

    pd.set_option('display.max_rows', None)
    projects_all = list(config_global.SUBJECT_SYSTEMS_YOUNG.keys()) + list(config_global.SUBJECT_SYSTEMS_MIDDLE.keys()) + list(config_global.SUBJECT_SYSTEMS_OLD.keys())
    spearmanr_list = []
    label_list = []
    for project in projects_all:
        # print("project: ", project)
        label_path = os.path.join(config_global.LABEL_PATH, f'20230912_{project}_3label_0.5_0.5_0.5.csv')
        if not os.path.exists(label_path):
            continue
    
        project_label_df = pd.read_csv(label_path)
        label_list.append(project_label_df)
    
        # For Spearman correlation
        # correlation_spearman = project_label_df['n_genealogy'].corr(project_label_df['cnt_siblings'], method='spearman')
        # Calculate the pairwise Spearman correlation for the three columns
        cor_AB, p_AB = spearmanr(project_label_df['n_genealogy'], project_label_df['cnt_siblings'])
        cor_AC, p_AC = spearmanr(project_label_df['n_genealogy'], project_label_df['bug_proneness'])
        cor_BC, p_BC = spearmanr(project_label_df['cnt_siblings'], project_label_df['bug_proneness'])
        # print(f"Spearman correlation between A and B: {cor_AB} {cor_AC} {cor_BC}")
    
        cor_AB, p_AB = spearmanr(project_label_df['rank_by_n_genealogy'], project_label_df['rank_by_prevalence'])
        cor_AC, p_AC = spearmanr(project_label_df['rank_by_n_genealogy'], project_label_df['rank_by_bugproneness'])
        cor_BC, p_BC = spearmanr(project_label_df['rank_by_prevalence'], project_label_df['rank_by_bugproneness'])
        spearmanr_tuple = (project, cor_AB, cor_AC, cor_BC)
        spearmanr_list.append(spearmanr_tuple)
    
    final_df = pd.concat(label_list, ignore_index=True)
    cor_AB, p_AB = spearmanr(final_df['rank_by_n_genealogy'], final_df['rank_by_prevalence'])
    cor_AC, p_AC = spearmanr(final_df['rank_by_n_genealogy'], final_df['rank_by_bugproneness'])
    cor_BC, p_BC = spearmanr(final_df['rank_by_prevalence'], final_df['rank_by_bugproneness'])
    print(cor_AB, cor_AC, cor_BC)
    
    df = pd.DataFrame(spearmanr_list, columns=['project', 'lifecycle-prevalence', 'lifecycle-bugproneness', 'prevalence-bugproneness',])
    # Apply the Styler
    styled_df = df.style.applymap(highlight_cells, subset=['lifecycle-prevalence', 'lifecycle-bugproneness', 'prevalence-bugproneness'])
    print(styled_df)
    

if __name__ == '__main__':
    check_correlation_dimensions()
    # main()