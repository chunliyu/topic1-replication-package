import os, multiprocessing, pickle, json, sys
sys.path.append("..")
from multiprocessing import Pool
from config import config_global, model_config
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE #,ADASYN
from collections import Counter

'''
不同模型在一个project表现好
20个project上也consistent
某个模型表现好无意义，专注点在reusable clone
'''

def load_projects_dataframe(projects):
    # read in all the dataframes
    project_df_dict = {}
    for project in projects:
        reusable_clone_path = os.path.join(config_global.DATASET_PATH, "%s_raw_dataset_20230118_0.5.csv" % project)
        reusable_clone_df = pd.read_csv(reusable_clone_path)
        reusable_clone_df = reusable_clone_df[config_global.FEATURES]
        project_df_dict[project] = reusable_clone_df
    return project_df_dict


def load_data_cross(project, project_df_dict, threshold=0.5):
    training_projects = list(filter(lambda x: x != project, project_df_dict.keys()))
    print("training_projects: ", training_projects)
    training_clones_list = [project_df_dict[proj] for proj in training_projects]
    training_clone_df = pd.concat(training_clones_list)
    y_train = training_clone_df['is_reusable']
    x_train = training_clone_df.drop('is_reusable', axis=1)
    y_test = project_df_dict[project]['is_reusable']
    x_test = project_df_dict[project].drop('is_reusable', axis=1)
    return x_train, x_test, y_train, y_test


def load_data(project, threshold=0.5):
    # load project data
    # reusable_clone_path = os.path.join(config_global.DATASET_PATH, "%s_raw_dataset.csv" % project)
    reusable_clone_path = os.path.join(config_global.DATASET_PATH, '%s_raw_dataset_20230118_%s.csv' % (project, threshold))
    reusable_clone_df = pd.read_csv(reusable_clone_path)
    reusable_clone_df = reusable_clone_df[config_global.FEATURES]
    print(reusable_clone_df.shape)
    x = reusable_clone_df.drop('is_reusable', axis=1)
    y = reusable_clone_df['is_reusable']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def grid_search(x_train, y_train, model, model_paras):
    #rcv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) # 10 * 10 repeated k-fold cv
    rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)  # 10 * 10 repeated k-fold cv

    grid_search = GridSearchCV(estimator=model #  #xgb0
                               , param_grid=model_paras   # PARAM_GRID_XGB
                               , cv=rcv  #, cv=10
                               , scoring='roc_auc'
                               #,return_train_score=True
                               , refit=True
                               , n_jobs=-1
                               ,verbose = 0
                               )

    grid_search.fit(x_train.values, y_train)

    # directly used the returned best_estimator model from cross-validation to predict testing datasets with roc_auc_score provided by cross-validation
    best_model = grid_search.best_estimator_

    return grid_search, best_model


def fine_tune_within(project, model, model_paras, threshold):
    print(project, threshold)
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    x_train, x_test, y_train, y_test = load_data(project, threshold)
    #x, y = sm.fit_resample(x, y)
    # counter1 = Counter(y_train)
    # print(counter1)

    x_train, y_train = sm.fit_resample(x_train, y_train)
    #counter2 = Counter(y_train_sm)
    #print(counter2)
    grid_search_model, best_model = grid_search(x_train, y_train, model, model_paras)

    #auc_score = best_model.score(x_test, y_test)
    auc_score_refit = grid_search_model.score(x_test, y_test)
    #auc_score_sklearn = roc_auc_score(y_test, grid_search_model.predict(x_test))
    # save auc
    classifier = type(model).__name__
    model_dict = dict()
    #model_dict[classifier] = auc_score
    print(classifier, auc_score_refit)
    #print(classifier, auc_score, auc_score_refit, auc_score_sklearn)

    # save checkpoint testing file
    tuple_objects = (grid_search_model, x_train, y_train, x_test, y_test, auc_score_refit)
    tuple_objects_path = os.path.join(config_global.MODEL_PATH, "20230118_%s_%s_%s_%s.pkl" % (project, classifier, threshold, auc_score_refit))
    pickle.dump(tuple_objects, open(tuple_objects_path, 'wb'))

    return classifier, auc_score_refit
    # load tuple
    # (grid_search_model, x_train, y_train, x_test, y_test, auc_score) = pickle.load(open("tuple_model.pkl", 'rb'))


def fine_tune_cross(project, model, model_paras, project_df_dict, threshold=0.5):
    print(project, threshold)

    # todo merge with fine-tune
    x_train, x_test, y_train, y_test = load_data_cross(project, project_df_dict, threshold)
    # counter1 = Counter(y_train)
    # print(counter1)

    sm = SMOTE(sampling_strategy='auto', random_state=42)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    #counter2 = Counter(y_train_sm)
    #print(counter2)
    grid_search_model, best_model = grid_search(x_train, y_train, model, model_paras)

    #auc_score = best_model.score(x_test, y_test)
    auc_score_refit = grid_search_model.score(x_test, y_test)
    #auc_score_sklearn = roc_auc_score(y_test, grid_search_model.predict(x_test))
    # save auc
    classifier = type(model).__name__
    model_dict = dict()
    #model_dict[classifier] = auc_score
    print(classifier, auc_score_refit)
    #print(classifier, auc_score, auc_score_refit, auc_score_sklearn)

    # save checkpoint testing file
    tuple_objects = (grid_search_model, x_train, y_train, x_test, y_test, auc_score_refit)
    tuple_objects_path = os.path.join(config_global.MODEL_PATH_CROSS, "cross_%s_%s_%s_%s.pkl" % (project, classifier, threshold, auc_score_refit))
    pickle.dump(tuple_objects, open(tuple_objects_path, 'wb'))

    return classifier, auc_score_refit
    # load tuple
    # (grid_search_model, x_train, y_train, x_test, y_test, auc_score) = pickle.load(open("tuple_model.pkl", 'rb'))


if __name__=='__main__':
    # with Pool(20) as pool:
    #    pool.map(fine_tune, config_global.PROJECTS)
    # clone_class_dict_4_clone = defaultdict(defaultdict)

    model_dict = {
        #'DecisionTrieeClassifier': model_config.PARAM_GRID_DT,
        #'RandomForestClassifier': model_config.PARAM_GRID_RF
        'XGBClassifier': model_config.PARAM_GRID_XGB,
        #'CatBoostClassifier': model_config.PARAM_GRID_CAT
        #'LGBMClassifier': model_config.PARAM_GRID_LGBM,
        #'AdaBoostClassifier': model_config.PARAM_GRID_ADA
        #'SVC': model_config.PARAM_GRID_SVC
    }

    perf_df = pd.DataFrame(columns=['project', 'classifier', 'auc'])

    # projects = ['RIOT'] # ['redis']
    projects = [ 'cleanflight', 'inav', 'collectd', 'libgit2', 'micropython', 'john', 'netdata', 'zfs', 'mpv', 'lxc']

    # projects = [config_global.PROJECT]
    #for project in [config_global.PROJECT]:
    project_df_dict = load_projects_dataframe(projects) # used for cross-project clones

    for project in tqdm(projects): #'muikku']: #'MinecraftForge']: #'netty', 'smarthome', 'framework', 'druid', 'checkstyle', 'gatk', 'grpc-java', 'jabref']:
        for classifier in model_dict:
            model = DecisionTreeClassifier(random_state=0)
            if classifier == 'DecisionTreeClassifier':
                model = DecisionTreeClassifier(random_state=0)
            elif classifier == 'RandomForestClassifier':
                model = RandomForestClassifier(oob_score=True, random_state=42)  # n_estimators=10000
            elif classifier == 'XGBClassifier':
                model = XGBClassifier()
            elif classifier == 'CatBoostClassifier':
                model = CatBoostClassifier(bootstrap_type='Bayesian', verbose=0)
            elif classifier == 'LGBMClassifier':
                model = LGBMClassifier(objective='binary', random_state=5)
            elif classifier == 'AdaBoostClassifier':
                model = AdaBoostClassifier(random_state=0)
            elif classifier == 'SVC':
                model = svm.SVC(random_state=42)

            model_paras = model_dict[classifier]
            print(model_paras)
            # classifier, auc_score_refit = fine_tune_within(project, model, model_paras, config_global.threshold)
            classifier, auc_score_refit = fine_tune_cross(project, model, model_paras, project_df_dict, config_global.threshold)
            perf_df.loc[len(perf_df)] = [project, classifier, auc_score_refit]

    perf_df.to_csv('result_spring-boot1.csv')
