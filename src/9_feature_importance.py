import eli5, os, sys, pickle
from eli5.sklearn import PermutationImportance
sys.path.append("..")
from config import config_global, model_config
import pandas as pd
import eli5, os, sys
import pandas as pd
import pickle
from glob import glob
from eli5.sklearn import PermutationImportance
sys.path.append("..")
from config import config_global, model_config


def load_df(project):
    reusable_clone_path = os.path.join(config_global.DATASET_PATH, "%s_raw_dataset.csv" % project)
    reusable_clone_df = pd.read_csv(reusable_clone_path)

    #model_path = os.path.normpath(model_config.model_results[project])
    model_path = os.path.join(config_global.MODEL_PATH, "%s_RandomForestClassifier.pkl" % project)
    print(model_path)
    with open(model_path, 'rb') as fp:
        grid_search_model, x_train, y_train, x_test, y_test, auc_score = pickle.load(fp)
        best_model = grid_search_model.best_estimator_

        perm = PermutationImportance(best_model, random_state=1, scoring='roc_auc').fit(x_test, y_test)
        eli5.show_weights(perm, feature_names=x_test.columns.tolist())

        #y_pred = grid_search_model.predict(x_test)
        #y_df = pd.DataFrame({'idx': y_test.index, 'label': y_test.values})
        #y_df['pred'] = y_pred
    #return y_df, reusable_clone_df


if __name__=='__main__':
    projects = config_global.SUBJECT_SYSTEMS.keys()
    projects = ['checkstyle', 'druid', 'framework', 'gatk', 'graylog2-server',
                'grpc-java', 'jabref', 'k', 'k-9', 'minecraftForge', 'molgenis',
                'muikku', 'netty', 'openhab1-addons', 'presto', 'product-apim',
                'realm-java', 'reddeer', 'RxJava', 'smarthome', 'spring-boot',
                'Terasology', 'XChange', 'xp', 'zaproxy', 'pinpoint'
                ]

    cnt = 0
    feature_imp_df_all = []
    for proj in projects:
        model_path_re = os.path.join(config_global.MODEL_PATH_202208, "20220819_%s_*.pkl" % proj)
        model_path_list = glob(model_path_re)
        best_auc = max(
            [float(os.path.basename(model_path).split('_')[-1].rsplit(".", 1)[0]) for model_path in model_path_list])

        if best_auc < 0.7:
            print("low auc of %s" % proj)
        else:
            cnt += 1
            best_model_path = [model_path for model_path in model_path_list if str(best_auc) in model_path][0]
            # print("best_model_path: ", best_model_path)

        with open(best_model_path, 'rb') as fp:
            grid_search_model, x_train, y_train, x_test, y_test, auc_score = pickle.load(fp)

        best_model = grid_search_model.best_estimator_
        perm = PermutationImportance(best_model, random_state=1, scoring='roc_auc').fit(x_test, y_test)
        # eli5.show_weights(perm, feature_names = x_test.columns.tolist())

        # x = eli5.explain_weights(perm, feature_names = x_test.columns.tolist())
        feature_imp_df = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names=x_test.columns.tolist(),
                                                                         top=3)
        feature_imp_df_all.append(feature_imp_df)

    res = pd.concat(feature_imp_df_all)
    res['feature'].value_counts()




