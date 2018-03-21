import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score
from matplotlib import pyplot as plt
import sklearn
import pandas

### define parameters
TRAIN_PATH = "../20k_dataset/train.csv"
TEST_PATH = "../20k_dataset/test.csv"
SEED = 1024




def get_data_from_csv(data_path, feature_label):
    df = pandas.read_csv(data_path)

    df = df.dropna()
    clean_mat_X = df.as_matrix(feature_label)
    clean_mat_Y = df.as_matrix(['label'])
    return (clean_mat_X, clean_mat_Y.flatten())

def prepare_dataset(train_path,test_path):

    ## get feature label
    df = pandas.read_csv(train_path)
    feature_label = list(df)[:6]

    # feature_label.remove('img_src')
    # feature_label.remove('label')

    X_train, Y_train = get_data_from_csv(train_path,feature_label)
    X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train, random_state=SEED)

    X_test, Y_test = get_data_from_csv(test_path, feature_label)
    X_test, Y_test = sklearn.utils.shuffle(X_test, Y_test, random_state=SEED)

    ## normalization
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)

    ##TODO:(mean,var) need to be saved in the production
    # print(scaler.mean_)
    # print(scaler.var_)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train,Y_train,X_test,Y_test

def  model_selection(model_name):

    if model_name == "RF":
        clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    elif model_name == 'CatBoost':
        clf = cb.CatBoostClassifier(eval_metric="AUC", depth=11, iterations=500, l2_leaf_reg= 9, learning_rate= 0.15)
    elif model_name == 'XGBoost':
        clf = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200, n_jobs=-1 , verbose=1,learning_rate=0.16)
    else:
        ## lightGBM
        clf = lgb.LGBMClassifier(max_depth=50,learning_rate=0.1,num_leaves=900,n_estimators=300)
    return {"model":clf,'name':model_name}

def model_evaluation(Y_true,Y_pred,name):

    ### ROC curve
    auc = roc_auc_score(Y_true,Y_pred)
    false_positive, true_positive, _= roc_curve(Y_true, Y_pred)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive, true_positive, color='darkorange', label=name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (auc = %0.4f)' % auc)
    plt.legend(loc='best')
    plt.show()

    ### metrics

    print name,"AUC:", auc


if __name__=="__main__":

    ### prepare dataset
    X_train,Y_train,X_test,Y_test = prepare_dataset(TRAIN_PATH,TEST_PATH)

    ### select model
    model_dic = model_selection('light')

    ### train model
    model_dic['model'].fit(X_train, Y_train)

    ### test model

    Y_pred = model_dic['model'].predict_proba(X_test)[:,1]

    ### evaluate model
    model_evaluation(Y_test,Y_pred,model_dic['name'])




