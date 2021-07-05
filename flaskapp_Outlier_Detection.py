from flask import Flask ,render_template
import pandas as pd
import datetime as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib
import pickle


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import optimizers,metrics,losses
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_validate
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler



app = Flask(__name__)
matplotlib.use('Agg')
balanceData= False # Change to False to use normal data
sampling_ratio= 0.5
# cross validation parts for classification
cross_val_splits=10



def read_preprocess_data(file):
    print ("in read process")
    data = pd.read_table(file)
    print ("read complete")
    data.columns = data.columns.str.strip()
    print ("preprocessing ")

    # a_no : calling no ; b_no : dialled_no ;
    data.dropna(inplace=True)
    data.drop(columns=["calling_number","dialled_number"], inplace=True)
    # time preprocessing
    data['start_time']=pd.to_datetime(data['start_time']).map(dt.datetime.toordinal)
    data['Label'] = np.where((data['c_number'] != data['a_number']) & (data['transaction_duration']< 6000) & (data["root_failure"]==2) & (data["conversation_time"]==0), 1, 0)
    print("Datset size", data.shape)
    
    #for classification
    X = data.drop('Label', 1)
    y = data['Label']
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(X)
    #X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(scaled_data, y, test_size=0.2)

    #for outlier detection
    X_normal_events = data[data['Label'] == 0].drop('Label', 1)
    X_abnormal_events = data[data['Label'] == 1].drop('Label', 1)
    y_normal_events = data[data['Label'] == 0]["Label"]
    y_abnormal_events = data[data['Label'] == 1]["Label"]
    
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_normal_events)
    scaled_normal_data = scaler.transform(X_normal_events)
    scaled_abnormal_data = scaler.transform(X_abnormal_events)
    X_train, X_test, y_train, y_test = train_test_split(scaled_normal_data, y_normal_events, test_size=0.2, random_state=42)
    X_test = np.concatenate((X_test, scaled_abnormal_data))
    y_test = np.concatenate((y_test, y_abnormal_events))
    print (X_train.shape, y_train.shape,X_test.shape , y_test.shape)
    print ("data ratio", y.value_counts())
    print ("preprocessing complete")
    
    return scaled_data, y,X_train, X_test, y_train, y_test

def balanced_data(file):
    print ("in balance read process")
    data = pd.read_table(file)
    print ("read complete")
    data.columns = data.columns.str.strip()
    print ("preprocessing ")

    # a_no : calling no ; b_no : dialled_no ;
    data.dropna(inplace=True)
    data.drop(columns=["calling_number","dialled_number"], inplace=True)
    # time preprocessing
    data['start_time']=pd.to_datetime(data['start_time']).map(dt.datetime.toordinal)
    data['Label'] = np.where((data['c_number'] != data['a_number']) & (data['transaction_duration']< 6000) & (data["root_failure"]==2) & (data["conversation_time"]==0), 1, 0)
    #balancing 
    print ("Balancing begin")
    X = data.drop('Label', 1)
    y = data['Label']
    oversample = RandomOverSampler(sampling_strategy=sampling_ratio)
    X_over, y_over = oversample.fit_resample(X, y)
    print ("Balancing complete")
    data= X_over.copy()
    data["Label"]=y_over
    
    #for classification
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(X_over)
    #X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(scaled_data, y, test_size=0.2)

    #for outlier detection
    X_normal_events = data[data['Label'] == 0].drop('Label', 1)
    X_abnormal_events = data[data['Label'] == 1].drop('Label', 1)
    y_normal_events = data[data['Label'] == 0]["Label"]
    y_abnormal_events = data[data['Label'] == 1]["Label"]
    
    scaler = preprocessing.MinMaxScaler()
    scaled_normal_data = scaler.fit_transform(X_normal_events)
    scaled_abnormal_data = scaler.transform(X_abnormal_events)
    X_train, X_test, y_train, y_test = train_test_split(scaled_normal_data, y_normal_events, test_size=0.2, random_state=42)
    X_test = np.concatenate((X_test, scaled_abnormal_data))
    y_test = np.concatenate((y_test, y_abnormal_events))
    print (X_train.shape, y_train.shape,X_test.shape , y_test.shape)
    print ("data ratio", y.value_counts())
    print ("preprocessing complete")
    
    return scaled_data, y_over,X_train, X_test, y_train, y_test
    

def conf_matrix(ytest ,predicted , name):
    cf_matrix= confusion_matrix(ytest, predicted)
    #print(name,cf_matrix,ytest.shape, predicted.shape)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    #print(group_counts)
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    #print(group_percentages)
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    #print("Labels",labels)
    #print(cf_matrix)
    #ax_osvm = plt.axes()
    #ax_osvm[0].set_ylabel('Actual')
    #ax_osvm[1].set_ylabel('Predicted')
    plt.clf()
    sns.heatmap(cf_matrix, annot=labels , fmt='', annot_kws={"size": 16})
    plt.title(name)
    plt.savefig('static/images/'+name+'.png')
    
def confusion_matrix_scorer_cv(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
             'fn': cm[1, 0], 'tp': cm[1, 1]} 

def confusion_matrix_classification(cv_results, name):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    cf_cv_matrix=np.array([[np.sum(cv_results['test_tn']),np.sum(cv_results['test_fp'])],[np.sum(cv_results['test_fn']),np.sum(cv_results['test_tp'])]])
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_cv_matrix.flatten()]
    #print(group_counts)
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_cv_matrix.flatten()/np.sum(cf_cv_matrix)]
    print(group_percentages)
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)
    #print("Labels",labels)
    #print(cf_cv_matrix)
    #ax_osvm = plt.axes()
    #ax_osvm[0].set_ylabel('Actual')
    #ax_osvm[1].set_ylabel('Predicted')
    plt.clf()
    sns.heatmap(cf_cv_matrix, annot=labels , fmt='', annot_kws={"size": 16})
    plt.title(name)
    plt.savefig('static/images/'+name+'.png')
    return cf_cv_matrix
                        
def calc_eval_metrics(cf_test_matrix):
    tp= cf_test_matrix[1][1]
    tn= cf_test_matrix[0][0]
    fp = cf_test_matrix[0][1]
    fn = cf_test_matrix[1][0]
    total = tp + fp +fn+ tn
    p= tp + fn
    n= tn +fp
    #print("Total",total, "tn",tn,"tp",tp,"fp",fp,"fn",fn)
    acc = (tp + tn)/total
    prec = tp / ( tp + fp ) 
    rec = tp / p
    f1 = 2 * prec* rec/ (prec+rec)
    return acc,prec,rec,f1

if balanceData:
    scaled_data, y,X_train, X_test, y_train, y_test = balanced_data("dummy_set.xlsx")
else : 
    scaled_data, y,X_train, X_test, y_train, y_test = read_preprocess_data("dummy_set.xlsx")


@app.route('/')
@app.route('/home', methods=["GET", "POST"])
def home():
    return render_template('home.html')

@app.route('/OSVM', methods=["GET", "POST"])
def OSVM():
    try:
        print ("OSVM model fit")
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        model = OneClassSVM(kernel = 'rbf', gamma = 'auto', nu = 0.02).fit(X_train)
        print ("OSVM train prediction")
        y_pred_train = model.predict(X_train)
        y_pred_train = np.where(y_pred_train == -1, 1, 0)
        
        conf_matrix(y_train,y_pred_train , "OSVM_performance_train")
        print ("OSVM test prediction")
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
        conf_matrix(y_test,y_pred , "OSVM_performance_test")
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test,y_pred)), "{0:.2%}".format(f1_score(y_test,y_pred)), "{0:.2%}".format(precision_score(y_test,y_pred)) , "{0:.2%}".format(recall_score(y_test,y_pred))
        #print(accuracy, f1 , precision , recall)
    
        return render_template('home.html', url_test ='/static/images/OSVM_performance_test.png' ,url_train= '/static/images/OSVM_performance_train.png' ,accuracy = accuracy, f1=f1 , precision =precision , recall =recall ,algo ="One Class Support Vector Machine")
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
@app.route('/iForest', methods=["GET", "POST"])
def iForest():
    try:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print ("iforest model fit")
        model = IsolationForest(n_estimators=150 ).fit(X_train)
    
        y_pred_train = model.predict(X_train)
        y_pred_train = np.where(y_pred_train == -1, 1, 0)
        conf_matrix(y_train,y_pred_train , "iForest_performance_train")
        #5typrint ("ifore",X_train.shape, y_train.shape,X_test.shape , y_test.shape)
        print ("OSVM test prediction")
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
        #print (y_pred.shape)
        conf_matrix(y_test,y_pred , "iForest_performance_test")
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test,y_pred)), "{0:.2%}".format(f1_score(y_test,y_pred)), "{0:.2%}".format(precision_score(y_test,y_pred)) , "{0:.2%}".format(recall_score(y_test,y_pred))
        pickle.dump(model, open('models/final_prediction_iforest.pickle', 'wb'))


        return render_template('home.html', url_test ='/static/images/iForest_performance_test.png' ,url_train= '/static/images/iForest_performance_train.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Isolation Forest" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))

@app.route('/AutoEnc', methods=["GET", "POST"])
def AutoEnc():
    try:
        n_features = X_train.shape[1]
        # model
        encoder = keras.Sequential(name='encoder')
        encoder.add(layer=layers.Dense(units=20, activation="relu", input_shape=[n_features]))
        encoder.add(layers.Dropout(0.5))
        encoder.add(layer=layers.Dense(units=20, activation="relu"))
        encoder.add(layer=layers.Dense(units=10, activation="relu"))

        decoder = keras.Sequential(name='decoder')
        decoder.add(layer=layers.Dense(units=20, activation="relu", input_shape=[10]))
        decoder.add(layer=layers.Dense(units=20, activation="relu"))
        #decoder.add(layer=layers.Dense(units= n_features, activation="relu"))
        #decoder.add(layers.Dropout(0.1))
        decoder.add(layer=layers.Dense(units=n_features, activation="sigmoid"))

        autoencoder = keras.Sequential([encoder, decoder])

        autoencoder.compile(loss=losses.MSE,optimizer=optimizers.Adam(),metrics=[metrics.mean_squared_error])
        print ("Auto encoder model fit")
        history = autoencoder.fit(x=X_train, y=X_train, epochs=20, verbose=0, validation_data=(X_test, X_test))
        print ("Auto encoder model prediction")
        X_pred_train = autoencoder.predict(X_train)
        train_events_mse = losses.mean_squared_error(X_train, X_pred_train)
        cut_off = np.percentile(train_events_mse, 95)
    
        #print("Cut off", cut_off)
        
        predicted_test= autoencoder.predict(X_test)
        test_mse = losses.mean_squared_error(X_test, predicted_test)
        
        pred_test = np.where(test_mse > cut_off, 1, 0)
        pred_train = np.where(train_events_mse > cut_off, 1, 0)
        

        conf_matrix(y_train,pred_train , "AutoEncoder_performance_train")
        
        conf_matrix(y_test,pred_test , "AutoEncoder_performance_test")
        
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test,pred_test)), "{0:.2%}".format(f1_score(y_test,pred_test)), "{0:.2%}".format(precision_score(y_test,pred_test)) , "{0:.2%}".format(recall_score(y_test,pred_test))
    
        return render_template('home.html', url_test ='/static/images/AutoEncoder_performance_test.png' ,url_train= '/static/images/AutoEncoder_performance_train.png', accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="AutoEncoder" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))

@app.route('/naiveBayes', methods=["GET", "POST"])
def naiveBayes():
    try:
        modelNB = GaussianNB()
        print ("Naive Bayes model ")
        cv_results = cross_validate(modelNB,scaled_data, y, cv=cross_val_splits
                                    ,scoring=confusion_matrix_scorer_cv)
        cf_test_matrix = confusion_matrix_classification(cv_results , "naiveBayes_performance_test")
        acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        return render_template('home.html', url_test ='/static/images/naiveBayes_performance_test.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Naive Bayes" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
@app.route('/randomForest', methods=["GET", "POST"])
def randomForest():
    try:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #print("RF",y_train_c.value_counts())
        #kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 

        #for train_index, test_index in kf.split(X):
       #     X_train, X_test = X[train_index], X[test_index] 
      #      y_train, y_test = y[train_index], y[test_index]
        #print("Train:", train_index, "Validation:",test_index)
        print ("Random Forest model fit")
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(scaled_data, y, test_size=0.2)
 
        modelRC = RandomForestClassifier(n_estimators=150).fit(X_train_c,y_train_c)
 
        y_pred_train_c = modelRC.predict(X_train_c)
 
        y_pred_c = modelRC.predict(X_test_c)
 
        conf_matrix(y_train_c,y_pred_train_c , "randomForest_performance_train")
 
        conf_matrix(y_test_c, y_pred_c , "randomForest_performance_test")
 
        accuracy, f1 , precision , recall = "{0:.2%}".format(accuracy_score(y_test_c,y_pred_c)), "{0:.2%}".format(f1_score(y_test_c,y_pred_c)), "{0:.2%}".format(precision_score(y_test_c,y_pred_c)) , "{0:.2%}".format(recall_score(y_test_c,y_pred_c))
 
        print("Random Forest results", accuracy, f1 , precision , recall)
 
        return render_template('home.html', url_test ='/static/images/randomForest_performance_test.png' ,url_train= '/static/images/randomForest_performance_train.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Random Forest" )
 
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
        #cv_results = cross_validate(modelRC,scaled_data, y, cv=cross_val_splits
        #                            ,scoring=confusion_matrix_scorer_cv)
        #cf_test_matrix = confusion_matrix_classification(cv_results , "randomForest_performance_test")
        #acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        #accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        #print("Random Forest results", accuracy, f1 , precision , recall)
        #return render_template('home.html', url_test ='/static/images/randomForest_performance_test.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Random Forest" )
    #except Exception as e:
     #   return render_template('home.html', message="Error recieved :" + str(e))

@app.route('/SVC', methods=["GET", "POST"])
def SVC():
    try:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #print("SVC",y_train_c.value_counts())
        print ("SVC model fit")
        modelSVC = svm.SVC(C=1.5)
        cv_results = cross_validate(modelSVC,scaled_data, y, cv=cross_val_splits
                                    ,scoring=confusion_matrix_scorer_cv)
        cf_test_matrix = confusion_matrix_classification(cv_results , "SVM_performance_test")
        acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        print("SVC results", accuracy, f1 , precision , recall)
        return render_template('home.html', url_test ='/static/images/SVM_performance_test.png', accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="Support Vector Machine" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
@app.route('/xgboost', methods=["GET", "POST"])
def xgboost():
    try:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #print("SVC",y_train_c.value_counts())
        print ("XGBoost model fit")
        modelxgb = xgb.XGBClassifier()
        cv_results = cross_validate(modelxgb,scaled_data, y, cv=cross_val_splits
                                    ,scoring=confusion_matrix_scorer_cv)
        cf_test_matrix = confusion_matrix_classification(cv_results , "xgboost_performance_test")
        acc,prec,rec,f1=calc_eval_metrics(cf_test_matrix)
        accuracy, f1 , precision , recall = "{0:.2%}".format(acc), "{0:.2%}".format(f1), "{0:.2%}".format(prec) , "{0:.2%}".format(rec)
        print("xgboost results", accuracy, f1 , precision , recall)
        return render_template('home.html', url_test ='/static/images/XGBoost_performance_test.png' , accuracy = accuracy, f1=f1 , precision =precision , recall =recall , algo ="XGBoost Algorithm" )
    except Exception as e:
        return render_template('home.html', message="Error recieved :" + str(e))
    
    