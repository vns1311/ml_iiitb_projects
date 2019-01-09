# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:58:03 2018

@author: vnshy
"""

import pandas as pd
import numpy as np
import numpy.random as rand
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys
import pickle
import os


seq2 = pd.Series(np.arange(2))

def get_correlation_plot(feature_df,output_folder):
    import seaborn as sns
    sns.set(style="white")
    plt.figure(figsize=(36,20))
    heatmap = sns.heatmap(feature_df.corr(), square=True, linewidths=.3)
    plt.savefig(output_folder + "/Correlaton.png")
    plt.close()

def get_metrics(cutoff, y_proba, y_test):
    '''
    Function to compute the various metrics required for plotting the Precision Recall Curve
    '''
    predict = [int(x > cutoff) for x in y_proba]
    cm = confusion_matrix(y_test, predict)
    Sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    Specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    Accuracy = accuracy_score(y_test, predict)
    return [Specificity, Sensitivity, Accuracy, cutoff, abs(Specificity - Sensitivity)]


def plotROC(y_test, y_proba, output_folder,model_name):
    """
        Function to plot the ROC Curce
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # ROC-AUC
    plt.figure(figsize=(12, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_folder + "/" + model_name + "_AUC.png")
    plt.close()
    return roc_auc


def get_optimum_cutoff(y_test, y_proba, output_folder, model_name):
    """
    Function to compute the optimum Cutoff
    """
    
    cutoff = np.arange(0.0, 1.0, 0.01)
    results = []
    for i in range(0, len(cutoff)):
        results.append(get_metrics(cutoff[i], y_proba, y_test))
    results = pd.DataFrame(results)
    np.apply_along_axis(sorted, 1, results[[4]])
    plt.figure(figsize=(10, 6))
    plt.plot(results[[3]], results[[0]], "green")
    plt.plot(results[[3]], results[[1]], "red")
    plt.plot(results[[3]], results[[2]], "blue")
    plt.text(0.2, 0.8, r'Cutoff = ' + str(results.loc[results[4].idxmin()][3]))
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.savefig(output_folder + "/" + model_name + "_Cutoff.png")
    plt.close()
    return results.loc[results[4].idxmin()][3]


def plot_confusion_matrix(cm, classes,output_folder,fname,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_folder + "/" + fname)
    plt.close()


def compute_cost(fp, fn):
    cost = 10 * fp + 500 * fn
    return cost


class APS(object):
    seq2 = pd.Series(np.arange(2))

    """docstring for APS"""

    def __init__(self, trainFile, testFile):
        self.trainFile = trainFile
        self.testFile = testFile
        self.__lr = LogisticRegression()
        self.__rforest = RandomForestClassifier()
        self.__lightgm = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.predicted_labels = None
        
    def getFinalTrainFeatures(self):
        print("Dropping Columns which have more than 0.1 Empty Data")
        num_rows = self.train_data.shape[0]
        num_nas = self.train_data.isnull().sum()
        num_nas_vals = num_nas.values/num_rows
        column_list = self.train_data.columns.values.tolist()
        for col_index in range(self.train_data.shape[1]):
            if num_nas_vals[col_index] > 0.1:
                self.train_data.pop(column_list[col_index])
        print("Imputing Median Values for other missing values")
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        for column in self.train_data:
            self.train_data[[column]] = imputer.fit_transform(self.train_data[[column]])


    def getFinalTestFeatures(self):
        print("Imputing Median Values for other missing values")
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        for column in self.test_data:
            self.test_data[[column]] = imputer.fit_transform(self.test_data[[column]])
    
    @staticmethod
    def balanceTrainData(train_data, train_labels):
        """
        Function to apply SMOTE on the training data to balance the classes
        """
        sm = SMOTE(random_state=42)
        col_names = train_data.columns
        train_data, train_labels = sm.fit_sample(train_data, train_labels)
        train_data = pd.DataFrame(train_data, columns=col_names)
        return train_data, train_labels

    def trainingData(self):
        df = pd.read_csv(self.trainFile)
        print("Replacing na with NAN")
        df.replace({'na': np.nan}, inplace=True, regex=True)
        print("Converting Class Labels to Numeric Values")
        self.train_labels = df['class'].apply(lambda x: 0 if x == "neg" else 1)
        self.train_data = df.drop('class', axis=1)
        self.getFinalTrainFeatures()
        print("Final Training Data: " + str(self.train_data.shape[0]) + ", "+ str(self.train_data.shape[1]))

    def testingData(self):
        df = pd.read_csv(self.testFile)
        df.replace({'na': np.nan}, inplace=True, regex=True)
        self.test_labels = df['class'].apply(lambda x: 0 if x == "neg" else 1)
        self.test_data = df.drop('class', axis=1)[self.train_data.columns]
        self.getFinalTestFeatures()
        print("Final Test Data: " + str(self.test_data.shape[0]) + ", "+ str(self.test_data.shape[1]))


    
    def data(self):
        print("Loading Training Data from file")
        self.trainingData()
        print("Loading Test Data from file")
        self.testingData()

    def trainLogisticRegression(self,output_folder):
        # Splitting the data into train and test to decide the cutoff
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.train_labels, test_size=0.2, stratify=self.train_labels, random_state=42)
        X_train_res, y_train_res = self.balanceTrainData(X_train, y_train)
        self.__lr.fit(X_train_res,y_train_res)    
        y_pred = self.__lr.predict(X_test)
        y_pred_proba = self.__lr.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm,np.array([0,1]),output_folder,"cm_lr_default",title="Confusion Matrix - Default Threshold")
        # Getting the Precision Recal Curve and figuring out the optimal cutoff
        cutoff = get_optimum_cutoff(y_test,y_pred_proba,output_folder,"LR")
        y_pred_cutoff = np.where(y_pred_proba>cutoff,1,0)
        cm_cutoff = confusion_matrix(y_test, y_pred_cutoff)
        plot_confusion_matrix(cm_cutoff,np.array([0,1]),output_folder,"cm_lr_custom",title="Confusion Matrix - Custom Threshold")
        plotROC(y_test,y_pred_proba,output_folder,"LR")
        pickle.dump(self.__lr, open(output_folder + "/Model_LR.sav", 'wb'))
        print("Custom Threshold: "+str(cutoff))
        return cutoff

    def testLogisticRegression(self,cutoff,output_folder):
        y_proba = self.__lr.predict_proba(self.test_data)[:, 1]
        self.predicted_labels = np.where(y_proba>cutoff,1,0)
        cm = confusion_matrix(self.test_labels, self.predicted_labels)
        tn,fp,fn,tp=cm.ravel()
        plot_confusion_matrix(cm,np.array([0,1]),output_folder,"cm_lr_test",title="Confusion Matrix - Threshold="+str(cutoff))
        print("Optimised Cost with Custom Threshold - "+str(compute_cost(fp,fn)))
        raw_test_data = pd.read_csv(self.testFile)
        raw_test_data["predicted_class"] = self.predicted_labels
        raw_test_data["predicted_class"] = raw_test_data["predicted_class"].apply(lambda x: "neg" if x == 0 else "pos")
        raw_test_data.to_csv(output_folder + "/lr_test_results.csv")
  
    def trainRandomForest(self, output_folder):
        # Setting the parameters for the RF Classifier
        self.__rforest.n_estimators=200
        self.__rforest.class_weight='balanced'
        self.__rforest.random_state=42
        self.__rforest.n_jobs=-1
        # Splitting the data into train and test to decide the cutoff
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.train_labels, test_size=0.2, stratify=self.train_labels, random_state=42)
        X_train_res, y_train_res = self.balanceTrainData(X_train, y_train)
        self.__rforest.fit(X_train_res, y_train_res)
        y_pred = self.__rforest.predict(X_test)
        y_pred_proba = self.__rforest.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm,np.array([0,1]),output_folder,"cm_rf_default",title="Confusion Matrix - Default Threshold")
        # Getting the Precision Recal Curve and figuring out the optimal cutoff
        cutoff = get_optimum_cutoff(y_test,y_pred_proba,output_folder,"RF")
        y_pred_cutoff = np.where(y_pred_proba>cutoff,1,0)
        cm_cutoff = confusion_matrix(y_test, y_pred_cutoff)
        plot_confusion_matrix(cm_cutoff,np.array([0,1]),output_folder,"cm_rf_custom",title="Confusion Matrix - Custom Threshold")
        plotROC(y_test,y_pred_proba,output_folder,"RF")
        pickle.dump(self.__rforest, open(output_folder + "/Model_RF.sav", 'wb'))
        print("Custom Threshold: "+str(cutoff))
        return cutoff

    def testRandomForest(self,cutoff,output_folder):
        y_proba = self.__rforest.predict_proba(self.test_data)[:, 1]
        self.predicted_labels = np.where(y_proba>cutoff,1,0)
        cm = confusion_matrix(self.test_labels, self.predicted_labels)
        tn,fp,fn,tp=cm.ravel()
        plot_confusion_matrix(cm,np.array([0,1]),output_folder,"cm_rf_test",title="Confusion Matrix - Threshold="+str(cutoff))
        print("Optimised Cost with Custom Threshold - "+str(compute_cost(fp,fn)))
        raw_test_data = pd.read_csv(self.testFile)
        raw_test_data["predicted_class"] = self.predicted_labels
        raw_test_data["predicted_class"] = raw_test_data["predicted_class"].apply(lambda x: "neg" if x == 0 else "pos")
        raw_test_data.to_csv(output_folder + "/rf_test_results.csv")

 
if __name__ == "__main__":
    """
    To trigger this code, use the below syntax
    python APS_Falure_Prediction.py -t <Train File Path> -p <Test File Path> -o <Output Folder>
    Train File Path - Path where the training file resides
    Test File Path - Path where the test file resides
    Output Folder - Directory Path where the graph and the pickled model will be available. Will be created if not existing
    Classifier - Type of Classifier to run. Either LR or RF.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", help="Path to TRAIN DATA FILE",required=True)
    parser.add_argument("-p", "--test", help="Path to TEST DATA FILE",required=True)
    parser.add_argument("-o", "--output", help="Output Directory For Plots and Saved Model",required=True)
    parser.add_argument("-c", "--classifier",help="Classifier Type(RF, LR)",default="RF",required=False)
    #parser.add_argument("-m", "--model",help="Model Pickle File",required=False)
    args = parser.parse_args()
    train_data_name = args.train
    test_data_name = args.test
    plot_folder = args.output
    if(not os.path.exists(train_data_name) or not os.path.exists(test_data_name)):
        print("Specified Train/Test File doesnt exist")
        exit(1)
    if(not os.path.exists(plot_folder)):
        print("Output Directory doesnt exist. Creating it now")
        os.mkdir(plot_folder)
    model = APS(train_data_name,test_data_name)
    model.data()
    get_correlation_plot(model.train_data,plot_folder)
    print("Model Building")
    if args.classifier == "RF":
        cutoff = model.trainRandomForest(plot_folder)
        print("Model Evaluation on the test file")
        model.testRandomForest(cutoff,plot_folder)
    elif args.classifier == "LR":
        cutoff = model.trainLogisticRegression(plot_folder)
        print("Model Evaluation on the test file")
        model.testLogisticRegression(cutoff,plot_folder)
    else:
        print("Invalid Classifier Selected")