import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import pickle
from xgboost import XGBClassifier

import CrimeDataExplorer


class CrimeClassifier(object):
    def __init__(self, train_data_path, test_data_path, output_folder):
        self.__lr = LogisticRegression()
        self.__rforest = RandomForestClassifier()
        self.__xgb = XGBClassifier()
        self.__nb = BernoulliNB()
        self.train_labels = None
        self.raw_train_labels = None
        self.cat_encoder = LabelEncoder()
        self.test_labels = None
        self.predicted_labels = None
        self.predicted_proba = None
        self.train_file = train_data_path
        self.test_file = test_data_path
        self.train_data = None
        self.test_data = None
        self.output_dir = output_folder

    @staticmethod
    def split_date_fields(data):
        data["Year"] = data["Dates"].dt.year
        data["Month"] = data["Dates"].dt.month
        data["Day"] = data["Dates"].dt.day
        data["Hour"] = data["Dates"].dt.hour
        data["Minute"] = data["Dates"].dt.minute
        return data

    def get_time_features(self, df_type):
        print("Adding Time Based Features")
        if df_type == "train":
            self.train_data = self.split_date_fields(self.train_data)
            self.train_data["Morning"] = self.train_data["Hour"].apply(lambda x: 1 if 6 <= x < 12 else 0)
            self.train_data["Noon"] = self.train_data["Hour"].apply(lambda x: 1 if 12 <= x < 17 else 0)
            self.train_data["Evening"] = self.train_data["Hour"].apply(lambda x: 1 if 17 <= x < 20 else 0)
            self.train_data["Night"] = self.train_data["Hour"].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
            self.train_data["Fall"] = self.train_data["Month"].apply(lambda x: 1 if 3 <= x <= 5 else 0)
            self.train_data["Winter"] = self.train_data["Month"].apply(lambda x: 1 if 6 <= x <= 8 else 0)
            self.train_data["Spring"] = self.train_data["Month"].apply(lambda x: 1 if 9 <= x <= 11 else 0)
            self.train_data["Summer"] = self.train_data["Month"].apply(lambda x: 1 if x >= 12 or x <= 2 else 0)
        elif df_type == "test":
            self.test_data = self.split_date_fields(self.test_data)
            self.test_data["Morning"] = self.test_data["Hour"].apply(lambda x: 1 if 6 <= x < 12 else 0)
            self.test_data["Noon"] = self.test_data["Hour"].apply(lambda x: 1 if 12 <= x < 17 else 0)
            self.test_data["Evening"] = self.test_data["Hour"].apply(lambda x: 1 if 17 <= x < 20 else 0)
            self.test_data["Night"] = self.test_data["Hour"].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
            self.test_data["Fall"] = self.test_data["Month"].apply(lambda x: 1 if 3 <= x <= 5 else 0)
            self.test_data["Winter"] = self.test_data["Month"].apply(lambda x: 1 if 6 <= x <= 8 else 0)
            self.test_data["Spring"] = self.test_data["Month"].apply(lambda x: 1 if 9 <= x <= 11 else 0)
            self.test_data["Summer"] = self.test_data["Month"].apply(lambda x: 1 if x >= 12 or x <= 2 else 0)

    def encode_category(self):
        self.cat_encoder.fit(self.train_labels)
        self.train_labels = self.cat_encoder.transform(self.train_labels)

    def get_address_features(self, df_type):
        print("Extracting Features from Address Field")
        add_encoder = LabelEncoder()
        if df_type == "train":
            self.train_data['StreetNo'] = self.train_data['Address'].apply(
                lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
            self.train_data["Intersection"] = self.train_data["Address"].apply(lambda x: 1 if "/" in x else 0)
            self.train_data['Address'] = self.train_data['Address'].apply(
                lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
            add_encoder.fit(self.train_data["Address"])
            self.train_data["Address"] = add_encoder.transform(self.train_data["Address"])
        elif df_type == "test":
            self.test_data['StreetNo'] = self.test_data['Address'].apply(
                lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
            self.test_data["Intersection"] = self.test_data["Address"].apply(lambda x: 1 if "/" in x else 0)
            self.test_data['Address'] = self.test_data['Address'].apply(
                lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
            add_encoder.fit(self.test_data["Address"])
            self.test_data["Address"] = add_encoder.transform(self.test_data["Address"])

    def training_data(self):
        print("Preparing Training Data")
        df = pd.read_csv(self.train_file, index_col=['Id'], parse_dates=['Dates'])
        self.train_data = df.drop(["Descript", "Resolution", "Category"], axis=1)
        self.raw_train_labels = df["Category"]
        self.train_labels = df["Category"]
        self.encode_category()
        self.get_time_features("train")
        self.get_address_features("train")
        self.train_data = pd.get_dummies(self.train_data, columns=['PdDistrict', 'DayOfWeek'])
        self.train_data = self.train_data.drop(['Dates', 'StreetNo', 'Address'], axis=1)
        print("Final Training Features", str(len(self.train_data.columns)))

    def testing_data(self):
        print("Preparing Test Data")
        df = pd.read_csv(self.test_file, index_col=['Id'], parse_dates=['Dates'])
        self.test_data = df.drop(["Descript", "Resolution"], axis=1)
        self.get_time_features("test")
        self.get_address_features("test")
        self.test_data = pd.get_dummies(self.test_data, columns=['PdDistrict', 'DayOfWeek'])
        self.test_data = self.test_data.drop(['Dates','StreetNo', 'Address'], axis=1)
        print("Final Test Features", str(len(self.train_data.columns)))

    def data(self):
        self.training_data()
        self.testing_data()

    def write_processed_data(self):
        print("Writing Final Processed Data for future use")
        if not os.path.exists(self.output_dir + "/processed_data"):
            print("Output Directory doesnt exist. Creating it now")
            os.mkdir(self.output_dir + "/processed_data")
        processed_dir = self.output_dir + "/processed_data"
        processed_train = self.train_data.copy()
        processed_train['Category'] = self.raw_train_labels
        processed_train.to_csv(processed_dir + "/train.csv", index=True)
        self.test_data.to_csv(processed_dir + "/test.csv", index=True)
        print("Writing Done")

    def plot_variable_importance(self):
        feature_names = self.train_data.columns
        importances = self.__rforest.feature_importances_
        indices = np.argsort(importances)
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(self.output_dir + "/rf_variable_importance.png")

    def train_random_forest(self):
        print("Random Forest Training")
        self.__rforest.set_params(n_estimators=40)
        self.__rforest.set_params(min_samples_split=100)
        self.__rforest.fit(self.train_data, self.train_labels)
        self.plot_variable_importance()
        print("Training log loss", str(log_loss(self.train_labels, self.__rforest.predict_proba(self.train_data))))
        pickle.dump(self.__rforest, open(self.output_dir + "/Model_RF.sav", 'wb'))

    def test_random_forest(self):
        print("Predicting Random Forest")
        final_test_pred = self.__rforest.predict_proba(self.test_data)
        submission = pd.DataFrame(final_test_pred, columns=self.cat_encoder.classes_)
        submission['Id'] = self.test_data.index.tolist()
        cols_at_start = ['Id']
        submission = submission[
            [c for c in cols_at_start if c in submission] + [c for c in submission if c not in cols_at_start]]
        submission.head()
        submission.to_csv(self.output_dir + "RF_Submission.csv", index=False)
        print("Done")

    def train_naive_bayes(self):
        print("Naive Bayes Training")
        self.__nb.set_params(alpha=1.0)
        self.__nb.set_params(fit_prior=True)
        self.__nb.fit(self.train_data, self.train_labels)
        print("Training log loss", str(log_loss(self.train_labels, self.__nb.predict_proba(self.train_data))))
        pickle.dump(self.__nb, open(self.output_dir+"/Model_NB.sav", "wb"))

    def test_naive_bayes(self):
        final_test_pred = self.__nb.predict_proba(self.test_data)
        submission = pd.DataFrame(final_test_pred, columns=self.cat_encoder.classes_)
        submission['Id'] = self.test_data.index.tolist()
        cols_at_start = ['Id']
        submission = submission[
            [c for c in cols_at_start if c in submission] + [c for c in submission if c not in cols_at_start]]
        submission.head()
        submission.to_csv(self.output_dir + "NB_Submission.csv", index=False)
        print("Done")

    def train_logistic_regression(self):
        print("Logistic Regression Training")
        self.__lr.set_params(multi_class="multinomial")
        self.__lr.set_params(solver="saga")
        self.__lr.set_params(penalty="l1")
        self.__lr.fit(self.train_data,self.train_labels)
        print("Training log loss", str(log_loss(self.train_labels, self.__lr.predict_proba(self.train_data))))
        pickle.dump(self.__nb, open(self.output_dir + "/Model_LR.sav", "wb"))

    def test_logistic_regression(self):
        final_test_pred = self.__lr.predict_proba(self.test_data)
        submission = pd.DataFrame(final_test_pred, columns=self.cat_encoder.classes_)
        submission['Id'] = self.test_data.index.tolist()
        cols_at_start = ['Id']
        submission = submission[
            [c for c in cols_at_start if c in submission] + [c for c in submission if c not in cols_at_start]]
        submission.head()
        submission.to_csv(self.output_dir + "LR_Submission.csv", index=False)
        print("Done")

    def train_xgb(self):
        print("XGBoost Training")
        # data_dmatrix = self.__xgb.DMatrix(data=self.train_data, label=self.train_labels)
        self.__xgb.set_params(max_depth=3)
        self.__xgb.set_params(n_estimators=300)
        self.__xgb.set_params(learning_rate=0.05)
        self.__xgb.fit(self.train_data,self.train_labels)
        print("Training log loss", str(log_loss(self.train_labels, self.__xgb.predict_proba(self.train_data))))
        pickle.dump(self.__nb, open(self.output_dir + "/Model_XGB.sav", "wb"))

    def test_xgb(self):
        final_test_pred = self.__xgb.predict_proba(self.test_data)
        submission = pd.DataFrame(final_test_pred, columns=self.cat_encoder.classes_)
        submission['Id'] = self.test_data.index.tolist()
        cols_at_start = ['Id']
        submission = submission[
            [c for c in cols_at_start if c in submission] + [c for c in submission if c not in cols_at_start]]
        submission.head()
        submission.to_csv(self.output_dir + "XGB_Submission.csv", index=False)
        print("Done")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--train", help="Path to TRAIN DATA FILE", required=True)
    parser.add_argument("-p", "--test", help="Path to TEST DATA FILE", required=True)
    parser.add_argument("-o", "--output", help="Output Directory For Plots and Saved Model", required=True)
    # parser.add_argument("-c", "--classifier", help="Classifier Type(RF, LR)", default="RF", required=False)
    # parser.add_argument("-m", "--model",help="Model Pickle File",required=False)
    args = parser.parse_args()
    train_data_name = args.train
    test_data_name = args.test
    plot_folder = args.output
    if not os.path.exists(train_data_name) or not os.path.exists(test_data_name):
        print("Specified Train/Test File doesnt exist")
        exit(1)
    if not os.path.exists(plot_folder):
        print("Output Directory doesnt exist. Creating it now")
        os.mkdir(plot_folder)
    model = CrimeClassifier(train_data_name, test_data_name, plot_folder)
    explorer = CrimeDataExplorer.CrimeDataExplorer(train_data_name, plot_folder)
    explorer.explore_data()
    model.data()
    model.write_processed_data()
    explorer.get_correlation_plot(model.train_data)

    model.train_random_forest()
    model.test_random_forest()

    model.train_logistic_regression()
    model.test_logistic_regression()

    model.train_naive_bayes()
    model.test_naive_bayes()

    model.train_xgb()
    model.test_xgb()
 