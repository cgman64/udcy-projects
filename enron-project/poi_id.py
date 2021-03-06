#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'long_term_incentive','loan_advances',
                 'deferred_income','exercised_stock_options',
                 'total_stock_value', 'from_poi_to_this_person', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi',
                 'from_messages', 'to_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace(to_replace='NaN', value=np.nan, inplace=True)
df = df[features_list]
# Remove negative numbers
df = df.abs()
# Outlier detection
# Ref: http://stamfordresearch.com/outlier-removal-in-python-using-iqr-rule/
for feat in features_list[1:]:
    q75, q25 = np.percentile(df[feat].dropna(), [75, 25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print feat.upper(), "outliers:\n"
    print "Under 25th percentile:"
    if len(df.loc[df[feat] < min,feat]) == 0:
        print "None"
    else:
        print df.loc[df[feat] < min,feat]
    print "\nOver 75th percentile:"
    if len(df.loc[df[feat] > max,feat]) == 0:
        print "None"
    else:
        print df.loc[df[feat] > max,feat], "\n"
# Null ratio
null_ratio = pd.DataFrame(df[features_list[1:]].isnull().sum() / len(df))
null_ratio.columns = ['null ratio']
null_ratio = null_ratio.round(3)
null_ratio = null_ratio.sort_values(by='null ratio', ascending=False)
print null_ratio
# Remove loan_advances feature
if 'loan_advances' in df.columns:
    df = df.drop(['loan_advances'], axis=1)
if 'loan_advances' in features_list:
    features_list.remove('loan_advances')

# Return data back to dictionary
df.replace(to_replace=np.nan, value='NaN', inplace=True)  
data_dict = df.to_dict('index')
### Task 2: Remove outliers
if 'TOTAL' in data_dict.keys():
    data_dict.pop('TOTAL', 0)
if 'THE TRAVEL AGENCY IN THE PARK' in data_dict.keys(): 
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Task 3: Create new feature(s)
for name in data_dict:
    #print "Adding Total Emails:", data_dict[name]['from_messages'], "and", data_dict[name]['to_messages']
    total_emails = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
    #print "Adding Total POI Emails:", data_dict[name]['from_poi_to_this_person'], "and", data_dict[name]['from_this_person_to_poi']
    total_poi_related_emails = data_dict[name]['from_poi_to_this_person'] + data_dict[name]['from_this_person_to_poi']
    if (total_emails not in [0, 'NaN', 'NaNNaN']) and data_dict[name]['from_poi_to_this_person'] != 'NaN' and data_dict[name]['from_this_person_to_poi'] != 'NaN':
        data_dict[name]['poi_email_ratio'] = total_poi_related_emails / total_emails
        #print "RATIO:", data_dict[name]['poi_email_ratio'] , "\n"
    else:
        data_dict[name]['poi_email_ratio'] = 'NaN'
        #print "RATIO:", 'NaN\n'

# Update features_list
features_list.append('poi_email_ratio')

#for name in data_dict:
#    point = data_dict[name]
#    print point['poi_email_ratio'], "type:", type(point['poi_email_ratio'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
from sklearn.feature_selection import SelectKBest

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

df.replace(to_replace='NaN', value=0, inplace=True)
features_list_df = np.array(df.columns)
data_df = featureFormat(my_dataset, features_list_df, sort_keys = True)
labels_df, features_df = targetFeatureSplit(data_df)

selector_df = SelectKBest(k=8)
selector_df.fit(features_df, labels)

selector_new = SelectKBest(k=8)
selector_new.fit(features, labels)


def print_scores(s, features):
    selected_feature_indices = s.get_support(indices=True)
    selected_features = [features[1:][i] for i in selected_feature_indices]
    kbest_scores = {}
    k = 0
    for feat in selected_features:
        kbest_scores[feat] = s.scores_[k]
        k += 1

    df_kbest_scores = pd.DataFrame.from_dict(kbest_scores, orient='index')
    df_kbest_scores.columns = ['score']
    return df_kbest_scores.sort_values(by='score', ascending=False)
# Feature scores without new feature
print_scores(selector_df, features_list)
# Feature scores with new feature
print_scores(selector_new, features_list)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

# Experimentation with other classifiers (Pipeline).
from sklearn.pipeline import Pipeline
psvm = Pipeline([("scale", MinMaxScaler()),("kbest", SelectKBest()),("pca", PCA()), ("svc", SVC())])
pgnb = Pipeline([("kbest", SelectKBest()), ("pca", PCA()), ("gnb", GaussianNB())])
pdt = Pipeline([("kbest", SelectKBest()), ("pca", PCA()), ("tree", DecisionTreeClassifier())])
prf = Pipeline([("kbest", SelectKBest()), ("pca", PCA()), ("rfc", RandomForestClassifier())])
plr = Pipeline([("kbest", SelectKBest()), ("pca", PCA()), ("lr", LogisticRegression())])

from sklearn.metrics import f1_score, precision_score, recall_score 
pdt.set_params(tree__min_samples_split=2, kbest__k=6, pca__n_components=3)
psvm.set_params(svc__kernel='rbf', kbest__k=6, pca__n_components=3)

from tester import test_classifier
if False:
    print "DECISION TREE:"
    test_classifier(pdt, data_dict, features_list)
    print "\nNAIVE BAYES:"
    test_classifier(pgnb, data_dict, features_list)
    print "\nSVC:"
    test_classifier(psvm, data_dict, features_list)
    print "\nRANDOM FOREST:"
    test_classifier(prf, data_dict, features_list)
    print "\nLOGISTIC REGRESSION:"
    test_classifier(plr, data_dict, features_list)

clf = Pipeline([("kbest", SelectKBest()), ("pca", PCA()), ("tree", DecisionTreeClassifier(random_state=42))])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Ref: http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#sphx-glr-auto-examples-plot-digits-pipe-py
# Ref: https://stackoverflow.com/questions/45394527/do-i-need-to-split-data-when-using-gridsearchcv
from sklearn.model_selection import GridSearchCV
parameters = {'tree': {'kbest__k': range(6, 10),'pca__n_components': range(2,6), 'tree__min_samples_split':[2, 5, 7, 9]}}
# Splitting method change to be more consistent with tester.py
# Ref: https://discussions.udacity.com/t/how-can-i-get-tester-py-to-print-multiple-accuracy-and-recall-values-for-documentation/301255
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(random_state=42)
gs = GridSearchCV(clf, param_grid=parameters['tree'], scoring='f1', cv=sss)
gs.fit(features_train, labels_train)
if True:
    # Selected features
    selected_feature_indices = gs.best_estimator_.named_steps['kbest'].get_support(indices=True)
    selected_features = [features_list[1:][i] for i in selected_feature_indices]
    # feature importances for decision tree
    feat_imp = gs.best_estimator_.named_steps['tree'].feature_importances_

clf = gs.best_estimator_
# Showing Scores of kbest features and feature importances
scores = clf.named_steps['kbest'].scores_
selected_feature_indices = clf.named_steps['kbest'].get_support(indices=True)
selected_features = [features_list[1:][i] for i in selected_feature_indices]
kbest_scores = {}
k = 0
for feat in selected_features:
    kbest_scores[feat] = scores[k]
    k += 1

df_kbest_scores = pd.DataFrame.from_dict(kbest_scores, orient='index')
df_kbest_scores.columns = ['score']
print df_kbest_scores.sort_values(by='score', ascending=False)

# Evaluation on 30% of data
from sklearn.metrics import classification_report
pred = gs.predict(features_test)
print "CLASSIFICATION REPORT:\n", classification_report(labels_test, pred)

test_classifier(clf, data_dict, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)