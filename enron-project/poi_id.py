#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'long_term_incentive','loan_advances',
                 'deferred_income','exercised_stock_options',
                 'total_stock_value', 'from_poi_to_this_person', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi',
                 'from_messages', 'to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

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
from time import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
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

clf = Pipeline([("kbest", SelectKBest()), ("pca", PCA()), ("tree", DecisionTreeClassifier())])

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
    # feature importances
    feat_imp = gs.best_estimator_.named_steps['tree'].feature_importances_

clf = gs.best_estimator_

test_classifier(clf, data_dict, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)