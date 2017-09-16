#!/usr/bin/python

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
### Task 3: Create new feature(s)
for name in data_dict:
    total_emails = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
    total_poi_related_emails = data_dict[name]['from_poi_to_this_person'] + data_dict[name]['from_this_person_to_poi']
    if (total_emails not in [0, 'NaN', 'NaNNaN']) and data_dict[name]['from_poi_to_this_person'] != 'NaN' and data_dict[name]['from_this_person_to_poi'] != 'NaN':
        data_dict[name]['poi_related_ratio'] = total_poi_related_emails / total_emails
    else:
        data_dict[name]['poi_related_ratio'] = 'NaN'
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
clf = GaussianNB()


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_rescaled =  scaler.fit_transform(features)

# Select KBest
from sklearn.feature_selection import SelectKBest
kbest = SelectKBest(k=5)
features = kbest.fit_transform(features_rescaled, labels)
print "\n", kbest.scores_

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(features_rescaled)
print pca.explained_variance_ratio_
first_pc = pca.components_[0]
second_pc = pca.components_[1]

# Train
clf.fit(features, labels)
# Prediction
pred = clf.predict(features)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels)
print "\n", accuracy

# Experimentation with other classifiers (Pipeline).
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scale", MinMaxScaler()),("select", SelectKBest(k=9)),("reduce", PCA(n_components=3)), ("classify", GaussianNB())])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)