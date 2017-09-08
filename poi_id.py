#!/usr/bin/python

import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean

# %matplotlib inline

sys.path.append("../tools/")

from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation



######## Task 1: Select what features you'll use. #################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'

features_list = ['poi',
                 'salary',
                 'total_payments',
                 'bonus',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',]
                 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Convert data into a pandas df    
df = pd.DataFrame.from_dict(data_dict, orient = 'index' , dtype = float)    

#==============================================================================
### Load the pkl file *** Investigate the pkl files ***
""" find how many POIs are in the dataset"""
i = 0
for key in data_dict:
    # value of poi field is type Boolean so don't use of string
    if data_dict[key]["poi"] == True:
        i = i + 1

print(i) # 18 POIs in the dataset

#==============================================================================
### Task 2: Remove outliers
df['poi'] # notice the TRAVEL AGENCY IN THE PARK poi entry

# Remove THE TRAVEL AGENCY IN THE PARK from the list of POIs since it's not 
# an actual person.  Data error entry.
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace = True)
df['poi']

df.plot.scatter(x = 'salary', y = 'bonus')

# outlier of a salary > 2.5 * 10e^7
# see who it belongs to
df['salary']
df['salary'].idxmax()

# remove this as it's the total of all the salaries of the employees in the Enron dataset
df.drop('TOTAL', inplace = True)
df.plot.scatter(x = 'salary', y = 'bonus')

# remove any outliers permanently
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# clean up the the data permanently
for ii in data_dict:
    for jj in data_dict[ii]:
        if data_dict[ii][jj] == 'NaN':
            data_dict[ii][jj] = 0

#==============================================================================

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Create new feature(s)

## create new features in the df
#df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
#df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['to_messages']
#
#features_list.extend(['fraction_from_poi','fraction_to_poi'])

# save to new features list
my_feature_list = features_list + ['fraction_from_poi','fraction_to_poi']

# k-best features: use for the SelectKBest parameter "k"
num_features = 10 

# function for SelectKBest
def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    #format the scores
    formatted_scores = ['%.5f' % elem for elem in scores]
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    #create a dictionary out of the best features and their scores 
    table_dict = dict(zip(k_best_features, formatted_scores))
    print(table_dict)
    return k_best_features


best_features = get_k_best(my_dataset, features_list, num_features)

my_feature_list = [target_label] + best_features.keys()
                         
# plot the features against each other
# plot non-poi
#plot1 = (df[df['poi'] == False].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', c='r', label='Non-POI'))
## plot of poi
#df[df['poi'] == True].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', c='b', label='POI', ax=plot1)


### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)

# properly scale the features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
print(features)

#==============================================================================
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# gaussian NB
gaussian_clf = GaussianNB()

# random forest
randomForest_clf = RandomForestClassifier(max_depth = 5, max_features = 'sqrt', n_estimators = 10, random_state = 42)

# svm
svm_clf = SVC(kernel='rbf', C=1000, gamma = 0.0001, random_state = 42, class_weight = 'auto')

# k-means clustering
k_clf = KMeans(n_clusters=2, tol=0.001)

# logistic regression
l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty = 'l2', tol = 0.001, C = .000001, random_state = 42))])

#==============================================================================
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Random Forest had better evaluator metrics than Gaussian NB
### Perform tuning techniques to improve Random Forest

# function for evaluating the different classifiers
def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        
        accuracy.append(accuracy_score(labels_test, pred))
        precision.append(precision_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))
        f1.append(f1_score(labels_test, pred))

    print "Accuracy: {} \n".format(mean(accuracy)), "Precision: {} \n".format(mean(precision)), \
        "Recall: {} \n".format(mean(recall)), "f1: {} \n".format(mean(f1))


### evaluate all classifiers using the evaluate_clf function
"""evaluate_clf(gaussian_clf, features, labels)
evaluate_clf(randomForest_clf, features, labels)
evaluate_clf(svm_clf, features, labels)
evaluate_clf(k_clf, features, labels)"""
evaluate_clf(l_clf, features, labels)

### Select Logistic Regression as final algorithm
clf = l_clf

#==============================================================================
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
