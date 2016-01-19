#!/usr/bin/python
import numpy as np
import sys
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import cross_validation
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
# from sklearn import cross_validation

'''
Select what features 
features_list is a list of strings, each of which is a feature name.
The first feature must is  "poi".
'''
financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other',
                      'long_term_incentive', 'restricted_stock',
                      'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person',
                  'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

features_list_All = ['poi', 'salary', 'deferral_payments', 'total_payments',
                     'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees', 'Norm_Email',
                     'Norm_Email_T', 'Norm_Email_F']
                     
features_list1 = ['poi', 'salary', 'deferral_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'Norm_Email_T', 'Norm_Email_F']
                 
features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'expenses',
                  'exercised_stock_options', 'other', 'restricted_stock', 
                 'Norm_Email_T']


# Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

# Remove outliers
data_dict.pop('TOTAL', 0)

# Scaling financial_features

for i_feat in financial_features:
    vec = []
    for i in data_dict:
        if data_dict[i][i_feat] == 'NaN':
            continue
        else:
            vec.append(data_dict[i][i_feat])
    for i in data_dict:
        if data_dict[i][i_feat] == 'NaN':
            data_dict[i][i_feat] = 0

        else:
            data_dict[i][i_feat] = ((data_dict[i][i_feat] - min(vec)) /
                                    float(max(vec) - min(vec)))
'''
Create new feature(s)
Store to my_dataset for easy export below.
Extract features and labels from dataset for local testing
'''
# Scaling Email feature

for i in data_dict:
        A_F_S = data_dict[i]['shared_receipt_with_poi']
        A_F_P = data_dict[i]['from_poi_to_this_person']
        A_T_P = data_dict[i]['from_this_person_to_poi']
        A_F = data_dict[i]['from_messages']
        A_T = data_dict[i]['to_messages']
        if A_F_P == "NaN" or A_T_P == "NaN" or A_F == "NaN" or A_T == "NaN":
            data_dict[i]['Norm_Email'] = 0
        else:
            data_dict[i]['Norm_Email'] = float(A_F_P + A_T_P) / (A_F + A_T)

        if A_F_P == "NaN" or A_T_P == "NaN" or A_T == "NaN":
            data_dict[i]['Norm_Email_T'] = 0
        else:
            data_dict[i]['Norm_Email_T'] = float(A_F_P + A_T_P) / (A_T)

        if A_F_P == "NaN" or A_T_P == "NaN" or A_F == "NaN":
            data_dict[i]['Norm_Email_F'] = 0
        else:
            data_dict[i]['Norm_Email_F'] = float(A_F_P + A_T_P) / (A_F)

my_dataset = data_dict

# Correlation matrix
data = featureFormat(my_dataset, features_list_All, sort_keys=True,
                     remove_NaN=True)
corr_matrix = np.corrcoef(data.T)
sm.graphics.plot_corr(corr_matrix, xnames=features_list_All)
plt.show()

# The data set after scalling and selecting the relevant features.
data = featureFormat(my_dataset, features_list, sort_keys=True,
                     remove_NaN=True)

# The final features of the data base arranged as  labels and features
labels, features = targetFeatureSplit(data)

# A varity of classifiers


# AdaBoostClassifier using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter

'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf1 = GaussianNB()
'''
'''
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=30, learning_rate=1.0,
                                 max_depth=1, random_state=0)
clf1 = GradientBoostingClassifier(n_estimators=30, learning_rate=1.0,
                                 max_depth=1, random_state=0)  
'''

clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=30,
    learning_rate=1)
# features importance    
clf.fit(features, labels)
importances = clf.feature_importances_
print features_list, importances

clf1 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=30,
    learning_rate=1)

# fiting and predicting the data using the model
# calculate the confusion_matrix for the test subset
def Conf_Mat(predictions, true_labels):
    confusion_matrix = Counter()
    positives = [1]
    binary_truth = [x in positives for x in true_labels]
    binary_prediction = [x in positives for x in predictions]
    for t, p in zip(binary_truth, binary_prediction):
        confusion_matrix[t, p] += 1
    return confusion_matrix


# First subset
features_train1, features_test1, labels_train1, labels_test1 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=0)

clf1.fit(features_train1, labels_train1)
predictions = clf1.predict(features_test1)
true_labels = labels_test1

confusion_matrix1 = Conf_Mat(predictions, true_labels)

# Secound subset
features_train2, features_test2, labels_train2, labels_test2 = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=43)

clf1.fit(features_train2, labels_train2)
predictions = clf1.predict(features_test2)
true_labels = labels_test2

confusion_matrix2 = Conf_Mat(predictions, true_labels)

# calculate the confusion_matrix for the two subset
confusion_matrix = (confusion_matrix2 + confusion_matrix1)

print "TP: {} TN: {} FP: {} FN: {}".format(confusion_matrix1[True, True],
                                           confusion_matrix1[False, False],
                                           confusion_matrix1[False, True],
                                           confusion_matrix1[True, False])

print "TP: {} TN: {} FP: {} FN: {}".format(confusion_matrix2[True, True],
                                           confusion_matrix2[False, False],
                                           confusion_matrix2[False, True],
                                           confusion_matrix2[True, False])

print "The precision of this classifier?", (confusion_matrix[True, True])/float(confusion_matrix[True, True] + confusion_matrix[False, True])
print "The recall of this classifier?", (confusion_matrix[True, True])/(float(confusion_matrix[True, True] + confusion_matrix[True, False]))

# optimization using GridSearchCV
'''
from sklearn.grid_search import GridSearchCV

param_grid = {
    'n_estimators': [25, 30]
}
'''

# clf = GridSearchCV(estimator=clf1, param_grid=param_grid)
# clf.fit(features, labels)
# print clf.best_params_

# #############################################################################
# Different Models that were tested,

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# Provided to give you a starting point. Try a varity of classifiers.

# from sklearn import tree
# clf = tree.DecisionTreeClassifier(min_samples_split = 20)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=6,n_estimators=5)

# from sklearn.grid_search import GridSearchCV
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier




# #############################################################################

# # Tune your classifier to achieve better than  precision and recall

# ## Because of the small size of the dataset, the script uses stratified
# ## shuffle split cross validation. 

test_classifier(clf, my_dataset, features_list)



dump_classifier_and_data(clf, my_dataset, features_list)
