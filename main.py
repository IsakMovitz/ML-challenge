import numpy as np
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from helper_funcs import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier

evaluation_data = createEvaluationDataset()
train, labels = finalTrainingDataset()

### CROSS-VALIDATION TESTING ###

rfc = RandomForestClassifier(n_estimators= 200)
rfc_score = cross_val_score(rfc, train, labels, cv=5)
print("Random Forest mean: ", np.mean(rfc_score))

qda = QuadraticDiscriminantAnalysis()
qda_score = cross_val_score(rfc, train, labels, cv=5)
print("QDA mean: ", np.mean(qda_score))

decisionTree = tree.DecisionTreeClassifier(max_depth=8)
dtree_adaboost = AdaBoostClassifier(base_estimator=decisionTree, n_estimators=50,
                          learning_rate=2)

dtree_adaboost_score = cross_val_score(rfc, train, labels, cv=5)
print("Dtree Adaboost mean: ", np.mean(dtree_adaboost_score))


### STACKING ENSEMBLE METHOD ###
stack_classifier = StackingClassifier(estimators=[('qda', qda), ('rfc', rfc), ('dtree_ada', dtree_adaboost)])
stack_score = cross_val_score(stack_classifier, train, labels, cv=5)
print(stack_score)
print("Stacking Classifier mean: ", np.mean(stack_score))

### FINAL CLASSIFICATION ###
final_classifier = stack_classifier
final_classifier.fit(train, labels)
classifications = final_classifier.predict(evaluation_data)
writeToTxt(classifications)

