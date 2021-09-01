from helper_funcs import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

test_size = 0.3
train,test, train_labels,  test_labels = createTrainingTestDataset(test_size)

#### TESTING ####

#1: Random Forest Classifier
#2: Random Forest Classifer Adaboost
#3: Quadratic Discriminant Analysis / MLP
#4: Decision Tree with Adaboost, maxdepth = 8

### Quadratic Discrimination Analysis ###
qda = QuadraticDiscriminantAnalysis()
qda.fit(train, train_labels)
classifications = qda.predict(test)
print("Accuracy Quadratic Discriminant Analysis: ", metrics.accuracy_score(test_labels, classifications))

### RANDOM FOREST CLASSIFIER ###
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(train, train_labels)
classifications = rfc.predict(test)
print("Random Forest Classifier Metrics: ", metrics.classification_report(test_labels, classifications))
print("Accuracy Random Forest Classifier:",metrics.accuracy_score(test_labels, classifications))

### RANDOM FOREST WITH ADABOOST ###
rfc = RandomForestClassifier(n_estimators=200)
rfc_adaboost = AdaBoostClassifier(base_estimator=rfc, n_estimators=50,
                         learning_rate=1)
rfc_model = rfc_adaboost.fit(train, train_labels)
classifications = rfc_model.predict(test)
print("Accuracy Random Forest with Adaboost:", metrics.accuracy_score(test_labels, classifications))

### DECISION TREE ###
decisionTree = tree.DecisionTreeClassifier(max_depth=8)
clf = decisionTree.fit(train, train_labels)
classifications = clf.predict(test)
print("Accuracy Decision Tree:", metrics.accuracy_score(test_labels, classifications))

### DECISION TREE  WITH ADABOOST ###
dtree_adaboost = AdaBoostClassifier(base_estimator=decisionTree, n_estimators=50,
                         learning_rate=2)
decisionTree_model = dtree_adaboost.fit(train, train_labels)
classifications = decisionTree_model.predict(test)
print("Accuracy Decision Tree with Adaboost:", metrics.accuracy_score(test_labels, classifications))

## Multi Layer Perceptron Neural Network ###
mlp = MLPClassifier(random_state=0, max_iter=100).fit(train, train_labels)
classifications = mlp.predict(test)
print("Accuracy MLP Neural Network: ", metrics.accuracy_score(test_labels, classifications))

### K-Nearest-Neighbour ###
neigh = KNeighborsClassifier(n_neighbors=19)
neigh.fit(train, train_labels)
classifications = neigh.predict(test)
print("Accuracy K-Nearest-Neighbour: ", metrics.accuracy_score(test_labels, classifications))

###############################################################################################

### NAIVE BAYES ###
nb = GaussianNB()
nb.fit(train,train_labels)
classifications = nb.predict(test)
print("Accuracy sklearn Naive Bayes:",metrics.accuracy_score(test_labels, classifications))

### NAIVE BAYES WITH ADABOOST ###
nb_adaboost = AdaBoostClassifier(base_estimator=nb, n_estimators=100,
                         learning_rate=1)
nb_model = nb_adaboost.fit(train, train_labels)
classifications = nb_model.predict(test)
print("Accuracy sklearn Naive Bayes with Adaboost:",metrics.accuracy_score(test_labels, classifications))

### SUPPORT VECTOR MACHINE ###
svm = make_pipeline(StandardScaler(), SVC(kernel='poly',degree=3, gamma='auto'))
svm.fit(train, train_labels)
classifications = svm.predict(test)
print("Accuracy Support Vector Machine:", metrics.accuracy_score(test_labels, classifications))

## Gaussian Process ###
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,
        random_state=0).fit(train, train_labels)

classifications = gpc.predict(test)
print("Accuracy Gaussian Process: ", metrics.accuracy_score(test_labels, classifications))