from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def finalTrainingDataset():

    df = pd.read_csv('TrainOnMe.csv')

    df.drop(['Unnamed: 0'], inplace=True, axis=1)

    df.dropna(
        axis=0,
        how="any",
        inplace=True
    )

    df['x12'] = df['x12'].str.replace('False', '0').astype(str)
    df['x12'] = df['x12'].str.replace('Flase', '0').astype(str)
    df['x12'] = df['x12'].str.replace('True', '1').astype(str)

    df['x6'] = df['x6'].str.replace('Bayesian Inference', '0').astype(str)
    df['x6'] = df['x6'].str.replace('Bayesian Interference', '0').astype(str)
    df['x6'] = df['x6'].str.replace('GMMs and Accordions', '1').astype(str)

    df['y'] = df['y'].str.replace('Atsuto', '0').astype(str)
    df['y'] = df['y'].str.replace('Shoogee', '1').astype(str)
    df['y'] = df['y'].str.replace('Jorg', '2').astype(str)
    df['y'] = df['y'].str.replace('Bob', '3').astype(str)

    df = df.apply(pd.to_numeric, errors='ignore', downcast='float')

    df.drop(['x2'], inplace=True, axis=1)
    df.drop(['x12'], inplace=True, axis=1)
    df.drop(['x3'], inplace=True, axis=1)

    df = df[df.x1 < 1000]
    df = df[df.x1 > -1000]

    labels = df['y']

    train = df.drop(['y'], axis=1)

    sc = preprocessing.StandardScaler()
    train = sc.fit_transform(train)
    labels = labels.to_numpy()

    return train, labels

def createEvaluationDataset():
    df = pd.read_csv('EvaluateOnMe.csv')
    df.drop(['Unnamed: 0'], inplace=True, axis=1)

    ### Dropping redundant columns ###
    df.drop(['x2'], inplace=True, axis=1)
    df.drop(['x3'], inplace=True, axis=1)
    df.drop(['x12'], inplace=True, axis=1)

    ### Changing values to floats ###
    # ['False', 'True' ] =  [0, 1]
    df.replace({False: 0, True: 1}, inplace=True)

    # ['Bayesian Inference','GMMs and Accordions'] = [0,1], 'Bayesian interference'
    df['x6'] = df['x6'].str.replace('Bayesian Inference', '0').astype(str)
    df['x6'] = df['x6'].str.replace('GMMs and Accordions', '1').astype(str)

    evaluation_data = df.apply(pd.to_numeric, errors='ignore', downcast='float')
    sc = preprocessing.StandardScaler()
    evaluation_data = sc.fit_transform(evaluation_data)

    return evaluation_data

def writeToTxt(classifications):
    # ['Atsuto', 'Shoogee', 'Jorg', 'Bob'  = [0, 1, 2, 3]

    file = open("classification.txt", "w")

    for classification in classifications:
        if (classification == 0):
            file.write("Atsuto\n")
        elif (classification == 1):
            file.write("Shoogee\n")
        elif (classification == 2):
            file.write("Jorg\n")
        elif (classification == 3):
            file.write("Bob\n")

    file.close()



def createTrainingTestDataset(test_size):
    df = pd.read_csv('TrainOnMe.csv')

    df.drop(['Unnamed: 0'], inplace=True, axis=1)       #Removes index column

    ### Remove null rows / rows including null ###

    df.dropna(
        axis=0,
        how="any",
        inplace=True
    )

    ### Change values of strings to floats and removing misspellings ###

    #Finding misspellings
    # print(df['x6'].value_counts())
    # print(df['x12'].value_counts())
    # print(df['y'].value_counts())

    # ['False', 'True' ] =  [0, 1]
    df['x12'] = df['x12'].str.replace('False','0').astype(str)
    df['x12'] = df['x12'].str.replace('Flase','0').astype(str)
    df['x12'] = df['x12'].str.replace('True','1').astype(str)

    # ['Bayesian Inference','GMMs and Accordions'] = [0,1], 'Bayesian interference'
    df['x6'] = df['x6'].str.replace('Bayesian Inference','0').astype(str)
    df['x6'] = df['x6'].str.replace('Bayesian Interference','0').astype(str)
    df['x6'] = df['x6'].str.replace('GMMs and Accordions','1').astype(str)

    # ['Atsuto', 'Shoogee', 'Jorg', 'Bob'  = [0, 1, 2, 3]
    df['y'] = df['y'].str.replace('Atsuto','0').astype(str)
    df['y'] = df['y'].str.replace('Shoogee','1').astype(str)
    df['y'] = df['y'].str.replace('Jorg','2').astype(str)
    df['y'] = df['y'].str.replace('Bob','3').astype(str)


    df = df.apply(pd.to_numeric,errors='ignore', downcast='float')

    #print(df.dtypes)

    ### 4 FIND CORRELATIONS ###

    # corr = df.corr(method ='pearson')
    # plt.figure()
    # sns.heatmap(corr, annot=True)
    # plt.show()

    #Correlation between x2 and x11 is 1, x12 has high correlation with x6 and x3 as well so we can try removing them
    #Remove x2 based on variance,  variance(x2) < variance(x11), small difference so likely does not matter too much which one we pick
    x2 = df['x2'].describe()
    x11 = df['x11'].describe()
    x3 = df['x3'].describe()
    x12 = df['x12'].describe()

    df.drop(['x2'], inplace=True, axis=1)
    df.drop(['x12'], inplace=True, axis=1)
    df.drop(['x3'], inplace=True, axis=1)
    ### REMOVE OUTLIERS  ###

    #Drop huge outliers in column x1

    sorted = df.sort_values(by=['x1'])
    # print(sorted)
    df = df[df.x1 < 1000]
    df = df[df.x1 > -1000]

    ### DIVIDING DATA AND SCALING ###

    testing = df['y']
    training = df.drop(['y'], axis=1)

    train, test, train_labels, test_labels = train_test_split(training, testing, test_size=test_size, random_state=30) #  random_state=30

    # ### METHOD 1 ###
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train = min_max_scaler.fit_transform(train)
    # test = min_max_scaler.transform(test)

    ## METHOD 2 ###
    sc = preprocessing.StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)
    train_labels = train_labels.to_numpy()
    test_labels = test_labels.to_numpy()

    return train, test, train_labels, test_labels
