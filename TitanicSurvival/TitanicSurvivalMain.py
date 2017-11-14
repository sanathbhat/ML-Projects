import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
import re
if __name__ == '__main__':
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 100)
    df_train = pd.read_csv('data/train.csv', index_col=0)
    df_test = pd.read_csv('data/test.csv', index_col=0)

    nrows = df_train.shape[0]

    #prints % of missing values in each feature

    # print '% missing values'
    # for (k,v) in df.count(0).iteritems():
    #     print k.rjust(10) + str(round((1 - float(v)/nrows)*100, 2)).rjust(10)


    #'Name' and 'Ticket' are seemingly useless so remove from dataframe
    df_train_Mod1 = df_train.drop(['Name', 'Ticket'], 1)
    df_classify_test = df_test.drop(['Name', 'Ticket'], 1)

    #print dfMod1.head()

    #survival based on having Cabin
    #print(dfMod1[(dfMod1.Survived==0) & (dfMod1.Cabin.notnull())])

    #Duplicates in Cabin
    #print(dfMod1.duplicated('Cabin', keep='first'))

    #survival based on Embarkation
    #print(dfMod1.groupby(['Embarked', 'Survived']).count())

    # Basic Feature engineering

    #fill missing embarked values
    df_train_Mod1.Embarked.fillna('N', inplace=True)
    df_classify_test.Embarked.fillna('N', inplace=True)

    #fill missing cabin values
    df_train_Mod1.Cabin.fillna(0, inplace=True)
    df_classify_test.Cabin.fillna(0, inplace=True)
    df_train_Mod1.Cabin.replace('\w', 1, regex=True, inplace=True)
    df_classify_test.Cabin.replace('\w', 1, regex=True, inplace=True)

    #transformation to numbers for creating numerical feature vectors
    df_train_Mod1.Sex.replace(['male', 'female'], [0, 1], inplace=True)
    df_classify_test.Sex.replace(['male', 'female'], [0, 1], inplace=True)
    df_train_Mod1.Embarked.replace(['S', 'C', 'Q', 'N'], [0, 1, 2, 99], inplace=True)
    df_classify_test.Embarked.replace(['S', 'C', 'Q', 'N'], [0, 1, 2, 99], inplace=True)

    # sex = dfMod1.Sex.as_matrix()
    # age = dfMod1.Age.as_matrix()
    # survival = dfMod1.Survived.as_matrix()
    # hasCabin = dfMod1.Cabin.as_matrix()

    # X_axis = [age[i] for i in range(nrows) if ~np.isnan(age[i])]             #age without nulls
    #
    # sexAxis_mod = [(3 * sex[i] - 1) if survival[i] == 1 else sex[i] for i in range(nrows)]  #separate survivors from the dead
    # Y_axis = [sexAxis_mod[i] for i in range(nrows) if ~np.isnan(age[i])]
    #
    # survival_labels = [[1, 0, 0] if hasCabin[i] == 0 else [0, 1, 0] for i in range(nrows)]
    # colors = [survival_labels[i] for i in range(nrows) if ~np.isnan(age[i])]

    # plt.scatter(X_axis, Y_axis, c=colors, s=15, alpha=1, lw=0)
    # plt.title('Survival based on Age and Sex')
    # plt.axis([-10, 110, -2, 3])
    # plt.xlabel('Age')
    # plt.ylabel('Sex')
    # plt.show()

#Filling the missing age values
    #Divide the dataset based on missing age values
    # df_WithoutMissingAge = dfMod1[~np.isnan(dfMod1.Age)]
    # df_MissingAge = dfMod1[np.isnan(dfMod1.Age)]
    #
    # #Checking dependencies on other features
    # # df_WithoutMissingAge.plot.scatter('Embarked', 'Age', s=15, c='r')
    # # plt.show()
    #
    # #Pclass, Cabin, Embarked not significantly correlated to Age, hence not used for regression
    # df_Age_Features = df_WithoutMissingAge[['Survived', 'Sex', 'SibSp', 'Parch', 'Fare']]
    # df_Age_Target = df_WithoutMissingAge['Age']
    # df_AgePredict =  df_MissingAge[['Survived', 'Sex', 'SibSp', 'Parch', 'Fare']]
    #
    # # print(df_Age_Features.head())
    # # print(df_Age_Target.head())
    #
    # X_train, X_test, y_train, y_test = train_test_split(df_Age_Features, df_Age_Target, test_size=0.1)
    # print(X_train)
    # # print(X_test)
    # # # print(y_train)
    # # print(y_test)
    #
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    #
    # age_test_pred = model.predict(X_test)
    #
    # for (pred, actual) in zip(age_test_pred, y_test):
    #     print(str(actual) + "    " +str(round(pred,2)))
    #
    # print(model.score(X_test, y_test))

#Conclusion: Age regression model not conclusive/high error

    #Fill missing age values with a large value so that they are considered as outliers by the classification models
    df_train_Mod1.Age.fillna(-9999, inplace=True)
    df_classify_test.Age.fillna(-9999, inplace=True)
    # print(dfMod1.head(10))

    #Data for classification
    df_classify_features = df_train_Mod1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
    df_classify_target = df_train_Mod1['Survived']

    df_classify_test.Fare.fillna(-9999, inplace=True)

    # print df_Classify_Features.head(10)
    # print df_test_Mod1.head(10)

    # print(df_Classify_Features.head(10))
    # print(df_Classify_Target.head(10))

    # X_train, X_test, y_train, y_test = train_test_split(df_Classify_Features, df_Classify_Target, test_size=0.1)

    #1: GaussianNB
    # model = GaussianNB()

    #2: DecisionTree
    # model = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)

    #3: RandomForest
    model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=20, max_depth=5, min_samples_leaf=5)

    #4: AdaBoost
    # modelA = AdaBoostClassifier(n_estimators=5, base_estimator=model, algorithm='SAMME.R')

    # model.fit(X_train, y_train)
    # pred = model.predict(X_test)
    # print(accuracy_score(y_test, pred))

#For final submission
    model.fit(df_classify_features, df_classify_target)
    pred = model.predict(df_classify_test)

    with open('gender_submission.csv', 'w') as f:
        f.write('PassengerId,Survived\n')
        for pid,sur in zip(df_classify_test.index, pred):
            f.write(str(pid) + ',' + str(sur)+ '\n')

    #print(accuracy_score(y_test, pred))
    # print(X_test)
    # print(X_test[[p != a for p,a in zip(pred, y_test)]])



