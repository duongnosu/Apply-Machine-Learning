import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def blight_model():

    scaler = MinMaxScaler()
    # Read in the data sets
    train_data = pd.read_csv(
        '/Users/Nhu/Documents/Coursera/Apply_ML/train.csv', encoding="ISO-8859-1")
    test_data = pd.read_csv(
        '/Users/Nhu/Documents/Coursera/Apply_ML/test.csv', encoding="ISO-8859-1")
    addresses = pd.read_csv(
        '/Users/Nhu/Documents/Coursera/Apply_ML/addresses.csv', encoding="ISO-8859-1")

    ###Clean data####
    # Drop all the compliance with NA
    train_data = train_data.dropna(subset=['compliance'])
    train_data = pd.merge(train_data, addresses, how='inner',
                          left_on='ticket_id', right_on='ticket_id')
    test_data = pd.merge(test_data, addresses, how='inner',
                         left_on='ticket_id', right_on='ticket_id')
    train_features = ['fine_amount', 'admin_fee', 'state_fee',
                      'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount']

    X = train_data[train_features]
    y = train_data['compliance']

    ###Training model###
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)

    ###Checking the model, making sure nothing crazy happened###
    # print('Accuracy of RF classifier on training set: {:.2f}'
    #      .format(clf.score(X_train, y_train)))
    # print('Accuracy of RF classifier on test set: {:.2f}'
    #      .format(clf.score(X_test, y_test)))

    ###Get the model to  make prediction on the test data###
    test_data.index = test_data['ticket_id']
    X_predict = clf.predict_proba(test_data[train_features])

    ###Evaluation model with GridSearchSV with evaluation of  roc_auc###

    # #Scale it because of the number different
    # X_scaled_train = scaler.fit_transform(X_train)
    # X_scaled_test = scaler.fit_transform(X_test)
    # #Set parameters GridSearchCV
    # grid_values = {'max_features': [1, 2, 3, 4, 6, 7], 'max_depth': [3, 4, 5, 6]}
    # grid_values1 = {'max_features': [1, 2, 3, 4, 6, 7], 'max_depth': [3, 4]}
    # #performing the GridSearchCV
    # grid_clf = GridSearchCV(clf, param_grid=grid_values, scoring='roc_auc', n_jobs=-1)
    # grid_clf1 = GridSearchCV(clf, param_grid=grid_values1, scoring='roc_auc', n_jobs=-1)
    # grid_clf.fit(X_scaled_train, y_train)
    # grid_clf1.fit(X_scaled_train, y_train)
    # print(grid_clf.best_params_, grid_clf.best_score_)
    # print(grid_clf1.best_params_, grid_clf1.best_score_)

    # format the answer
    ans = pd.Series(data=X_predict[:, 1], index=test_data['ticket_id'], dtype='float32')

    return ans


blight_model()
print(blight_model().shape)
print(blight_model().head(5))
