from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from numpy import array
from numpy import argmax
import keras
import sys
import pandas as pd
import numpy as np


np.set_printoptions(threshold=sys.maxsize)


# load the dataset
data = pd.read_csv("origin/spanish-paradigm.csv")


# prepare X
X = pd.DataFrame()
for i in range(4, 0, -1):
    X["n%d" % i] = data['stem'].str[-i]
X = X.fillna("0")
"""
X = data['stem'].str[-4:]
X = X.str.zfill(4)  # padding to the left
X = pd.DataFrame(data=X)
X["n4"] = X['stem'].str.strip().str[-4]
X["n3"] = X['stem'].str.strip().str[-3]
X["n2"] = X['stem'].str.strip().str[-2]
X["n1"] = X['stem'].str.strip().str[-1]
X = X.drop('stem', 1)
"""

#enc = OneHotEncoder()
enc = OrdinalEncoder()
X = enc.fit_transform(X)


# prepare y
y = data['class']
y = pd.Series(y).str.replace('-.*', '', regex=True)
y = pd.get_dummies(y)
col = y.columns
y = y.values
y_count = y.shape[1]


cvscores = []
nfold = 5
kfold = KFold(n_splits=nfold, shuffle=True, random_state=1)
for train_index, test_index in kfold.split(X, y):
    # define the keras model
    model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_count, activation='softmax'))

    # compile the keras model
    sgd = optimizers.adam(lr=0.02)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train, validation_split=0.2,
              epochs=8, batch_size=4, verbose=1)
    cvscores.append(model.evaluate(x_test, y_test))
    print('Model evaluation ', cvscores[-1])
    print('\n')
    cfm = confusion_matrix(np.argmax(y_test, axis=1),
                           model.predict_classes(x_test),
                           labels=[i for i in range(y_count)]
                           )
    cfm = pd.DataFrame(cfm, col, col)
    print(cfm)


print('\n')

print('mean accuracy is at: %s' % np.mean(list(zip(*cvscores))[1]))
print('accuracy std is at: %s' % np.std(list(zip(*cvscores))[1]))
print('mean val_loss is at: %s' % np.mean(list(zip(*cvscores))[0]))
print('val_loss std is at: %s' % np.std(list(zip(*cvscores))[0]))

print('\n')


def test(x):
    namestem = x
    test_dummy = namestem
    test_dummy = pd.DataFrame({"stem": [test_dummy]})
    test_dummy["n4"] = ""
    test_dummy["n4"] = test_dummy['stem'].str.strip().str[-4]
    test_dummy["n3"] = ""
    test_dummy["n3"] = test_dummy['stem'].str.strip().str[-3]
    test_dummy["n2"] = ""
    test_dummy["n2"] = test_dummy['stem'].str.strip().str[-2]
    test_dummy["n1"] = ""
    test_dummy["n1"] = test_dummy['stem'].str.strip().str[-1]
    test_dummy = test_dummy.drop('stem', 1)
    test_dummy = enc.transform(test_dummy)
    testmodel = model.predict_classes(test_dummy)
    print('prediction class for')
    print(namestem)
    print('is')
    print((col[int(testmodel)]))
