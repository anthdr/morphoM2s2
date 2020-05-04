import pandas as pd
import sys
import numpy as np
from numpy import argmax
from numpy import array

import keras
from keras import optimizers
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


np.set_printoptions(threshold=sys.maxsize)


# load the dataset
data = pd.read_csv("origin/spanish-paradigm.csv")

# prepare X
cv = CountVectorizer(ngram_range=(2, 2), analyzer='char_wb', lowercase=False)
X = cv.fit_transform(data['stem'])
X = X.todense()


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
    model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(y_count, activation='softmax'))

    # compile the keras model
    sgd = optimizers.adam(lr=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train, validation_split=0.2,
              epochs=32, batch_size=16, verbose=1)
    cvscores.append(model.evaluate(x_test, y_test))
    print('Model evaluation ', cvscores[-1])
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
    testem = namestem
    test_dummy = testem
    test_dummy = [test_dummy]
    test_dummy = cv.transform(test_dummy)
    test_dummy = test_dummy.todense()
    testmodel = model.predict_classes(test_dummy)
    print('prediction class for')
    print(namestem)
    print('is')
    print((col[int(testmodel)]))
