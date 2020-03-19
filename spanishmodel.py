import keras
import sys
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# load the dataset
data = pd.read_csv("origin/spanish-paradigm.csv")



cv = CountVectorizer(ngram_range=(2,2),analyzer='char_wb', lowercase=False)
X = cv.fit_transform(data['stem'])
print(cv.vocabulary_)
X = X.todense()



# split into input (X) and output (y) variables
y = data['class']
y = pd.get_dummies(y)
col = y.columns
y = y.values
y_count = y.shape[1]

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(y_count, activation='sigmoid'))

# compile the keras model
sgd = optimizers.adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
scores, histories = list(), list()
nfold = 3
kfold = KFold(nfold, shuffle=True, random_state=1)
print('number of fold:')
print(nfold)
for train_index,test_index in KFold(nfold).split(X):
    x_train,x_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=16, batch_size=8)
    print('Model evaluation ',model.evaluate(x_test,y_test))
    _, acc = model.evaluate(x_train, y_train, verbose=1)
    #scores.append(acc)
    histories.append(history)



print(history.history.get('accuracy')[-1])
print(history.history.get('val_loss')[-1])

print('\n \n \n \n')

test_dummy = 'abalanz'
test_dummy = [test_dummy]

cv_test = CountVectorizer(ngram_range=(2,2),analyzer='char_wb', lowercase=False)
test_dummy = cv_test.fit_transform(test_dummy)
print(cv_test.vocabulary_)
print(type(test_dummy))
print(test_dummy)
test_dummy = test_dummy.todense()
print(type(test_dummy))
print(test_dummy)


fill_value = 0
test_dummy = np.ones(X.shape)*fill_value

#test = model.predict_classes(test_dummy)

#print(test)

#{' a': 3, 'ab': 42, 'ba': 62, 'al': 50, 'la': 170, 'an': 52, 'nz': 215, 'z ': 327,
#{' a': 0, 'ab': 1, 'ba': 5, 'al': 2, 'la': 6, 'an': 3, 'nz': 7, 'za': 9, 'ar': 4, 'r ': 8}