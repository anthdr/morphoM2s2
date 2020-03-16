# first neural network with keras tutorial
import pandas
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# load the dataset
dataset = pandas.read_csv('spanish-paradigm.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset['last']
labelencoder = LabelEncoder()
X = labelencoder.fit_transform(X)
X2 = dataset['last2']
X3 = dataset['last3']
X4 = dataset['last4']
y = dataset['regular']


# define the keras model
model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# fit the keras model on the dataset
scores, histories = list(), list()
nfold = 6
kfold = KFold(nfold, shuffle=True, random_state=1)
print('number of fold:')
print(nfold)
for train_index,test_index in KFold(nfold).split(X):
    x_train,x_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    history = model.fit(x_train, y_train, validation_split=0.5, epochs=350, batch_size=30)
    print('Model evaluation ',model.evaluate(x_test,y_test))
    _, acc = model.evaluate(x_train, y_train, verbose=1)
    #scores.append(acc)
    histories.append(history)
