import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# load the dataset
data = pd.read_csv("origin/spanish-paradigm.csv")
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), min_df=1) 
X = vectorizer.fit_transform(data['stem'])
vectorizer.get_feature_names()
X = pd.DataFrame(X.todense())
print(X)
print(type(X))
# split into input (X) and output (y) variables
y = data['class']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# define the keras model
model = Sequential()
model.add(Dense(128, input_dim=X.shape, activation='relu'))
model.add(Dense(activation='sigmoid'))

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
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=350, batch_size=30)
    print('Model evaluation ',model.evaluate(x_test,y_test))
    _, acc = model.evaluate(x_train, y_train, verbose=1)
    #scores.append(acc)
    histories.append(history)

print(history.history.get('accuracy')[-1])