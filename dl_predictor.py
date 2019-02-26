%matplotlib inline

import numpy as np
import pandas as pd

ratings = pd.read_csv('/gdrive/My Drive/app/ratings.csv', encoding = 'ISO-8859-1')
print(ratings.head())

print(ratings.shape)

n_users = int(ratings.user_id.nunique())
n_books = int(ratings.book_id.nunique())
print(n_users, n_books)


# create a cross-tab for better visualization of user ids and book ids
g = ratings.groupby('user_id')['rating'].count()
topg = g.sort_values(ascending = False)[:30]

i = ratings.groupby('book_id')['rating'].count()
topi = i.sort_values(ascending = False)[:30]

join1 = ratings.join(topg, on = 'user_id', how = 'inner', rsuffix = '_r')
join2 = join1.join(topi, on = 'book_id', how = 'inner', rsuffix = '_r')

pd.crosstab(join2.user_id, join2.book_id, join2.rating, aggfunc = np.sum)


from keras.layers import Dense, Dropout, Flatten, Embedding, Input
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.utils

X = ratings[['user_id', 'book_id']]
y = ratings['rating']

def get_new_model(input_shape = (2, )):
    model = Sequential()
    model.add(Embedding(input_dim = n_users+2, output_dim = 200, input_length = 2))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', input_shape = input_shape))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(6, activation = 'softmax'))
    model.summary()
    return(model)
    
model = get_new_model()
#my_optimizer = SGD(lr = 1e-4, decay = 1e-6) # Stochastic gradient descent 
my_optimizer = 'adam'
bs = 1000
model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
early_stopping_monitor = EarlyStopping(patience = 2)
model.fit(X_train, y_train, validation_split = 0.3, epochs = 15, batch_size = bs, callbacks = [early_stopping_monitor])
#model.fit(X_train, y_train, validation_split = 0.3, epochs = 15, batch_size = bs)

score = model.evaluate(X_test, y_test, verbose = 0, batch_size = bs)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


books = pd.read_csv('/gdrive/My Drive/app/books.csv', encoding = 'ISO-8859-1', index_col = 'id')

#pred_id = [1, 1]
#pred_user = [9731, 9806]
#results = model.predict_classes(pd.DataFrame({'user_id': pred_user, 
                                                'book_id': pred_id}))
#for i in range(len(pred_id)):
#    print('Predicted rating by user ', pred_user[i], ' for ', books.loc[pred_id[i]]['title'], ' is ', results[i])


# Predict user's rating for centain book
def DL_predict(user_id, book_id):
    result = model.predict_classes(pd.DataFrame({'user_id': [user_id], 
                                                'book_id': [book_id]}))                                                
    return result[0]    
    
    
DL_predict(9731, 1)
