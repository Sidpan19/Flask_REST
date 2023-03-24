from flask import Flask, render_template,url_for,request
import numpy as np
import pandas as pd
import pickle
from keras.datasets import imdb
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten
import joblib
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    (X_train,y_train),(X_test,y_test) = imdb.load_data()
    X_train = pad_sequences(X_train,padding='post',maxlen=50)
    X_test = pad_sequences(X_test,padding='post',maxlen=50)
    
    model = Sequential()

    model.add(SimpleRNN(32,input_shape=(50,1),return_sequences=False))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X_train,y_train,epochs=3,validation_data=(X_test,y_test))
    
    cv=TfidfVectorizer()
    #Save Model
    joblib.dump(model, 'model.pkl')
    print("Model dumped!")
    
    #ytb_model = open('spam_model.pkl', 'rb')
    model = joblib.load('model.pkl')
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        print(data)
        vect1 = cv.fit_transform(data).toarray()
        vect = pad_sequences(vect1,padding='post',maxlen=50)
        print(vect)
        print(vect1)
        vect=np.reshape(vect,(50,1))
        my_prediction = model.predict(vect)
        print(my_prediction.mean())
        if my_prediction.mean()>=0.5:
            m=1
        elif my_prediction.mean()<0.5:
            m=0
    return render_template('result.html', prediction = m)

if __name__ == '__main__':
    app.run(debug=True)