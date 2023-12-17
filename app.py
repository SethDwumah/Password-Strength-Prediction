from flask import Flask, render_template, request
import re
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model = pickle.load(open('dt_model.pkl','rb'))

def word_to_character(inputs):
    characters = [char for char in inputs]
    return characters

vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    password = request.form['password']
    pass_array = np.array([password])
    # Transform the password into a format that the model can use.
    transformed_password = vectorizer.transform(pass_array)
    
    # Predict the strength of the password.
    prediction = model.predict(transformed_password)
    

    # Display the prediction.
    return render_template('results.html', prediction=prediction[0])

if __name__== "__main__":
    app.run(debug=True)