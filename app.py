from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    rand = pickle.load(file)

# Load SVM model
with open('svm_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Render home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form.get('name')
    statuses_count = float(request.form.get('statuses_count'))
    followers_count = float(request.form.get('followers_count'))
    friends_count = float(request.form.get('friends_count'))
    favourites_count = float(request.form.get('favourites_count'))
    listed_count = float(request.form.get('listed_count'))
    sex_code = float(request.form.get('sex_code'))
    lang_code = float(request.form.get('lang_code'))

    # Assuming you have processed the categorical variables (sex_code and language_code) appropriately in your ML model

    # Create input array
    input_data = np.array([[statuses_count, followers_count, friends_count, favourites_count, listed_count,sex_code,lang_code]])

    # Make predictions for each model
    rand_prediction = rand.predict(input_data)
    svm_prediction = classifier.predict(input_data)

    # Prepare prediction messages
    rand_message = 'Fake Profile Detected' if rand_prediction[0] == 1 else 'Real Profile Detected'
    svm_message = 'Fake Profile Detected' if svm_prediction[0] == 1 else 'Real Profile Detected'

    # Render the index.html template with the prediction messages
    return render_template('index.html', rand_message=rand_message, svm_message=svm_message)
    # Render the index.html template with the prediction result
    return render_template('index.html', prediction_result=prediction_result)