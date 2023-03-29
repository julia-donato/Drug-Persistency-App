import pickle
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('ada_model.pkl', 'rb'))

# Define a function to preprocess the user input
def preprocess_input(feature1, feature2, feature3, feature4, feature5):
    # Create a dictionary with the user input
    input_dict = {
        'Dexa_Freq_During_Rx': [feature1],
        'Dexa_During_Rx': [feature2],
        'Comorb_Long_Term_Current_Drug_Therapy': [feature3],
        'Comorb_Encounter_For_Screening_For_Malignant_Neoplasms': [feature4],
        'Comorb_Encounter_For_Immunization': [feature5]
    }
    # Convert the dictionary to a pandas dataframe
    input_df = pd.DataFrame.from_dict(input_dict)
    return input_df

# Define a function to make predictions
def make_prediction(input_df):
    # Make a prediction using the loaded model
    prediction = model.predict(input_df)
    # Return the prediction
    return prediction[0]

# Define the route for the home page
@app.route('/')
def home():
    # Render the home page
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    feature4 = request.form['feature4']
    feature5 = request.form['feature5']
    # Preprocess the user input
    input_df = preprocess_input(feature1, feature2, feature3, feature4, feature5)
    # Make a prediction
    prediction = make_prediction(input_df)
    # Render the prediction page with the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
