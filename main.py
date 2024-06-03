from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
with open('Model/Iris.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
