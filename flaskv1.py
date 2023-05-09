from flask import Flask, request, jsonify

app = Flask(__name__)

import joblib

model = joblib.load('assets/trained_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # assumes JSON data
    # make predictions with your model
    predictions = model.predict(data)
    # return predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
