from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("best_model.pkl")

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
    