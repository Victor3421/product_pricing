
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('regression_model.pkl')
feature_names = joblib.load('features.pkl')

@app.route('/')
def home():
    return 'Regression Model is Running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    try:
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)
        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)
        return jsonify({'predicted_total_price': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
