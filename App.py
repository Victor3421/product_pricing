from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Assume model and feature_names are already loaded globally

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
        elif request.method == 'GET':
            data = request.args.to_dict()
            # Convert all query parameters to correct numeric types
            for k in data:
                try:
                    data[k] = float(data[k])
                except:
                    pass  # Leave as is if not convertible

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode and align with training features
        input_df = pd.get_dummies(input_df)
        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:            input_df[col] = 0
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)
        return jsonify({'predicted_total_price': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
