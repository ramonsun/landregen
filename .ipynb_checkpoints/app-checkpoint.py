from flask import Flask, request, jsonify
import numpy as np
import joblib  # You might need to install this package

app = Flask(__name__)

# Load your trained model
model = joblib.load('your_model_file.pkl')  # Replace with your model file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    # Convert prediction to a native Python list or a simple data type
    output = int(prediction[0])  # Assuming prediction is a single value
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
