from flask import Flask, request, jsonify
import pickle

# Load model and vectorizer
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    prob = model.predict_proba(X)[0, 1]
    return jsonify({'subscription_probability': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)
