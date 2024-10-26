import pickle

# Load the model and DictVectorizer
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

client = {"job": "management", "duration": 400, "poutcome": "success"}
X = dv.transform([client])

probability = model.predict_proba(X)[0, 1]
print("Probability of subscription:", probability)
