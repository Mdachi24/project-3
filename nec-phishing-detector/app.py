
from flask import Flask, request, jsonify
import joblib
from utils import load_model_and_vectorizer, preprocess

app = Flask(__name__)

model, vectorizer = load_model_and_vectorizer("model.pkl", "vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    X = preprocess([text], vectorizer)
    prediction = model.predict(X)[0]
    result = "phishing" if prediction == 1 else "legitimate"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
