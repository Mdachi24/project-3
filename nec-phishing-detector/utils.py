
import joblib

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def preprocess(texts, vectorizer):
    return vectorizer.transform(texts)
