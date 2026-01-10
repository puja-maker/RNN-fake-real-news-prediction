from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("train_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 🔴 UI ROUTE
@app.route("/")
def home():
    return render_template("index.html")

# 🔴 API ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if text == "":
        return jsonify({"error": "No text provided"}), 400

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prob = model.predict(padded)[0][0]
    prediction = "Real" if prob > 0.5 else "Fake"

    return jsonify({
        "prediction": prediction,
        "confidence": float(prob)
    })

if __name__ == "__main__":
    app.run()
    


