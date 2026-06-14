from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from io import BytesIO


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
model = tf.keras.models.load_model("civic_issue_model.h5")

CLASS_NAMES = [
    "Electricity",
    "Roads",
    "Sanitation",
    "WasteManagement"
]

@app.route("/predict", methods=["POST"])
def predict():


    file = request.files["image"]

    img = tf.keras.utils.load_img(
        BytesIO(file.read()),
        target_size=(224,224)
    )
    img = tf.keras.utils.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    index = np.argmax(predictions)

    confidence = float(np.max(predictions))

    return jsonify({
        "department": CLASS_NAMES[index],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)