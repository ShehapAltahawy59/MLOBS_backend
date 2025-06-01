from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
from flask_cors import CORS
CORS(app) 
# Load model
model = joblib.load("Models/SVM_best_model.pkl")

def get_prediction(landmarks):
    if landmarks:
        # Convert landmarks from list of dicts to NumPy array
        landmarks_list = []
        for point in landmarks:
            landmarks_list.append([point['x'], point['y'], point['z']])
        landmarks = np.array(landmarks_list)

        # Normalize landmarks
        wrist_x, wrist_y, wrist_z = landmarks[0]
        landmarks[:, 0] -= wrist_x
        landmarks[:, 1] -= wrist_y
        mid_finger_x, mid_finger_y, _ = landmarks[12]
        scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
        landmarks[:, 0] /= scale_factor
        landmarks[:, 1] /= scale_factor

        # Prepare features and predict
        features = landmarks.flatten().reshape(1, -1)
        prediction = model.predict(features)[0]
        print(prediction)
        return prediction

@app.route('/')
def index():
    return "Server is running"

@app.route('/get_prediction', methods=['POST'])
def get_prediction_endpoint():
    
    
        # Parse JSON data from the request
        data = request.get_json()
        landmarks = data.get('landmarks')
        if not landmarks:
            
            return jsonify({"error": "No landmarks provided"}), 400

        # Get prediction
        prediction = get_prediction(landmarks)
        print(prediction ,flush=True)
        return jsonify({"prediction": prediction})

    

if __name__ == '__main__':
    app.run(debug=True)
