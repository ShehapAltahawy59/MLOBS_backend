from flask import Flask, request, jsonify
import numpy as np
import joblib
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Custom metrics
# 1. Model-related metric: Prediction confidence/probability distribution
prediction_counter = Counter(
    'ml_predictions_total', 
    'Total number of ML predictions made',
    ['prediction_class']
)

prediction_confidence = Histogram(
    'ml_prediction_confidence_seconds',
    'Time taken to make ML predictions',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

model_accuracy_gauge = Gauge(
    'ml_model_loaded',
    'Whether the ML model is successfully loaded (1=loaded, 0=not loaded)'
)

# 2. Data-related metric: Input data quality
data_quality_counter = Counter(
    'ml_data_quality_total',
    'Count of data quality issues',
    ['issue_type']
)

landmark_count_histogram = Histogram(
    'ml_landmark_count',
    'Distribution of landmark counts in requests',
    buckets=[0, 5, 10, 15, 20, 21, 25, 30, 50]
)

# 3. Server-related metric: Memory usage and response times
memory_usage_gauge = Gauge(
    'ml_app_memory_usage_bytes',
    'Memory usage of the ML application'
)

# Load model with error handling
try:
    model = joblib.load("Models/SVM_best_model.pkl")
    model_accuracy_gauge.set(1)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_accuracy_gauge.set(0)

def get_memory_usage():
    """Get current memory usage of the process"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    except ImportError:
        return 0

def get_prediction(landmarks):
    start_time = time.time()
    
    if not landmarks:
        data_quality_counter.labels(issue_type='no_landmarks').inc()
        return None
    
    # Record landmark count
    landmark_count_histogram.observe(len(landmarks))
    
    # Validate landmark structure
    if len(landmarks) != 21:
        data_quality_counter.labels(issue_type='invalid_landmark_count').inc()
        return None
    
    try:
        # Convert landmarks from list of dicts to NumPy array
        landmarks_list = []
        for i, point in enumerate(landmarks):
            if not all(key in point for key in ['x', 'y', 'z']):
                data_quality_counter.labels(issue_type='missing_coordinates').inc()
                return None
            landmarks_list.append([point['x'], point['y'], point['z']])
        
        landmarks_array = np.array(landmarks_list)

        # Normalize landmarks
        wrist_x, wrist_y, wrist_z = landmarks_array[0]
        landmarks_array[:, 0] -= wrist_x
        landmarks_array[:, 1] -= wrist_y
        
        # Check for valid mid finger landmark
        if len(landmarks_array) <= 12:
            data_quality_counter.labels(issue_type='insufficient_landmarks').inc()
            return None
            
        mid_finger_x, mid_finger_y, _ = landmarks_array[12]
        scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
        
        if scale_factor == 0:
            data_quality_counter.labels(issue_type='zero_scale_factor').inc()
            return None
            
        landmarks_array[:, 0] /= scale_factor
        landmarks_array[:, 1] /= scale_factor

        # Prepare features and predict
        features = landmarks_array.flatten().reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Record prediction metrics
        prediction_counter.labels(prediction_class=str(prediction)).inc()
        prediction_time = time.time() - start_time
        prediction_confidence.observe(prediction_time)
        
        print(f"Prediction: {prediction}, Time: {prediction_time:.4f}s")
        return prediction
        
    except Exception as e:
        data_quality_counter.labels(issue_type='processing_error').inc()
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def index():
    # Update memory usage
    memory_usage_gauge.set(get_memory_usage())
    return "ML API Server is running"

@app.route('/health')
def health():
    """Health check endpoint"""
    memory_usage_gauge.set(get_memory_usage())
    
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "memory_usage_bytes": get_memory_usage()
    }
    return jsonify(health_status)

@app.route('/get_prediction', methods=['POST'])
def get_prediction_endpoint():
    # Update memory usage
    #memory_usage_gauge.set(get_memory_usage())
    
    try:
        # Parse JSON data from the request
        data = request.get_json()
        
        if not data:
            data_quality_counter.labels(issue_type='no_json_data').inc()
            return jsonify({"error": "No JSON data provided"}), 400
            
        landmarks = data.get('landmarks')
        
        if not landmarks:
            data_quality_counter.labels(issue_type='no_landmarks_field').inc()
            return jsonify({"error": "No landmarks provided"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get prediction
        prediction = get_prediction(landmarks)
        
        if prediction is None:
            return jsonify({"error": "Failed to make prediction"}), 400
            
        print(f"Prediction result: {prediction}", flush=True)
        return jsonify({"prediction": prediction})
        
    except Exception as e:
        data_quality_counter.labels(issue_type='endpoint_error').inc()
        print(f"Error in endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Custom metrics endpoint for additional info
@app.route('/custom_metrics')
def custom_metrics():
    """Endpoint to view custom metrics summary"""
    return jsonify({
        "model_loaded": model is not None,
        "memory_usage_bytes": get_memory_usage(),
        "metrics_info": {
            "predictions_tracked": "ml_predictions_total",
            "data_quality_tracked": "ml_data_quality_total", 
            "prediction_timing": "ml_prediction_confidence_seconds",
            "memory_usage": "ml_app_memory_usage_bytes"
        }
    })

if __name__ == '__main__':
    # Set memory usage on startup
    #memory_usage_gauge.set(get_memory_usage())
    app.run(host='0.0.0.0', port=5000, debug=False)
