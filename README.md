# ML API Server

This project is a Flask-based API server that provides machine learning predictions based on hand landmarks. It includes monitoring and visualization tools using Prometheus and Grafana.

## Overview

The server loads a pre-trained SVM model (stored in `Models/SVM_best_model.pkl`) and exposes an endpoint (`/get_prediction`) that accepts hand landmark data. It processes the landmarks, normalizes them, and returns a prediction.

## Features

- **ML Prediction API**: Endpoint to submit hand landmarks and receive predictions.
- **Monitoring**: Prometheus metrics for tracking predictions, data quality, and server performance.
- **Visualization**: Grafana dashboards for real-time monitoring.
- **Containerization**: Docker and Docker Compose setup for easy deployment.
- **Testing**: Comprehensive unit tests for API endpoints and prediction logic.

## Prerequisites

- Python 3.9
- Docker and Docker Compose

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the pre-trained model is available in the `Models` directory.

## Running the Application

### Using Docker Compose

To run the entire stack (API, Prometheus, Grafana, Node Exporter):

```bash
docker-compose up
```

This will start:
- ML API server on port 5000
- Prometheus on port 9090
- Grafana on port 3000
- Node Exporter on port 9100

### Running Locally

To run the Flask application locally:

```bash
python app.py
```

The server will be available at `http://localhost:5000`.

## API Endpoints

- **GET /**: Returns a simple message indicating the server is running.
- **GET /health**: Health check endpoint returning server status and memory usage.
- **POST /get_prediction**: Accepts JSON data with hand landmarks and returns a prediction.
- **GET /custom_metrics**: Returns a summary of custom metrics.

## Testing

Run the unit tests using:

```bash
python test_app.py
```

## Monitoring

- **Prometheus**: Access the Prometheus UI at `http://localhost:9090` to view metrics.
- **Grafana**: Access Grafana at `http://localhost:3000` (default credentials: admin/admin123) to view dashboards.

### Metrics in Grafana

The following metrics are displayed in Grafana dashboards:

- **ML Predictions Total**: Tracks the total number of predictions made by the model, categorized by prediction class.
- **ML Prediction Confidence**: Measures the time taken to make predictions, helping to identify performance bottlenecks.
- **ML Model Loaded**: Indicates whether the ML model is successfully loaded (1=loaded, 0=not loaded).
- **ML Data Quality Total**: Counts data quality issues, such as missing landmarks or invalid data.
- **ML Landmark Count**: Shows the distribution of landmark counts in requests, aiding in understanding data patterns.
- **ML App Memory Usage**: Monitors the memory usage of the ML application, ensuring efficient resource utilization.

These metrics provide insights into the performance, reliability, and efficiency of the ML API server, helping to maintain and optimize the system.

## Project Structure

- `app.py`: Main Flask application with API endpoints and prediction logic.
- `test_app.py`: Unit tests for the API and prediction functions.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Instructions for building the Docker image.
- `docker-compose.yml`: Configuration for running the entire stack.
- `Models/`: Directory containing the pre-trained model.
- `prometheus/`: Prometheus configuration files.
- `grafana/`: Grafana dashboards and provisioning files.



