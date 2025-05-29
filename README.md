# Hand Landmarks Prediction Project

This project predicts hand gestures using machine learning models trained on hand landmarks data. It compares multiple models, selects the best based on evaluation metrics, and tracks experiments using MLflow.

---

## Project Overview

This repository includes:

* **Data Preprocessing**: Preparing the dataset for training and testing.
* **Model Training**: Training various machine learning models like Logistic Regression, SVM, KNN, and Random Forest.
* **Model Evaluation**: Evaluating models using metrics like accuracy and F1-score.
* **Experiment Tracking**: Using MLflow to log parameters, metrics, and models.
* **Best Model Selection**: SVM was chosen as the best model based on its superior F1-score and accuracy.

---

## Prerequisites

### Install Required Libraries

Run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

### Required Python Libraries

* pandas
* numpy
* scikit-learn
* matplotlib
* mlflow
* joblib

---

## Dataset

The dataset (`hand_landmarks_data.csv`) contains hand landmarks and their corresponding gesture labels. It is used for training and testing the models.

---

## Model Training and Evaluation

### Models Used

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **K-Nearest Neighbors (KNN)**
4. **Random Forest**

### Results Summary

| Model                           | Accuracy  | F1 Score  |
| ------------------------------- | --------- | --------- |
| Logistic Regression             | 0.849     | 0.848     |
| SVM with GridSearchCV           | **0.986** | **0.985** |
| KNN with GridSearchCV           | 0.979     | 0.979     |
| Random Forest with GridSearchCV | 0.977     | 0.977     |

**Selected Model**: SVM was chosen for its superior performance in terms of both accuracy and F1 score.

---

## How to Run

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Preprocess the data:

   The preprocessing is handled in the `preprocessing.py` script.

3. Train the models:

   Use the `train.py` script to train the models. MLflow will automatically log results.

   ```bash
   python train.py
   ```

4. Track Experiments:

   Start the MLflow server:

   ```bash
   mlflow ui
   ```

   Navigate to `http://localhost:5000` to view experiment results.

---

## Files

* `DataSet/hand_landmarks_data.csv`: Dataset for training and testing.
* `preprocessing.py`: Preprocessing script.
* `train.py`: Model training script.
* `requirements.txt`: Python library dependencies.

---

## Experiment Tracking with MLflow

The project uses MLflow for:

* Logging model parameters and metrics.
* Saving trained models.
* Comparing model performance.

To view experiment results:

1. Start the MLflow server with:

   ```bash
   mlflow ui
   ```

2. Open your browser and go to `http://localhost:5000`.

---

## Future Work

* Add deep learning models for better accuracy.
* Implement real-time hand gesture recognition using a webcam.
* Integrate the system into a web or mobile application.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

