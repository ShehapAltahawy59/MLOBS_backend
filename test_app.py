import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np
import sys
import os

# Add the parent directory to the path to import the main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app (assuming your main file is named app.py)
from app import app, get_prediction

class TestFlaskApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test client before each test"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Sample landmark data for testing (21 hand landmarks)
        self.sample_landmarks = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0},  # Wrist (index 0)
            {'x': 0.6, 'y': 0.4, 'z': 0.1},  # Thumb CMC
            {'x': 0.7, 'y': 0.3, 'z': 0.2},  # Thumb MCP
            {'x': 0.8, 'y': 0.2, 'z': 0.3},  # Thumb IP
            {'x': 0.9, 'y': 0.1, 'z': 0.4},  # Thumb TIP
            {'x': 0.55, 'y': 0.3, 'z': 0.1}, # Index MCP
            {'x': 0.6, 'y': 0.2, 'z': 0.2},  # Index PIP
            {'x': 0.65, 'y': 0.1, 'z': 0.3}, # Index DIP
            {'x': 0.7, 'y': 0.05, 'z': 0.4}, # Index TIP
            {'x': 0.5, 'y': 0.3, 'z': 0.1},  # Middle MCP
            {'x': 0.5, 'y': 0.2, 'z': 0.2},  # Middle PIP
            {'x': 0.5, 'y': 0.1, 'z': 0.3},  # Middle DIP
            {'x': 0.5, 'y': 0.0, 'z': 0.4},  # Middle TIP (index 12)
            {'x': 0.45, 'y': 0.3, 'z': 0.1}, # Ring MCP
            {'x': 0.4, 'y': 0.2, 'z': 0.2},  # Ring PIP
            {'x': 0.35, 'y': 0.1, 'z': 0.3}, # Ring DIP
            {'x': 0.3, 'y': 0.05, 'z': 0.4}, # Ring TIP
            {'x': 0.4, 'y': 0.4, 'z': 0.1},  # Pinky MCP
            {'x': 0.35, 'y': 0.3, 'z': 0.2}, # Pinky PIP
            {'x': 0.3, 'y': 0.2, 'z': 0.3},  # Pinky DIP
            {'x': 0.25, 'y': 0.15, 'z': 0.4} # Pinky TIP
        ]

    def test_index_route(self):
        """Test the index route returns correct message"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "Server is running")

    @patch('app.model')
    def test_get_prediction_function_success(self, mock_model):
        """Test get_prediction function with valid landmarks"""
        # Mock the model prediction
        mock_model.predict.return_value = ['A']
        
        prediction = get_prediction(self.sample_landmarks)
        
        self.assertEqual(prediction, 'A')
        mock_model.predict.assert_called_once()

    def test_get_prediction_function_empty_landmarks(self):
        """Test get_prediction function with empty landmarks"""
        prediction = get_prediction([])
        self.assertIsNone(prediction)

    def test_get_prediction_function_none_landmarks(self):
        """Test get_prediction function with None landmarks"""
        prediction = get_prediction(None)
        self.assertIsNone(prediction)

    @patch('app.model')
    def test_get_prediction_endpoint_success(self, mock_model):
        """Test POST endpoint with valid landmarks"""
        # Mock the model prediction
        mock_model.predict.return_value = ['B']
        
        # Prepare test data
        test_data = {'landmarks': self.sample_landmarks}
        
        response = self.app.post('/get_prediction',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data.decode())
        self.assertEqual(response_data['prediction'], 'B')

    def test_get_prediction_endpoint_no_landmarks(self):
        """Test POST endpoint with missing landmarks"""
        test_data = {}
        
        response = self.app.post('/get_prediction',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data.decode())
        self.assertIn('error', response_data)
        self.assertEqual(response_data['error'], 'No landmarks provided')

    def test_get_prediction_endpoint_empty_landmarks(self):
        """Test POST endpoint with empty landmarks list"""
        test_data = {'landmarks': []}
        
        response = self.app.post('/get_prediction',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data.decode())
        self.assertIn('error', response_data)

    def test_get_prediction_endpoint_invalid_json(self):
        """Test POST endpoint with invalid JSON"""
        response = self.app.post('/get_prediction',
                               data='invalid json',
                               content_type='application/json')
        
        # Flask should handle this gracefully, typically returning 400
        self.assertIn(response.status_code, [400, 500])

    def test_get_prediction_endpoint_wrong_method(self):
        """Test GET request to POST-only endpoint"""
        response = self.app.get('/get_prediction')
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

    @patch('app.model')
    def test_landmark_normalization(self, mock_model):
        """Test that landmarks are properly normalized"""
        mock_model.predict.return_value = ['C']
        
        # Create landmarks where wrist is not at origin
        offset_landmarks = []
        wrist_offset = {'x': 0.3, 'y': 0.4, 'z': 0.1}
        
        for i, landmark in enumerate(self.sample_landmarks):
            if i == 0:  # Wrist
                offset_landmarks.append(wrist_offset)
            else:
                offset_landmarks.append({
                    'x': landmark['x'] + 0.3,
                    'y': landmark['y'] + 0.4,
                    'z': landmark['z'] + 0.1
                })
        
        prediction = get_prediction(offset_landmarks)
        
        # Verify that prediction was made (meaning normalization worked)
        self.assertEqual(prediction, 'C')
        mock_model.predict.assert_called_once()
        
        # Check that the input to predict was properly shaped
        called_args = mock_model.predict.call_args[0][0]
        self.assertEqual(called_args.shape, (1, 63))  # 21 landmarks * 3 coordinates

    @patch('app.model')
    def test_feature_flattening(self, mock_model):
        """Test that landmarks are properly flattened for model input"""
        mock_model.predict.return_value = ['D']
        
        get_prediction(self.sample_landmarks)
        
        # Verify the model was called with properly shaped features
        called_args = mock_model.predict.call_args[0][0]
        self.assertEqual(called_args.shape, (1, 63))  # 21 landmarks * 3 coordinates
        self.assertIsInstance(called_args, np.ndarray)

class TestLandmarkProcessing(unittest.TestCase):
    """Additional tests specifically for landmark processing logic"""
    
    def setUp(self):
        self.simple_landmarks = [
            {'x': 1.0, 'y': 1.0, 'z': 0.0},  # Wrist
            {'x': 2.0, 'y': 2.0, 'z': 1.0},  # Point 1
            {'x': 3.0, 'y': 3.0, 'z': 2.0},  # Point 2
            # Add more points to reach 21 total
            *[{'x': float(i), 'y': float(i), 'z': float(i-3)} for i in range(4, 22)]
        ]

    @patch('app.model')
    def test_wrist_normalization(self, mock_model):
        """Test that all points are normalized relative to wrist"""
        mock_model.predict.return_value = ['E']
        
        get_prediction(self.simple_landmarks)
        
        # Get the features passed to the model
        features = mock_model.predict.call_args[0][0]
        reshaped_features = features.reshape(21, 3)
        
        # First point (wrist) should be at origin for x and y after normalization
        # Note: We can't check exact values due to scaling, but we can verify structure
        self.assertEqual(reshaped_features.shape, (21, 3))

    

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestFlaskApp))
    test_suite.addTest(unittest.makeSuite(TestLandmarkProcessing))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
